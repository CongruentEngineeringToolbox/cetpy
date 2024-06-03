"""
RegressionDataSet
=======

This file extends the cetpy DataSet to add further functions for training regression models.
"""

from typing import List, Any
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS, RegressionResults
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import keras
from keras import backend as K
import keras_tuner

from cetpy.CaseTools import DataSet


class RegressionDataSet(DataSet):
    """Congruent Engineering Toolbox RegressionDataSet, a wrapper around a pandas DataFrame with additional analysis
    functionality. The RegressionDataSet adds further functions for training regression models."""

    # region Extensions
    def get_ols(self, output_key: str,
                input_keys: str | List[str] | None = None,
                add_constant: bool = True) -> RegressionResults:
        """Return Ordinary Least Squares (OLS) model of the dataset."""
        x, y = self.get_model_xy(output_key, input_keys, add_constant)
        return OLS(y, x).fit()

    def get_random_forest_regressor(
            self, output_key: str,
            input_keys: str | List[str] | None = None,
            add_constant: bool = True, **kwargs) -> RandomForestRegressor:
        """Return Scikit-Learn Random Forest Regressor of the dataset."""
        regressor = RandomForestRegressor(**kwargs)
        regressor.fit(*self.get_model_xy(output_key, input_keys, add_constant))
        return regressor

    def get_mlp_regressor(
            self, output_key: str,
            input_keys: str | List[str] | None = None,
            add_constant: bool = True, **kwargs) -> MLPRegressor:
        """Return Scikit-Learn MLP Neural Network Regressor of the dataset."""
        regressor = MLPRegressor(**kwargs)
        regressor.fit(*self.get_model_xy(output_key, input_keys, add_constant))
        return regressor

    @staticmethod
    def __get_compiled_keras_regressor__(
            n_output_dim: int,
            normalizer: keras.layers.Normalization,
            out_denormalizer: keras.layers.Normalization,
            n_hidden_layers: int,
            dense_layer_units: int,
            dense_activation: str = 'relu6',
            learning_rate: float = 1e-2,
            loss: str = 'mean_squared_error') -> keras.Model:
        """Get Tensorflow Keras regressor."""
        K.clear_session()
        model = keras.Sequential([
            normalizer,
            *[keras.layers.Dense(dense_layer_units, activation=dense_activation, name='dense_' + str(n))
              for n in range(n_hidden_layers)],
            keras.layers.Dense(n_output_dim, name='dense_output'),
            out_denormalizer
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=loss)

        return model

    @staticmethod
    def __get_keras_normalizers__(
            X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame
            ) -> (keras.layers.Normalization, keras.layers.Normalization):
        """Return keras normalisation and de-normalisation layers for deep neural networks."""
        # Normalize to mean 0, variance 1
        normalizer = keras.layers.Normalization(name='normalizer')
        normalizer.adapt(np.asarray(X).astype(float))
        # De-Normalize from mean 0, variance 1
        de_normalizer = keras.layers.Normalization(invert=True, name='de-normalizer')
        if len(X.shape) == 2:
            normalizer.adapt(np.asarray(X).astype(float))
        else:
            normalizer.adapt(np.atleast_2d(X).T.astype(float))
        if len(y.shape) == 2:
            de_normalizer.adapt(np.asarray(y).astype(float))
        else:
            de_normalizer.adapt(np.atleast_2d(y).T.astype(float))
        return normalizer, de_normalizer

    def get_keras_regressor(
            self, output_key: str,
            input_keys: str | List[str] | None = None,
            add_constant: bool = True,
            test_size: float = 0.1,
            n_hidden_layers: int = None,
            dense_units: int = None,
            batch_size: int = None,
            epochs: int = None,
            ** kwargs) -> (keras.Model, Any):
        """Get Tensorflow Keras regressor."""
        X, y = self.get_model_xy(output_key, input_keys, add_constant)
        n_input_dim = X.shape[1]
        try:
            n_output_dim = y.shape[1]
        except IndexError:
            n_output_dim = 1
        if n_hidden_layers is None:
            n_hidden_layers = 2
        if dense_units is None:
            dense_units = int(2 ** (np.ceil(np.log2(n_input_dim)) + 2))
        if batch_size is None:
            batch_size = X.shape[0]
        if epochs is None:
            epochs = 500

        normalizer, de_normaliser = self.__get_keras_normalizers__(X, y)

        model = self.__get_compiled_keras_regressor__(
            n_output_dim, normalizer, de_normaliser, n_hidden_layers, dense_units,
            dense_activation=kwargs.get('dense_activation', 'relu6'), learning_rate=kwargs.get('learning_rate', 1e-2),
            loss=kwargs.get('loss', 'mean_squared_error')
        )

        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]

        history = model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=test_size, callbacks=callbacks)

        return model, history

    def get_keras_regressor_optimized(
            self, output_key: str,
            input_keys: str | List[str] | None = None,
            add_constant: bool = True,
            test_size: float = 0.1,
            batch_size: int = None,
            optimization_epochs: int = None,
            final_epochs: int = None,
            tuner_model: str = 'bayesian',
            n_hidden_layers: List[int] | int | None = None,
            dense_units: List[int] | int | None = None,
            dense_activation: List[str] | str | None = None,
            learning_rate: List[float] | float | None = None,
            **kwargs) -> keras.Model:
        """Get Tensorflow Keras regressor."""
        X, y = self.get_model_xy(output_key, input_keys, add_constant)
        try:
            n_output_dim = y.shape[1]
        except IndexError:
            n_output_dim = 1
        if batch_size is None:
            batch_size = X.shape[0]
        if optimization_epochs is None:
            optimization_epochs = 100
        if final_epochs is None:
            final_epochs = 500
        if n_hidden_layers is None:
            n_hidden_layers = [1, 4, 1]
        if dense_units is None:
            dense_units = [16, 1024, 16]
        if dense_activation is None:
            dense_activation = ['relu6', 'tanh']
        if learning_rate is None:
            learning_rate = [1e-4, 0.25]

        normalizer, de_normaliser = self.__get_keras_normalizers__(X, y)

        def build_model(hp):
            if isinstance(n_hidden_layers, list):
                n_layers = hp.Int(
                    'n_hidden_layers', min_value=n_hidden_layers[0], max_value=n_hidden_layers[1],
                    step=n_hidden_layers[2])
            else:
                n_layers = n_hidden_layers
            if isinstance(dense_units, list):
                units = hp.Int(
                    'dense_units', min_value=dense_units[0], max_value=dense_units[1], step=dense_units[2])
            else:
                units = dense_units
            if isinstance(dense_activation, list):
                activation = hp.Choice('dense_activation', values=dense_units)
            else:
                activation = dense_activation
            if isinstance(learning_rate, list):
                l_rate = hp.Float('learning_rate', min_value=learning_rate[0], max_value=learning_rate[1])
            else:
                l_rate = learning_rate

            return self.__get_compiled_keras_regressor__(
                n_output_dim, normalizer, de_normaliser, n_layers, units,
                dense_activation=activation, learning_rate=l_rate, loss=kwargs.get('loss', 'mean_squared_error')
            )

        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]

        if tuner_model == 'bayesian':
            tuner = keras_tuner.tuners.BayesianOptimization(
                build_model,
                objective=keras_tuner.Objective('val_loss', direction="min"),
                max_trials=kwargs.get('max_trials', 200),
                overwrite=True)

            tuner.search(X, y, batch_size=batch_size, epochs=optimization_epochs, validation_split=test_size,
                         callbacks=callbacks)

        elif tuner_model == 'hyperband':
            tuner = keras_tuner.tuners.Hyperband(
                build_model,
                objective='val_loss',
                max_epochs=optimization_epochs,
                executions_per_trial=kwargs.get('executions_per_trial', 2),
                overwrite=True)

            tuner.search(X, y, batch_size=batch_size, validation_split=test_size, callbacks=callbacks)
        else:
            raise ValueError("Unrecognised tuner_model, must be 'bayesian' or 'hyperband'.")

        model = tuner.get_best_models(num_models=2)[0]

        if final_epochs > optimization_epochs:
            # Recreate the callbacks to ensure they can be deep-copied.
            callbacks = [keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True)]

            model.fit(X, y, batch_size=batch_size,  epochs=final_epochs, validation_split=test_size,
                      callbacks=callbacks)

        return model
    # endregion
