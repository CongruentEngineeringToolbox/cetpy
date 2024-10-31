"""
2D line plot functions
======================

This file defines a standardised format for cetpy Block 2D line plots for vector ValueProperty properties.
"""
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt

from cetpy.Modules.SysML import ValueProperty


def get_plot_2d_value_properties(
        instance,
        x: ValueProperty | str,
        y: ValueProperty | str | Iterable[ValueProperty | str],
        ax: plt.Axes = None,
        title: str = None,
        aspect1: bool = False,
        dpi: float = 150,
        **kwargs) -> (plt.Figure, plt.Axes, plt.Axes | None):
    """Plot a 2d line plot for an x and y value property set.

    Parameters
    ----------
    instance: Block
        The cetpy Block instance for which the ValueProperty values should be plotted.
    x: ValueProperty | str
        X values cetpy ValueProperty. Can be defined by the ValueProperty itself or its name within the class.
    y: ValueProperty | str | Iterable[ValueProperty | str]
        Y values cetpy ValueProperty. Can be defined by the ValueProperty itself or its name within the class. Must
        have at most two different axis labels and units.
    ax: plt.Axes, optional, default = None
        Optionally pass an existing Axes to plot to same figure.
    title: str, optional, default = None
        Manually set plot title.
    aspect1: bool, optional, default = False
        Bool trigger to set the aspect ratio to 1. Typically used for geometry plots without distortion.
    dpi: float, default = 150
        Dots per inch for plot resolution. If an axis is passed, this property is not used.
    kwargs: dict, optional
        Keyword arguments are passed on to the matplotlib plot command.

    Returns
    -------
    plt.Figure | None
        matplotlib figure object. None if an axis is passed into the function using the 'ax' argument.
    plt.Axes
        matplotlib Axes object containing the plot.
    plt.Axes | None
        matplotlib Axes object containing the plot of the second y-axis and associated plots if multiple axis labels
        or units are utilized. None otherwise.
    """
    # region convert to ValueProperty
    if isinstance(x, str):
        x = instance.__deep_get_vp__(x)

    if isinstance(y, Iterable) and len(y) == 1:
        y = y[0]  # Ensure iterable path is only used when more than one y ValueProperty is defined.
    if isinstance(y, str):
        y = instance.__deep_get_vp__(y)
    elif isinstance(y, Iterable):
        for i, y_i in enumerate(y):
            if isinstance(y_i, str):
                y[i] = instance.__deep_get_vp__(y_i)
    # endregion

    if isinstance(y, ValueProperty):
        axis_label = y.axis_label
        unit = y.unit_latex
        title_y_label = y.name_display
        twin_axis = False
        axis_labels = None
        units = None
        axis_label_2, unit_2 = None, None
    elif isinstance(y, Iterable):
        axis_labels = [y_i.axis_label for y_i in y]
        units = [y_i.unit_latex for y_i in y]

        # Test if max two axis labels and units.
        if len(np.unique(axis_labels)) > 2 or len(np.unique(units)) > 2:
            names = [y_i.name for y_i in y]
            raise ValueError("This plot functions supports a maximum of two sets of axis labels and units. However "
                             "more were detected.\n" + "\n".join(["{:20s}: {:15s}, {:10s}".format(n, l, u)
                                                                  for n, l, u in zip(names, axis_labels, units)]))
        elif len(np.unique(axis_labels)) > 1 or len(np.unique(units)) > 1:
            twin_axis = True
            if len(np.unique(axis_labels)) > 1:
                axis_label, axis_label_2 = np.unique(axis_labels)
            else:
                axis_label, axis_label_2 = axis_labels[0], axis_labels[0]  # Label shared by both axis
            if len(np.unique(units)) > 1:
                unit, unit_2 = np.unique(units)
            else:
                unit, unit_2 = units[0], units[0]  # Unit shared by both axis

            # Rudimentary pluralisation
            if axis_label[-1] == 's':
                title_y_label_1 = axis_label + 'es'
            else:
                title_y_label_1 = axis_label + 's'
            if axis_label_2[-1] == 's':
                title_y_label_2 = axis_label_2 + 'es'
            else:
                title_y_label_2 = axis_label_2 + 's'
            title_y_label = title_y_label_1 + ' and ' + title_y_label_2

        else:
            twin_axis = False
            axis_label = axis_labels[0]
            unit = units[0]
            axis_label_2, unit_2 = None, None

            # Rudimentary pluralisation
            if axis_label[-1] == 's':
                title_y_label = axis_label + 'es'
            else:
                title_y_label = axis_label + 's'

    else:
        raise ValueError("value_y must be ValueProperty or Iterable[ValueProperty].")

    if ax is None:
        fig, ax = plt.subplots(dpi=dpi)
        ax.grid()
    else:
        fig = None

    if twin_axis:
        ax2 = ax.twinx()
    else:
        ax2 = None

    if isinstance(y, ValueProperty):
        ax.plot(x.value(instance), y.value(instance), label=y.name, **kwargs)
        if fig is None:  # Already has lines on axis
            ax.legend()
    elif isinstance(y, Iterable):
        if not twin_axis:
            for val_y in y:
                ax.plot(x.value(instance), val_y.value(instance), label=val_y.name, **kwargs)
            lines, line_labels = ax.get_legend_handles_labels()
        else:
            label_1 = axis_labels[0]
            unit_1 = units[0]
            for val_y, l, u in zip(y, axis_labels, units):
                if l == label_1 and u == unit_1:
                    ax.plot(x.value(instance), val_y.value(instance), label=val_y.name, **kwargs)
                else:
                    ax2.plot(x.value(instance), val_y.value(instance), '--', label=val_y.name, **kwargs)
            lines, line_labels = ax.get_legend_handles_labels()
            lines_2, line_labels_2 = ax2.get_legend_handles_labels()
            lines = lines + lines_2
            line_labels = line_labels + line_labels_2
        ax.legend(lines, line_labels)

    if aspect1:
        # noinspection PyTypeChecker
        ax.set_aspect(1)

    ax.set_xlabel(x.axis_label + ", " + x.unit_latex)
    ax.set_ylabel(axis_label + ", " + unit)
    if ax2 is not None:
        ax2.set_ylabel(axis_label_2 + ", " + unit_2)

    if title is None:
        ax.set_title(instance.name_display + " " + title_y_label)
    else:
        ax.set_title(title)

    if fig is not None:
        fig.tight_layout()

    return fig, ax, ax2
