"""
SysML Block Test
=================

This file implements tests for a basic SysML Block.
"""

from typing import Callable
from logging import Logger
import pytest

from cetpy.Modules.SysML import Block
from cetpy.Modules.Report import Report


class TestBlock:

    @pytest.fixture
    def test_class(self) -> type(Block):
        return Block

    @pytest.fixture
    def init_kwargs(self) -> dict:
        return {'name': 'test'}

    @pytest.fixture
    def instance(self, test_class, init_kwargs) -> Block:
        return test_class(**init_kwargs)

    @pytest.fixture
    def child_pair(self, instance) -> (Block, Block):
        child1 = Block('child1', parent=instance)
        return instance, child1

    @pytest.fixture
    def child_pair2(self, instance) -> (Block, Block, Block):
        child1 = Block('child1', parent=instance)
        child2 = Block('child2', parent=instance)
        return instance, child1, child2

    @pytest.fixture
    def parent_pair(self, instance) -> (Block, Block):
        parent = Block('parent')
        instance.parent = parent
        return instance, parent

    def test_initialisation(self, instance):
        assert instance is not None

    def test_initialisation2(self, instance):
        assert isinstance(instance.name, str)
        assert isinstance(instance.name_display, str)
        assert isinstance(instance.abbreviation, str)
        assert isinstance(instance.tolerance, float)
        assert isinstance(instance._get_init_parameters, Callable)
        assert isinstance(instance.parts, list)
        assert isinstance(instance.solvers, list)
        assert isinstance(instance.ports, list)
        assert isinstance(instance.report, Report)
        assert instance.report._parent is instance
        assert isinstance(instance._logger, Logger)

    def test_parent_initialisation(self, parent_pair):
        instance, parent = parent_pair
        assert instance.parent is parent
        assert instance in parent.parts
        assert instance.tolerance <= parent.tolerance

    def test_part_initialisation(self, child_pair):
        instance, child1 = child_pair
        assert child1.parent is instance
        assert child1 in instance.parts
        assert child1.tolerance <= instance.tolerance

    def test_part_initialisation2(self, child_pair2):
        instance, child1, child2 = child_pair2
        assert child1.parent is instance
        assert child2.parent is instance
        assert child1 in instance.parts
        assert child2 in instance.parts
        assert child1.tolerance <= instance.tolerance
        assert child2.tolerance <= instance.tolerance

    def test_parent_unset(self, parent_pair, mocker):
        instance, parent = parent_pair
        instance_reset = mocker.spy(instance, 'reset_self')
        parent_reset = mocker.spy(parent, 'reset_self')
        assert instance.parent is parent
        assert instance in parent.parts
        instance.parent = None
        assert instance.parent is None
        assert instance not in parent.parts
        instance_reset.assert_called_once()
        parent_reset.assert_called_once()

    def test_parent_replace(self, parent_pair, mocker):
        instance, parent = parent_pair
        new_parent = Block('new_parent')
        instance_reset = mocker.spy(instance, 'reset_self')
        parent_reset = mocker.spy(parent, 'reset_self')
        new_parent_reset = mocker.spy(new_parent, 'reset_self')
        assert instance.parent is parent
        assert instance in parent.parts
        instance.parent = new_parent
        assert instance not in parent.parts
        assert instance.parent is new_parent
        assert instance in new_parent.parts
        instance_reset.assert_called_once()
        parent_reset.assert_called_once()
        new_parent_reset.assert_called_once()

    def test_report(self, instance, capsys):
        instance()
        captured = capsys.readouterr()
        assert captured.out is not None

    def test_child_reset(self, child_pair, mocker):
        instance, child1 = child_pair
        instance_reset = mocker.spy(instance, 'reset_self')
        child1_reset = mocker.spy(child1, 'reset_self')
        instance.reset()
        instance_reset.assert_called_once()
        child1_reset.assert_called_once()

    def test_parent_reset(self, parent_pair, mocker):
        instance, parent = parent_pair
        instance_reset = mocker.spy(instance, 'reset_self')
        parent_reset = mocker.spy(parent, 'reset_self')
        instance.reset()
        instance_reset.assert_called_once()
        parent_reset.assert_called_once()

    def test_child_hard_reset(self, child_pair, mocker):
        instance, child1 = child_pair
        instance_reset = mocker.spy(instance, 'hard_reset')
        child1_reset = mocker.spy(child1, 'hard_reset')
        instance.hard_reset()
        instance_reset.assert_called_once()
        child1_reset.assert_called_once()

    def test_parent_hard_reset(self, parent_pair, mocker):
        instance, parent = parent_pair
        instance_reset = mocker.spy(instance, 'hard_reset')
        # Explicitly not hard_reset
        parent_reset = mocker.spy(parent, 'reset_self')
        instance.hard_reset()
        instance_reset.assert_called_once()
        parent_reset.assert_called_once()

    def test_child_tolerance(self, child_pair):
        instance, child1 = child_pair
        instance.tolerance = 1e-4
        assert instance.tolerance == 1e-4
        assert child1.tolerance == 1e-4
        instance.tolerance = 1e-5
        assert instance.tolerance == 1e-5
        assert child1.tolerance == 1e-5

