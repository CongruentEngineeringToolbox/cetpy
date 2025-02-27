"""
ReferenceProperty
=============

This file implements a Reference Property for SysML Blocks. The reference property allows the block internals to
access information from another block with improved resilience and traceability. Specifically resilience through an
independence of changes to the relative structure between the blocks.

The model developer is able to specify a specific signature of the referenced block as well as a rule-set by which
the block may be found within the model structure graph. The connection is evaluated once and stored until the model
structure graph is rebuild, e.g. as a result of the addition or removal of a part.
"""

from typing import List
import numpy as np
import networkx as nx

from cetpy.Modules.Utilities.Labelling import name_2_display


class ReferencePropertyDoc:
    """Simple Descriptor to pass a more detailed doc string to a reference property."""

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        else:
            return f"{getattr(instance, '_name')} ReferenceProperty."

    def __set__(self, instance, value: str) -> None:
        pass


class ReferenceProperty:
    """ReferenceProperty that allows the linking of blocks independent of the relative model structure through a
    defined signature and search rule-set.

    The target block signature may be specified either by, in order of priority, name, class, or a list of required
    attributes. Search parameters are shared across all instances of a given block.

    Parameters
    ==========
    target_class: str | type | None, optional, default = None
        Optional class name of the target block. Not case-sensitive and compared to the full method resolution order
        of the blocks in the structure graph. The class can be specified by its name as a str or by directly passing
        a python type. If no class is specified, then the target_name or required_attribute_list are used to
        determine the applicability.

    target_name: str | None, optional, default = None
        Optional object name for the target block. Case-sensitive and compared to the name property of the blocks in
        the structure graph. This offers a faster and more specific but less flexible identification compared to the
        target_class specification. If no name is specified, either the target class or the required_attribute_list
        are used to determine applicability.

    required_attribute_list: List[str] | None, optional, default = None
        Optional list of attributes that the target block must have. The ReferenceProperty will select all cetpy
        Block objects which contain all listed attributes in accordance to the search criteria. This method of
        specifying the signature of the target block is the most flexible, but also risky. Another developer may use
        matching attribute names to express different values and trigger before the intended block type regardless.

    search_goal: str, optional, default = 'nearest'
        Choice of method to determine the most applicable target block. By default the nearst block is used. All
        options are:
            - nearest: closest graph distance
            - nearest_below: closest graph distance, but prioritising nodes below over nodes above
            - nearest_above: closest graph distance, but prioritising nodes above over nodes below
            - top_level: closest graph distance to the root graph node, the top-level system
            - fixed: evaluated from a user given object.
            - hard_coded_path: evaluated along the hard_coded_path from the current block

    hard_coded_path: str | None, optional, default = None
        str path from the owned block to the target block. E.g. 'parent.parent.subsystem_b.component_d'

    search_above: bool, optional, default = True
        Bool switch to allow the search above the current block. If disabled only internal parts will be search
        depending on the search_below setting. The search above can be further modified using the search_parts_above
        property.

    search_parts_above: bool, optional, default = True
        Bool switch to also search within the parts of the blocks above the current block. To illustrate this
        option: In a system of systems with a root system 'SystemA' and two sub-systems 'SystemB' and 'SystemC',
        then the ReferenceProperty systemC owned by subsystem 'SystemB' would behave as follows. If search_above is
        False, then the ReferenceProperty would only search within the systems contained in 'SystemB'. If
        search_above is True and search_parts_above is False, then it could also find 'SystemA'. If both search_above
        and search_parts_above are True, then it could navigate from 'SystemB' to its parent system 'SystemA' and
        then also search through its parts to find the desired 'SystemC'. If the target system is strictly in the
        parent system path to the root system, then it is recommended to turn this option off to significantly reduce
        the search scope.

    search_below: bool, optional, default = True
        Bool switch to allow the search within the parts of the target block. If disabled the internals will not be
        included in the search.

    maximum_distance: int, default = -1
        Maximum search distance as an integer of the number of edges from the current block. Set to -1 to disable.
        Disabled by default.

    allow_multiple: bool, optional, default = False
        Bool switch to allow the return of a list of objects. If enabled, all blocks matching the target signature
        within the search scope are returned, not just the first one. If enabled, the ReferenceProperty will always
        return a list, even if only one block or none at all are found.
    """
    __slots__ = ['_name', '_name_instance', 'target_class', 'target_name', 'required_attribute_list', 'search_goal',
                 'hard_coded_path', 'search_above', 'search_parts_above', 'search_below', 'maximum_distance',
                 'allow_multiple']

    __doc__ = ReferencePropertyDoc()

    def __init__(self, target_class: str | type | None = None,
                 target_name: str | None = None,
                 required_attribute_list: List[str] | None = None,
                 hard_coded_path: str | None = None,
                 search_goal: str = 'nearest',
                 search_above: bool = True,
                 search_parts_above: bool = True,
                 search_below: bool = True,
                 maximum_distance: int = -1,
                 allow_multiple: bool = False) -> None:
        self._name = ''
        self._name_instance = ''

        self.target_class = target_class
        self.target_name = target_name
        self.required_attribute_list = required_attribute_list

        self.search_goal = search_goal

        self.hard_coded_path = hard_coded_path
        self.search_above = search_above
        self.search_parts_above = search_parts_above
        self.search_below = search_below
        self.maximum_distance = maximum_distance
        self.allow_multiple = allow_multiple

    # region Getters and Setters
    def __set_name__(self, instance, name):
        self._name = name
        self._name_instance = '_' + name

    def str(self, instance) -> str:
        """Return formatted display name of the property value."""
        try:
            value = self.__get__(instance)
            return str(value)
        except ValueError:
            return self.name_display

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        value = None
        search_goal = self.search_goal
        allow_multiple = self.allow_multiple
        match search_goal:
            case 'fixed':
                value = getattr(instance, self._name_instance)
            case 'hard_coded_path':
                path = self.hard_coded_path
                if not isinstance(path, str):
                    raise ValueError(f"ReferenceProperty {self._name} of block {instance.name} search criteria is "
                                     f"hard_coded_path, but no str path is specified. A target block could not be "
                                     f"identified.")
                if path[0] == '.':  # Eliminate potential user error.
                    path = path[1:]
                value = instance.__deep_get_attr__(path)
            case 'nearest' | 'nearest_below' | 'nearest_above' | 'top_level':
                target_name = self.target_name
                target_class = self.target_class
                required_attribute_list = self.required_attribute_list

                search_below = self.search_below
                search_above = self.search_above
                search_parts_above = self.search_parts_above
                max_distance = self.maximum_distance

                if target_name is not None:
                    def valid_block(block: object) -> bool:
                        # noinspection PyUnresolvedReferences
                        return block.name == target_name
                elif target_class is not None:
                    def valid_block(block: object) -> bool:
                        return any([s.__name__.lower() == target_class.lower() for s in type(block).__mro__])
                elif required_attribute_list is not None:
                    def valid_block(block: object) -> bool:
                        return all([hasattr(block, a) for a in self.required_attribute_list])
                else:
                    def valid_block(block: object) -> bool:
                        # noinspection PyUnresolvedReferences
                        return block.name == self._name or any(
                            [s.__name__.lower() == self._name.lower().replace('_', '') for s in type(block).__mro__])

                # region Graph Pre-Processing
                graph = instance.composition_structure_graph

                # Generate a second graph without node content. This graph is selectively cut depending on the search
                # criteria without modifying the original graph. The original graph is used to test the validity of
                # the nodes using the full node data.
                graph_reduced = nx.DiGraph()
                graph_reduced.add_nodes_from(graph)
                graph_reduced.add_edges_from(graph.edges)

                idx_this_block = [i for i, n in graph.nodes.items() if n.get('instance', None) == instance][0]

                if not search_below:
                    # Remove immediate successors
                    for n in list(graph_reduced.successors(idx_this_block)):
                        graph_reduced.remove_node(n)
                if not search_above:
                    # Remove immediate predecessors
                    for n in list(graph_reduced.predecessors(idx_this_block)):
                        graph_reduced.remove_node(n)
                elif not search_parts_above:
                    def remove_different_predecessor_successors(node_id: int) -> None:
                        for i in list(graph_reduced.nodes[node_id].predecessors()):
                            for n in [j for j in graph_reduced.successors(i) if j != node_id]:
                                graph_reduced.remove_node(n)
                            remove_different_predecessor_successors(i)
                    remove_different_predecessor_successors(idx_this_block)
                # Remove disconnected nodes
                for n in [i for i in graph_reduced.nodes.keys()
                          if i not in nx.node_connected_component(graph_reduced.to_undirected(), idx_this_block)]:
                    graph_reduced.remove_node(n)
                # endregion

                valid_nodes = [i for i, n in graph.nodes.items()
                               if ('instance' in n.keys() and valid_block(n['instance']))
                               and i in graph_reduced.nodes]
                dist = [nx.shortest_path_length(graph_reduced.to_undirected(), idx_this_block, i) for i in valid_nodes]

                if max_distance != -1:
                    valid_nodes = [i for i, d in zip(valid_nodes, dist) if d < max_distance]

                if len(valid_nodes) == 0:
                    pass
                elif allow_multiple:
                    value = [graph.nodes[i]['instance'] for i in valid_nodes]
                else:
                    if search_goal == 'nearest' or (
                            search_goal in ['nearest_below', 'nearest_above']
                            and (False in [search_below, search_above])):
                        value = graph.nodes[valid_nodes[np.argmin(dist)]]['instance']
                    elif search_goal in ['nearest_below', 'nearest_above']:
                        # Create a new graph with just the nodes following on the current block. Then separate the
                        # nodes and distances by below and above
                        graph_below = nx.DiGraph()
                        graph_below.add_nodes_from(graph_reduced)
                        graph_below.add_edges_from(graph_reduced.edges)
                        graph_below.remove_node(graph_below.predecessors(idx_this_block))
                        # Remove disconnected nodes
                        for n in [i for i, n in graph_below.nodes
                                  if i not in nx.node_connected_component(graph_below.to_undirected(), idx_this_block)]:
                            graph_below.remove_node(n)

                        # split below and above
                        nodes_below = [(n, d) for n, d in zip(valid_nodes, dist) if n in graph_below.nodes]
                        nodes_above = [(n, d) for n, d in zip(valid_nodes, dist) if n not in graph_below.nodes]
                        closest_below = graph.nodes[nodes_below[np.argmin([d for n, d in nodes_below])]]['instance']
                        closest_above = graph.nodes[nodes_above[np.argmin([d for n, d in nodes_above])]]['instance']

                        if search_goal == 'nearest_below':
                            if closest_below is not None:
                                value = closest_below
                            else:
                                value = closest_above
                        else:
                            if closest_above is not None:
                                value = closest_above
                            else:
                                value = closest_below
                    elif search_goal == 'top_level':
                        dist_top = [nx.shortest_path_length(graph_reduced.to_undirected(), 0, i) for i in valid_nodes]
                        value = graph.nodes[valid_nodes[np.argmin(dist_top)]]['instance']

            case _:
                raise ValueError("Unrecognized search goal, must be one of: nearest, nearest_below, nearest_above, "
                                 "top_level, fixed, hard_coded_path")

        # region Output
        if value is None:
            if allow_multiple:
                return []
            else:
                raise ValueError(f"No block matching the desired signature could be found within the search criteria "
                                 f"of ReferenceProperty {self._name} of block {instance.name}. Please examine the "
                                 f"reference property settings and system structure or consider directly passing the "
                                 f"desired object.")
        if allow_multiple:
            if isinstance(value, list):
                return value
            else:
                return [value]
        else:
            return value
        # endregion

    def __set__(self, instance, value) -> None:
        setattr(instance, self._name_instance, value)
        self.search_goal = 'fixed'
    # endregion

    # region Labelling
    @property
    def name(self) -> str:
        """ReferenceProperty name"""
        return self._name

    @property
    def name_display(self) -> str:
        """Return the display formatted name of the value property."""
        return name_2_display(self._name)
    # endregion
