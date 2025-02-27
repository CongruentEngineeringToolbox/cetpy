"""
ModelStructureGraph
===================


This file specifies methods for generating and modifying structure graphs of the model.

The graph structure uses the networkx Graph package.
"""
import networkx as nx

node_colors = {
    'block': 'blue',
    'solver': 'red',
    'port': 'green',
    'attribute': 'orange'
}

edge_colors = {
    'composition': 'blue',
    'aggregation': 'orange',
    'flow': 'green',
    'attribute': 'black'
}


def get_model_structure_graph(
        block,
        include_solvers: bool = True,
        include_ports: bool = True,
        include_parts: bool = True,
        include_references: bool = True,
        include_value_properties: bool = False,
        include_connected_blocks: bool = False,
        include_instances: bool = False,
        graph: nx.DiGraph = None,
        root_index: int = 0) -> nx.DiGraph:
    """Generate a networkx directed graph of the model elements and connections."""
    if graph is None:
        graph = nx.DiGraph()
    ni = root_index * 1  # Non-Mutable, Node index incremented with each initialization to avoid double registries.
    if include_instances:
        block_instance = block
    else:
        block_instance = None

    graph.add_node(ni, type='block', uuid=block.__uuid__, instance=block_instance,
                   **block.report.get_header_attributes())
    ni += 1
    if include_solvers:
        for p in block.solvers:
            uuid = p.__uuid__  # Solvers can be shared
            matching_nodes = [idx for idx, node in graph.nodes.items() if node.get('uuid', None) == uuid]
            if len(matching_nodes) == 0:
                graph.add_node(ni, type='solver', uuid=p.__uuid__, **p.report.get_header_attributes())
                graph.add_edge(root_index, ni, type='composition')
                idx_this_solver = ni * 1  # Non-Mutable
                ni += 1
            else:
                idx_existing_node = matching_nodes[0]
                idx_this_solver = idx_existing_node * 1  # Non-Mutable
                graph.add_edge(root_index, idx_existing_node, type='composition')

            # For continuous flow solvers that can be nested.
            if hasattr(p, 'parent_solver') and p.parent_solver is not None:
                uuid_parent = p.parent_solver.__uuid__
                matching_nodes = [idx for idx, node in graph.nodes.items() if node.get('uuid', None) == uuid_parent]
                if len(matching_nodes) == 0:
                    graph.add_node(ni, type='solver', uuid=p.__uuid__, **p.report.get_header_attributes())
                    graph.add_edge(ni, idx_this_solver, type='composition')
                    ni += 1
                else:
                    idx_existing_node = matching_nodes[0]
                    graph.add_edge(idx_existing_node, idx_this_solver, type='composition')

    if include_ports:
        for p in block.ports:
            uuid = p.__uuid__  # Ports are shared
            matching_nodes = [idx for idx, node in graph.nodes.items() if node.get('uuid', None) == uuid]
            if p.upstream is None:
                uuid_up = 'None'
            else:
                uuid_up = p.upstream.__uuid__
            if p.downstream is None:
                uuid_down = 'None'
            else:
                uuid_down = p.downstream.__uuid__

            if len(matching_nodes) == 0:
                graph.add_node(ni, type='port', uuid=p.__uuid__, uuid_up=uuid_up,
                               uuid_down=uuid_down, **p.report.get_header_attributes())
                if uuid_down == block.__uuid__:
                    graph.add_edge(ni, root_index, type='flow')
                else:
                    graph.add_edge(root_index, ni, type='flow')
                ni += 1
            else:
                idx_existing_node = matching_nodes[0]
                if uuid_down == block.__uuid__:
                    graph.add_edge(idx_existing_node, root_index, type='flow')
                else:
                    graph.add_edge(root_index, idx_existing_node, type='flow')

    if include_parts:
        for p in block.parts:
            if include_instances:
                block_instance = p
            else:
                block_instance = None

            # Check if the same part has already been added as a reference property, if so delete the previous
            # node and reconnect all existing edges to the new node once it has been added. Extract matching blocks
            # and edges before making any additions. Make the reconnections after the additions.
            uuid = p.__uuid__  # Ports are shared
            matching_nodes = [idx for idx, node in graph.nodes.items() if node.get('uuid', None) == uuid]
            if len(matching_nodes) == 0:
                matching_edges = []
            else:
                matching_edges = [(idx, edge) for idx, edge in graph.edges.items() if matching_nodes[0] in idx]

            graph.add_node(ni, type='block', uuid=p.__uuid__, instance=block_instance,
                           **p.report.get_header_attributes())
            graph.add_edge(root_index, ni, type='composition')
            _ = get_model_structure_graph(p, include_solvers=include_solvers, include_ports=include_ports,
                                          include_parts=include_parts, include_references=include_references,
                                          include_value_properties=include_value_properties,
                                          include_connected_blocks=True, include_instances=include_instances,
                                          graph=graph, root_index=ni * 1)  # Index as non-mutable for safety.
            # Connected blocks always included in recursive function and only checked once at the end.

            if len(matching_nodes) > 0:
                graph.remove_node(matching_nodes[0])
                # [graph.remove_edge(*idx) for idx, edge in matching_edges]
                # Replace the matching node in the list of edges
                matching_edges = [((ni, idx[1]), edge) if idx[0] == matching_nodes[0] else ((idx[0], ni), edge)
                                  for idx, edge in matching_edges]
                # Reconnect to the new node.
                [graph.add_edge(*idx, **edge) for idx, edge in matching_edges]

            ni = max([i for i, n in graph.nodes.items()])  # Recursive function in line above may add multiple nodes.
            ni += 1

    if include_references:
        for rp in block.report.reference_properties:
            try:
                p = rp.__get__(block)
            except ValueError:
                continue

            uuid = p.__uuid__  # Ports are shared
            matching_nodes = [idx for idx, node in graph.nodes.items() if node.get('uuid', None) == uuid]

            if len(matching_nodes) == 0:
                if include_instances:
                    block_instance = p
                else:
                    block_instance = None

                graph.add_node(ni, type='block', uuid=p.__uuid__, instance=block_instance,
                               **p.report.get_header_attributes())
                graph.add_edge(root_index, ni, type='aggregation')
                ni += 1
            else:
                idx_existing_node = matching_nodes[0]
                graph.add_edge(root_index, idx_existing_node, type='aggregation')

    if include_value_properties:
        for vp in block.report.value_properties:
            graph.add_node(ni, type='attribute', name=vp.name_display, unit=vp.unit)
            graph.add_edge(root_index, ni, type='aggregation')
            ni += 1

    if not include_connected_blocks:
        list_blocks = [(i, n) for i, n in graph.nodes.items() if n['type'] == 'block']
        list_composition_edges = [i for i, e in graph.edges.items() if e['type'] == 'composition']

        def add_contained_nodes(node_index: int, list_nodes) -> None:
            """Add composition nodes of the node at index to the node list as well as their composition nodes.

            Recursive function.
            """
            new_nodes = [e[1] for e in list_composition_edges if (e[0] == node_index and e[1] not in list_nodes)]
            list_nodes += new_nodes
            for n in new_nodes:
                add_contained_nodes(node_index=n, list_nodes=list_nodes)

        list_root_composition_nodes = [root_index]
        add_contained_nodes(node_index=root_index, list_nodes=list_root_composition_nodes)

        # Add ports with at least one connection in the existing list.
        list_first_degree_flow = [i for i, e in graph.edges.items() if e['type'] == 'flow'
                                  and (i[0] in list_root_composition_nodes or i[1] in list_root_composition_nodes)]
        list_root_composition_nodes += [i[0] for i in list_first_degree_flow if i[0] not in list_root_composition_nodes]
        list_root_composition_nodes += [i[1] for i in list_first_degree_flow if i[1] not in list_root_composition_nodes]

        # Inverse selection
        list_nodes_not_included = [i for i, n in list_blocks if i not in list_root_composition_nodes]

        [graph.remove_node(i) for i in list_nodes_not_included]
        removal_edges = [(e[0], e[1]) for e in graph.edges if (e[0] in list_nodes_not_included
                                                               or e[1] in list_nodes_not_included)]
        [graph.remove_edge(*e) for e in removal_edges]

    return graph
