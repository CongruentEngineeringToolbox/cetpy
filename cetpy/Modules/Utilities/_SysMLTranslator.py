import networkx as nx


class SysMLTranslator:
    """
    Translates a cetpy model structure networkX DiGraph representing a system structure into SysML 2.0 textual notation.

    Known Issues
    ------------
    - Does not translate reference properties.
    - Does not translate constraints.
    """

    def __init__(self, graph: nx.DiGraph):
        """
        Initializes the translator with a given system structure graph.
        The graph is expected to contain nodes representing blocks, attributes, and ports,
        and edges representing different types of relationships.
        """
        self.graph = graph
        self.sysml_output = []

    def translate(self):
        """
        Translates the entire system structure into SysML 2.0 notation.
        This function serves as the entry point for the translation process.
        """
        self.sysml_output.append("package SystemModel {")
        self._translate_blocks()
        self.sysml_output.append("}")
        return "\n".join(self.sysml_output)

    def _translate_blocks(self):
        """
        Processes and translates all blocks found in the graph.
        Blocks are the primary structural components in SysML and encapsulate attributes and ports.
        """
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "block":
                self.sysml_output.append(f"  part {data.get('name')} {{")
                self._translate_attributes(node)
                self._translate_ports(node)
                self.sysml_output.append("  }")

    def _translate_attributes(self, block):
        """
        Translates attributes associated with a given block.
        Attributes define specific properties of a block and are linked through composition relationships.
        """
        for neighbor in self.graph.neighbors(block):
            edge_data = self.graph.get_edge_data(block, neighbor)
            if edge_data and edge_data.get("type") == "composition":
                neighbor_data = self.graph.nodes[neighbor]
                if neighbor_data.get("type") == "block":
                    self.sysml_output.append(f"    part {neighbor_data.get('name')};")
                elif neighbor_data.get("type") == "attribute":
                    self.sysml_output.append(f"    attribute {neighbor_data.get('name')};")

    def _translate_ports(self, block):
        """
        Translates ports associated with a given block.
        Ports define interaction points for blocks and may be shared between multiple blocks.
        """
        for neighbor in self.graph.neighbors(block):
            edge_data = self.graph.get_edge_data(block, neighbor)
            if self.graph.nodes[neighbor].get("type") == "port":
                port_name = self.graph.nodes[neighbor].get(
                    "upstream_name" if block in self.graph.predecessors(neighbor) else "downstream_name")
                self.sysml_output.append(f"    port {port_name};")
