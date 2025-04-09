def simple_print(node, level=0):
    """
    Recursively prints the node and its children in a tree-like structure.
    """
    tabs = "\t" * level
    print(
        f"{tabs}{node['name']} ({node['id']}) {node['predicate']}: {node['attributes']}"
    )

    for child in node["children"]:
        simple_print(child, level + 1)


def parse_tree(graph):
    """
    Parses the graph and returns the objects and relationships.
    """
    objects = {}
    relationships = {}

    def _parse_tree_helper(node, parent=None):
        if not node:
            return

        # add object
        objects[node["id"]] = {
            "name": node["name"],
            "bbox": node["bbox"],
        }

        # add edge to parent
        if parent and node["predicate"] != "root":
            relationships[(node["id"], parent["id"])] = node["predicate"]

        for child in node["children"]:
            _parse_tree_helper(child, node)

    _parse_tree_helper(graph)
    return objects, relationships
