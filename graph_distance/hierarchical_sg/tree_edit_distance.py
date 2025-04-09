from zss import simple_distance, Node
from collections import defaultdict
import json
from ..util import iou

"""
Tree edit distance using Zhang-Shasha algorithm.
"""


def make_node(obj):
    """
    Creates a zss node from a given object.
    Args:
        obj (dict): The object to create a node from.

    Note that the label for each node is a json string with the id, name, bbox, and predicate of the object.
    """
    info = {
        "id": obj["id"],
        "name": obj["name"],
        "bbox": obj["bbox"],
        "predicate": obj["predicate"],
    }

    label = json.dumps(info)

    node = Node(label)

    for child in obj["children"]:
        child_node = make_node(child)
        node.addkid(child_node)

    return node


def label_dist(a, b):
    """
    Computes the label distance between two objects.

    Cost definition:
    - Add one if the id, name, predicate, or bbox are different.
    - A bbox is considered different if the iou is less than 0.5.
    """
    a = json.loads(a) if a else defaultdict(lambda: "")
    b = json.loads(b) if b else defaultdict(lambda: "")

    cost = 0

    for k in ["id", "name", "predicate"]:
        if a[k] != b[k]:
            cost += 1

    if not a["bbox"] or not b["bbox"]:
        cost += 1
    else:
        iou_score = iou(a["bbox"], b["bbox"])
        if iou_score < 0.5:
            cost += 1

    return cost


def tree_edit_distance(a, b):
    """
    Computes the tree edit distance between two trees.
    """
    a_node = make_node(a)
    b_node = make_node(b)

    # Compute the tree edit distance
    distance = simple_distance(a_node, b_node)

    return distance
