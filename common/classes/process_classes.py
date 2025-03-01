"""
Processes the raw text files to generate a dictionary of classes and their IDs.

Output format:
{
    "objects": [(0, "tail"), (1, "television"), ...],
    "attributes": [(0, "alive"), (1, "aluminum"), ...],
    "predicates": [(0, "above"), (1, "across from"), ...],
}

These classes are borrowed from https://cs.stanford.edu/people/jcjohns/cvpr15_supp/.
"""

import json

def process(raw):
    """
    Convert raw text to list of words.
    """
    result = []
    raw = raw.strip()
    for line in raw.split("\n"):
        result += line.split("\t")
    print(f"Found {len(result)} words")
    return result

with open("objects_raw.txt") as f:
    objects_raw = f.read()
with open("attributes_raw.txt") as f:
    attributes_raw = f.read()
with open("predicates_raw.txt") as f:
    predicates_raw = f.read()

object_bank = process(objects_raw)
attribute_bank = process(attributes_raw)
predicate_bank = process(predicates_raw)

id_bank = {
    "objects": list(enumerate(object_bank)),
    "attributes": list(enumerate(attribute_bank)),
    "predicates": list(enumerate(predicate_bank)),
}

with open("classes.json", "w") as f:
    json.dump(id_bank, f)
print("Classes saved to classes.json")
