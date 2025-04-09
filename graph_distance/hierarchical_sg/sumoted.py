import os
from graph_distance.hierarchical_sg.sumoted_external import sudoted_main

TEMP_DIR = "temp_sumoted"


def create_edge_file(t, filename):
    """
    Converts the input tree into an edge file format.
    """

    def _recurse(fp, node):
        for child in node["children"]:
            fp.write(f"{node['id']}, {child['id']}\n")
            _recurse(fp, child)

    fp = open(filename, "w+")
    _recurse(fp, t)
    fp.close()


def sumoted_distance(a, b, normalized=False):
    # create temp folder if not exists
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    try:
        # convert to edge file
        create_edge_file(a, f"{TEMP_DIR}/a.txt")
        create_edge_file(b, f"{TEMP_DIR}/b.txt")

        # run sumoted and get the result
        result, normalized_result = sudoted_main(TEMP_DIR)
        result = result[0,1]
        normalized_result = normalized_result[0,1]

    except Exception as e:
        print(f"Error: {e}")
        result = None
        normalized_result = None

    # finally:

    #     # delete temp
    #     for file in os.listdir(TEMP_DIR):
    #         os.remove(os.path.join(TEMP_DIR, file))
    #     os.rmdir(TEMP_DIR)

    return normalized_result if normalized else result


