"""
Utility functions for drawing scene graphs.
"""

from PIL import ImageDraw
import random
import json

with open("common/classes/classes.json") as f:
    id_bank = json.load(f)

# a list of colors for drawing
colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'pink',
    'purple',
    'brown',
    'gray',
    'beige',
    'turquoise',
    'cyan',
    'magenta',
    'lime',
    'navy',
    'maroon',
    'teal',
    'olive',
    'coral',
    'lavender',
    'violet',
    'gold',
    'silver'
]

def denormalize(bounding_box, width, height):
    """
    Given a set of normalized coordinates of a bounding box, returns the absolute coordinates.

    Gemini works in normalized coordinates, so we need to process it to draw it.

    Takes (y1, x1, y2, x2) in a normalized format.

    Returns (x1, y1, x2, y2) relative to the original image.
    """
    y1, x1, y2, x2 = bounding_box
    res = [x1/1000 * width, y1/1000 * height, x2/1000 * width, y2/1000 * height]
    return (int(z) for z in res)

def plot_bounding_boxes(img, object_tuples, color=None, font_size=None, line_width=None):
    """
    Plots bounding boxes on an image with markers for each noun phrase, using PIL, normalized coordinates, and different colors.

    Args:
    - img: the PIL image.
    - object_tuples: A list of tuples containing the noun phrases and their positions in normalized [y1 x1 y2 x2] format.
    - color (str): The color to use for the lines and text. If not provided, a random color is chosen for each relationship.
    - font_size (int): The font size to use for the text. If not provided, it is calculated based on the image width.
    - line_width (int): The width of the lines. If not provided, it is calculated based on the image width.
    """
    width, height = img.size

    if not font_size:
        font_size = width // 60
    if not line_width:
        line_width = width // 600

    draw = ImageDraw.Draw(img)

    for object_id, bounding_box in object_tuples:
        _color = colors[random.randint(0, len(colors)-1)] if color is None else color

        # denormalize coordinates (see helper)
        abs_x1, abs_y1, abs_x2, abs_y2 = denormalize(bounding_box, width, height)

        # draw bounding box
        draw.rectangle(
            ((abs_x1, abs_y1), (abs_x2, abs_y2)),
            outline=_color,
            width=line_width
        )

        # draw text
        draw.text((abs_x1 + 8, abs_y1 + 6), object_id, fill=_color, font_size=font_size)

def center(bounding_box):
    """
    Returns the center coordinate of the bounding box.

    Assumes the bounding box is already denormalized.

    Returns:
    - Center point as (x, y)
    """
    x1, y1, x2, y2 = bounding_box
    return (x1 + x2) // 2, (y1 + y2) // 2

def draw_relationships(image, graph, label="full", color=None, font_size=None, line_width=None, verbose=False):
    """
    Draws relationship lines on an image between bounding boxes.

    Args:
    - image (PIL.Image): The image to draw on.
    - graph (dict): The graph object containing the objects and relationships.
    - label (str): The type of label to draw. Can be "full" or "short".
    - color (str): The color to use for the lines and text. If not provided, a random color is chosen for each relationship.
    - font_size (int): The size of the font to use.
    - line_width (int): The width of the line to draw.
    - verbose (bool): Whether to print errors.

    Returns:
    - PIL.Image: The image with relationships drawn.
    """
    width, height = image.size

    if not font_size:
        font_size = width // 60
    if not line_width:
        line_width = width // 600
    
    errors = 0

    draw = ImageDraw.Draw(image)

    for i, (obj_id1, obj_id2, relation_id) in enumerate(graph["relations"]):
        relation_name = id_bank["predicates"][relation_id][1]

        if obj_id1 not in graph["objects"] or obj_id2 not in graph["objects"]:
            if verbose:
                if obj_id1 not in graph["objects"]:
                    print(f"Skipping relation {obj_id1} {relation_name} {obj_id2} because {obj_id1} not found in objects")
                if obj_id2 not in graph["objects"]:
                    print(f"Skipping relation {obj_id1} {relation_name} {obj_id2} because {obj_id2} not found in objects")
            errors += 1
            continue
        
        _color = colors[random.randint(0, len(colors)-1)] if color is None else color

        # get center of both objects
        bb1 = graph["objects"][obj_id1]["bounding_box"]
        bb2 = graph["objects"][obj_id2]["bounding_box"]
        center1 = center(denormalize(bb1, width, height))
        center2 = center(denormalize(bb2, width, height))

        # compute midpoint for labeling
        mid_x = (center1[0] + center2[0]) // 2
        mid_y = (center1[1] + center2[1]) // 2

        # get categories
        c1 = graph["objects"][obj_id1]["category"]
        c2 = graph["objects"][obj_id2]["category"]

        if c1 >= len(id_bank["objects"]) or c2 >= len(id_bank["objects"]):
            if verbose:
                    if c1 >= len(id_bank["objects"]):
                        print(f"Skipping relation {obj_id1} {relation_name} {obj_id2} because id {c1} not found in object bank")
                    if c2 >= len(id_bank["objects"]):
                        print(f"Skipping relation {obj_id1} {relation_name} {obj_id2} because id {c2} not found in object bank")
            errors += 1
            continue

        # draw line connecting centers
        draw.line([center1, center2], fill=_color, width=line_width)

        # draw relation text
        relation_text = f"{id_bank['objects'][c1][1]} {relation_name} {id_bank['objects'][c2][1]}" if label == "full" else relation_name
        draw.text((mid_x, mid_y), relation_text, fill=_color, font_size=font_size)
    
    if verbose and errors > 0:
        print(f"Skipped {errors} relations")
    
    return errors
