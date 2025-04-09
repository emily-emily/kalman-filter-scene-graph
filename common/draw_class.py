from PIL import ImageDraw, Image
import random


class Drawer:
    """
    USAGE
    ```
    img = Image.open("img.jpg")

    draw = Drawer(img)

    draw.add_objects(objects)
    draw.add_relationships(relationships)

    draw.draw()
    draw.show()
    ```

    INPUT FORMAT

    objects
    ```
    {
        "id": {
            "name": "asdf",
            "bbox": [y_min, x_min, y_max, x_max]
        },
        ...
    }
    ```

    relationships
    ```
    {
        (subject_id, object_id): "predicate",
        ...
    }
    ```
    """

    colors = [
        "red",
        "green",
        "blue",
        "yellow",
        "orange",
        "pink",
        "purple",
        "brown",
        "gray",
        "beige",
        "turquoise",
        "cyan",
        "magenta",
        "lime",
        "navy",
        "maroon",
        "teal",
        "olive",
        "coral",
        "lavender",
        "violet",
        "gold",
        "silver",
    ]

    def __init__(self, image, color=None, font_size=None, line_width=None):
        self.image = image
        self.color = color

        self.width, self.height = image.size

        self.font_size = font_size or self.width // 60
        self.line_width = line_width or self.width // 600

        self.objects = {}
        self.relationships = {}

    def _get_color(self):
        if self.color is None:
            return self.colors[random.randint(0, len(self.colors) - 1)]
        return self.color

    def denormalize(self, bounding_box):
        """
        Given a set of normalized coordinates of a bounding box, returns the absolute coordinates.

        Gemini works in normalized coordinates, so we need to process it to draw it.

        Takes (y1, x1, y2, x2) in a normalized format.

        Returns (x1, y1, x2, y2) relative to the original image.
        """
        y1, x1, y2, x2 = bounding_box
        res = [
            x1 / 1000 * self.width,
            y1 / 1000 * self.height,
            x2 / 1000 * self.width,
            y2 / 1000 * self.height,
        ]
        return (int(z) for z in res)

    def center(self, bounding_box):
        """
        Returns the center coordinate of the bounding box.

        Assumes the bounding box is already denormalized.

        Returns:
        - Center point as (x, y)
        """
        x1, y1, x2, y2 = bounding_box
        return (x1 + x2) // 2, (y1 + y2) // 2

    def draw_object(self, label, bounding_box):
        """
        Draws one object.

        Includes bounding box and label.
        """
        _color = self._get_color()

        # denormalize coordinates (see helper)
        abs_x1, abs_y1, abs_x2, abs_y2 = self.denormalize(bounding_box)

        # draw bounding box
        self._draw.rectangle(
            ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=_color, width=self.line_width
        )

        # draw text
        self._draw.text(
            (abs_x1 + 8, abs_y1 + 6),
            label,
            fill=_color,
            font_size=self.font_size,
        )

    def draw_relationship(self, subject, object, label, verbose=False):
        """
        Draws a single relationship.
        """
        # check if objects exist
        if subject not in self.objects:
            if verbose:
                print(f"WARNING: skipping unknown object {subject}")
            return
        if object not in self.objects:
            if verbose:
                print(f"WARNING: skipping unknown object {object}")
            return

        _color = self._get_color()

        # get center of both objects
        bb1 = self.objects[subject]["bbox"]
        bb2 = self.objects[object]["bbox"]
        center1 = self.center(self.denormalize(bb1))
        center2 = self.center(self.denormalize(bb2))

        # compute midpoint for labeling
        mid_x = (center1[0] + center2[0]) // 2
        mid_y = (center1[1] + center2[1]) // 2

        # draw line connecting centers
        self._draw.line([center1, center2], fill=_color, width=self.line_width)

        # draw predicate text
        self._draw.text((mid_x, mid_y), label, fill=_color, font_size=self.font_size)

    def add_objects(self, objects):
        """
        Adds objects.

        Args:
        - objects: dict as follows
        {
            "id": {
                "name": "asdf",
                "bbox": [y_min, x_min, y_max, x_max]
            },
            ...
        }
        where bounding box is unnormalized [y_min, x_min, y_max, x_max]
        """
        self.objects.update(objects)

    def add_relationships(self, relationships):
        """
        Adds new relationships.

        Args:
        - relationships: dict as follows
        {
            (subject_id, object_id): "predicate",
            ...
        }
        """
        self.relationships.update(relationships)

    def draw(self):
        """
        Draws the stored objects and relationships.

        Does not modify the original image.
        (so you can run this consecutively and it will work)
        """
        self.image_copy = self.image.copy()
        self._draw = ImageDraw.Draw(self.image_copy)

        # object bounding boxes
        for id, obj in self.objects.items():
            self.draw_object(f"{obj['name']} ({id})", obj["bbox"])

        # relationship lines
        for (subject, object), predicate in self.relationships.items():
            subject_name = self.objects[subject]["name"]
            object_name = self.objects[object]["name"]
            label = f"{subject_name} {predicate} {object_name}"
            self.draw_relationship(subject, object, label)

    def get_resized(self, image, max_size=1200):
        """
        Displays a resized version of the image for faster loading.

        Args:
        - image (PIL.Image): The original image.
        - max_size (int): The maximum width or height of the resized image.

        Returns:
        - PIL.Image: The resized image (does not modify the original).
        """
        # Compute new size while maintaining aspect ratio
        scale = min(max_size / self.width, max_size / self.height)
        new_size = (int(self.width * scale), int(self.height * scale))

        # Resize and show
        resized_image = image.resize(new_size, Image.Resampling.LANCZOS)

        return resized_image

    def show(self):
        """
        Displays a resized version of the image.

        draw() must have been called.
        """
        return self.get_resized(self.image_copy)
