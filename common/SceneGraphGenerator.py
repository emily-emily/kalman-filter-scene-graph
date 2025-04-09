import json
import sys
from google import genai
from dotenv import load_dotenv
import os

GOOGLE_COLAB = "google.colab" in sys.modules

GEMINI_API_KEY = None
if GOOGLE_COLAB:
    from google.colab import userdata

    GEMINI_API_KEY = userdata.get("GOOGLE_API_KEY")
else:
    load_dotenv()
    GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")


class SceneGraphGenerator:
    def __init__(self, model_name="gemini-1.5-flash"):
        """
        Initializes the SceneGraphGenerator with the specified model name.

        Currently only supports Gemini models.
        """
        self.gemini_model = model_name
        self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    def response_to_json(self, res):
        """
        Takes the output of a model and returns the parsed result.

        May need to clean up the output if it is a code chunk.
        """
        if res.startswith("```json"):
            res = res[7:-4]

        try:
            return json.loads(res)
        except:
            raise ValueError(f"Could not parse JSON: {res}")

    def generate_scene_graph(self, img, prompt, context=[]):
        """
        Generates a scene graph from an image and a prompt.
        Args:
            img: The image to generate the scene graph from.
            prompt: The prompt to guide the generation.
            context: Optional context to provide additional information (eg. previous scene graph).
        """
        contents = [img, prompt]

        if context:
            # make sure everything is consumable by the model
            for i in range(len(context)):
                if isinstance(context[i], dict):
                    context[i] = json.dumps(context[i])

            contents.extend(context)

        response = self.gemini_client.models.generate_content(
            model=self.gemini_model,
            contents=contents,
            config={"response_mime_type": "application/json"},
        ).text

        graph = self.response_to_json(response)
        return graph
