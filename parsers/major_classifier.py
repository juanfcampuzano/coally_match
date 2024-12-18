import openai
import os
import json
import logging
import yaml

logger = logging.getLogger("Major Classifier")

class MajorClassifier:

    def __init__(self, majors):
        self.majors = majors
        self.majors_dict = {val["major"]:val["description"] for val in self.majors}
        logger.debug("Initialized Major Classifier")

    def cargar_carreras(self, path):
        with open(path, 'r', encoding='utf-8') as archivo:
            data = yaml.safe_load(archivo)
        return data

    def run(self, text):
        content = f"""###Majors###\n{self.majors_dict}\n\n###Instructions###\nCategorise the following resume or job offer to one or more majors.
        You will give a resume a label or labels if the person studied that major or a very related one. A person might study more than one major.\n\n
        You will give a job offer a label or labels if the job accepts people with that major or majors. Ex: If the job requires. A software engineer or related field,
        you will label it with computer science, software architecture, etc.\n\n
        {text}\n"""

        function = {
        "name": "predict_major",
        "description": "Predict the majors for given resume or job offer",
        "parameters": {
            "type": "object",
            "properties": {
                "prediction": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": list(self.majors_dict.keys())
                    },
                    "description": "The predicted majors."
                }
            },
            "required": [
                "prediction"
            ]
        }
        }

        r = openai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[{"role": "user", "content": content}],
        functions=[function],
        function_call={"name": "predict_major"}
        )

        logger.debug("Classified majors for the given text")
        return json.loads(r.choices[0].message.function_call.arguments)["prediction"]