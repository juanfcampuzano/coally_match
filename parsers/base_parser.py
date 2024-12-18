import os
import json
import openai
import yaml
import logging
from .major_classifier import MajorClassifier
from .logger import AppLogger

from pydantic import BaseModel

logger = AppLogger("Base Parser").get_logger()

class BaseParser:

    def __init__(self, template="", output_model=None):
        config = self.load_config("./config.yaml")
        self.majors = config.get("majors", [])
        self.major_classifier = MajorClassifier(majors=self.majors)

        self.output_model = output_model or {"type": "json_object"}

        if isinstance(self.output_model, BaseModel):
            self.output_model = self.output_model.model_json_schema()

        self.template = template

        openai.api_key = os.getenv("OPENAI_API_KEY")

    def load_config(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        client = os.getenv("CLIENT")
        client_config = config['clients'].get(client, None)
        if client_config:
            return client_config
        else:
            raise ValueError(f"Cliente '{client}' no encontrado en el archivo de configuración.")

    def run(self, inputs):
        logger.debug("Initialized BaseParser")
        majors = self.major_classifier.run(inputs)

        formatted_prompt = self.template.format(**inputs)

        try:
            response = openai.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Eres un asistente útil."},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0,
                response_format=self.output_model
            )

            result_content = response.choices[0].message.content
            result = json.loads(result_content)
        except Exception as e:
            logger.error(f"Error en la llamada al API de OpenAI: {e}")
            raise

        result["majors"] = majors

        logger.debug("Finished running BaseParser")
        return result
