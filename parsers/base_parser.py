import os
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import logging
from .major_classifier import MajorClassifier
from .logger import AppLogger
import yaml

logger = AppLogger("Base Parser").get_logger()

class BaseParser:

    def __init__(self, template="", output_model = None):

        config = self.load_config("./config.yaml")
        self.majors = config.get("majors", [])
        self.major_classifier = MajorClassifier(majors=self.majors)

        if output_model is None:
            self.output_model = {"type": "json_object"}
        else:
            self.output_model = output_model

        self.template = template

        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            api_key = os.getenv("OPENAI_API_KEY"),
            temperature = 0
            )
        self.llm=self.llm.bind(response_format=output_model)

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

        prompt = ChatPromptTemplate.from_template(self.template)
        chain = prompt | self.llm
        result = chain.invoke(inputs)
        result = json.loads(result.content)
        result["majors"] = majors

        logger.debug("Finished running base parser")
        return result