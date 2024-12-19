from pymongo import MongoClient
import os
import logging
from parsers.decorators import handle_exception
from bson import ObjectId
from parsers.logger import AppLogger
import yaml
from dotenv import load_dotenv

load_dotenv()

logger = AppLogger('MongoDB Handler').get_logger()


class MongoDBConnection:
    def __init__(self):
        self.put_uri = os.environ.get("PUT_MONGO_URI")
        self.read_uri = os.environ.get("READ_MONGO_URI")
        self.connection = None
        self.old_connection = None

    def __enter__(self):
        self.connection = MongoClient(self.put_uri)
        self.old_connection = MongoClient(self.read_uri)
        logger.debug("Connected to MongoDB")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self.connection:
                self.connection.close()
                logger.debug("Put MongoDB connection closed.")
            if self.old_connection:
                self.old_connection.close()
                logger.debug("Read MongoDB connection closed.")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")

class MongoDBHandler:
    def __init__(self):
        config = self.load_config("./config.yaml")
        self.app_database_name = config.get("mongo_databases", {}).get("app")
        self.projects_collection_name = config.get("mongo_collections", {}).get("projects")
        self.resumes_collection_name = config.get("mongo_collections", {}).get("resumes")
        self.ml_database_name = config.get("mongo_databases", {}).get("ai")
        self.parsed_projects_collection_name = config.get("mongo_collections", {}).get("parsed_projects")
        self.parsed_resumes_collection_name = config.get("mongo_collections", {}).get("parsed_resumes")

    def load_config(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        client = os.getenv("CLIENT")
        client_config = config['clients'].get(client, None)
        if client_config:
            return client_config
        else:
            raise ValueError(f"Cliente '{client}' no encontrado en el archivo de configuración.")

    def get_project(self, project_id):
        query = {
            "_id": ObjectId(project_id)
        }

        with MongoDBConnection() as mongo:
            found_project = mongo.old_connection[self.app_database_name][self.projects_collection_name].find_one(query)

            if found_project is None:
                logger.warning(f"Couldn't find project with ID {project_id}")
                return {}
            
            return {
                "project_name": found_project.get("NombreOportunidad"),
                "project_description": found_project.get("DescribeProyecto"),
                "majors": found_project.get("SeleccionaCarrera", []).split(', ') + found_project.get("empleos_alternativos", []),
                "hard_skills": found_project.get("habilidadesTecnicas"),
                "approved_by": found_project.get("approvedBy", []),
                "type": found_project.get("tipoDeServicioDeseado"),
                "source": found_project.get("tipo"),
                "status": found_project.get("status"),
                "contract_type": found_project.get("modalidadDeContratacion")
            }

    def get_resume(self, resume_id):
        query = {
            "_id": ObjectId(resume_id)
        }

        with MongoDBConnection() as mongo:
            found_cv = mongo.old_connection[self.app_database_name][self.resumes_collection_name].find_one(query)

            if found_cv is None:
                logger.warning(f"Couldn't find resume with ID {resume_id}")
                return {}
            
            return {
                "main_skills": found_cv.get("aptitudes_principales"),
                "resume_abstract": found_cv.get("extracto"),
                "education": [edu.get("Titulo_Certificacion", "") for edu in found_cv.get("educacion", [])],
                "experience": self.parse_experience(found_cv),
                "current_position": found_cv.get("info_personal", {}).get("profesion_actual")
            }

    def upsert_parsed_document(self, document, collection):
        query = {
            "id": document["id"]
        }

        if collection == "projects":
            collection_name = self.parsed_projects_collection_name
        else:
            collection_name = self.parsed_resumes_collection_name

        with MongoDBConnection() as mongo:
            result = mongo.connection[self.ml_database_name][collection_name].update_one(query, {"$set": document}, upsert=True)

            if result.upserted_id:
                logger.debug(f"Inserted new document with ID: {result.upserted_id}")
            else:
                logger.debug("Updated document.")

    def find_documents_with_matching_items(self, collection, search_list):
        query = {
            "majors": {
                "$in": search_list
            }
        }

        if collection == "projects":
            collection_name = self.projects_collection_name
        else:
            collection_name = self.resumes_collection_name

        with MongoDBConnection() as mongo:
            result = mongo.connection[self.ml_database_name][collection_name].find(query)
            return result
        return []

    def parse_experience(self, cv):
        if not isinstance(cv, dict) or 'experiencia' not in cv:
            return ''
        
        job_titles = []
        for exp in cv.get('experiencia', []):
            positions = exp.get('cargos', [])
            
            if isinstance(positions, list) and positions:
                for position in positions:
                    job_title = position.get('nombrecargo', 'Sin título de cargo')
                    job_titles.append(job_title)
            
            start_date = exp.get('fecha_inicio')
            end_date = exp.get('fecha_finalizacion')
            if start_date and end_date:
                job_titles.append(f"From {start_date} to {end_date}")
            elif start_date:
                job_titles.append(f"From {start_date}")
            elif end_date:
                job_titles.append(f"Until {end_date}")
        
        return ', '.join(job_titles) if job_titles else 'No experience available'