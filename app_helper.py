from mongodb_handler import MongoDBHandler
from postgres_handler import PostgresHandler
from pipeline import Pipeline
import pickle as pkl
from parsers.project_parser import ProjectParser
from parsers.resume_parser import ResumeParser
from parsers.logger import AppLogger
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

class AppHelper:
    def __init__(self):
        self.mongo_handler = MongoDBHandler()
        self.project_parser = ProjectParser()
        self.resume_parser = ResumeParser()
        self.pipeline = Pipeline()
        self.model = pkl.load(open('./objects/model.pkl', 'rb'))
        self.logger = AppLogger("App").get_logger()

    
    def parse_project(self, project_id):
        self.logger.debug(f"Parsing project with id {project_id}")
        project = self.mongo_handler.get_project(project_id=project_id)
        if not project:
            self.logger.warning(f"Project with ID {project_id} not found.")
            return None
        parsed_project = self.project_parser.run(project)
        parsed_project["id"] = ObjectId(project_id)
        parsed_project["approved_by"] = project.get("approved_by")
        parsed_project["type"] = project.get("type")
        parsed_project["status"] = project.get("status")
        self.logger.info(f"Parsed project with id {project_id}")
        return parsed_project

    
    def parse_resume(self, resume_id):
        self.logger.debug(f"Parsing resume with id {resume_id}")
        resume = self.mongo_handler.get_resume(resume_id=resume_id)
        if not resume:
            self.logger.warning(f"Resume with ID {resume_id} not found.")
            return None
        parsed_resume = self.resume_parser.run(resume)
        parsed_resume['id'] = ObjectId(resume_id)
        self.logger.info(f"Parsed resume with id {resume_id}")
        return parsed_resume

    
    def calculate_compatibility(self, parsed_resume, parsed_project, model):
        if parsed_resume['experience'] > 12 and parsed_project['experience'] == 0:
            return 0
        features = self.pipeline.run_pipeline(parsed_project=parsed_project, parsed_resume=parsed_resume)
        return model.predict(features)[0]


    
    def calculate_compatible_resumes(self, parsed_project, model):
        majors = parsed_project['majors']
        compatible_resumes = list(self.mongo_handler.find_documents_with_matching_items(
            collection="resumes",
            search_list=majors
        ))
        self.logger.info("compatible_resumes")

        self.logger.info(compatible_resumes)
        
        compatibilities = {
            str(resume['id']): max(0, min(self.calculate_compatibility(parsed_resume=resume, parsed_project=parsed_project, model=model), 100))
            for resume in compatible_resumes
        }
        return compatibilities

    
    def calculate_compatible_projects(self, parsed_resume, model):
        majors = parsed_resume['majors']
        compatible_projects = self.mongo_handler.find_documents_with_matching_items(
            collection="projects",
            search_list=majors
        )
        
        compatibilities = {
            str(project['id']): max(0, min(self.calculate_compatibility(parsed_resume=parsed_resume, parsed_project=project, model=model), 100))
            for project in compatible_projects
        }
        return compatibilities


    
    def create_resume(self, request):
        id_resume = request.id_cv
        parsed_resume = self.parse_resume(id_resume)

        if not parsed_resume:
            return False
        self.mongo_handler.upsert_parsed_document(document=parsed_resume, collection="resumes")
        compatibilities = self.calculate_compatible_projects(parsed_resume, self.model)
        with PostgresHandler() as handler:
            handler.upsert_compatibilities(key_id=id_resume, compatibilities=compatibilities, entity_type="resume")
        return True


    def create_project(self, request):
        id_project = request.id_project
        parsed_project = self.parse_project(id_project)
        if not parsed_project or parsed_project.get("source") == "externo":
            return False
        with PostgresHandler() as postgres_handler:
            postgres_handler.upsert_project_filters(parsed_project=parsed_project)
        self.mongo_handler.upsert_parsed_document(document=parsed_project, collection="projects")
        compatibilities = self.calculate_compatible_resumes(parsed_project, self.model)
        with PostgresHandler() as postgres_handler:
            postgres_handler.upsert_compatibilities(key_id=id_project, compatibilities=compatibilities, entity_type="project")
        return True
    
    def perform_delete_project(self):
        pass

    def perform_delete_resume(self):
        pass