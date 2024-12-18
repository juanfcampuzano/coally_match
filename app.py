from bson.objectid import ObjectId
import pickle as pkl
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from parsers.models.create_resume_request import CreateResumeRequest
from parsers.models.create_project_request import CreateProjectRequest
from parsers.models.update_project_filters_request import UpdateProjectRequest
from parsers.models.update_resume_request import UpdateResumeRequest
from parsers.models.delete_project_request import DeleteProjectRequest
from parsers.models.delete_resume_request import DeleteResumeRequest
from parsers.models.message_request import MessageRequest
from parsers.decorators import handle_exception
from parsers.logger import AppLogger
from app_helper import AppHelper
import boto3
import json
import time

logger = AppLogger("App").get_logger()

load_dotenv()

app = FastAPI(docs_url="/api/docs")

sqs = boto3.client("sqs", region_name="us-east-2")
QUEUE_URL = "https://sqs.us-east-2.amazonaws.com/203152832070/ml_empleo_uniandes.fifo"

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app_helper = AppHelper()


@app.get("/api/health")
async def health():
    return {"message": "Success!"}

@handle_exception(logger)
@app.post("/api/project")
def add_project(request: CreateProjectRequest):
    created = app_helper.create_project(request)
    if created:
        logger.info(f"Created project with id {request.id_project}")
        return {"message": f"Created project with id {request.id_project}"}
    logger.warning(f"Couldn't create project with id {request.id_project}. It is an external project")
    return {"message": f"Couldn't create project with id {request.id_project}. It is an external project"}

@handle_exception(logger)
@app.post("/api/resume")
def add_cv(request: CreateResumeRequest):
    created = app_helper.create_resume(request)
    if created:
        logger.info(f"Created resume with id {request.id_cv}")
        return {"message": f"Created resume with id {request.id_cv}"}
    logger.warning(f"Couldn't create resume with id {request.id_cv}.")
    return {"message": f"Couldn't create resume with id {request.id_cv}."}

@handle_exception(logger)
@app.put("/api/resume")
def update_cv(request: CreateResumeRequest):
    created = app_helper.create_resume(request)
    if created:
        logger.info(f"Updated resume with id {request.id_cv}")
        return {"message": f"Updated resume with id {request.id_cv}"}
    logger.warning(f"Couldn't update resume with id {request.id_cv}.")
    return {"message": f"Couldn't update resume with id {request.id_cv}."}

@handle_exception(logger)
@app.put("/api/project")
def update_project(request: CreateProjectRequest):
    created = app_helper.create_project(request)
    if created:
        logger.info(f"Updated project with id {request.id_project}")
        return {"message": f"Updated project with id {request.id_project}"}
    logger.warning(f"Couldn't update project with id {request.id_project}. It is an external project")
    return {"message": f"Couldn't update project with id {request.id_project}. It is an external project"}

@handle_exception(logger)
@app.delete("/api/project")
def delete_project(request: DeleteProjectRequest):
    deleted = app_helper.perform_delete_project(request)
    if deleted:
        logger.info(f"Deleted project with id {request.id_project}")
        return {"message": f"Deleted project with id {request.id_project}"}
    logger.warning(f"Couldn't delete project with id {request.id_project}. It is an external project")
    return {"message": f"Couldn't delete project with id {request.id_project}. It is an external project"}

@handle_exception(logger)
@app.delete("/api/resume")
def delete_resume(request: DeleteResumeRequest):
    created = app_helper.perform_delete_resume(request)
    if created:
        logger.info(f"Updated resume with id {request.id_cv}")
        return {"message": f"Updated resume with id {request.id_cv}"}
    logger.warning(f"Couldn't update resume with id {request.id_cv}.")
    return {"message": f"Couldn't update resume with id {request.id_cv}."}


def process_message(message_body):
    message = json.loads(message_body)
    operation = message["operation"]
    entity = message["entity"]
    data = message["data"]

    if entity == "project":
        if operation == "create":
            add_project(data)
        elif operation == "update":
            update_project(data)
        elif operation == "delete":
            delete_project(data)
    elif entity == "resume":
        if operation == "create":
            add_cv(data)
        elif operation == "update":
            update_cv(data)
        elif operation == "delete":
            delete_resume(data)


def consume_queue():
    while True:
        response = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=10
        )
        if "Messages" in response:
            for message in response["Messages"]:
                logger.info(f"Processing message: {message["Body"]}")
                try:
                    process_message(message["Body"])
                    sqs.delete_message(
                        QueueUrl=QUEUE_URL,
                        ReceiptHandle=message["ReceiptHandle"]
                    )
                    logger.info("Message processed and removed from queue.")
                except Exception as e:
                    logger.error(f"Error processing message", e)
        else:
            logger.debug("Waiting for messages.")
            time.sleep(5)