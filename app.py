from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from parsers.models.create_resume_request import CreateResumeRequest
from parsers.models.create_project_request import CreateProjectRequest
from parsers.models.update_resume_request import UpdateResumeRequest
from parsers.models.update_project_request import UpdateProjectRequest
from parsers.models.delete_project_request import DeleteProjectRequest
from parsers.models.delete_resume_request import DeleteResumeRequest
from parsers.models.close_project_request import CloseProjectRequest
from parsers.models.modify_approvals_project_request import ModifyApprovalsProjectRequest
from parsers.models.apply_request import ApplyRequest
from parsers.decorators import handle_exception
from parsers.logger import AppLogger
from app_helper import AppHelper
import boto3
import json
import re
import asyncio
import yaml
import os

logger = AppLogger("App").get_logger()

load_dotenv()

app = FastAPI(docs_url="/api/docs")

sqs = boto3.client("sqs", region_name="us-east-2")

def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    client = os.getenv("CLIENT")
    client_config = config['clients'].get(client, None)
    if client_config:
        return client_config
    else:
        raise ValueError(f"Cliente '{client}' no encontrado en el archivo de configuración.")
    
config = load_config("./config.yaml")

QUEUE_URL = config.get("sqs_url")

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
        logger.info(f"Created project with id {request.id}")
        return {"message": f"Created project with id {request.id}"}
    logger.warning(f"Couldn't create project with id {request.id}. It is an external project")
    return {"message": f"Couldn't create project with id {request.id}. It is an external project"}

@handle_exception(logger)
@app.post("/api/resume")
def add_cv(request: CreateResumeRequest):
    created = app_helper.create_resume(request, method="post")
    if created:
        logger.info(f"Created resume with id {request.id}")
        return {"message": f"Created resume with id {request.id}"}
    logger.warning(f"Couldn't create resume with id {request.id}.")
    return {"message": f"Couldn't create resume with id {request.id}."}

@handle_exception(logger)
@app.put("/api/resume")
def update_cv(request: UpdateResumeRequest):
    created = app_helper.create_resume(request, method="put")
    if created:
        logger.info(f"Updated resume with id {request.id}")
        return {"message": f"Updated resume with id {request.id}"}
    logger.warning(f"Couldn't update resume with id {request.id}.")
    return {"message": f"Couldn't update resume with id {request.id}."}

@handle_exception(logger)
@app.put("/api/project")
def update_project(request: UpdateProjectRequest):
    created = app_helper.create_project(request)
    if created:
        logger.info(f"Updated project with id {request.id}")
        return {"message": f"Updated project with id {request.id}"}
    logger.warning(f"Couldn't update project with id {request.id}. It is an external project")
    return {"message": f"Couldn't update project with id {request.id}. It is an external project"}

@handle_exception(logger)
@app.delete("/api/project")
def delete_project(request: DeleteProjectRequest):
    deleted = app_helper.perform_delete_project(request)
    if deleted:
        logger.info(f"Deleted project with id {request.id}")
        return {"message": f"Deleted project with id {request.id}"}
    logger.warning(f"Couldn't delete project with id {request.id}. It is an external project")
    return {"message": f"Couldn't delete project with id {request.id}. It is an external project"}

@handle_exception(logger)
@app.delete("/api/resume")
def delete_resume(request: DeleteResumeRequest):
    created = app_helper.perform_delete_resume(request)
    if created:
        logger.info(f"Updated resume with id {request.id}")
        return {"message": f"Updated resume with id {request.id}"}
    logger.warning(f"Couldn't update resume with id {request.id}.")
    return {"message": f"Couldn't update resume with id {request.id}."}

@handle_exception(logger)
@app.patch("/api/project/close")
def close_project(request: CloseProjectRequest):
    closed = app_helper.perform_close_project(request)
    if closed:
        logger.info(f"Closed project with id {request.id}")
        return {"message": f"Closed project with id {request.id}"}
    logger.warning(f"Couldn't close project with id {request.id}.")
    return {"message": f"Couldn't close project with id {request.id}."}

@handle_exception(logger)
@app.patch("/api/project/approvedby")
def modify_approved_institutions(request: ModifyApprovalsProjectRequest):
    updated = app_helper.perform_modify_approved_institutions(request)
    if updated:
        logger.info(f"Modified approvals for project with id {request.id}")
        return {"message": f"Modified approvals for project with id {request.id}"}
    logger.warning(f"Couldn't modify approvals for project with id {request.id}.")
    return {"message": f"Couldn't modify approvals for project with id {request.id}."}

@handle_exception(logger)
@app.post("/api/resume/apply")
def apply(request: ApplyRequest):
    updated = app_helper.apply(request)
    if updated:
        logger.info(f"Applied resume with id {request.id_resume} to project with id {request.id_project}")
        return {"message": f"Applied resume with id {request.id_resume} to project with id {request.id_project}"}
    logger.warning(f"Couldn't apply for project with id {request.id} with resume with id {request.id_resume}.")
    return {"message": f"Applied resume with id {request.id_resume} to project with id {request.id_project}"}

@app.get("/api/get_feedback_data")
async def get_feedback_data():
    try:
        rows = app_helper.get_feedback_data()

        response_data = {
            "timestamps": [row["week_start"].strftime("%Y-%m-%d") for row in rows],
            "likes": [row["likes"] for row in rows],
            "dislikes": [row["dislikes"] for row in rows],
        }

        return response_data
    except Exception as e:
        logger.error(f"Error in /api/get_feedback_data: {e}")
        return {"error": "Unable to fetch feedback data"}
    

@app.get("/api/dashboard", response_class=HTMLResponse)
async def dashboard():
    with open("./templates/dashboard.html", "r") as file:
        return HTMLResponse(content=file.read())

def limpiar_json(json_sucio):
    json_limpio = re.sub(r'\\[nrt]', '', json_sucio)
    json_limpio = re.sub(r'\\"', '"', json_limpio)
    json_limpio = re.sub(r'[\n\r\t]', '', json_limpio)
    json_limpio = re.sub(r'\s{2,}', ' ', json_limpio)
    json_limpio = json_limpio.strip()
    return json_limpio

def process_message(message_body: str):
    try:
        cleaned_body = limpiar_json(message_body)
        message = json.loads(cleaned_body)

        operation = message["action"]
        entity = message["entity"]
        data = message["body"]

        if entity == "project":
            if operation == "create":
                add_project(CreateProjectRequest.model_validate(data))
            elif operation == "update":
                update_project(UpdateProjectRequest.model_validate(data))
            elif operation == "delete":
                delete_project(DeleteProjectRequest.model_validate(data))
            elif operation == "update status":
                close_project(CloseProjectRequest.model_validate(data))
            elif operation == "update approvals":
                modify_approved_institutions(ModifyApprovalsProjectRequest.model_validate(data))
        elif entity == "resume":
            if operation == "create":
                add_cv(CreateResumeRequest.model_validate(data))
            elif operation == "update":
                update_cv(UpdateResumeRequest.model_validate(data))
            elif operation == "delete":
                delete_resume(DeleteResumeRequest.model_validate(data))
            elif operation == "apply":
                apply(ApplyRequest.model_validate(data))

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e} - Body: {message_body}")
    except KeyError as e:
        logger.error(f"Missing key in message: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


async def consume_queue():
    while True:
        response = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=10,
            VisibilityTimeout=120
        )
        if "Messages" in response:
            for message in response["Messages"]:
                logger.info(f"Processing message: {message['Body']}")
                try:
                    process_message(message["Body"])
                    sqs.delete_message(
                        QueueUrl=QUEUE_URL,
                        ReceiptHandle=message["ReceiptHandle"]
                    )
                    logger.info("Message processed and removed from queue.")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        else:
            logger.debug("Waiting for messages.")
        await asyncio.sleep(5)


@app.on_event("startup")
async def startup_event():
    logger.info("Starting SQS consumer in the background...")
    asyncio.create_task(consume_queue())