import os
import langchain
from pymongo import MongoClient
import psycopg2
import psycopg2.extras
from bson.objectid import ObjectId
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.output_parsers import JsonOutputParser
import unicodedata
import re
from custom_vectorizer import CustomTfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle as pkl
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from psycopg2 import OperationalError, InterfaceError, DatabaseError
from psycopg2.errors import InFailedSqlTransaction
from datetime import datetime, timezone

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

def connect_to_mongodb(uri, db_name, projects_col, cvs_col):
    """
    Establish a connection to the MongoDB database and return collections.
    
    Parameters:
    uri (str): The MongoDB connection URI.
    db_name (str): The name of the database to connect to.
    projects_col (str): The name of the projects collection.
    cvs_col (str): The name of the user CVs collection.
    
    Returns:
    tuple: A tuple containing the projects collection and the user CVs collection.
    """
    client = MongoClient(uri)
    db = client[db_name]
    projects_collection = db[projects_col]
    cvs_collection = db[cvs_col]
    return projects_collection, cvs_collection

def get_project_string(project):
    """
    Generate a descriptive string for a project document.
    
    Parameters:
    project (dict): A dictionary representing the project document from the database.
    
    Returns:
    str: A concatenated string describing the project's details.
    """

    string = ""
    
    if 'NombreOportunidad' in project:
        string += str(project['NombreOportunidad']) + '. '
    
    if 'DescribeProyecto' in project:
        string += str(project['DescribeProyecto']) + '. '
    
    if 'SeleccionaCarrera' in project:
        string += str(project['SeleccionaCarrera']) + '. '
    
    if 'empleos_alternativos' in project:
        string += ', '.join(project['empleos_alternativos']) + '. '
    

    if 'habilidadesTecnicas' in project:
        string += ', '.join(project['habilidadesTecnicas'])
    
    return string

class Job(BaseModel):
    """
    Pydantic model to represent a job offer.
    """
    majors: list = Field(description="Lista de carreras que debería idealmente tener alguien para postularse en esta oferta")
    experience: int = Field(description="Experiencia requerida por la vacante en meses, si es intern o practicante este valor es cero.")
    education_level: str = Field(description="Nivel educativo requerido por la oferta, puede ser una de estas opciones: [basico, tecnico, tecnologico, profesional, maestria, doctorado] (profesional es universitario, basico es colegio)")
    technical_skills: list = Field(description="Lista de conocimientos de herramientas requeridas por la oferta. Usa el nombre completo de la habilidad de una forma estándar y sin abreviaciones.")
    keywords: list = Field(description="Palabras clave más importantes del rol (de las funciones)")

def remove_accents(text):
    """
    Remove accent marks from a given text.
    
    Parameters:
    text (str): The input text with potential accent marks.
    
    Returns:
    str: The text with accent marks removed.
    """
    accents = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
    }
    return ''.join(accents.get(c, c) for c in text)

def parse_job_offer(job_offer, majors):
    """
    Parse a job offer to extract relevant details using the OpenAI model.
    
    Parameters:
    job_offer (str): The job offer text.
    majors (list): A list of majors to match against the job offer.
    
    Returns:
    dict: A dictionary containing the parsed details of the job offer.
    """

    model = ChatOpenAI(temperature=0, model='gpt-4o-mini')
    
    query = f"""Extrae la experiencia en meses requerida por la oferta. Extrae las palabras clave más relevantes del rol (de las funciones). También extrae los conocimientos de herramientas requeridas por la oferta. También extrae el nivel educativo requerido por la oferta, puede ser: basico (high school), tecnico, tecnologico, profesional, maestria (si la vacante requiere posgrado, pon maestria), doctorado. También dime cuales de las carreras de esta lista: {str(majors)}, es requerida para la oferta. Si la vacante es ventas o asesoria comercial, la carrera es ingeniería comercial, si es operaciones o calidad de procesos probablemente sea ingenieria industrial, si es algo relacionado a electronica, automatizacion o mantenimiento probablemente la vacante sea de ingenieria electronica ciencias de la computacion y relacionados a ciencia de datos e inteligencia artificial, analisis de datos tómalo como ingenieria de sistemas :
    {job_offer}
    """

    # Initialize the parser with the Job model
    parser = JsonOutputParser(pydantic_object=Job)

    # Define the prompt template
    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Create a chain to process the prompt
    chain = prompt | model | parser

    # Invoke the chain with the query
    response = chain.invoke({"query": query})
    
    # Process the 'required_majors' field to remove accents and match against the provided list
    if 'majors' in response:
        response['majors'] = [remove_accents(str(major).lower()) for major in response['majors'] if remove_accents(str(major).lower()) in [remove_accents(m.lower()) for m in majors]]
    else:
        response['majors'] = []


    return response


def parse_experience(cv):
    """
    Parse and return the experience details from a CV.

    Parameters:
    cv (dict): A dictionary representing a CV.

    Returns:
    str: A string containing parsed experience details.
    """
    if 'experiencia' not in cv:
        return ''
    
    job_titles = []
    for exp in cv['experiencia']:
        positions = exp.get('cargos', [])
        job_titles.extend(
            position.get('nombrecargo', '') for position in positions if isinstance(positions, list)
        )
        job_titles.append(f"desde {exp['fecha_inicio']} hasta {exp['fecha_finalizacion']}") if 'fecha_inicio' in exp and 'fecha_finalizacion' in exp else None

    return ', '.join(job_titles)

def parse_education(cv):
    """
    Parse and return the education details from a CV.

    Parameters:
    cv (dict): A dictionary representing a CV.

    Returns:
    str: A string containing parsed education details.
    """
    if 'educacion' not in cv:
        return ''
    
    degrees = [
        edu.get('Titulo_Certificacion', '') for edu in cv['educacion']
    ]

    return ', '.join(degrees)

def get_cv_string(cv):
        """
        Generate a descriptive string for a cv document.

        Parameters:
        cvs (list): A list of CV dictionaries containing CV details.

        Returns:
        list: A list of concatenated CV detail strings.
        """
        string = (
        (', '.join(cv['aptitudes_principales']) + '. ' if 'aptitudes_principales' in cv else '') + " " +
        (cv['extracto'] + '. ' if 'extracto' in cv else '') + " " +
        parse_experience(cv) + " " +
        parse_education(cv) + " " +
        (', '.join(cv['info_personal']['profesion_actual']) if 'info_personal' in cv and 'profesion_actual' in cv['info_personal'] else '')
        )

        aumentos_contexto = {"auxiliar contable": "contabilidad", "auxiliar administrativo": 'administración de empresas', 'Relaciones Internacionales': 'negocios internacionales', 'relaciones internacionales': 'negocios internacionales'}

        for key, value in aumentos_contexto.items():
            if key in string:
                string += ' '+value

        return string

class Resume(BaseModel):
    majors: list = Field(description="Lista de carreras que tiene o que está estudiando la persona")
    experience: int = Field(description="Experiencia que tiene la persona en meses, si es recién graduado este valor es cero")
    education_level: str = Field(description="""Eivel educativo que tiene la persona, puede ser una de estas opciones: [basico, 
                                                tecnico, tecnologico, profesional, maestria, doctorado] (profesional es universitario, basico es colegio)""")
    technical_skills: list = Field(description="""Lista de conocimientos de herramientas que tiene la persona. Usa el nombre completo de la habilidad 
                                                de una forma estándar y sin abreviaciones.""")
    keywords: list = Field(description="Palabras clave más importantes del cv")

def remove_accents(text):
    accents = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
    }
    return ''.join(accents.get(c, c) for c in text)

def parse_cv(cv, majors):

    model = ChatOpenAI(temperature=0, model='gpt-4o-mini')

    query = f"""Extrae la experiencia en meses que tiene el cv, puedes dar un estimado a partir de la duracion de los cargos o si en la descripción dice.
    Extrae las palabras clave más relevantes del cv. También extrae los conocimientos de herramientas que tiene la persona. 
    También extrae el nivel educativo que tiene la persona, puede ser: [basico, tecnico, tecnologico, profesional, maestria, doctorado]. 
    También dime cual o cuales de las carreras de esta lista: {str(majors)}, está estudiando o estudió la persona. 
    Si la persona es ventas o asesoria comercial, la carrera es ingeniería comercial, si es operaciones o calidad de 
    procesos probablemente sea ingenieria industrial, si es algo relacionado a electronica, automatizacion o mantenimiento probablemente 
    la persona sea de ingenieria electronica. si menciona algo administrativo, probablemente la carrera sea administracion. ciencias de la computacion y relacionados a ciencia
    de datos e inteligencia artificial, analisis de datos tomalo como ingenieria de sistemas, administracion de mercadeo puede ser marketing:
    {cv}
    """

    parser = JsonOutputParser(pydantic_object=Resume)

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model | parser

    response = chain.invoke({"query": query})
    
    if 'majors' in response:
        response['majors'] = [remove_accents(str(i).lower()) for i in response['majors'] if remove_accents(str(i).lower()) in [c.lower() for c in majors]]
    else:
        response['majors'] = []
    
    return response


def preprocess_project(projects_collection, project_id, majors):
    project = projects_collection.find_one({'_id':ObjectId(project_id)})
    project_string = get_project_string(project)
    parsed_project = parse_job_offer(project_string, majors)
    parsed_project['id'] = project_id

    extra_info = {'aprobados':'-'.join(project['approvedBy'])+'-coally' if len(project['approvedBy']) > 0 else 'coally',
                  'tipos': project['tipoDeServicioDeseado']+'-Todas' if 'tipoDeServicioDeseado' in project else 'Todas-Oferta laboral'}
    return parsed_project, extra_info

def preprocess_cv(cvs_collection, cv_id, majors):
    print(type(ObjectId(cv_id)), ObjectId(cv_id))
    cv = cvs_collection.find_one({'_id':ObjectId(cv_id)})
    cv_string = get_cv_string(cv)
    parsed_string = parse_cv(cv_string, majors)
    parsed_string['id'] = cv_id
    return parsed_string


def connect_to_postgres(host, database, user, password):
    connection = psycopg2.connect(host=host, database=database, user=user, password=password)
    cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
    return connection, cursor


# alamcenar un cv parseado en la base de datos
def store_parsed_cv(parsed_cv, cursor):
    query = f"""INSERT INTO public.parsed_cvs (id, majors, experience, education_level, technical_skills, keywords) VALUES ('{parsed_cv['id']}', 
    '{'|'.join(parsed_cv['majors'])}', {parsed_cv['experience']}, '{parsed_cv['education_level']}', '{'|'.join(parsed_cv['technical_skills'])}', '{'|'.join(parsed_cv['keywords'])}')"""
    cursor.execute(query)

# almacenar un project parseado en la base de datos
def store_parsed_project(parsed_project, cursor, extra_info):
    query = f"""INSERT INTO public.parsed_projects (id, majors, experience, education_level, technical_skills, keywords) VALUES ('{parsed_project['id']}', 
    '{'|'.join(parsed_project['majors'])}', {parsed_project['experience']}, '{parsed_project['education_level']}', '{'|'.join(parsed_project['technical_skills'])}', '{'|'.join(parsed_project['keywords'])}')"""
    cursor.execute(query)

    tipos = extra_info['tipos'] if 'tipos' in extra_info else 'Oferta laboral-Todas'
    aprobados = extra_info['aprobados'] if 'aprobados' in extra_info else 'coally'

    query = f"""INSERT INTO public.filters_projects(
    id, tipo_oferta, approved_by)
    VALUES ('{parsed_project['id']}', '{tipos}', '{aprobados}');"""
    cursor.execute(query)


# get all projects for majors
def get_projects_from_majors(majors, cursor):
    total_projects = []
    for major in majors:
        query = f"""SELECT * FROM public.parsed_projects WHERE majors LIKE '%{major}%'"""
        cursor.execute(query)
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        total_projects += results

    seen = set()
    final_result = []
    for result in total_projects:
        if result['id'] not in seen:
            seen.add(result['id'])
            final_result.append(result)

    total = []
    for result in final_result:
        total.append({'majors':result['majors'].split('|'), 'experience':result['experience'], 
        'education_level':result['education_level'], 'technical_skills':result['technical_skills'].split('|'), 
        'keywords':result['keywords'].split('|'), 'id':result['id']})

    return total

# get all cvs for majors
def get_cvs_from_majors(majors, cursor):
    total_cvs = []
    for major in majors:
        print('major', major)
        query = f"""SELECT * FROM public.parsed_cvs WHERE majors LIKE '%{major}%'"""
        cursor.execute(query)
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        total_cvs += results
    
    seen = set()
    final_result = []
    for result in total_cvs:
        if result['id'] not in seen:
            seen.add(result['id'])
            final_result.append(result)
    total = []
    for result in final_result:
        total.append({'majors':result['majors'].split('|'), 'experience':result['experience'], 
        'education_level':result['education_level'], 'technical_skills':result['technical_skills'].split('|'), 
        'keywords':result['keywords'].split('|'), 'id':result['id']})

    return total


# get features from cv and project

def quitar_puntuacion(text):
    # Primero, identificamos y conservamos los puntos seguidos de una letra.
    # Utilizamos una expresión regular que encuentra puntos seguidos de una letra y los marca temporalmente.
    text_with_marks = re.sub(r'(\.\w)', r'__KEEP__\1', text)

    # Reemplazamos caracteres de puntuación no deseados por espacios
    text_no_punctuation = re.sub(r'[.,;:]', ' ', text_with_marks)
    
    # Restauramos los puntos que estaban seguidos de una letra
    cleaned_text = re.sub(r'__KEEP__ ', '.', text_no_punctuation)
    
    # Reemplazar múltiples espacios por uno solo
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    # Eliminar espacios extra al principio y al final
    return cleaned_text.strip()

def normalizar_texto(texto):
    # Convertir a minúsculas
    texto = texto.lower()
    
    # Normalización Unicode
    texto = unicodedata.normalize('NFD', texto)
    texto = texto.encode('ascii', 'ignore')
    texto = texto.decode('utf-8')
    
    # Reemplazar múltiples espacios por uno solo y eliminar espacios al principio y final
    texto = re.sub(r'\s+', ' ', texto).strip()

    texto = quitar_puntuacion(texto)
    
    return texto

def comparten_carrera(cv_parsed, oferta_parsed):
    return int(len(set(cv_parsed['majors']).intersection(set(oferta_parsed['majors']))) > 0)

def diferencia_experiencia(cv_parsed, oferta_parsed):
    return abs(cv_parsed['experience'] - oferta_parsed['experience'])

def porcentaje_habilidades(cv_parsed, oferta_parsed):
    if len(oferta_parsed['technical_skills']) == 0:
        return 1
    return len(set(cv_parsed['technical_skills']).intersection(set(oferta_parsed['technical_skills'])))/len(set(oferta_parsed['technical_skills']))

def numero_keywords(cv_parsed, oferta_parsed):
    if len(oferta_parsed['keywords']) == 0:
        return 1
    return len(set(' '.join(cv_parsed['keywords']).split()).intersection(set(' '.join(oferta_parsed['keywords']).split())))

def score_nivel_educativo(cv_parsed, oferta_parsed):
    return int(cv_parsed['education_level'] ==oferta_parsed['education_level'])

def reemplazar_sinonimos(texto):
    texto_lista = texto.split(', ')
    reemplazos = {"sst": "seguridad y salud en el trabajo", "hse":"seguridad y salud en el trabajo", "machine learning":"ciencia de datos", "deep learning":"ciencia de datos", "analisis de datos":"ciencia de datos"}
    resultado = []
    for subfrase in texto_lista:
        if subfrase in reemplazos:
            resultado.append(reemplazos[subfrase])
        else:
            resultado.append(subfrase)

    return ', '.join(resultado)

def calcular_similitud_promedio(frase1, frase2, vectorizer):

    if frase1 == "" or frase2 == "":
        return 0
    
    # Obtener los vectores TF-IDF de cada palabra en las frases
    palabras_frase1 = frase1.split()
    palabras_frase2 = frase2.split()
    
    # Crear un vector TF-IDF para cada palabra
    tfidf_palabras_frase1 = vectorizer.transform(palabras_frase1)
    tfidf_palabras_frase2 = vectorizer.transform(palabras_frase2)
    
    # Calcular la similitud de coseno entre todas las combinaciones de palabras
    similitudes = []
    for vector1 in tfidf_palabras_frase1:
        for vector2 in tfidf_palabras_frase2:
            similitud = cosine_similarity([vector1], [vector2])[0][0]
            similitudes.append(similitud)
    
    # Seleccionar las 5 similitudes más altas
    similitudes = sorted(similitudes, reverse=True)[:5]
    
    # Calcular y devolver el promedio de las 5 similitudes más altas
    return np.mean(similitudes)

def calculate_features(cv_parsed, project_parsed, vectorizer):
    features_dict = {'keywords_job': ', '.join(project_parsed['keywords']), 
                     'keywords_cv': ', '.join(cv_parsed['keywords']),
                     'habilidades_job': ', '.join(project_parsed['technical_skills']), 
                     'habilidades_cv': ', '.join(cv_parsed['technical_skills']),
                     'diferencia_experiencia':diferencia_experiencia(cv_parsed, project_parsed),
                     'comparten_carrera':comparten_carrera(cv_parsed, project_parsed),
                     'porcentaje_habilidades': porcentaje_habilidades(cv_parsed, project_parsed),
                     'numero_keywords': numero_keywords(cv_parsed, project_parsed),
                     'score_nivel_educativo': score_nivel_educativo(cv_parsed, project_parsed)
                     }
    #   ['similitud_keywords','similitud_skills', 'diferencia_experiencia', 'numero_keywords', 'score_nivel_educativo']

    features_dict['keywords_job_normalizado_reemplazado'] = reemplazar_sinonimos(normalizar_texto(features_dict['keywords_job']))
    features_dict['keywords_cv_normalizado_reemplazado'] = reemplazar_sinonimos(normalizar_texto(features_dict['keywords_cv']))
    features_dict['habilidades_job_normalizado_reemplazado'] = reemplazar_sinonimos(normalizar_texto(features_dict['habilidades_job']))
    features_dict['habilidades_cv_normalizado_reemplazado'] = reemplazar_sinonimos(normalizar_texto(features_dict['habilidades_cv']))

    features_dict['similitud_keywords'] = calcular_similitud_promedio(features_dict['keywords_job_normalizado_reemplazado'], features_dict['keywords_cv_normalizado_reemplazado'], vectorizer)
    features_dict['similitud_skills'] = calcular_similitud_promedio(features_dict['habilidades_job_normalizado_reemplazado'], features_dict['habilidades_cv_normalizado_reemplazado'], vectorizer)

    return [[features_dict['similitud_keywords'], features_dict['similitud_skills'], features_dict['diferencia_experiencia'], features_dict['numero_keywords'], features_dict['score_nivel_educativo']]]

def calculate_compatibility(cv_parsed, project_parsed, vectorizer, scaler, model):
    if cv_parsed['experience'] > 12 and project_parsed['experience'] == 0:
        return 0
    
    features = calculate_features(cv_parsed, project_parsed, vectorizer)
    features = scaler.transform(features)
    return model.predict(features)[0]


# given a cv, get all projects for its majro and get compatibility percentage for each one

def calculate_compatible_cvs(project_parsed, scaler, vectorizer, model, cursor):
    majors = project_parsed['majors']
    compatible_cvs = get_cvs_from_majors(majors, cursor)
    print('len(compatible_cvs)', len(compatible_cvs))
    compatibilities = {cv['id']:max(0,min(calculate_compatibility(project_parsed, cv, vectorizer, scaler, model),100)) for cv in compatible_cvs}
    return compatibilities

 # the same for a given project

def calculate_compatible_projects(cv_parsed, scaler, vectorizer, model, cursor):
    majors = cv_parsed['majors']
    compatible_projects = get_projects_from_majors(majors, cursor)
    compatibilities = {project['id']:max(0, min(calculate_compatibility(cv_parsed, project, vectorizer, scaler, model), 100)) for project in compatible_projects}
    return compatibilities



def insert_compatibilities_for_cv(id_cv, compatibilities, cursor):
    for id_project, compatibility in compatibilities.items():
        query = f"""INSERT INTO public.compatibilities (id_project, id_cv, compatibility) VALUES ('{id_project}','{id_cv}', {compatibility} )"""
        cursor.execute(query)

def insert_compatibilities_for_project(id_project, compatibilities, cursor):
    for id_cv, compatibility in compatibilities.items():
        query = f"""INSERT INTO public.compatibilities (id_project, id_cv, compatibility) VALUES ('{id_project}','{id_cv}', {compatibility} )"""
        cursor.execute(query)


def create_cv(id_cv):
    global majors, cvs_collection, cursor, scaler, vectorizer, model, connection
    try:
        parsed_cv = preprocess_cv(cvs_collection, id_cv, majors)
        store_parsed_cv(parsed_cv, cursor)
        compatibilities = calculate_compatible_projects(parsed_cv, scaler, vectorizer, model, cursor)
        insert_compatibilities_for_cv(id_cv, compatibilities, cursor)
        connection.commit()
    except (OperationalError, InterfaceError, DatabaseError, InFailedSqlTransaction) as e:
        print(e)
        if connection:
            connection.rollback()
    

def create_project(id_project):
    global majors, projects_collection, cursor, scaler, vectorizer, model, connection
    try:
        parsed_project, extra_info = preprocess_project(projects_collection, id_project, majors)
        store_parsed_project(parsed_project, cursor, extra_info)
        compatibilities = calculate_compatible_cvs(parsed_project, scaler, vectorizer, model, cursor)
        insert_compatibilities_for_project(id_project, compatibilities, cursor)
        connection.commit()
    except (OperationalError, InterfaceError, DatabaseError, InFailedSqlTransaction) as e:
        print(e)
        if connection:
            connection.rollback()


majors = ['negocios internacionales', 'ingenieria de sistemas', 'ingenieria de minas', 'ingenieria civil', 'arquitectura', 'economia',  'lenguas modernas', 'comunicacion social', 'sociologia', 'medicina', 'ingenieria quimica', 'enfermeria', 'quimico', 'docente', 'marketing', 'ingenieria metalurgica', 'psicologia', 'project manager', 'gastronomia', 'matematicas', 'bioquimica', 'antropologia', 'diseño', 'fisioterapia', 'administracion', 'contabilidad', 'ingenieria mecanica', 'ingenieria electrica', 'agronomia', 'topografia', 'biomedico', 'biologia', 'odontologia', 'ingenieria electronica', 'ingenieria ambiental', 'trabajo social', 'geologia', 'farmacia', 'ingenieria comercial', 'derecho', 'ingenieria en alimentos', 'veterinaria', 'ingenieria industrial']

host = 'db-resumescreening-coally.c960wcwwcazt.us-east-2.rds.amazonaws.com'
database = 'postgres'
user = 'postgres'
password = 'CoallySecur3'

uri = 'mongodb+srv://danielCTO:S3cure-2024Co4llyn3w@coally.nqokc.mongodb.net/CoallyProd?authSource=admin&replicaSet=atlas-39r1if-shard-0&w=majority&readPreference=primary&retryWrites=true&ssl=true'
db_name = 'CoallyProd'
projects_col = 'projects'
cvs_col = 'usercvs'

connection, cursor = connect_to_postgres(host, database, user, password)
projects_collection, cvs_collection = connect_to_mongodb(uri, db_name, projects_col, cvs_col)

vectorizer = pkl.load(open('vectorizer.pkl', 'rb'))
scaler = pkl.load(open('scaler.pkl', 'rb'))
model = pkl.load(open('model.pkl', 'rb'))


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CreateCVRequest(BaseModel):
    id_cv: str

class CreateProjectRequest(BaseModel):
    id_project: str

@app.get("/api/")
async def root():
    return {"message": "Hello World"}

@app.post("/api/create_project")
def add_project(request: CreateProjectRequest):
    try:
        create_project(id_project=request.id_project)
    except Exception as e:
        dt = datetime.now(timezone.utc)
        cursor.execute(f"INSERT into public.error_log (fecha, tipo, mensaje, id) VALUES ('{dt}', 'project', '{str(e).replace("'",'"')}', '{request.id_project}')")
        connection.commit()
        raise e
    return {'message':'Proyecto creado'}

@app.post("/api/create_cv")
def add_cv(request:CreateCVRequest):
    try:
        create_cv(id_cv=request.id_cv)
    except Exception as e:
        dt = datetime.now(timezone.utc)
        cursor.execute(f"INSERT into public.error_log (fecha, tipo, mensaje, id) VALUES ('{dt}', 'cv', '{str(e).replace("'",'"')}', '{request.id_cv}')")
        connection.commit()
        raise e
    return {'message':'CV creado'}