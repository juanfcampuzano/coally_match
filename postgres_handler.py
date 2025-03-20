import psycopg2
import psycopg2.extras
import os
from parsers.logger import AppLogger
import yaml
from dotenv import load_dotenv

load_dotenv()

logger = AppLogger("Postgres Handler").get_logger()

class PostgresHandler:
    def __init__(self):
        self.host = os.environ.get("POSTGRES_HOST")
        self.database = os.environ.get("POSTGRES_DATABASE_NAME")
        self.user = os.environ.get("POSTGRES_USER")
        self.password = os.environ.get("POSTGRES_PASSWORD")
        self.conn = None
        self.cur = None

        config = self.load_config("./config.yaml")

        self.compatibility_table = config.get("postgres_tables", {}).get("compatibilities")
        self.project_filters_table = config.get("postgres_tables", {}).get("project_filters")
        self.feedback_table = config.get("postgres_tables", {}).get("feedback")

    def load_config(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        client = os.getenv("CLIENT")
        client_config = config['clients'].get(client, None)
        if client_config:
            return client_config
        else:
            raise ValueError(f"Cliente '{client}' no encontrado en el archivo de configuración.")

    def __enter__(self):
        self.conn = psycopg2.connect(
            dbname=self.database,
            user=self.user,
            password=self.password,
            host=self.host
        )
        self.cur = self.conn.cursor()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
                logger.error(f"An exception has occurred: {exc_value}")
        except Exception as e:
            logger.error(f"Error during commit/rollback: {e}")
        finally:
            if self.cur:
                self.cur.close()
            if self.conn:
                self.conn.close()

    def upsert_compatibilities(self, key_id, compatibilities, entity_type):
        insert_count = 0
        values_to_insert = []

        if entity_type == "resume":
            for project_id, percentage in compatibilities.items():
                percentage = float(percentage)
                values_to_insert.append((project_id, key_id, percentage))
                insert_count += 1
        elif entity_type == "project":
            for resume_id, percentage in compatibilities.items():
                percentage = float(percentage)
                values_to_insert.append((key_id, resume_id, percentage))
                insert_count += 1

        if values_to_insert:
            query_upsert = f"""
            INSERT INTO public.{self.compatibility_table} (project_id, resume_id, percentage) 
            VALUES %s
            ON CONFLICT (project_id, resume_id)
            DO UPDATE SET
                percentage = EXCLUDED.percentage
            """
            
            try:
                psycopg2.extras.execute_values(self.cur, query_upsert, values_to_insert)
                logger.debug(f"{insert_count} upserted or inserted compatibilities on table {self.compatibility_table}.")
            except Exception as e:
                logger.error(f"Error during upsert compatibilities: {e}")
                return

            ids_to_keep = tuple([key_id] + list(compatibilities.keys()))
            
            deleted_count = 0
            if entity_type == "resume":
                query_delete_obsolete = f"""
                DELETE FROM public.{self.compatibility_table}
                WHERE resume_id = %s
                AND project_id NOT IN %s
                RETURNING *;
                """
                try:
                    self.cur.execute(query_delete_obsolete, (key_id, ids_to_keep))
                    deleted_rows = self.cur.fetchall()
                    deleted_count = len(deleted_rows)
                except Exception as e:
                    logger.error(f"Error during cleanup of old compatibilities for resume_id {key_id}: {e}")

            elif entity_type == "project":
                query_delete_obsolete = f"""
                DELETE FROM public.{self.compatibility_table}
                WHERE project_id = %s
                AND resume_id NOT IN %s
                RETURNING *;
                """
                try:
                    self.cur.execute(query_delete_obsolete, (key_id, ids_to_keep))
                    deleted_rows = self.cur.fetchall()
                    deleted_count = len(deleted_rows)
                except Exception as e:
                    logger.error(f"Error during cleanup of old compatibilities for project_id {key_id}: {e}")

            logger.info(f"{insert_count} upserted compatibilities, {deleted_count} obsolete compatibilities deleted.")


    def upsert_project_filters(self, parsed_project):
        project_id = str(parsed_project.get("id"))
        approved_by = '-'.join(parsed_project.get("approved_by", []))
        opportunity_type = parsed_project.get("type")
        status = parsed_project.get("status")

        query = f"""
        INSERT INTO public.{self.project_filters_table} (project_id, approved_by, opportunity_type, status)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (project_id)
        DO UPDATE SET
            approved_by = EXCLUDED.approved_by,
            opportunity_type = EXCLUDED.opportunity_type,
            status = EXCLUDED.status
        """
        
        try:
            self.cur.execute(query, (project_id, approved_by, opportunity_type, status))
            logger.debug(f"Upserted project filters for project with id {project_id}")
        except Exception as e:
            logger.error(f"Error during upsert project filters: {e}")

    def delete_project(self, project_id):
        try:
            query = f"DELETE FROM public.{self.project_filters_table} WHERE project_id = '%s'"
            self.cur.execute(query, (project_id,))

            query2 = f"DELETE FROM public.{self.compatibility_table} WHERE project_id = '%s'"
            self.cur.execute(query2, (project_id,))

            self.conn.commit()

            logger.info(f"Deleted project with id {project_id}")

            return True

        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error deleting project with id {project_id}: {e}")
            return False


    def delete_resume(self, resume_id):
        try:
            query = f"DELETE FROM public.{self.compatibility_table} WHERE resume_id = '%s'"
            self.cur.execute(query, (resume_id,))

            self.conn.commit()

            logger.info(f"Deleted resume with id {resume_id}")
            return True

        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error deleting resume with id {resume_id}: {e}")
            return False
        

    def update_project_status(self, project_id, status):
        query = f"""
        UPDATE public.{self.project_filters_table}
        SET status = %s
        WHERE project_id = %s
        """

        try:
            self.cur.execute(query, (status, project_id))
            if self.cur.rowcount > 0:
                logger.debug(f"Updated status for project with ID {project_id} to {status}")
                return True
            else:
                logger.warning(f"Project ID {project_id} not found. No status updated.")
                return False
        except Exception as e:
            logger.error(f"Error during updating project status for project ID {project_id}: {e}")
            return False

    def update_project_approved_by(self, project_id, approved_by):
        query = f"""
        UPDATE public.{self.project_filters_table}
        SET approved_by = %s
        WHERE project_id = %s
        """

        try:
            self.cur.execute(query, (approved_by, project_id))
            if self.cur.rowcount > 0:
                logger.debug(f"Updated approved_by for project with ID {project_id} to {approved_by}")
                return True
            else:
                logger.warning(f"Project ID {project_id} not found. No approved_by updated.")
                return False
        except Exception as e:
            logger.error(f"Error during updating approved_by for project ID {project_id}: {e}")
            return False
        
    def fetch_feedback_data(self):
        query = f"""
        SELECT 
            DATE_TRUNC('week', date) AS week_start,
            COUNT(CASE WHEN feedback = 'like' THEN 1 END) AS likes,
            COUNT(CASE WHEN feedback = 'dislike' THEN 1 END) AS dislikes
        FROM {self.feedback_table}
        GROUP BY week_start
        ORDER BY week_start;
        """
        try:
            self.cur.execute(query)
            rows = self.cur.fetchall()
            columns = [desc[0] for desc in self.cur.description]
            result = [dict(zip(columns, row)) for row in rows]
            
            return result
        except Exception as e:
            logger.error(f"Error fetching feedback data grouped by weeks: {e}")
            return []
        

    def update_projects_status(self, projects_status):
        query = f"""
        UPDATE public.{self.project_filters_table}
        SET status = CASE project_id 
        """
        
        cases = " ".join([f"WHEN %s THEN %s" for _ in projects_status])
        query += cases + " END WHERE project_id IN ({})".format(", ".join(["%s"] * len(projects_status)))
        
        params = []
        for project_id, status in projects_status.items():
            params.extend([project_id, status])
        params.extend(projects_status.keys())
        
        try:
            self.cur.execute(query, params)
            if self.cur.rowcount > 0:
                logger.debug(f"Updated status for {len(projects_status)} projects")
                return True
            else:
                logger.warning("No projects found. No status updated.")
                return False
        except Exception as e:
            logger.error(f"Error updating project statuses: {e}")
            return False
