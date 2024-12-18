import re
import unicodedata
import numpy as np
import pickle as pkl
from sklearn.metrics.pairwise import cosine_similarity

class Preprocessor:

    def __init__(self):
        self.vectorizer = pkl.load(open("./objects/vectorizer.pkl", "rb"))
        self.scaler = pkl.load(open("./objects/scaler.pkl", "rb"))

    def remove_punctuation(self, text):
        text_with_marks = re.sub(r'(\.\w)', r'__KEEP__\1', text)
        text_no_punctuation = re.sub(r'[.,;:]', ' ', text_with_marks)
        cleaned_text = re.sub(r'__KEEP__ ', '.', text_no_punctuation)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        return cleaned_text.strip()

    def normalize_text(self, text):
        text = text.lower()
        text = unicodedata.normalize('NFD', text)
        text = text.encode('ascii', 'ignore')
        text = text.decode('utf-8')
        text = re.sub(r'\s+', ' ', text).strip()
        text = self.remove_punctuation(text)
        return text

    def remove_accents(self, text):
        accents = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        }
        return ''.join(accents.get(c, c) for c in text)

    def replace_synonyms(self, text):
        text_list = text.split(', ')
        replacements = {"sst": "seguridad y salud en el trabajo", "hse":"seguridad y salud en el trabajo", "machine learning":"ciencia de datos", "deep learning":"ciencia de datos", "analisis de datos":"ciencia de datos"}
        result = []
        for subsentence in text_list:
            if subsentence in replacements:
                result.append(replacements[subsentence])
            else:
                result.append(subsentence)
        return ', '.join(result)
    
    def share_major(self, parsed_resume, parsed_project):
        return int(any(major in parsed_project['majors'] for major in parsed_resume['majors']))

    def experience_diff(self, parsed_resume, parsed_project):
        diff = abs(parsed_resume["experience"] - parsed_project["experience"])
        return {"experience_difference": diff}

    def skills_percentage(self, parsed_resume, parsed_project):
        project_skills = parsed_project['technical_skills']
        if not project_skills: 
            return 1

        resume_skills = set(parsed_resume['technical_skills'])
        matching_skills = resume_skills.intersection(project_skills)

        return len(matching_skills) / len(project_skills)

    def keywords_number(self, parsed_resume, parsed_project):
        if not parsed_project['keywords']:
            return 1

        resume_keywords = set(word for phrase in parsed_resume['keywords'] for word in phrase.split())
        project_keywords = set(word for phrase in parsed_project['keywords'] for word in phrase.split())

        matching_keywords = resume_keywords.intersection(project_keywords)

        keywords_number = len(matching_keywords)
        
        return {"keywords_number": keywords_number}

    def education_level_score(self, parsed_resume, parsed_project):
        score = int(parsed_resume['education_level'] == parsed_project['education_level'])
        return {"education_level_score": score}
        
    def calculate_average_similarity(self, sentence1, sentence2):
        if not sentence1 or not sentence2:
            return 0

        sentences = [sentence1, sentence2]
        tfidf_vectors = self.vectorizer.transform(sentences)

        vector1 = np.array(tfidf_vectors[0])
        vector2 = np.array(tfidf_vectors[1])

        similarities = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0]
        top_similarities = np.sort(similarities)[-5:]
        return np.mean(top_similarities)
    
    def calculate_keywords_similarity(self, parsed_resume, parsed_project):
        sentence1 = self.replace_synonyms(self.normalize_text(', '.join(parsed_project['keywords'])))
        sentence2 = self.replace_synonyms(self.normalize_text(', '.join(parsed_resume['keywords'])))
        similitud = self.calculate_average_similarity(sentence1, sentence2)
        return {"keywords_similarity": similitud}
    
    def calculate_skills_similarity(self, parsed_resume, parsed_project):
        sentence1 = self.replace_synonyms(self.normalize_text(', '.join(parsed_project['technical_skills'])))
        sentence2 = self.replace_synonyms(self.normalize_text(', '.join(parsed_resume['technical_skills'])))
        similitud = self.calculate_average_similarity(sentence1, sentence2)
        return {"skills_similarity": similitud}
    
    def scale(self, data):
        return self.scaler.transform(data)