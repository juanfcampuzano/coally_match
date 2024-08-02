from sklearn.exceptions import NotFittedError
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class CustomTfidfVectorizer(TfidfVectorizer):
    def __init__(
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        analyzer="word",
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.float64,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    ):
        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
        )
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.tfidf_table = None

    def fit_transform(self, raw_documents, **kwargs):
        # Verifica que los documentos no estén vacíos
        if not raw_documents:
            raise ValueError("No documents provided for fitting and transformation.")

        # Verifica que raw_documents sea una lista de strings
        if not isinstance(raw_documents, (list, np.ndarray)):
            raise TypeError("raw_documents should be a list or numpy array of strings.")

        if isinstance(raw_documents, np.ndarray):
            if raw_documents.dtype != object:
                raise ValueError("If raw_documents is a numpy array, it should be of dtype object.")
            raw_documents = raw_documents.tolist()

        # Llama al método fit_transform original de TfidfVectorizer
        tfidf_matrix = super().fit_transform(raw_documents, **kwargs)

        # Manejo de casos en los que la matriz es vacía
        if tfidf_matrix.shape[0] == 0:
            raise ValueError("The transformation resulted in an empty matrix.")
        
        feature_names = self.get_feature_names_out()

        
        self.tfidf_table = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

        return tfidf_matrix

    def transform(self, raw_documents, **kwargs):
        # Verifica si el vectorizador ha sido ajustado
        if not hasattr(self, 'vocabulary_'):
            raise NotFittedError("This CustomTfidfVectorizer instance is not fitted yet.")
        
        # Verifica que los documentos no estén vacíos
        if not raw_documents:
            raise ValueError("No documents provided for transformation.")

        # Verifica que raw_documents sea una lista de strings
        if not isinstance(raw_documents, (list, np.ndarray)):
            raise TypeError("raw_documents should be a list or numpy array of strings.")

        if isinstance(raw_documents, np.ndarray):
            if raw_documents.dtype != object:
                raise ValueError("If raw_documents is a numpy array, it should be of dtype object.")
            raw_documents = raw_documents.tolist()

        all_average_vectors = []

        # Recorre la lista de listas de términos
        for terms in [doc.replace(',','').strip().split() for doc in raw_documents]:
            term_vectors = []

            # Recorre la lista de términos y busca las columnas en el DataFrame
            for term in terms:
                if term in self.get_feature_names_out():
                    if term in self.tfidf_table.columns:
                        # Agrega el vector de la columna encontrada
                        term_vector = self.tfidf_table[term].values
                        term_vectors.append(term_vector)
                    else:
                        pass
                        # print(f"Term '{term}' not found in DataFrame columns.")
                else:
                    pass
                    # print(f"Term '{term}' not found in vocabulary.")

            # Verifica si se encontraron vectores
            if term_vectors:
                # Concatena los vectores y calcula el promedio
                term_vectors = np.vstack(term_vectors)
                average_vector = np.mean(term_vectors, axis=0)
                all_average_vectors.append(average_vector)
            else:
                # Agrega un vector vacío si no se encontraron términos
                all_average_vectors.append(np.array([0.]*len(self.tfidf_table)))

        return all_average_vectors