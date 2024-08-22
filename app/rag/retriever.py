import numpy as np

class Retriever:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def retrieve(self, question, k=5):
        question_vector = self.preprocessor.vectorizer.transform([question])
        distances, indices = self.preprocessor.nn.kneighbors(question_vector, n_neighbors=k)
        return [self.preprocessor.chunks[i] for i in indices[0]]