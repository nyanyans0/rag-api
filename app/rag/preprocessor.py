from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import joblib

class Preprocessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.nn = None
        self.chunks = []

    def preprocess(self, document, chunk_size=200):
        # Split document into chunks
        words = document.split()
        self.chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        
        # Generate TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
        
        # Create NearestNeighbors model
        self.nn = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.nn.fit(tfidf_matrix)

    def save(self, filename):
        np.save(f"{filename}_chunks.npy", self.chunks)
        joblib.dump(self.vectorizer, f"{filename}_vectorizer.joblib")
        joblib.dump(self.nn, f"{filename}_nn.joblib")

    def load(self, filename):
        self.chunks = np.load(f"{filename}_chunks.npy", allow_pickle=True)
        self.vectorizer = joblib.load(f"{filename}_vectorizer.joblib")
        self.nn = joblib.load(f"{filename}_nn.joblib")