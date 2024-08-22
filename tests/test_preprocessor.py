import pytest
from app.rag.preprocessor import Preprocessor
import numpy as np
import os

@pytest.fixture
def sample_text():
    return "This is a sample document. It contains multiple sentences for testing purposes. The preprocessor should handle this text appropriately."

def test_preprocessor_initialization():
    preprocessor = Preprocessor()
    assert preprocessor.vectorizer is not None
    assert preprocessor.nn is None
    assert len(preprocessor.chunks) == 0

def test_preprocess(sample_text):
    preprocessor = Preprocessor()
    preprocessor.preprocess(sample_text, chunk_size=10)
    
    assert len(preprocessor.chunks) > 0
    assert preprocessor.nn is not None

def test_save_and_load(sample_text, tmp_path):
    preprocessor = Preprocessor()
    preprocessor.preprocess(sample_text)
    
    save_path = tmp_path / "test_index"
    preprocessor.save(save_path)
    
    assert os.path.exists(f"{save_path}_chunks.npy")
    assert os.path.exists(f"{save_path}_vectorizer.joblib")
    assert os.path.exists(f"{save_path}_nn.joblib")
    
    new_preprocessor = Preprocessor()
    new_preprocessor.load(save_path)
    
    assert np.array_equal(preprocessor.chunks, new_preprocessor.chunks)
    assert preprocessor.vectorizer.get_feature_names_out().tolist() == new_preprocessor.vectorizer.get_feature_names_out().tolist()