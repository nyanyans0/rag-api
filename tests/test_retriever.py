import pytest
from app.rag.preprocessor import Preprocessor
from app.rag.retriever import Retriever

@pytest.fixture
def sample_document():
    return """
    NVIDIA is a technology company.
    It specializes in designing graphics processing units (GPUs).
    NVIDIA also produces system on a chip units (SoCs) for the mobile computing and automotive market.
    The company is headquartered in Santa Clara, California.
    NVIDIA's GPUs are widely used in gaming, professional visualization, and artificial intelligence.
    """

@pytest.fixture
def preprocessor_and_retriever(sample_document):
    preprocessor = Preprocessor()
    preprocessor.preprocess(sample_document, chunk_size=20)
    retriever = Retriever(preprocessor)
    return preprocessor, retriever

def test_retriever_initialization(preprocessor_and_retriever):
    _, retriever = preprocessor_and_retriever
    assert retriever.preprocessor is not None

def test_retrieve(preprocessor_and_retriever):
    _, retriever = preprocessor_and_retriever
    question = "What does NVIDIA specialize in?"
    results = retriever.retrieve(question, k=2)
    
    assert len(results) == 2
    assert any("NVIDIA" in chunk for chunk in results)
    assert any("GPUs" in chunk for chunk in results)

def test_retrieve_irrelevant_question(preprocessor_and_retriever):
    _, retriever = preprocessor_and_retriever
    question = "What is the capital of France?"
    results = retriever.retrieve(question, k=1)
    
    assert len(results) == 1  # It will still return a result, but it might not be relevant