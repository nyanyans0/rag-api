import pytest
from app.rag.generator import Generator

@pytest.fixture
def generator():
    return Generator()

def test_generator_initialization(generator):
    assert generator.tokenizer is not None
    assert generator.model is not None

def test_generate(generator):
    question = "What does NVIDIA specialize in?"
    context = "NVIDIA is a technology company that specializes in designing graphics processing units (GPUs) for gaming and professional markets."
    answer = generator.generate(question, context)
    
    assert isinstance(answer, str)
    assert len(answer) > 0
    assert "GPU" in answer or "graphics" in answer

def test_generate_with_irrelevant_context(generator):
    question = "What does NVIDIA specialize in?"
    context = "The capital of France is Paris. It is known for its iconic Eiffel Tower."
    answer = generator.generate(question, context)
    
    assert isinstance(answer, str)
    assert len(answer) > 0
    # The model might generate an answer, but it should not contain specific information about NVIDIA
    assert "NVIDIA" not in answer.lower()