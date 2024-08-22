# RAG API Project

This project implements a Retrieval-Augmented Generation (RAG) API using FastAPI, Sentence Transformers, and scikit-learn. It processes a given NVIDIA document and answers questions based on the content.

## Prerequisites

- Python 3.12
- pip (latest version)

## Setup

1. Clone the repository:

   ```
   git clone https://github.com/nyanyans0/rag-api.git
   cd rag-api
   ```

2. Create a virtual environment:

   ```
   python -m venv venv
   ```

3. Activate the virtual environment:

   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```

4. Upgrade pip:

   ```
   pip install --upgrade pip
   ```

5. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

```
rag_api/
├── app/
│   ├── main.py
│   ├── rag/
│   │   ├── preprocessor.py
│   │   ├── retriever.py
│   │   └── generator.py
│   └── utils/
│       └── pdf_to_text.py - script to transform PDF to TXT
├── data/
│   └── nvidia_document.txt - extracted text from PDF
│   └── NVIDIAAn.pdf - input document
├── tests/
│   ├── test_preprocessor.py
│   ├── test_retriever.py
│   ├── test_generator.py
│   └── test_api.py
├── bruno/ - testing api configuration to use with bruno
├── requirements.txt
└── README.md
```

## Running the Project

1. Preprocess the NVIDIA document:

   ```
   python -c "from app.rag.preprocessor import Preprocessor; p = Preprocessor(); p.preprocess(open('data/nvidia_document.txt').read()); p.save('index')"
   ```

2. Start the FastAPI server:

   ```
   uvicorn app.main:app --reload
   ```

3. The API will be available at `http://127.0.0.1:8000`. You can use the `/rag` endpoint to ask questions about the NVIDIA document.

## Testing

Run the tests using pytest:

```
pytest
```

## API Usage

Send a POST request to the `/rag` endpoint with a JSON body containing a "question" field. For example:

```
curl -X POST "http://127.0.0.1:8000/rag" -H "Content-Type: application/json" -d "{\"question\":\"What was NVIDIA's revenue in Q2 2024?\"}"
```

The API will return a JSON response with an answer and relevant chunks from the document.

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
