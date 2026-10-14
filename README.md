# E-commerce Customer Support — Multi-Agent System

## 1. Project Overview

This project implements an e-commerce customer-support system using Python, LangGraph, FastAPI, and Streamlit. It retrieves predefined answers from a custom support dataset and routes queries through agents for intent identification, sentiment analysis, response validation, and escalation.

## 2. Technologies Used

- Python
- Pandas and scikit-learn
- LangChain
- LangGraph
- FastAPI and Uvicorn
- Streamlit
- Pytest
- TF-IDF and cosine similarity

## 3. Project Features

- Customer-support question and answer retrieval
- Query intent and category identification
- Rule-based sentiment analysis
- Response validation
- Conditional workflow routing and escalation
- Session-based conversation-memory component
- Feedback storage in JSONL format
- FastAPI endpoints for health checks and customer questions
- Streamlit chat interface
- Initial retrieval evaluation and automated tests

## 4. Project Structure

```text
capstone-rag-agents/
├── artifacts/
│   ├── vector_store/
│   ├── feedback.jsonl
│   └── support_graph.png
├── data/
│   └── my_support_data.csv
├── src/
│   ├── agents/
│   ├── api/
│   ├── evaluation/
│   ├── feedback/
│   ├── graph/
│   ├── ingestion/
│   ├── memory/
│   ├── retrieval/
│   ├── tools/
│   └── ui/
├── tests/
├── requirements.txt
└── README.md
```

## 5. Setup

Activate the project's virtual environment and install dependencies:

```bash
python -m pip install -r requirements.txt
```

## 6. Run the Application

Start the FastAPI backend in one terminal:

```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Open the API documentation at `http://127.0.0.1:8000/docs`.

Start Streamlit in a second terminal:

```bash
python -m streamlit run src/ui/app.py
```

Open the UI at `http://localhost:8501`.

## 7. Evaluation

Run the retrieval intent-match evaluation:

```bash
python -m src.evaluation.evaluate_support
```

This evaluation reports whether the retrieved intent matches the expected intent for each configured test question.

## 8. Automated Testing

Run the support component tests:

```bash
python -m pytest tests/test_support.py -v
```

## 9. Current Limitations

- The current small-dataset answer generator retrieves predefined responses rather than generating new responses with an LLM.
- Query understanding and sentiment analysis use rule-based logic.
- Conversation memory is implemented separately and is not yet integrated into the main graph.
- Feedback is stored locally and is not automatically used to retrain or improve the system.
- Human-support escalation currently returns a message; it does not create a ticket or connect to a live agent.
- The evaluation currently covers a small set of test questions and does not represent comprehensive model accuracy.

## 10. Conclusion

The project combines retrieval-based customer support, multi-agent processing, conditional routing, API access, and a chat interface. Initial component tests and end-to-end API/UI checks have been performed. Further evaluation and integration can improve reliability and completeness.