# MedLlama — Medical AI Chatbot (TinyLlama + GGUF)

This project is a simple medical chatbot built using a fine-tuned TinyLlama model.
The model was trained on MedQuAD, converted to GGUF format, and served through a FastAPI backend.

The goal of this project is to show the full pipeline:
training → fine-tuning → conversion → backend → working API.

---

## What this project does

* Takes user questions about medical topics
* Sends them to a local LLM (GGUF model via llama.cpp)
* Returns a generated response through an API
* Can be connected to any frontend (HTML/JS, React, etc.)

---

## Tech stack

* Python (FastAPI)
* llama-cpp-python (GGUF inference)
* TinyLlama (fine-tuned with LoRA)
* Pydantic (data validation)
* Uvicorn (server)

---

## Project structure

```
backend/
 ├── app/
 │   ├── main.py        # FastAPI entry point
 │   ├── llm.py         # model loading + generation
 │   ├── config.py      # settings
 │   ├── routes/
 │   │   └── chat.py    # API endpoints
 │   └── schemas/       # request/response models
 ├── models/
 │   └── model-q4.gguf  # quantized model
 └── .env               # configuration
```

---

## How to run locally

1. Clone the repository:

```
git clone <your-repo-link>
cd MedLlama/backend
```

2. Create virtual environment:

```
python -m venv venv
venv\Scripts\activate   # Windows
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Make sure your GGUF model is inside:

```
backend/models/model-q4.gguf
```

5. Run the server:

```
uvicorn app.main:app --reload --port 8001
```

---

## Test the API

Open in browser:

```
http://127.0.0.1:8001/docs
```

You can test `/api/chat` directly from Swagger UI.

---

## Example request

```
POST /api/chat
```

```
{
  "messages": [
    {
      "role": "user",
      "content": "What are the symptoms of diabetes?"
    }
  ]
}
```

---

## Notes

* This is a local LLM setup (no external API required)
* Performance depends on your computer
* Lower temperature is recommended for more accurate medical responses
* The model is not a substitute for professional medical advice

---

## Author

Built as a learning + portfolio project to understand end-to-end LLM systems.
