# RAG Lab

Django app with a screen for each RAG architecture (Multimodal, Advanced, Adaptive, Corrective, Graph, Agentic), a LangGraph pipeline that wraps every request in input and output guardrails, MLOps logging and evaluation, and AWS ECS deployment files.

Start here: run the steps below, then open `/guide/` for the full beginner guide (also `BEGINNER_GUIDE.html`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && export $(grep -v '^#' .env | xargs)
python manage.py migrate && python manage.py test && python manage.py runserver
```
