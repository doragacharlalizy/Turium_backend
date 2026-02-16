# RAG Inbox Project (Turium_backend)
 
 A lightweight Django project that provides a Retrieval-Augmented Generation (RAG) style API for managing content, splitting it into chunks, and querying via an LLM-backed service.
 
 This repository contains a Django project (`config/`) and a single app `rag_inbox/` with services and utilities to integrate with an LLM client.
 
 ## Repository layout
 
 - `manage.py` - Django management entrypoint.
 - `requirements.txt` - Python dependencies.
 - `config/` - Django project settings and WSGI/ASGI entrypoints.
 - `rag_inbox/` - main Django app:
	 - `models.py`, `serializers.py`, `views.py`, `urls.py`
	 - `services/llm_client.py` - adapter to LLM provider(s)
	 - `utils/chunking.py` - text chunking utilities
	 - `utils/logging.py` - project logging utilities
 - `Procfile`, `runtime.txt` - deployment hints (Heroku-like).
 
 ## Purpose / Overview
 
 This project exposes an API to:
 
 - Create and manage content resources.
 - Split content into chunks suitable for vectorization or retrieval.
 - Query the content using a Query endpoint that integrates with an LLM backend.
 
 The REST endpoints are registered via a DRF `DefaultRouter` in `rag_inbox/urls.py`. Routes available (registered in the router):
 
 - `/content/` - ContentViewSet (create, list, retrieve, update, delete)
 - `/chunks/` - ChunkViewSet (manage content chunks)
 - `/query/` - QueryViewSet (query the stored content/execute RAG flows)
 

 
 ## Quick start (Windows / PowerShell)
 
 1. Create and activate a virtual environment:
 
 ```powershell
 python -m venv .venv
 .\.venv\Scripts\Activate.ps1
 ```
 
 2. Install dependencies:
 
 ```powershell
 pip install --upgrade pip
 pip install -r requirements.txt
 ```
 
 3. Configure environment variables (example, see "Environment variables" below):
 
 - Set at least `DJANGO_SECRET_KEY` and a database connection (or leave default SQLite settings in `config/settings.py` for local dev):
 
 ```powershell
 $env:DJANGO_SECRET_KEY = 'replace-me-with-a-secure-key'
 $env:DEBUG = 'True'
 # optionally, for an external DB:
 # $env:DATABASE_URL = 'postgres://USER:PASS@HOST:PORT/DBNAME'
 # $env:OPENAI_API_KEY = 'sk-...'
 ```
 
 4. Run migrations and start the dev server:
 
 ```powershell
 python manage.py migrate
 python manage.py createsuperuser  # optional
 python manage.py runserver 0.0.0.0:8000
 ```
 
 Open http://127.0.0.1:8000/ to see the API root.
 
 ## Environment variables
 
 The project expects (or commonly benefits from) the following environment variables:
 
 - DJANGO_SECRET_KEY - a secret key for Django.
 - DEBUG - set to `False` in production.
 - DATABASE_URL - optional; if provided, used to configure the DB.
 - OPENAI_API_KEY or other LLM provider keys - if `services/llm_client.py` uses them.
 - OPTIONAL: any settings referenced in `config/settings.py` for external storage, cache, or SENTRY.
 
 If these are not set, the project will likely run locally with default SQLite configuration. Check `config/settings.py` for exact names.
 
 ## API Usage examples
 
 Assuming the server runs at `http://localhost:8000/` and the router is mounted at the root:
 
 - List content:
 
	 GET http://localhost:8000/content/
 
 - Create content (example JSON):
 
	 POST http://localhost:8000/content/
	 {
		 "title": "My document",
		 "body": "Long text of the document..."
	 }
 
 - List chunks:
 
	 GET http://localhost:8000/chunks/
 
 - Query (RAG) endpoint (example):
 
	 POST http://localhost:8000/query/
	 {
		 "query": "Explain the main idea of the document",
		 "top_k": 5
	 }
 
 The exact request/response formats are defined by the serializers and views in `rag_inbox/serializers.py` and `rag_inbox/views.py`.
 
 ## Development notes
 
 - Code related to LLM integration is in `rag_inbox/services/llm_client.py`. Check that file to see what environment variables or credentials it requires.
 - Chunking logic is in `rag_inbox/utils/chunking.py` — adjust chunk sizes and overlap strategies there if you need different behavior.
 - The project uses Django REST Framework (DRF) and the `DefaultRouter` (see `rag_inbox/urls.py`). If you add new ViewSets, register them with the router.
 
 ## Running tests
 
 If `rag_inbox/tests.py` or other test modules exist, run the test suite with:
 
 ```powershell
 python manage.py test
 ```
 

 
 ## Deployment hints
 
 - `Procfile` and `runtime.txt` are present — they indicate the project may be set up for Heroku-like deployments. Check `Procfile` for the exact web command.
 - Make sure `DEBUG=False`, `ALLOWED_HOSTS` are configured, and secret keys/credentials are set as environment variables in production.
 
 ## Assumptions
 
 - The app exposes `ContentViewSet`, `ChunkViewSet`, and `QueryViewSet` in `rag_inbox/views.py` (these are registered in `rag_inbox/urls.py`).
 - `services/llm_client.py` contains an adapter to an external LLM provider (OpenAI or similar) and will require API keys.
 - Database defaults in `config/settings.py` will allow local development with SQLite unless `DATABASE_URL` or other settings are changed.

## 🔗 Related Repositories

- **Frontend Application**: [Turium Frontend](https://github.com/doragacharlalizy/Turium_frontend) - React js
