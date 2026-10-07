# RAG Lab

A Django web app that demonstrates **Retrieval-Augmented Generation (RAG)**: answering a question by first *finding* the relevant documents, then *building an answer only from them*.

It ships with a small demo knowledge base (a dental clinic: refund policy, opening hours, insurance, prices, and so on), six different RAG strategies you can compare side by side, security guardrails, quality monitoring, and a ready-made deployment to AWS.

You can **upload your own files** (PDF, Word, text, CSV, images, audio) on the Upload page and ask questions about them. Without uploads it answers from the built-in demo documents.

---

## What it does

You type a question, for example *"What is the refund policy?"*. The app then:

1. **Checks the question** (input guardrail): removes personal data such as emails and card numbers, and blocks prompt-injection attempts and harmful requests.
2. **Retrieves** the most relevant documents from the knowledge base using one of six strategies.
3. **Generates an answer** from those documents only. By default it picks the best sentences from the sources. If `USE_LLM=1`, an LLM (Groq) writes the answer from the sources.
4. **Checks the answer** (output guardrail): every sentence must be supported by the retrieved sources, otherwise the answer is replaced with a refusal.
5. **Logs** the question, latency, and groundedness score for monitoring.

The dashboard shows every step, so you can see exactly what each strategy did.

## Using the dashboard

1. **Choose a RAG method.** Click one of the six method cards, or "Compare all six" to run them side by side.
2. **Choose your data.** Search everything, only the demo clinic data, or only your uploaded files. Pick the access level (public, or staff after sign-in).
3. **Ask a question** and press Run.

The result shows the answer, a groundedness score (how well the answer is backed by the sources), response time, the sources used, and a step-by-step trace of what happened.

## The six RAG strategies

| Strategy | Idea |
|---|---|
| **Multimodal** | Searches text, plus images and audio stored as captions and transcripts |
| **Advanced** | Query expansion, then re-ranking of the results |
| **Adaptive** | Chooses how much retrieval effort to spend based on the question |
| **Corrective** | Checks result quality and falls back to a second source (a "web" set) if poor |
| **Graph** | Uses a small knowledge graph of entities and relations (for example Dr. Rivera, works at, Downtown branch) |
| **Agentic** | A step-by-step plan that combines the other tools |

The pipeline is orchestrated with **LangGraph**. Retrieval uses **BM25** (keyword ranking), so the project runs without embeddings or API keys.

## Pages and API

| URL | Purpose |
|---|---|
| `/` | Dashboard: choose a method, choose your data, ask, read the result |
| `/upload/` | Upload and manage your files (sign in with `RAG_API_KEY`) |
| `/lab/<strategy>/` | Opens the dashboard with that method already selected |
| `/guardrails/` | Try attacks and see how the input guardrail reacts |
| `/mlops/` | Latency, groundedness, blocked counts, recent queries, evaluation runs |
| `/guide/` | Beginner guide |
| `/healthz` | Health check |
| `POST /api/ask/` | JSON API (needs an `X-API-Key` header) |

```bash
curl -X POST http://localhost:8000/api/ask/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your RAG_API_KEY>" \
  -d '{"question": "What is the refund policy?", "pipeline": "advanced", "role": "public"}'
```

The `role` can be `public` or `staff`. Documents marked `staff` (such as the payroll note) are **never retrieved** for `public` callers.

---

## Run it locally

You need Python 3.12.

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt

$env:DJANGO_DEBUG = "1"
$env:RAG_API_KEY = "dev-api-key"
python manage.py migrate
python manage.py test
python manage.py runserver
```

On macOS or Linux use `source venv/bin/activate` and `export DJANGO_DEBUG=1 RAG_API_KEY=dev-api-key`.

Open http://127.0.0.1:8000.

### Settings (environment variables)

| Variable | Meaning | Default |
|---|---|---|
| `DJANGO_SECRET_KEY` | Required when `DJANGO_DEBUG` is not `1` | none |
| `DJANGO_DEBUG` | `1` for local development | `0` |
| `DJANGO_ALLOWED_HOSTS` | Comma-separated hostnames or IPs | `localhost,127.0.0.1` |
| `CSRF_TRUSTED_ORIGINS` | Comma-separated origins, for example `https://example.com` | empty |
| `DATABASE_URL` | Postgres URL. Empty means SQLite | SQLite |
| `RAG_API_KEY` | Key required by `/api/ask/` | empty (API disabled) |
| `RATE_LIMIT_PER_MIN` | Requests per minute per IP | `30` |
| `SECURE_SSL_REDIRECT` | `1` when serving over HTTPS | `1` |
| `USE_LLM` | `1` to let an LLM write answers | `0` |
| `GROQ_API_KEY` | Your Groq key (only when `USE_LLM=1`) | none |
| `LLM_MODEL` | Groq model that writes answers | `openai/gpt-oss-120b` |
| `VISION_MODEL` | Groq model that reads images and scanned PDF pages | `qwen/qwen3.8-27b` |
| `WHISPER_MODEL` | Groq model that transcribes audio | `whisper-large-v3-turbo` |
| `USE_DEMO_KB` | `0` removes the demo clinic documents so answers come only from your uploads | `1` |

Never commit real keys. `.env` and `.env.prod` are in `.gitignore`.

### Optional: use an LLM (Groq)

```powershell
$env:USE_LLM = "1"
$env:GROQ_API_KEY = "<your key>"
```

The LLM only sees the retrieved sources, and the output guardrail still checks its answer. If the LLM call fails, the app falls back to the extractive answer.

---

## Using your own documents

Open **Upload** in the menu and sign in with the server's `RAG_API_KEY`. Then choose files.

| Type | Formats | How it is read |
|---|---|---|
| Documents | PDF, Word (.docx), .txt, .md, .csv | Text is extracted on the server. Scanned PDF pages (no text layer) are read by the vision model, up to 8 pages per file |
| Images | PNG, JPG, WebP, GIF | The vision model describes the image and transcribes any visible text |
| Audio | MP3, WAV, M4A, OGG, FLAC, WebM | Whisper transcribes the speech |

Images, audio and scanned pages need `GROQ_API_KEY` on the server. Without it, add your own description or transcript in the optional box on the Upload page.

What happens to a file:

1. The text is extracted and cut into chunks of about 700 characters.
2. The chunks are stored in the database (the original file is **not** kept) and become searchable straight away.
3. Every strategy can retrieve them. The **Multimodal** strategy shows text, image and audio hits side by side.

Each file is marked **public** (anyone) or **staff** (only signed-in users). Choosing the staff role in the Lab requires the sign-in from the Upload page.

Limits: 10 MB per document, 12 MB per image, 24 MB per audio file, 4000 chunks in total. The demo clinic documents stay searchable unless you set `USE_DEMO_KB=0`. The Graph strategy uses the built-in clinic entity graph only, so it does not use your uploads.

To replace the demo data in code instead, edit `rag/kb.py` and `rag/eval_set.py`.

---

## Quality: tests and evaluation

```bash
python manage.py test                         # unit tests
python manage.py run_eval --min-hit-rate 0.8  # golden-question check; fails if hit rate is below 80%
```

CI runs both before every deploy.

## Deployment (AWS)

Pushing to `main` triggers `.github/workflows/ci.yml`:

1. **Test**: unit tests and the evaluation check.
2. **Build**: Docker image, pushed to Amazon ECR, tagged with the commit.
3. **Deploy**: GitHub Actions asks AWS Systems Manager (SSM) to run `deploy/remote-deploy.sh` on an EC2 server. No SSH and no open port 22.
4. **Verify**: the script waits for `/healthz` before reporting success.

On the server, Docker Compose (`docker-compose.prod.yml`) runs three containers: the Django app (gunicorn), Postgres, and Caddy as the web server.

**Authentication:** GitHub gets short-lived AWS credentials through OIDC. There are no long-lived AWS keys stored in GitHub.

### GitHub settings needed

| Type | Name | Value |
|---|---|---|
| Secret | `AWS_DEPLOY_ROLE_ARN` | ARN of the IAM role GitHub assumes |
| Variable | `AWS_REGION` | for example `us-east-1` |
| Variable | `EC2_INSTANCE_ID` | the EC2 instance ID |

### Server configuration

App settings and secrets (database password, `DJANGO_SECRET_KEY`, `GROQ_API_KEY`) live in `/opt/raglab/.env.prod` **on the server**, never in git. To change one, edit that file over SSM, then redeploy so the container restarts.

### HTTPS

Without a domain name the app runs on plain HTTP. For HTTPS, point a domain at the server, set `SITE_ADDRESS` to the domain, set `DJANGO_ALLOWED_HOSTS` and `CSRF_TRUSTED_ORIGINS`, and keep `SECURE_SSL_REDIRECT=1`.

---

## Project layout

```
config/            Django settings, URLs, WSGI
rag/
  kb.py            Demo knowledge base (uploads are added to it at search time)
  retrieval.py     BM25 search, filters, re-ranking, answer extraction
  pipelines.py     The six RAG strategies
  orchestrator.py  LangGraph flow: input guard, retrieve, generate, output guard
  guardrails.py    Input and output safety checks
  llm.py           Optional Groq LLM generation
  ingest.py        Reads PDF / Word / text / image / audio uploads into chunks
  services.py      Runs a question and logs it
  models.py        QueryLog, EvalRun, Upload and Chunk tables
  views.py         Dashboard, upload page and the JSON API
  eval_set.py      Golden questions for evaluation
  templates/       Web pages (dashboard.html, upload.html, ...)
  static/rag/      app.css
deploy/            Server deploy script
Dockerfile         Production image
docker-compose.prod.yml, Caddyfile
.github/workflows/ci.yml   Test, build, deploy
```

## Notes

- `deploy/ecs-task-def.json` is a leftover from an earlier ECS-based setup and is not used by the current EC2 deployment.
- The demo knowledge base is fictional.
- Uploaded text is stored in the database, so it survives redeploys. Uploaded documents may contain instructions aimed at the AI; the system prompt tells the model to ignore them and the output guardrail checks every sentence against the sources.
