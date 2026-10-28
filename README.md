<div align="center">

# 🗄️ Enterprise Knowledge Management System

### A ChromaDB knowledge server for RAG applications, with a bilingual (Arabic/English) admin dashboard

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange.svg)](https://www.trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Features](#-features) • [Quick Start](#-quick-start) • [API](#-api-reference) • [Deployment](#%EF%B8%8F-deployment) • [Testing](#-testing)

</div>

---

## 📖 Overview

Organizations keep their knowledge in scattered PDFs, Word files and Google Docs. This project turns those
documents into a searchable vector knowledge base that a chatbot or any RAG (Retrieval-Augmented Generation)
application can query over a simple REST API.

Staff upload documents through a web dashboard; the server extracts the text, splits it into chunks, generates
embeddings and stores them in ChromaDB. External applications then call `POST /search` with an API key to retrieve
the most relevant passages for a question.

```
Upload → Extract text → Chunk → Generate embeddings (OpenRouter) → Store in ChromaDB → Semantic search API
```

---

## ✨ Features

### Document processing
| Source | What is extracted |
|--------|-------------------|
| **PDF** | Text layer (scanned PDFs without a text layer are not supported) |
| **DOCX** | Paragraphs and tables |
| **TXT** | Plain text, with UTF-8 / Windows-1256 encoding detection for Arabic files |
| **Google Docs** | Imported directly from a shared document link |

Each document's language is detected (Arabic, English or mixed) and stored with its metadata.

### Access control
- Session-based login with Werkzeug password hashing (Flask-Login)
- Three roles:
  | Role | Can do |
  |------|--------|
  | **Staff** | Upload documents, browse files |
  | **Admin** | + delete files, view system status, health and analytics |
  | **System Manager** | + manage user accounts |
- Separate API-key authentication for external clients (`/search`, `/api/stats`)
- Rate limiting (Flask-Limiter), including a stricter limit on login attempts

### Web dashboard (Arabic UI)
- Statistics, file manager and drag-and-drop upload
- Service health and diagnostics (ChromaDB + embeddings provider)
- Query and activity logs, user management

### Built for small servers
Runs on a 512 MB instance (e.g. Render's free tier): chunk count per document is capped, memory is
garbage-collected after each upload, and gunicorn workers are recycled.

---

## 🚀 Quick Start

```bash
git clone https://github.com/Salman-Naif/chromadb-server-public.git
cd chromadb-server-public

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # then fill in the values below
python app.py                     # http://localhost:8000
```

Minimum `.env`:

```env
SECRET_KEY=<random string>          # python -c "import secrets; print(secrets.token_hex(32))"
OPENROUTER_API_KEY=sk-or-v1-...     # used to generate embeddings
ADMIN_EMAIL=you@example.com         # first System Manager account
ADMIN_PASSWORD=<strong password>
CHROMADB_API_KEY=<random string>    # key external apps send to /search
```

Log in with `ADMIN_EMAIL` / `ADMIN_PASSWORD`. There is **no default password**: if `ADMIN_PASSWORD` is empty on
first run, a random one-time password is printed to the server log.

---

## 📡 API Reference

### Authentication

| Endpoints | Auth |
|-----------|------|
| `POST /search`, `GET /api/stats` | API key: `Authorization: Bearer <CHROMADB_API_KEY>` or `X-API-Key: <key>` |
| Dashboard and `/api/upload`, `/api/file/...`, `/api/user/...` | Logged-in session (role-based) |
| `GET /api/health-check`, `GET /wake-up` | Public |

If `CHROMADB_API_KEY` is not set on the server, API-key endpoints return `503` (API access disabled).

### Semantic search

```bash
curl -X POST https://<your-server>/search \
  -H "Authorization: Bearer $CHROMADB_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the graduation requirements?", "top_k": 5}'
```

```json
{
  "success": true,
  "results": {
    "documents": ["Graduation requires completion of 132 credit hours..."],
    "metadatas": [{"source": "academic_guide.pdf", "file_id": "…", "chunk_index": 12, "lang": "en"}],
    "distances": [0.21]
  },
  "query_time_ms": 38.4
}
```

`top_k` is optional (default 5, max 10). Lower distance means a closer match.

### Main endpoints

| Method | Path | Description | Access |
|--------|------|-------------|--------|
| `POST` | `/search` | Semantic search over the knowledge base | API key |
| `GET` | `/api/stats` | Chunk, file and query counts | API key |
| `POST` | `/api/upload` | Upload a PDF / DOCX / TXT file (multipart `file`) | Session |
| `POST` | `/api/upload/google-docs` | Import a Google Doc (`{"url": "..."}`) | Session |
| `GET` | `/api/file/<file_id>` | File metadata | Session |
| `DELETE` | `/api/file/<file_id>` | Delete a file and its vectors | Admin |
| `POST` | `/api/user/<id>/toggle-status` | Enable / disable a user | System Manager |
| `DELETE` | `/api/user/<id>/delete` | Delete a user | System Manager |
| `GET` | `/api/diagnose` | Full service diagnosis | Admin |
| `GET` | `/api/health-check` | ChromaDB + embeddings status | Public |

The dashboard pages (`/dashboard`, `/upload`, `/files`, `/status`, `/health`, `/analysis`, `/users`, `/profile`)
make up the rest of the 26 routes.

---

## 🏗️ Architecture

```
            ┌───────────────────────────────┐
 Browser ──►│  Flask app (Jinja2 dashboard) │◄── RAG client / chatbot
 (session)  │  Flask-Login · Flask-Limiter  │    (API key → POST /search)
            └──────┬───────────────┬────────┘
                   │               │
          ┌────────▼─────┐   ┌─────▼─────────────┐      ┌──────────────────────┐
          │  ChromaDB    │   │ SQLite / Postgres │      │ OpenRouter embeddings│
          │  (vectors)   │   │ users, files, logs│      │ qwen3-embedding-8b   │
          └──────────────┘   └───────────────────┘      └──────────────────────┘
```

- **ChromaDB** stores the chunk vectors and their metadata.
- **SQLAlchemy** (SQLite locally, PostgreSQL via `DATABASE_URL`) stores users, file metadata and logs.
- **OpenRouter** generates the embeddings (`EMBEDDING_MODEL`, default `qwen/qwen3-embedding-8b`).

---

## ☁️ Deployment

### Render (blueprint included)
`render.yaml` defines the web service and a persistent disk for ChromaDB. `SECRET_KEY` is generated automatically;
set `OPENROUTER_API_KEY`, `ADMIN_PASSWORD` and `CHROMADB_API_KEY` in the Render dashboard.

### Docker

```bash
cp .env.example .env    # fill in SECRET_KEY, ADMIN_PASSWORD, OPENROUTER_API_KEY, CHROMADB_API_KEY
docker compose up -d    # http://localhost:5000
```

Compose refuses to start without `SECRET_KEY` and `ADMIN_PASSWORD`, and data is kept in a named volume.

---

## ⚙️ Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | Flask session key | random per process (sessions reset on restart) |
| `OPENROUTER_API_KEY` | Embeddings provider key | — (required for upload/search) |
| `CHROMADB_API_KEY` | Key for `/search` and `/api/stats` | — (API disabled) |
| `ADMIN_EMAIL` | First System Manager login | `admin@example.com` |
| `ADMIN_PASSWORD` | Its password | — (random, printed once) |
| `DATABASE_URL` | PostgreSQL URL | local SQLite |
| `CHROMADB_PATH` | Vector store path | `./data/chromadb` |
| `CHROMADB_COLLECTION` | Collection name | `knowledge_base` |
| `EMBEDDING_MODEL` | OpenRouter embedding model | `qwen/qwen3-embedding-8b` |
| `EMBEDDING_DIMENSION` | Vector size | `4096` |

Uploads are limited to 50 MB.

---

## 🧪 Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

The test suite covers authentication and security behaviour (no default credentials, open-redirect protection,
API-key checks) and runs without network access.

---

## 📝 License

MIT — see [LICENSE](LICENSE).

## 📧 Contact

**Salman Naif Almuhaysin** — [X: @Snma7_](https://x.com/snma7_) · almuhaysins@gmail.com
