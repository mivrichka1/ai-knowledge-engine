# Portfolio notes — draft requiring personal review

This project is under development and was created with agent assistance. These descriptions describe observed project behavior, not proof that the portfolio owner independently implemented or fully understands each component. Before using a statement as a personal accomplishment, study the referenced code and adapt the wording to reflect your actual contribution.

**English description:** AI Knowledge Engine is a work-in-progress document search application with a FastAPI backend, PostgreSQL/pgvector, background indexing, multilingual embeddings, hybrid retrieval, reranking and an optional OpenAI-compatible RAG adapter.

**Repository description:** Work-in-progress document intelligence platform exploring hybrid search, pgvector and evidence-linked RAG.

**Project-oriented résumé wording, subject to verification and personal contribution:**

- Built a document indexing and search prototype with FastAPI, PostgreSQL/pgvector, page-aware PDF extraction, multilingual embeddings and hybrid retrieval.
- Added an evaluation workflow for labelled queries and compared retrieval configurations on a ten-question, partially labelled seed set; the sample is too small for general quality conclusions.
- Added automated API, algorithm and browser tests, durable background jobs, database migrations and local Docker Compose configuration.

**Technologies present:** Python, FastAPI, SQLAlchemy, Alembic, PostgreSQL, pgvector, PyMuPDF, Sentence Transformers, PyTorch, React, TypeScript, Vite, Docker Compose, pytest and Playwright.

**Measured results:** The five- and ten-question seed experiments and their limitations are recorded in `docs/evaluation.md`. Test counts and coverage change as development proceeds; use only freshly verified values. PostgreSQL migrations and integration tests have been run locally; a real external LLM integration still requires the checks described in the README.

**Suggested 60-second pitch:** “This project explores how to turn uploaded documents into a searchable knowledge base. A background worker extracts and chunks text, calculates embeddings and stores them in PostgreSQL with pgvector. The application can compare lexical, semantic and hybrid retrieval and provides source-linked answers through a configurable language-model adapter. I would explain the difference between retrieval and generation, show how the evaluation pipeline works, and be explicit that the current dataset is small and the system is still being developed.”
