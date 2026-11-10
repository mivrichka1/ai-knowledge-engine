# Verification record

## 2026-11-10 local development run

- Docker Compose images were rebuilt; PostgreSQL, migration, API, worker and frontend started. The existing Compose volume was retained.
- A separate empty PostgreSQL database was created, migrated through Alembic head, checked for schema drift, exercised by five PostgreSQL integration tests and then removed. All five tests passed.
- The full Python suite with PostgreSQL and the locally cached real embedding/reranking models enabled: **104 passed, 2 skipped**. The two skipped tests require a configured external LLM provider and would make real requests. This is a test count, not a statement of production readiness.
- Ruff and mypy checks passed; the frontend TypeScript/Vite production build passed.
- A Playwright browser test passed on the Compose application: account creation, PDF upload and indexing, keyword search, strategy comparison, chunk browsing, labelled evaluation, reindexing, opening the PDF page and deletion. The screenshot in `docs/screenshots/workspace.png` uses the test fixture and was captured during this test.
- A ten-query partially labelled retrieval experiment on *Attention Is All You Need* was run against the Compose app; the full results and caveats are in `docs/evaluation.md` and `evaluation/runs/attention-seed-v2.json`.

These checks do not validate real-provider generation or extraction, factual entailment of citations, a representative dataset, production security, sustained load or an actual GitHub Actions run. Some API tests use test doubles by design. The Compose application was not run against a fresh empty Compose volume in this particular verification; the separate clean PostgreSQL migration and test described above used a disposable database.

## Additional evaluation and multi-document smoke run

- Downloaded BERT v2 and RoBERTa v1 from arXiv, recorded PDF hashes and ran ten anchor queries per paper across four retrieval strategies; see `docs/evaluation.md` for results and limitations.
- Indexed Attention, BERT and RoBERTa into one knowledge base and confirmed that each of three filtered search queries returned five results belonging to the selected document. One unrestricted query returned results from two documents in its top 20. This did not involve an LLM.
- Investigated the two-column extraction issue on page 4 of both BERT and RoBERTa: `page.get_text(sort=True)` orders text across columns by vertical position, interleaving unrelated lines. Unsorted extraction happens to read those pages more naturally, but PDF content order can vary, so switching all documents without a representative layout regression suite would introduce a different ordering risk. No global parser change was made.

## PDF column handling follow-up

After reproducing interleaved columns, the parser gained a conservative two-column ordering heuristic with fallback for mixed layouts. Synthetic two-column, single-column and mixed-layout checks passed. Page 4 of each BERT and RoBERTa was also inspected through actual extraction, and their anchor experiments were repeated. A full suite with PostgreSQL and real local ML models reported **107 passed, 2 skipped**; the skipped tests require a configured real LLM provider. The Playwright workspace scenario passed following the parsing change. This heuristic is not a general layout analysis or OCR solution; complex PDFs still require manual review.

## Local LLM investigation

A temporary local OpenAI-compatible test server ran real CPU inference with `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`, on a synthetic document. Adapter smoke checks for ordinary and streamed responses passed; after adding a citation-format example to `rag-v2`, the ordinary single-source adapter smoke also included the expected source ID. A separate full application test with real PostgreSQL, embeddings, retrieval and the same model failed citation validation for ordinary and streamed RAG, and extraction returned 502 because the model output did not satisfy its required schema/evidence. The streamed response was discarded and no conversation messages were saved. These are actual model failures, not evidence that generation or extraction works reliably. The test server was a temporary local development adapter, not a production provider implementation. No private document or external LLM API was used.

A second local CPU run used `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, through the same temporary test server and disposable PostgreSQL database. On one synthetic question, both ordinary and genuinely streamed generation returned an answer with an allowed source ID and the conversation stored two messages. On a second synthetic question, the model omitted the source ID; both responses were rejected and no messages were saved. A question lacking an answer was also rejected for invalid citations, not confirmed as a successful abstention. Structured extraction returned 502 on both synthetic examples; direct inspection showed incorrect schema and invented details. A separate run against the open Attention PDF again returned missing/invalid citations for both ordinary and streamed responses; no messages were saved. The temporary test server and synthetic examples do not establish general LLM quality. Citation membership is not an entailment check, and a failed response should not be interpreted as successful abstention.
