# Current architecture and boundaries

The React client calls FastAPI through the Vite proxy. FastAPI enforces ownership in database queries and writes uploads to private local storage with generated keys. It writes job records to PostgreSQL. A separate worker claims jobs with row locks and leases, extracts text, chunks it and computes embeddings. It writes chunks and embeddings in one transaction. PostgreSQL and pgvector serve lexical and vector search; RRF merges rankings and a cross-encoder can rerank candidates. The optional LLM adapter sends a bounded set of untrusted excerpts to a configured OpenAI-compatible endpoint after per-request consent.

The worker uses a five-minute lease and three attempts. It retries transient errors with a bounded delay, while malformed input produces an error. The atomic transaction prevents partial chunk replacement by a stale worker, but model execution has no hard timeout and file/database storage cleanup is incomplete. SQLite is used in selected API unit tests; PostgreSQL-specific search and migrations are tested separately.

The tokenizer budget is 120 tokens with 16 tokens of overlap for the configured multilingual MiniLM model. This is a conservative choice for its context size, not a measured optimum. Evaluation and performance baselines have not been established.

## Streaming contract

`POST /api/v1/chat/conversations/{id}/stream` returns `text/event-stream` after authorization, explicit transfer consent and retrieval. `delta` events contain provisional provider text and `provisional: true`; the client labels it as unverified. A `final` event is emitted only after the completed answer passes citation-ID checks and is saved to the conversation. An `error` event instructs the client to discard provisional text. An incomplete provider stream, an abnormal finish reason, invalid citation IDs, overlong output or failure to save prevents a final answer. The client can disconnect to cancel; an in-flight provider request may already have completed, so the history is refreshed afterwards. The generation has a 120-second server-side timeout, subject to cancellation propagation and provider behavior.

New assistant messages record the prompt version, configured model name when available, outcome and elapsed processing time. Messages saved before migration 0014 have no generation metadata. This is partial provenance: the application does not currently record provider token usage or all generation settings for each answer. Citation membership and coverage of selected documents do not establish whether the claims are supported by their cited text.

These checks validate membership and presence of citations for selected documents, not that claims are entailed by those citations. Streaming is tested against deterministic HTTP and provider doubles; a real LLM streaming call and the complete PostgreSQL/browser path remain unverified.

For PDF pages with an unambiguous separation into two vertically overlapping columns, the parser reads each column top to bottom. If a line crosses the middle within the column area or the layout is ambiguous, it falls back to PyMuPDF sorted extraction. This is a heuristic, not general layout analysis: mixed text, figures, tables and OCR require review. Newly indexed PDF documents store the parser version; existing indexes require reindexing to use the new ordering.
