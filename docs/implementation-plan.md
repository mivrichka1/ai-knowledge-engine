# Implementation plan and current status

1. Establish supported Python and dependency lock, Compose services, database migrations, authentication and tenant isolation; test against real PostgreSQL/pgvector.
2. Implement validated document uploads and durable, retryable worker with atomic versioned indexing; test recovery and isolation.
3. Add parsing, page-aware chunking, batched embeddings, PostgreSQL lexical and vector retrieval, and reranking; integrate the standalone RRF function with actual result identifiers.
4. Add provider abstraction, grounded RAG, citation checks, extraction, history and streaming; distinguish unavailable provider from insufficient evidence.
5. Implement a verified evaluation dataset, experiment execution and aggregate metrics, reproducibility metadata and export. The existing functions are per-query primitives only.
6. Build accessible frontend, observability, integration and end-to-end tests, CI, documentation and real benchmarks. Verify each feature before claiming readiness.

Current verified scope is limited to the algorithms and tests listed in README. This plan does not imply that remaining steps have been implemented.
