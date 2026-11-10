# ADR 0002: Multilingual embeddings, RRF and a cross-encoder

**Context:** Both exact technical terms and paraphrased natural-language questions are important. Scores from full-text and vector retrieval are not directly comparable.

**Alternatives:** Keyword only; vector only; calibrated score fusion; rank fusion with an optional reranker.

**Decision:** Use multilingual MiniLM embeddings, PostgreSQL full-text search, reciprocal rank fusion and an optional cross-encoder. Fix model revisions and store them with vectors.

**Rationale:** RRF combines ranked lists without treating unlike raw scores as equivalent. The embedding model runs on CPU, while the cross-encoder is reserved for a smaller candidate set.

**Trade-offs:** A reranker cannot recover chunks missing from its candidate set; the selected cross-encoder is English-oriented; the current small seed dataset does not establish general performance. The configured chunk size and overlap are starting values, not measured optima.
