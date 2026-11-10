# Retrieval and model notes

The index worker extracts text per page, normalizes Unicode and creates overlapping chunks with the embedding model's tokenizer. The current budget is 120 tokens with 16 tokens of overlap. Offsets refer to normalized page text, not coordinates in the original PDF. `recursive_chunks` is an experimental character-bounded alternative and is not connected to the index worker; it does not enforce a model-token limit.

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` encodes each chunk and question into a normalized 384-dimensional vector on CPU. PostgreSQL pgvector orders candidates by cosine distance. Vectors are checked for dimensionality, finite values and nonzero norm. Search filters both the model name and revision so an incompatible model is not silently mixed with existing vectors. The HNSW index is approximate; its behavior under tenant and document filters requires further recall testing.

Keyword retrieval uses PostgreSQL `simple` full-text search and `ts_rank_cd`, not BM25. With no results for a plain query, it falls back to matching any term; explicit operators retain their original semantics. This may improve recall while reducing precision. The `simple` configuration does not provide Russian or English stemming; both languages can be tokenized, but inflectional forms are not normalized with language-specific dictionaries.

Hybrid retrieval combines ranked candidate lists by reciprocal rank fusion: for each chunk, sum `1 / (k + rank)` over lists where it occurs, using k=60. It operates on ranks rather than assuming lexical and cosine scores share a scale. A cross-encoder can rerank the hybrid candidates by jointly scoring question and chunk. The current cross-encoder is English-oriented.

These implementations are verified by tests and a small documented smoke experiment; their real-world retrieval accuracy has not been established. See `evaluation.md` for dataset, labels, recorded configurations and limitations.

When users explicitly select up to four documents for a question, retrieval runs separately for each and interleaves their candidates. If any selected document has no excerpt in the context, the answer abstains; generated answers must cite at least one chunk from each selected document. These checks do not verify entailment or prove that a comparison is complete. This multi-document path still requires end-to-end PostgreSQL and real-provider testing.
