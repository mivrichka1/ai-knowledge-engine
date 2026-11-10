# Evaluation status

Per-query retrieval metrics and aggregation are implemented and tested using hand-computed examples. Users can submit labelled datasets in the UI or API; a durable background job executes selected strategies against the same retrieval API and stores per-query results, latencies, model/configuration metadata, dataset hash and document index versions. An experiment retries if the document index changes during the run. The input format is a JSON array of records with `id`, `answerable`, `relevant_ids`, and `retrieved_ids`. Retrieval metrics are averaged only over answerable examples; unanswerable examples are counted separately. Empty relevance for an answerable example is rejected. The reported precision uses a fixed K denominator; repeated IDs occupy ranks but count only once.

No representative, exhaustively labelled scientific-document dataset or benchmark is available yet. The small anchor-based seed experiments below provide limited comparative observations. Example-based tests verify arithmetic and software integration, not system accuracy. A meaningful experiment must record document versions, chunking, model revisions, query labels, retrieval parameters, environment and latency, and review relevance judgments before reporting results.

## Seed experiment (2026-11-10)

A five-question seed set was manually anchored to short passages on pages 1–4 of *Attention Is All You Need* (arXiv:1706.03762v7). The file downloaded for the run had SHA-256 `bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697`; the PDF is not redistributed here. Labels include only chunks containing each chosen anchor and therefore omit other potentially relevant passages. They are not exhaustive ground truth.

On the single indexed paper, at K=5, anchor-based hit rate was 0/5 for keyword and 3/5 for both vector and hybrid. Recall@5 was 0.0 for keyword and 0.5 for vector and hybrid. This is a smoke experiment on a tiny partially labelled dataset, not evidence of general retrieval quality. Keyword uses PostgreSQL `websearch_to_tsquery('simple', ...)`, which effectively required the words in these unquoted questions to match together; that design is a likely contributor to the zero hits. No significance or improvement claim is justified.

The run used a Linux aarch64 container with CPU PyTorch 2.15.0+cpu, Transformers 4.57.6 and Sentence Transformers 5.1.2; the model revision and query results are in `evaluation/runs/attention-seed-v1.json`. The worker used 120 model tokens with 16 overlap. CPU model and memory were not captured. Individual latency measurements include service and model startup effects and are not performance benchmarks. To reproduce, download the same paper version from arXiv, verify its hash, upload and wait for `ready`, then pass the access token and IDs through the environment as documented in `evaluation/run_seed.py`. The user must supply their own credentials and knowledge base.

This run preceded the later addition of the HNSW index and deterministic chunk IDs. Its recorded chunk identifiers refer to that run; a fresh run must resolve the page and text anchors again. Filtered approximate nearest-neighbour retrieval can have different recall from exact retrieval and should be measured separately.

A second seed run added a keyword fallback: if a plain websearch query has no hits, PostgreSQL searches for any of its terms; the API reports `query_mode=any_terms`. Queries with explicit minus signs, quotes or `OR` retain their original semantics. In this run hit@5 was 2/5 keyword, 3/5 vector, 4/5 hybrid; anchor recall@5 was 0.4, 0.5 and 0.7 respectively. Results are in `evaluation/runs/attention-seed-fallback-v1.json`. The HNSW index and deterministic IDs were also introduced between runs, so this is not a controlled ablation and the changes cannot be assigned to the fallback alone. The same five-query and incomplete-label limitations apply.

A third run also evaluated hybrid plus the real `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker (revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`). On the same five partial labels, hit@5 was 2/5 for keyword, 3/5 for vector, 4/5 for hybrid and 5/5 for hybrid plus reranking; the corresponding anchor recall@5 was 0.4, 0.5, 0.7 and 0.9. Full per-query output and latencies are in `evaluation/runs/attention-seed-rerank-v1.json`. Ties and index state can affect the results, and the labels cover only selected anchors. These numbers are an engineering smoke test, not an estimated improvement on a representative dataset.

An evaluation run is rejected if documents are still being indexed. Start a new run after indexing finishes. Document statuses are recorded in the run's snapshot; failed documents require review before interpreting results.

A ten-question extension, `evaluation/attention_seed_v2.json`, was run end to end against the same PDF using the current Compose application. The results in `evaluation/runs/attention-seed-v2.json` show anchor hit@5 of 5/10 for keyword, 7/10 for vector, 8/10 for hybrid and 10/10 for hybrid with reranking; corresponding anchor recall@5 was 0.45, 0.60, 0.70 and 0.95. The result file records the PDF and dataset hashes, model settings and actual index metadata. Indexing used 120 model tokens with 16 overlap. These labels cover selected anchors only, not every relevant passage; ten questions from one paper are not representative, and no general quality or causal improvement claim follows from these numbers. This does not replace the requested 100–150 question dataset. Set `AKE_DATASET` to that file when running `evaluation/run_seed.py`.

Newly indexed documents record the actual chunking strategy, size, overlap, embedding model and revision in `index_metadata`. Earlier documents may have null metadata until reindexed; experiment exports preserve that unknown state rather than infer settings.

The current seed script restricts each search to the selected document, checks the index version before and after the run, and obtains index metadata and search model settings from the API. The source version in the dataset is a declaration: independently verify the downloaded PDF hash before interpreting the run. Reports created with earlier revisions of the script retain the configuration and scope used at the time.

## Two additional single-paper seed experiments (2026-11-10)

The same application and four strategies were run on ten anchor questions each for [BERT, arXiv:1810.04805v2](https://arxiv.org/abs/1810.04805v2) and [RoBERTa, arXiv:1907.11692v1](https://arxiv.org/abs/1907.11692v1). Their query sets and full outputs are `evaluation/bert_seed_v1.json`, `evaluation/roberta_seed_v1.json`, and the corresponding `evaluation/runs/*-seed-v1.json` files. The downloaded PDFs had SHA-256 `5692a5514787a8c6727b4ff3b726a3385798bc68e12138d1d4af83947e2acf6e` and `76a3872d244793563a5b000b818cbaf0ca8972ab1145d32be474c05b2a8f3070`, respectively.

| Dataset | Keyword hit@5 | Vector hit@5 | Hybrid hit@5 | Hybrid + reranker hit@5 |
| --- | ---: | ---: | ---: | ---: |
| BERT (10 questions) | 1/10 | 5/10 | 4/10 | 5/10 |
| RoBERTa (10 questions) | 2/10 | 6/10 | 5/10 | 7/10 |

These are partial anchor matches, not exhaustive relevance judgments: a result may answer the question but not contain the chosen anchor. Labels are not independently reviewed; they are not a held-out benchmark, and the questions cover only selected parts of each paper. The parser used for these original runs interleaved text from adjacent columns in some places; this can damage context and source quality. Figures and tables require manual checking. Do not average these numbers into a general model-quality claim. The requested 100–150 independently reviewed questions and multi-document relevance judgments remain unfinished.

A separate smoke test indexed all three PDFs together and returned five results per query when restricted to the intended document; an unrestricted sample query returned chunks from two of the three in its top 20. This verifies basic filtering and retrieval behavior for that run, not a multi-document quality benchmark or a generated comparison.

## Repeat after PDF column handling change

The parser now attempts column ordering only when it detects two separated, vertically overlapping columns; ambiguous mixed layouts retain PyMuPDF's default order. The BERT and RoBERTa seed experiments were repeated with fresh indexes after this change. At K=5, anchor hit rates for keyword, vector, hybrid and hybrid with reranking were respectively 1/10, 4/10, 4/10 and 4/10 for BERT; 5/10, 6/10, 7/10 and 8/10 for RoBERTa. See `evaluation/runs/bert-seed-column-v1.json` and `evaluation/runs/roberta-seed-column-v1.json`. These runs used the new column-handling code before the later addition of an explicit `parser_version` metadata field, so that field is absent in their saved metadata. The old results above were retained. These are not controlled quality comparisons: the small anchor labels are incomplete, chunks and index state changed, and extracted layouts can still be wrong.

Newly indexed PDF documents now record `parser_version=pdf-columns-v1` in index metadata. Older indexes do not gain the new text ordering until reindexed; previous source IDs can change when chunks change.

## Twenty-question Attention seed

`evaluation/attention_seed_v3.json` extends the first paper's set to 20 anchor questions. A fresh run using PDF parser `pdf-columns-v1` is saved in `evaluation/runs/attention-seed-v3.json`: keyword hit@5 11/20, vector 15/20, hybrid 17/20, and hybrid with reranking 20/20; anchor recall@5 was 0.525, 0.70, 0.80 and 0.975 respectively. The record contains the actual document and dataset hashes, index settings and per-query results. The 20 anchors were checked against extracted text, but are incomplete relevance labels without independent review. The sample remains a single paper, and performance must not be generalized from it.

## Expanded BERT and RoBERTa seeds

The second versions `bert_seed_v2.json` and `roberta_seed_v2.json` each contain 20 questions and were rerun against fresh indexes with `pdf-columns-v1`. At K=5, anchor hit rates for keyword, vector, hybrid and hybrid with reranking were 4/20, 10/20, 9/20 and 9/20 for BERT, and 12/20, 13/20, 15/20 and 17/20 for RoBERTa. The per-query results and hashes are in `evaluation/runs/bert-seed-v2.json` and `evaluation/runs/roberta-seed-v2.json`. The three current seeds total 60 selected questions, but anchor labels are partial and not independently reviewed; the requested 100–150 question verified dataset remains incomplete. These small, non-independent samples do not establish comparative quality.

## Current 100-query anchor sets

The current sets are `attention_seed_v5.json` (40 questions), `bert_seed_v3.json` (30), and `roberta_seed_v3.json` (30). Every selected anchor was found in a chunk on its indicated PDF page and each set was run across all four search strategies on a newly indexed copy of its own paper. Results, per-query IDs, hashes and index metadata are preserved in the corresponding `evaluation/runs/*-seed-v*.json` files. On these incomplete anchor labels, the observed hit@5 counts were:

| Paper | Keyword | Vector | Hybrid | Hybrid + reranker |
| --- | ---: | ---: | ---: | ---: |
| Attention (40) | 20 | 34 | 35 | 40 |
| BERT (30) | 5 | 13 | 17 | 14 |
| RoBERTa (30) | 16 | 18 | 21 | 27 |

These are 100 question–anchor pairs, **not** 100 independently verified complete relevance judgments. Some questions are narrow or overlap in meaning; all are answerable questions drawn from three closely related English papers. Missing relevant chunks are not exhaustively labelled, the sources are not held out and human-independent review, unanswerable questions, multi-document relevance judgments, and broader domains are absent. This is an engineering exploration, not a representative benchmark or a basis for general performance claims. Reindexing changes chunk boundaries and can alter which chunks contain each anchor.
