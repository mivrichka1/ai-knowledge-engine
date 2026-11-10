# ADR 0001: PostgreSQL for metadata, vectors and jobs

**Context:** A local portfolio system needs relational ownership checks, persistent work records and vector retrieval.

**Alternatives:** PostgreSQL plus pgvector and a database-backed job queue; PostgreSQL plus a separate vector store and Redis/Celery; or an in-process queue.

**Decision:** Use PostgreSQL with pgvector and persistent job rows claimed with row locks and leases. Long-running work runs in a separate worker.

**Rationale:** This reduces the number of services while keeping access checks and durable job state close to the stored records. The vector extension avoids coordinating a second database for a small local deployment.

**Trade-offs:** Database-backed polling and approximate vector search need load testing. Leases do not interrupt a hung model call; high-throughput or distributed deployments may warrant a dedicated queue and object store. File writes and database transactions are not atomic together.
