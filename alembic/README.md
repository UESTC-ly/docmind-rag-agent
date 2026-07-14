# DocMind database migrations

Alembic is the only production schema-management path from v2.2.0 onward.

- Fresh database: `alembic upgrade head`
- Existing v2.1 database previously created by SQLAlchemy `create_all`:
  verify that its schema matches v2.1, then run
  `alembic stamp 0001_v210_baseline` followed by `alembic upgrade head`.
- Inspect pending SQL without applying it:
  `alembic upgrade head --sql`

The v2.2 schema change and full-text index are separate revisions. Revision
`0002_v220_search` commits and stamps the evaluation columns transactionally;
revision `0003_v220_fts` builds the PostgreSQL index with `CONCURRENTLY`, outside
a transaction, so document ingestion can continue. SQLite local/test databases
apply the column revision and treat the PostgreSQL-only index revision as a
no-op.

PostgreSQL can leave an invalid index behind if a concurrent build is
interrupted, or it can finish the index before a process dies prior to stamping
the revision. Revision `0003_v220_fts` handles both retry states: it drops and
rebuilds an invalid index, and accepts an already-valid index only when its
definition matches the expected GIN expression. A valid index with a different
definition fails visibly as schema drift. Do not manually stamp past that error;
inspect and reconcile the database definition first.
