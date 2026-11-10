PYTHON ?= python3.12
K ?= 5

.PHONY: setup up down migrate test lint format typecheck eval seed-demo
setup:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install -r backend/requirements-dev.txt
	cd frontend && npm ci
up:
	docker compose up --build
down:
	docker compose down
migrate:
	cd backend && ../.venv/bin/alembic upgrade head
test:
	cd backend && ../.venv/bin/python -m pytest tests ../tests -q
lint:
	cd backend && ../.venv/bin/ruff check ake_backend tests alembic
typecheck:
	cd backend && ../.venv/bin/mypy ake_backend --ignore-missing-imports
format:
	cd backend && ../.venv/bin/ruff format ake_backend tests alembic
eval:
	@test -n "$(DATASET)" || (echo 'Set DATASET to a JSON file of labelled retrieval results' && exit 1)
	cd backend && ../.venv/bin/python -m ake_backend.evaluation "$(abspath $(DATASET))" --k $(K)
seed-demo:
	@test -n "$(PDF)" || (echo 'Set PDF to a locally obtained open paper PDF' && exit 1)
	cd backend && ../.venv/bin/python -m ake_backend.demo "$(abspath $(PDF))"
