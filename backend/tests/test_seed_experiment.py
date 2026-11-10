import importlib.util
from pathlib import Path

import httpx
import pytest


def test_seed_experiment_limits_search_and_records_actual_configuration():
    path = Path(__file__).parents[2] / "evaluation" / "run_seed.py"
    spec = importlib.util.spec_from_file_location("run_seed", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("/documents/doc"):
            return httpx.Response(
                200,
                json={"status": "ready", "index_version": 4, "index_metadata": {"chunk_size": 99}},
            )
        if request.url.path.endswith("/settings"):
            return httpx.Response(
                200,
                json={"embedding_model": "actual-model", "reranker_revision": "actual-revision"},
            )
        if request.url.path.endswith("/chunks"):
            return httpx.Response(200, json=[{"id": "chunk", "page": 1, "text": "anchor"}])
        if request.url.path.endswith("/search"):
            assert b'"document_ids":["doc"]' in request.content
            return httpx.Response(200, json={"results": [{"id": "chunk"}], "latency_ms": 1})
        return httpx.Response(404)

    config = {
        "name": "seed",
        "source_version": "v1",
        "queries": [{"id": "q", "question": "question", "page": 1, "anchor": "anchor"}],
    }
    with httpx.Client(
        base_url="http://test/api/v1", transport=httpx.MockTransport(handler)
    ) as client:
        result = module.run_experiment(client, config, "base", "doc", ["keyword"])
    assert result["document_index_metadata"] == {"chunk_size": 99}
    assert result["document_index_version"] == 4
    assert result["search_settings"]["embedding_model"] == "actual-model"
    assert result["results"]["keyword"]["evaluation"]["metrics"]["hit_at_k"] == 1


def test_seed_experiment_rejects_index_changed_during_run():
    path = Path(__file__).parents[2] / "evaluation" / "run_seed.py"
    spec = importlib.util.spec_from_file_location("run_seed", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    status_calls = 0

    def handler(request):
        nonlocal status_calls
        if request.url.path.endswith("/documents/doc"):
            status_calls += 1
            return httpx.Response(200, json={"status": "ready", "index_version": status_calls})
        if request.url.path.endswith("/settings"):
            return httpx.Response(200, json={})
        if request.url.path.endswith("/chunks"):
            return httpx.Response(200, json=[{"id": "chunk", "page": 1, "text": "anchor"}])
        return httpx.Response(200, json={"results": [{"id": "chunk"}], "latency_ms": 1})

    config = {
        "name": "seed",
        "source_version": "v1",
        "queries": [{"id": "q", "question": "question", "page": 1, "anchor": "anchor"}],
    }
    with httpx.Client(
        base_url="http://test/api/v1", transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(ValueError, match="index changed"):
            module.run_experiment(client, config, "base", "doc", ["keyword"])


def test_seed_experiment_rejects_missing_index_version():
    path = Path(__file__).parents[2] / "evaluation" / "run_seed.py"
    spec = importlib.util.spec_from_file_location("run_seed", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def handler(request):
        return httpx.Response(200, json={"status": "ready"})
    with httpx.Client(
        base_url="http://test/api/v1", transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(ValueError, match="index version"):
            module.run_experiment(client, {}, "base", "doc", ["keyword"])
