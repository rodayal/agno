"""Offline contracts; SDK calls are injected and no credentials are loaded."""

import asyncio
import time
from types import SimpleNamespace

import pytest
from agno.exceptions import EmbeddingError

from .config import ConfigurationError, OciIamSettings, Provider
from .embeddings import OciEmbedder
from .models import StaticEmbedder, build_live_embedder


def provider(**changes):
    raw = {"enabled": True, "provider": "oci", "model_id": "cohere.embed-v4.0"}
    raw.update(changes)
    iam = OciIamSettings(
        compartment_id="ocid1.tenancy.oc1..offlinefixture",
        config_file="/offline/config-not-read",
        embedding=raw,
    )
    return Provider(settings=iam.model_settings(), iam=iam)


def test_status_and_lazy_construction_do_not_load_keys(monkeypatch):
    import oci

    def deny(*args, **kwargs):
        pytest.fail("No key/config reads or API calls during status/construction")

    monkeypatch.setattr(oci.config, "from_file", deny)
    current = provider()
    embedder = build_live_embedder(current)
    assert isinstance(embedder, OciEmbedder) and embedder._client is None
    assert current.embedding_status()["synthetic"] is False
    assert current.embedding_status()["dimensions"] == 1536
    assert "compartment" not in str(current.embedding_status())
    assert "offline/config" not in str(current.embedding_status())
    assert isinstance(build_live_embedder(provider(enabled=False)), StaticEmbedder)


@pytest.mark.parametrize(
    "changes",
    [
        {"dimensions": 64},
        {"dimensions": True},
        {"region": "us-phoenix-1"},
        {"model_id": "wrong"},
        {"endpoint": "https://example.com"},
    ],
)
def test_bad_embedding_configuration_does_not_disable_chat(changes):
    current = provider(**changes)
    assert current.ready
    assert not current.embeddings_ready
    assert current.embedding_reason
    with pytest.raises(ConfigurationError):
        build_live_embedder(current)


def injected_embedder(vector=None):
    embedder = build_live_embedder(provider())
    calls = []

    def embed_text(details):
        calls.append(details)
        return SimpleNamespace(
            data=SimpleNamespace(
                embeddings=[vector if vector is not None else [1.0] * 1536], usage=None
            )
        )

    embedder._client = SimpleNamespace(embed_text=embed_text)
    return embedder, calls


def test_document_and_query_modes_and_exact_float_shape():
    embedder, calls = injected_embedder()
    assert len(embedder.get_embedding("question")) == 1536
    vector, usage = embedder.get_embedding_and_usage("passage")
    assert len(vector) == 1536 and usage is None
    assert [c.input_type for c in calls] == ["SEARCH_QUERY", "SEARCH_DOCUMENT"]
    assert [c.inputs for c in calls] == [["question"], ["passage"]]
    assert all(
        c.truncate == "NONE"
        and c.output_dimensions == 1536
        and c.embedding_types == ["float"]
        for c in calls
    )
    assert embedder.space_identity != "static-word-buckets-sha256-v2:64:local"


def test_typed_float_response_is_accepted():
    embedder, _ = injected_embedder()
    embedder._client.embed_text = lambda details: SimpleNamespace(
        data=SimpleNamespace(
            embeddings=None, embeddings_by_type={"float": [[1.0] * 1536]}, usage=None
        )
    )
    assert len(embedder.get_embedding("question")) == 1536


@pytest.mark.parametrize(
    "vector", [[1.0], [0.0] * 1536, [float("nan")] * 1536, [float("inf")] * 1536]
)
def test_bad_vectors_fail_without_synthetic_substitution(vector):
    embedder, _ = injected_embedder(vector)
    with pytest.raises(EmbeddingError, match="no static vector"):
        embedder.get_embedding("question")


def test_provider_failure_is_sanitized_and_not_replaced():
    embedder, _ = injected_embedder()

    def fail(details):
        raise RuntimeError("private-provider-error-details")

    embedder._client.embed_text = fail
    with pytest.raises(EmbeddingError) as error:
        embedder.get_embedding("question")
    assert "private-provider" not in str(error.value)


def test_async_embedding_keeps_event_loop_responsive():
    embedder, calls = injected_embedder()
    original = embedder._client.embed_text

    def delayed(details):
        time.sleep(0.04)
        return original(details)

    embedder._client.embed_text = delayed

    async def run():
        task = asyncio.create_task(embedder.async_get_embedding_and_usage("passage"))
        await asyncio.sleep(0.005)
        assert not task.done()
        vector, _ = await task
        query = await embedder.async_get_embedding("question")
        assert len(vector) == len(query) == 1536

    asyncio.run(run())
    assert [c.input_type for c in calls] == ["SEARCH_DOCUMENT", "SEARCH_QUERY"]
