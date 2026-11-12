"""OCI IAM text embeddings, using the saved Playground script's SDK path."""

import asyncio
import math
from pathlib import Path
from threading import RLock

from agno.exceptions import EmbeddingError


class OciEmbedder:
    enable_batch = False
    synthetic = False

    def __init__(self, settings):
        self.settings = settings
        self.id = settings.model_id
        self.dimensions = settings.dimensions
        self.space_identity = (
            f"oci:{self.id}:{self.dimensions}:{settings.region}:search-v1"
        )
        self._client = None
        self._lock = RLock()

    def _embed(self, text, input_type):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Embedding input must be nonempty text.")
        import oci

        try:
            # Keep key loading inside the SDK, as in embed_text_demo.py.
            with self._lock:
                if self._client is None:
                    profile = oci.config.from_file(
                        str(Path(self.settings.config_file).expanduser()),
                        self.settings.profile_name,
                    )
                    self._client = (
                        oci.generative_ai_inference.GenerativeAiInferenceClient(
                            config=profile,
                            service_endpoint=self.settings.endpoint,
                            retry_strategy=oci.retry.NoneRetryStrategy(),
                            timeout=(10, 240),
                        )
                    )
                details = oci.generative_ai_inference.models.EmbedTextDetails(
                    serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(
                        model_id=self.id
                    ),
                    inputs=[text],
                    compartment_id=self.settings.compartment_id,
                    truncate=self.settings.truncate,
                    input_type=input_type,
                    output_dimensions=self.dimensions,
                    embedding_types=["float"],
                    is_echo=False,
                )
                data = self._client.embed_text(details).data
            entries = data.embeddings or (
                getattr(data, "embeddings_by_type", None) or {}
            ).get("float")
            if not entries or len(entries) != 1:
                raise ValueError("OCI did not return exactly one embedding.")
            vector = [float(value) for value in entries[0]]
            if (
                len(vector) != self.dimensions
                or not all(math.isfinite(value) for value in vector)
                or not any(vector)
            ):
                raise ValueError("OCI returned an invalid embedding vector.")
            usage = oci.util.to_dict(data.usage) if data.usage is not None else None
            return vector, usage
        except Exception as error:
            status = getattr(error, "status", None)
            status = status if isinstance(status, int) else None
            message = "OCI embedding request failed; no static vector was substituted."
            if status is not None:
                message += f" HTTP {status}."
            raise EmbeddingError(
                message, status_code=status, model_id=self.id, provider="OCI"
            ) from None

    def get_embedding(self, text):
        """OracleVector calls this path for search queries."""
        return self._embed(text, "SEARCH_QUERY")[0]

    def get_embedding_and_usage(self, text):
        """Agno Document calls this path for stored documents."""
        return self._embed(text, "SEARCH_DOCUMENT")

    async def async_get_embedding(self, text):
        return await asyncio.to_thread(self.get_embedding, text)

    async def async_get_embedding_and_usage(self, text):
        return await asyncio.to_thread(self.get_embedding_and_usage, text)
