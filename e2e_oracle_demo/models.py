"""A labelled Agno model simulator and the separately configured live connector."""

import json
import math
import re
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
from uuid import uuid4

from agno.models.base import Model
from agno.models.response import ModelResponse

from .config import DEMO_EMBEDDING_DIMENSIONS, DEMO_EMBEDDING_MODEL

STATIC_REQUEST = ContextVar("static_demo_request", default=None)
STATIC_ACTIONS = {
    "auto",
    "add_numbers",
    "remember_note",
    "recall_notes",
    "save_learning",
    "find_learning",
    "search_documents",
    "publish_report",
    "make_report",
}


@dataclass
class StaticModel(Model):
    id: str = "static-simulator-v2"
    name: str = "Static simulator (not an LLM)"
    provider: str = "Local deterministic simulation"
    role_name: str = "Assistant"

    def invoke(self, messages, assistant_message=None, tools=None, **kwargs):
        request = STATIC_REQUEST.get() or {}
        user_index = next(
            (i for i in range(len(messages) - 1, -1, -1) if messages[i].role == "user"),
            -1,
        )
        recent = messages[user_index + 1 :]
        completed = [m for m in recent if m.role == "tool"]
        action = request.get("action", "auto") or "auto"
        available = {t.get("function", t).get("name") for t in tools or []}
        if action != "auto" and not completed and action in available:
            return ModelResponse(
                role="assistant",
                tool_calls=[
                    {
                        "id": "call_" + uuid4().hex,
                        "type": "function",
                        "function": {
                            "name": action,
                            "arguments": json.dumps(request.get("arguments", {})),
                        },
                    }
                ],
            )
        if (
            self.role_name == "Coordinator"
            and "delegate_task_to_member" in available
            and len(completed) < 2
        ):
            member = "worker" if not completed else "reviewer"
            task = (
                messages[user_index].get_content_string()
                if not completed and user_index >= 0
                else str(completed[-1].content)
            )
            return ModelResponse(
                role="assistant",
                tool_calls=[
                    {
                        "id": "call_" + uuid4().hex,
                        "type": "function",
                        "function": {
                            "name": "delegate_task_to_member",
                            "arguments": json.dumps(
                                {"member_id": member, "task": task}
                            ),
                        },
                    }
                ],
            )
        if completed:
            content = "\n".join(str(m.content)[:3500] for m in completed)
            return ModelResponse(
                role="assistant", content="[STATIC: no LLM] Tool returned:\n" + content
            )
        user = messages[user_index].get_content_string() if user_index >= 0 else ""
        history = [
            m.get_content_string()[:180]
            for m in messages[:user_index]
            if m.role == "user"
        ][-3:]
        content = f"[STATIC: no LLM] {self.role_name} received: {user}\nThis is a deterministic test response, not AI reasoning."
        if history:
            content += "\nEarlier user context supplied by Agno: " + " | ".join(history)
        return ModelResponse(role="assistant", content=content)

    async def ainvoke(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)

    def invoke_stream(self, *args, **kwargs):
        response = self.invoke(*args, **kwargs)
        if response.tool_calls:
            yield response
        else:
            for offset in range(0, len(response.content or ""), 36):
                yield ModelResponse(
                    role="assistant", content=response.content[offset : offset + 36]
                )

    async def ainvoke_stream(self, *args, **kwargs):
        for response in self.invoke_stream(*args, **kwargs):
            yield response

    def _parse_provider_response(self, response, **kwargs):
        return response

    def _parse_provider_response_delta(self, response):
        return response


class StaticEmbedder:
    """Synthetic SHA-256 word buckets, not semantic or provider embeddings."""

    id = DEMO_EMBEDDING_MODEL
    dimensions = DEMO_EMBEDDING_DIMENSIONS
    enable_batch = False
    synthetic = True

    def get_embedding(self, text):
        vector = [0.0] * self.dimensions
        for word in re.findall(r"\w+", text.lower()):
            vector[
                int.from_bytes(sha256(word.encode()).digest()[:4], "big")
                % self.dimensions
            ] += 1.0
        length = math.sqrt(sum(value * value for value in vector))
        # Oracle cosine operations cannot use a zero vector.
        return (
            [value / length for value in vector]
            if length
            else [1.0] + [0.0] * (self.dimensions - 1)
        )

    def get_embedding_and_usage(self, text):
        return self.get_embedding(text), {"total_tokens": 0}

    async def async_get_embedding(self, text):
        return self.get_embedding(text)

    async def async_get_embedding_and_usage(self, text):
        return self.get_embedding_and_usage(text)


def build_live_model(provider):
    from agno.models.openai import OpenAIResponses

    if not provider.ready:
        raise ValueError("Live provider is not configured.")
    model_class = OpenAIResponses
    extra = {}
    if provider.iam is not None:
        from .iam import OciIamResponses

        model_class = OciIamResponses
        extra["iam_settings"] = provider.iam
    return model_class(
        id=provider.settings.model_id,
        api_key=provider.api_key,
        base_url=provider.settings.base_url,
        store=False,
        max_output_tokens=provider.settings.max_output_tokens,
        timeout=60,
        max_retries=0,
        **extra,
    )


def build_live_embedder(provider):
    """Use OCI embeddings when explicitly enabled; never mask provider failures."""
    if provider.uses_oci_embeddings:
        from .embeddings import OciEmbedder

        return OciEmbedder(provider.oci_embedding_settings)
    return StaticEmbedder()
