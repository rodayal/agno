"""Lazy OCI IAM clients for Agno's synchronous and asynchronous Responses paths."""

from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
from threading import RLock

from agno.models.openai import OpenAIResponses

from .config import OciIamSettings

_CLIENT_LOCK = RLock()


def build_signed_client(settings: OciIamSettings, *, asynchronous=False):
    # Constructing the signer is the point at which OCI loads the private key.
    # The key remains inside the SDK; the demo never reads or logs its contents.
    import httpx
    from oci_openai import OciUserPrincipalAuth
    from openai import AsyncOpenAI, OpenAI

    auth = OciUserPrincipalAuth(
        config_file=str(Path(settings.config_file).expanduser()),
        profile_name=settings.profile_name,
    )
    # The SDK accepts legacy HTTPX clients. Its default client aliases now
    # select HTTPX2, whose Auth class differs from Oracle's HTTPX signer.
    http_class = httpx.AsyncClient if asynchronous else httpx.Client
    http_client = http_class(
        auth=auth,
        timeout=60,
        follow_redirects=False,
        headers={
            "CompartmentId": settings.compartment_id,
            "opc-compartment-id": settings.compartment_id,
        },
    )
    client_class = AsyncOpenAI if asynchronous else OpenAI
    return client_class(
        api_key="OCI",
        base_url=settings.base_url,
        http_client=http_client,
        timeout=60,
        max_retries=0,
    )


@dataclass
class OciIamResponses(OpenAIResponses):
    iam_settings: OciIamSettings | None = field(default=None, repr=False)

    def _format_messages(self, messages, compress_tool_results=False, tools=None):
        formatted = super()._format_messages(messages, compress_tool_results, tools)
        if self.id not in {"openai.gpt-oss-20b", "openai.gpt-oss-120b"}:
            return formatted
        # OCI returns reasoning_text but rejects it when replayed as input.
        # Replay public conversation and paired tool calls/results instead;
        # keep the original message/provider metadata intact in Oracle.
        return [
            item
            for item in formatted
            if (
                item.get("type")
                if isinstance(item, dict)
                else getattr(item, "type", None)
            )
            != "reasoning"
        ]

    def _format_tool_params(self, messages, tools=None):
        # Agno persists approval policy with a Function, but OCI accepts only
        # provider tool fields. Keep that policy on the original local tool.
        formatted = super()._format_tool_params(messages, deepcopy(tools))
        for tool in formatted:
            if tool.get("type") == "function":
                for key in (
                    "requires_confirmation",
                    "external_execution",
                    "approval_type",
                ):
                    tool.pop(key, None)
        return formatted

    def get_client(self):
        with _CLIENT_LOCK:
            if self.client is None or self.client.is_closed():
                if self.iam_settings is None:
                    raise ValueError("OCI IAM settings are required.")
                self.client = build_signed_client(self.iam_settings)
            return self.client

    def get_async_client(self):
        with _CLIENT_LOCK:
            if self.async_client is None or self.async_client.is_closed():
                if self.iam_settings is None:
                    raise ValueError("OCI IAM settings are required.")
                self.async_client = build_signed_client(
                    self.iam_settings, asynchronous=True
                )
            return self.async_client
