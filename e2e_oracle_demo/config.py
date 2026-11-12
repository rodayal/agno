"""Persistent deployment configuration, separate from source and secrets."""

import json
import os
import re
import stat
from importlib.util import find_spec
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = Path.home() / ".config" / "agno-oracle-demo"
STATE_DIR = Path.home() / ".local" / "state" / "agno-oracle-demo-v2"
DEMO_EMBEDDING_MODEL = "static-word-buckets-sha256-v2"
DEMO_EMBEDDING_DIMENSIONS = 64


class ConfigurationError(RuntimeError):
    """A safe, user-facing deployment configuration error."""


def read_private_value(path: Path, label: str) -> str:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise ConfigurationError(f"{label} is not configured.") from None
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077
    ):
        raise ConfigurationError(
            f"{label} must be a user-owned regular file with mode 600."
        )
    value = path.read_text().strip()
    if not value or any(c.isspace() for c in value):
        raise ConfigurationError(f"{label} must be nonempty and contain no whitespace.")
    return value


def write_private(path: Path, value: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as output:
        output.write(value + "\n")


@dataclass(frozen=True)
class Settings:
    prefix: str
    port: int = 7777
    expected_user: str = "AGNO_TEST_USER"
    expected_container: str = "CDB1_PDB1"
    expected_port: int = 1550

    @classmethod
    def load(cls):
        try:
            result = cls(**json.loads((STATE_DIR / "deployment.json").read_text()))
        except (OSError, ValueError, TypeError):
            raise ConfigurationError(
                "Run python -m e2e_oracle_demo.cli init first."
            ) from None
        if (
            not re.fullmatch(r"ae_[a-f0-9]{8}", result.prefix)
            or not 1024 <= result.port <= 65535
        ):
            raise ConfigurationError("Invalid demo deployment settings.")
        return result

    @classmethod
    def initialize(cls):
        result = cls(prefix="ae_" + uuid4().hex[:8])
        write_private(
            STATE_DIR / "deployment.json", json.dumps(asdict(result), indent=2)
        )
        return result

    def tables(self, mode: str):
        if mode not in {"static", "live"}:
            raise ConfigurationError("Unknown execution mode.")
        names = [
            "session",
            "runs",
            "memory",
            "metrics",
            "eval",
            "knowledge",
            "traces",
            "spans",
            "versions",
            "components",
            "component_configs",
            "component_links",
            "learnings",
            "schedules",
            "schedule_runs",
            "job",
            "approvals",
            "auth_tokens",
            "service_accounts",
            "mcp_oauth_clients",
            "mcp_oauth_transactions",
            "mcp_oauth_codes",
            "mcp_oauth_refresh_tokens",
            "mcp_oauth_keys",
        ]
        return {
            name + "_table": f"{self.prefix}_{mode[0]}_{index:02}"
            for index, name in enumerate(names)
        }


def validate_url(value: str, settings: Settings):
    from sqlalchemy.engine import make_url

    try:
        url = make_url(value)
        valid = (
            url.drivername == "oracle+oracledb"
            and url.username
            and url.password
            and url.username.upper() == settings.expected_user.upper()
            and url.port == settings.expected_port
            and url.query.get("service_name")
            and url.host
            in {
                "localhost",
                "127.0.0.1",
                "100.94.147.94",
                "phoenix990761",
                "phoenix990761.dev3sub4phx.databasede3phx.oraclevcn.com",
            }
        )
    except Exception:
        valid = False
    if not valid:
        raise ConfigurationError(
            "Oracle URL does not match the approved account, host and port; its value is not logged."
        )
    return url


def read_database_url(settings: Settings):
    value = read_private_value(CONFIG_DIR / "oracle-url", "Oracle credential")
    validate_url(value, settings)
    return value


@dataclass(frozen=True)
class ModelSettings:
    model_id: str | None = None
    embedding_model: str = "text-embedding-3-small"
    dimensions: int = 1536
    base_url: str | None = None
    max_output_tokens: int = 1024

    @property
    def is_oci(self):
        hostname = urlsplit(self.base_url or "").hostname or ""
        return hostname.startswith("inference.generativeai.") and hostname.endswith(
            ".oci.oraclecloud.com"
        )

    def validate(self):
        if (
            not self.model_id
            or len(self.model_id) > 160
            or any(c.isspace() for c in self.model_id)
        ):
            raise ConfigurationError("Choose a valid provider model ID.")
        if not self.embedding_model or not 1 <= self.dimensions <= 4096:
            raise ConfigurationError(
                "Choose an embedding model and dimensions between 1 and 4096."
            )
        if (
            type(self.max_output_tokens) is not int
            or not 1 <= self.max_output_tokens <= 16384
        ):
            raise ConfigurationError(
                "Choose an output token limit between 1 and 16384."
            )
        if self.base_url:
            url = urlsplit(self.base_url)
            if (
                url.scheme != "https"
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
            ):
                raise ConfigurationError(
                    "A custom Responses-compatible endpoint must use HTTPS without credentials or query parameters."
                )
            if self.is_oci and url.path.rstrip("/") != "/20231130/actions/v1":
                raise ConfigurationError(
                    "The OCI Responses base URL must end in /20231130/actions/v1."
                )
        return self


@dataclass(frozen=True)
class OciIamSettings:
    compartment_id: str
    model_id: str = "openai.gpt-oss-20b"
    region: str = "us-chicago-1"
    profile_name: str = "DEFAULT"
    config_file: str = str(Path.home() / ".oci" / "config")
    max_output_tokens: int = 1024
    embedding: dict | None = None

    @property
    def base_url(self):
        return (
            f"https://inference.generativeai.{self.region}.oci.oraclecloud.com"
            "/20231130/actions/v1"
        )

    def validate(self):
        if not re.fullmatch(
            r"ocid1\.(?:compartment|tenancy)\.[a-z0-9]+\.\.[A-Za-z0-9]+",
            self.compartment_id,
        ):
            raise ConfigurationError(
                "Choose a valid OCI compartment or root-tenancy OCID."
            )
        if self.region != "us-chicago-1":
            raise ConfigurationError("This demo's IAM deployment targets us-chicago-1.")
        if (
            not self.profile_name
            or len(self.profile_name) > 128
            or any(c in self.profile_name for c in "\r\n[]")
            or not Path(self.config_file).expanduser().is_absolute()
        ):
            raise ConfigurationError(
                "Choose an OCI profile and an absolute configuration path."
            )
        self.model_settings().validate()
        return self

    def model_settings(self):
        return ModelSettings(
            model_id=self.model_id,
            base_url=self.base_url,
            max_output_tokens=self.max_output_tokens,
        )


@dataclass(frozen=True)
class OciEmbeddingSettings:
    compartment_id: str = field(repr=False)
    config_file: str = field(repr=False)
    profile_name: str
    model_id: str = "cohere.embed-v4.0"
    region: str = "us-chicago-1"
    dimensions: int = 1536
    truncate: str = "NONE"

    @property
    def endpoint(self):
        return f"https://inference.generativeai.{self.region}.oci.oraclecloud.com"

    @classmethod
    def from_iam(cls, iam):
        raw = iam.embedding or {}
        settings = cls(
            compartment_id=iam.compartment_id,
            config_file=iam.config_file,
            profile_name=iam.profile_name,
            model_id=raw.get("model_id", "cohere.embed-v4.0"),
            region=raw.get("region", iam.region),
            dimensions=raw.get("dimensions", 1536),
            truncate=raw.get("truncate", "NONE"),
        )
        if (
            raw.get("provider") != "oci"
            or settings.model_id != "cohere.embed-v4.0"
            or settings.region != "us-chicago-1"
            or type(settings.dimensions) is not int
            or settings.dimensions not in {256, 512, 1024, 1536}
            or settings.truncate != "NONE"
            or raw.get("endpoint", settings.endpoint) != settings.endpoint
        ):
            raise ConfigurationError("OCI embedding settings are invalid.")
        return settings


@dataclass(frozen=True)
class Provider:
    settings: ModelSettings = field(default_factory=ModelSettings)
    api_key: str | None = field(default=None, repr=False)
    reason: str | None = None
    iam: OciIamSettings | None = field(default=None, repr=False)

    @property
    def ready(self):
        return bool(
            self.settings.model_id and (self.api_key or self.iam) and not self.reason
        )

    @property
    def embeddings_ready(self):
        return self.embedding_reason is None

    @property
    def uses_oci_embeddings(self):
        return bool(self.iam and (self.iam.embedding or {}).get("enabled") is True)

    @property
    def oci_embedding_settings(self):
        return OciEmbeddingSettings.from_iam(self.iam)

    @property
    def embedding_reason(self):
        if self.uses_oci_embeddings:
            try:
                self.oci_embedding_settings
            except ConfigurationError as error:
                return str(error)
        return None

    def embedding_status(self):
        if self.uses_oci_embeddings:
            raw = self.iam.embedding or {}
            return {
                "ready": self.embeddings_ready,
                "model_id": raw.get("model_id", "cohere.embed-v4.0"),
                "dimensions": raw.get("dimensions", 1536),
                "provider": "OCI Generative AI / EmbedText (IAM signing)",
                "synthetic": False,
                "reason": self.embedding_reason,
            }
        return {
            "ready": self.embeddings_ready,
            "model_id": DEMO_EMBEDDING_MODEL,
            "dimensions": DEMO_EMBEDDING_DIMENSIONS,
            "provider": "Local static word buckets; no embedding API",
            "synthetic": True,
            "reason": self.embedding_reason,
        }

    def status(self):
        return {
            "ready": self.ready,
            "auth_mode": "oci_iam" if self.iam else "api_key",
            "region": self.iam.region if self.iam else None,
            "provider": "OCI Generative AI / Responses (IAM signing)"
            if self.iam
            else "OCI Generative AI / Responses"
            if self.settings.is_oci
            else "OpenAI Responses / compatible endpoint",
            "model_id": self.settings.model_id,
            "embedding_model": self.embedding_status()["model_id"],
            "max_output_tokens": self.settings.max_output_tokens,
            "reason": self.reason,
            "live_model_tested": False,
        }


def load_provider():
    iam_file = CONFIG_DIR / "oci-iam.json"
    if iam_file.exists() or iam_file.is_symlink():
        try:
            metadata = iam_file.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or metadata.st_mode & 0o077
            ):
                raise ConfigurationError(
                    "OCI IAM settings must be user-owned with mode 600."
                )
            iam = OciIamSettings(**json.loads(iam_file.read_text())).validate()
            if find_spec("oci_openai") is None:
                raise ConfigurationError("OCI signing support is not installed.")
            return Provider(settings=iam.model_settings(), iam=iam)
        except (OSError, ValueError, TypeError, ConfigurationError):
            return Provider(
                reason="OCI IAM configuration is missing or invalid; check oci-iam.json and signing dependencies."
            )
    model_file = CONFIG_DIR / "model.json"
    if not model_file.exists():
        return Provider(
            reason="No live model configured; static fallback will be clearly labelled."
        )
    try:
        settings = ModelSettings(**json.loads(model_file.read_text())).validate()
        key = read_private_value(CONFIG_DIR / "openai-api-key", "Model API key")
        return Provider(settings, key)
    except (OSError, ValueError, TypeError, ConfigurationError):
        return Provider(
            reason="Live configuration is missing or invalid. Check private model.json and API key permissions."
        )


def resolve_mode(requested: str, provider: Provider):
    if requested not in {"static", "live"}:
        raise ConfigurationError("Choose static or live mode.")
    fallback = requested == "live" and not provider.ready
    return {
        "requested_mode": requested,
        "effective_mode": "static" if fallback else requested,
        "fallback": fallback,
        "notice": "Static fallback—no LLM called."
        if fallback
        else "Static simulation—no LLM called."
        if requested == "static"
        else "Live provider selected; requests may incur charges.",
    }
