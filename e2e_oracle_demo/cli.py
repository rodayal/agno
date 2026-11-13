"""Private configuration and loopback service entry point."""

import argparse
import getpass
import json
from dataclasses import asdict
from pathlib import Path

from . import config
from .config import (
    ConfigurationError,
    ModelSettings,
    OciIamSettings,
    Settings,
    validate_url,
    write_private,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "init",
            "configure",
            "configure-model",
            "configure-iam",
            "serve",
            "check",
            "inventory",
        ],
    )
    parser.add_argument(
        "--db-host",
        action="append",
        help="Approved Oracle host; repeat for an allowlist (init only)",
    )
    parser.add_argument(
        "--db-user", default="DEMO_USER", help="Expected Oracle account (init only)"
    )
    parser.add_argument(
        "--db-container", default="DEMO_PDB", help="Expected Oracle PDB (init only)"
    )
    parser.add_argument(
        "--db-port", type=int, default=1521, help="Expected Oracle port (init only)"
    )
    args = parser.parse_args()
    try:
        if args.command == "init":
            path = config.STATE_DIR / "deployment.json"
            settings = (
                Settings.load()
                if path.exists()
                else Settings.initialize(
                    expected_user=args.db_user,
                    expected_container=args.db_container,
                    expected_port=args.db_port,
                    expected_hosts=tuple(args.db_host or ("localhost", "127.0.0.1")),
                )
            )
            print("Deployment prefix:", settings.prefix)
            print("Persistent deployment:", path)
            print("Existing Oracle/model credentials are not replaced.")
        elif args.command == "configure":
            settings = Settings.load()
            path = config.CONFIG_DIR / "oracle-url"
            if path.exists():
                raise ConfigurationError(
                    "Oracle credential already exists; it has not been overwritten."
                )
            value = getpass.getpass(
                "Complete approved Oracle connection URL (hidden): "
            )
            validate_url(value, settings)
            write_private(path, value)
            print("Oracle credential saved outside the repository with mode 600.")
        elif args.command == "configure-iam":
            path = config.CONFIG_DIR / "oci-iam.json"
            if path.exists():
                raise ConfigurationError(
                    "OCI IAM settings already exist; review them privately."
                )
            settings = OciIamSettings(
                compartment_id=input("Approved compartment OCID: ").strip(),
                model_id=input("Model [openai.gpt-oss-20b]: ").strip()
                or "openai.gpt-oss-20b",
                profile_name=input("OCI profile [DEFAULT]: ").strip() or "DEFAULT",
                config_file=input("OCI config path [~/.oci/config]: ").strip()
                or str(Path.home() / ".oci/config"),
                max_output_tokens=int(
                    input("Maximum output tokens [1024]: ").strip() or "1024"
                ),
            ).validate()
            write_private(path, json.dumps(asdict(settings), indent=2))
            print(
                "Chicago IAM settings saved. Restart the service to load them; no API request was made."
            )
        elif args.command == "configure-model":
            model_file = config.CONFIG_DIR / "model.json"
            key_file = config.CONFIG_DIR / "openai-api-key"
            if model_file.exists() or key_file.exists():
                raise ConfigurationError(
                    "Model configuration already exists; it has not been overwritten. Review/rotate it privately."
                )
            model_id = input("Your approved model ID: ").strip()
            base_url = (
                input(
                    "Approved HTTPS Responses endpoint, or Enter for OpenAI: "
                ).strip()
                or None
            )
            embedding_model = (
                input("Embedding model [text-embedding-3-small]: ").strip()
                or "text-embedding-3-small"
            )
            dimensions = int(input("Embedding dimensions [1536]: ").strip() or "1536")
            max_output_tokens = int(
                input("Maximum output tokens per model response [1024]: ").strip()
                or "1024"
            )
            settings = ModelSettings(
                model_id=model_id,
                base_url=base_url,
                embedding_model=embedding_model,
                dimensions=dimensions,
                max_output_tokens=max_output_tokens,
            ).validate()
            key = getpass.getpass("API key (hidden): ").strip()
            if not key or any(c.isspace() for c in key):
                raise ConfigurationError(
                    "API key must be nonempty and contain no whitespace."
                )
            write_private(key_file, key)
            write_private(model_file, json.dumps(asdict(settings), indent=2))
            print(
                "Model settings saved privately. No API request was made. Restart the service to load them."
            )
        elif args.command == "serve":
            import uvicorn

            settings = Settings.load()
            uvicorn.run(
                "e2e_oracle_demo.app:create_app",
                factory=True,
                host="127.0.0.1",
                port=settings.port,
                workers=1,
                log_level="warning",
                access_log=False,
            )
        else:
            from .app import safe_error
            from .runtime import Runtime

            runtime = Runtime()
            try:
                result = (
                    runtime.inventory()
                    if args.command == "inventory"
                    else runtime.status()
                )
                print(json.dumps(result, indent=2, default=str))
            except Exception as error:
                print(json.dumps({"status": "FAIL", "error": safe_error(error)}))
                raise SystemExit(1) from None
            finally:
                runtime.close()
    except (ConfigurationError, ValueError, FileExistsError) as error:
        print("Configuration stopped:", str(error))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
