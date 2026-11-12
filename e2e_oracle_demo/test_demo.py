"""Offline contracts for deployment configuration and the two model paths.

Run with the VM's existing dependencies, without installing anything here::

    python -m pytest -q e2e_oracle_demo/test_demo.py

Coverage checklist: private persistence; approved Oracle URLs; explicit mode
labels; deterministic Agno messages, tools and streams; synthetic embeddings;
construction of real, unused live connectors; and the injected app's lifespan,
request boundaries and SSE records. No test constructs a real Oracle runtime.
"""

import asyncio
import json
import math
import re
import socket
import stat
from dataclasses import asdict
from threading import RLock
from types import SimpleNamespace

import pytest
from agno.models.message import Message
from agno.models.response import ModelResponse
from sqlalchemy.engine import URL

from e2e_oracle_demo import config, models


SYNTHETIC_KEY = "sk-offline-synthetic-never-sent"


@pytest.mark.parametrize("returned_error", [False, True])
@pytest.mark.parametrize("auth_mode", ["api_key", "oci_iam"])
def test_configured_provider_failure_does_not_switch_to_static(
    returned_error, auth_mode
):
    from e2e_oracle_demo.runtime import DemoError, Runtime

    runtime = object.__new__(Runtime)
    runtime.provider = config.Provider(
        config.ModelSettings(model_id="offline-model"),
        SYNTHETIC_KEY if auth_mode == "api_key" else None,
        iam=config.OciIamSettings(compartment_id="ocid1.compartment.oc1..offline")
        if auth_mode == "oci_iam"
        else None,
    )
    runtime._lock = RLock()
    runtime._session_locks = {}
    selected = []

    class FailedProvider:
        def run(self, *args, **kwargs):
            if not returned_error:
                raise RuntimeError("synthetic provider failure")
            return SimpleNamespace(
                run_id="offline-run",
                status="ERROR",
                content="synthetic provider details",
                to_dict=lambda: {"content": "synthetic provider details"},
            )

    def components(mode):
        selected.append(mode)
        return {"agent": FailedProvider()}

    runtime.components = components
    with pytest.raises(DemoError if returned_error else RuntimeError):
        runtime.chat({"user_id": "alice", "mode": "live", "message": "offline check"})
    assert selected == ["live"]
    assert models.STATIC_REQUEST.get() is None


APPROVED_HOSTS = (
    "localhost",
    "127.0.0.1",
    "100.94.147.94",
    "phoenix990761",
    "phoenix990761.dev3sub4phx.databasede3phx.oraclevcn.com",
)


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch, tmp_path):
    """Deny sockets and isolate every persisted value and explicit action."""

    def deny_network(*args, **kwargs):
        pytest.fail("Network access is forbidden in offline tests", pytrace=False)

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)
    monkeypatch.setattr(socket, "getaddrinfo", deny_network)
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_ORG_ID",
        "OPENAI_ORGANIZATION",
        "OPENAI_PROJECT",
        "OPENAI_PROJECT_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    token = models.STATIC_REQUEST.set(None)
    try:
        yield
    finally:
        models.STATIC_REQUEST.reset(token)


@pytest.fixture
def ready_provider():
    return config.Provider(
        settings=config.ModelSettings(
            model_id="offline-responses-model",
            embedding_model="offline-embedding-model",
            dimensions=64,
            base_url="https://provider.invalid/v1",
        ),
        api_key=SYNTHETIC_KEY,
    )


def oracle_url(**overrides):
    fields = {
        "drivername": "oracle+oracledb",
        "username": "AGNO_TEST_USER",
        "password": "offline-password-never-sent",
        "host": "localhost",
        "port": 1550,
        "query": {"service_name": "CDB1_PDB1"},
    }
    fields.update(overrides)
    return URL.create(**fields).render_as_string(hide_password=False)


def write_provider_files(settings=None, key=SYNTHETIC_KEY):
    settings = settings or config.ModelSettings(model_id="offline-responses-model")
    config.write_private(config.CONFIG_DIR / "model.json", json.dumps(asdict(settings)))
    if key is not None:
        config.write_private(config.CONFIG_DIR / "openai-api-key", key)


def tool_spec(name="add_numbers"):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    }


def assert_static(response):
    assert isinstance(response, ModelResponse)
    assert response.role == "assistant"
    assert response.content.startswith("[STATIC: no LLM]")
    assert not response.tool_calls


def assert_unit_vector(vector):
    assert len(vector) == 64
    assert all(isinstance(value, float) and math.isfinite(value) for value in vector)
    assert any(value != 0.0 for value in vector)
    assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(1.0)


def test_network_guard_blocks_connection_entry_points():
    with pytest.raises(pytest.fail.Exception, match="Network access is forbidden"):
        socket.create_connection(("127.0.0.1", 9))
    with socket.socket() as client:
        with pytest.raises(pytest.fail.Exception, match="Network access is forbidden"):
            client.connect(("127.0.0.1", 9))
        with pytest.raises(pytest.fail.Exception, match="Network access is forbidden"):
            client.connect_ex(("127.0.0.1", 9))
    with pytest.raises(pytest.fail.Exception, match="Network access is forbidden"):
        socket.getaddrinfo("provider.invalid", 443)


def test_settings_initialize_persists_and_loads_private_deployment():
    initialized = config.Settings.initialize()
    path = config.STATE_DIR / "deployment.json"
    assert re.fullmatch(r"ae_[a-f0-9]{8}", initialized.prefix)
    assert initialized.port == 7777
    assert initialized.expected_user == "AGNO_TEST_USER"
    assert initialized.expected_container == "CDB1_PDB1"
    assert initialized.expected_port == 1550
    assert json.loads(path.read_text()) == asdict(initialized)
    assert config.Settings.load() == initialized
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_settings_initialize_and_private_write_refuse_overwrite():
    original = config.Settings.initialize()
    path = config.STATE_DIR / "deployment.json"
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        config.Settings.initialize()
    with pytest.raises(FileExistsError):
        config.write_private(path, "replacement")
    assert path.read_bytes() == before
    assert config.Settings.load() == original


def test_settings_load_rejects_missing_corrupt_and_invalid_deployments():
    with pytest.raises(config.ConfigurationError, match="init first"):
        config.Settings.load()
    config.STATE_DIR.mkdir()
    path = config.STATE_DIR / "deployment.json"
    invalid_payloads = (
        "{",
        "[]",
        "{}",
        json.dumps({"prefix": "ae_deadbeef", "unexpected": True}),
        json.dumps({"prefix": "ae_DEADBEEF"}),
        json.dumps({"prefix": "ae_deadbeef;DROP"}),
        json.dumps({"prefix": "ae_deadbee"}),
        json.dumps({"prefix": "ae_deadbeef", "port": 1023}),
        json.dumps({"prefix": "ae_deadbeef", "port": 65536}),
    )
    for payload in invalid_payloads:
        path.write_text(payload)
        with pytest.raises(config.ConfigurationError):
            config.Settings.load()
    for port in (1024, 65535):
        path.write_text(json.dumps({"prefix": "ae_deadbeef", "port": port}))
        assert config.Settings.load().port == port


def test_settings_tables_are_stable_unique_and_separate_by_mode():
    settings = config.Settings(prefix="ae_deadbeef")
    static = settings.tables("static")
    live = settings.tables("live")
    assert static == settings.tables("static")
    assert static.keys() == live.keys()
    assert len(static) == len(set(static.values())) == 24
    assert len(live) == len(set(live.values())) == 24
    assert set(static.values()).isdisjoint(live.values())
    assert all(key.endswith("_table") for key in static)
    assert all(re.fullmatch(r"ae_deadbeef_s_\d{2}", value) for value in static.values())
    assert all(re.fullmatch(r"ae_deadbeef_l_\d{2}", value) for value in live.values())
    with pytest.raises(config.ConfigurationError, match="Unknown execution mode"):
        settings.tables("automatic")


def test_private_value_reports_missing_secret_without_exposing_path(tmp_path):
    path = tmp_path / "missing-secret"
    with pytest.raises(
        config.ConfigurationError, match="Model API key is not configured"
    ) as error:
        config.read_private_value(path, "Model API key")
    assert str(path) not in str(error.value)


def test_private_value_accepts_owner_only_permissions_and_trims_file_newline(tmp_path):
    path = tmp_path / "secret"
    config.write_private(path, SYNTHETIC_KEY)
    for mode in (0o600, 0o400):
        path.chmod(mode)
        assert config.read_private_value(path, "Model API key") == SYNTHETIC_KEY


def test_private_value_rejects_wrong_owner(monkeypatch, tmp_path):
    path = tmp_path / "secret"
    config.write_private(path, SYNTHETIC_KEY)
    owner_uid = path.stat().st_uid
    monkeypatch.setattr(config.os, "getuid", lambda: owner_uid + 1)
    with pytest.raises(
        config.ConfigurationError, match="user-owned regular file"
    ) as error:
        config.read_private_value(path, "Model API key")
    assert SYNTHETIC_KEY not in str(error.value)


def test_private_value_rejects_group_or_other_permissions(tmp_path):
    path = tmp_path / "secret"
    config.write_private(path, SYNTHETIC_KEY)
    for mode in (0o644, 0o640, 0o620, 0o610, 0o604, 0o602, 0o601):
        path.chmod(mode)
        with pytest.raises(config.ConfigurationError, match="mode 600") as error:
            config.read_private_value(path, "Model API key")
        assert SYNTHETIC_KEY not in str(error.value)


def test_private_value_rejects_symlinks_and_nonregular_files(tmp_path):
    target = tmp_path / "target"
    config.write_private(target, SYNTHETIC_KEY)
    link = tmp_path / "link"
    link.symlink_to(target)
    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "absent")
    directory = tmp_path / "directory"
    directory.mkdir(mode=0o700)
    for path in (link, dangling, directory):
        with pytest.raises(config.ConfigurationError, match="user-owned regular file"):
            config.read_private_value(path, "Model API key")
    with pytest.raises(FileExistsError):
        config.write_private(link, "replacement")
    assert config.read_private_value(target, "Model API key") == SYNTHETIC_KEY


def test_private_value_rejects_empty_and_embedded_whitespace(tmp_path):
    path = tmp_path / "secret"
    for value in ("", " \t\r\n", "sk two", "sk\tkey", "sk\nkey", "sk\u2003key"):
        path.write_text(value)
        path.chmod(0o600)
        with pytest.raises(
            config.ConfigurationError, match="nonempty.*no whitespace"
        ) as error:
            config.read_private_value(path, "Model API key")
        assert str(path) not in str(error.value)


def test_read_database_url_uses_private_config_and_validates_it():
    settings = config.Settings(prefix="ae_deadbeef")
    with pytest.raises(
        config.ConfigurationError, match="Oracle credential is not configured"
    ):
        config.read_database_url(settings)
    path = config.CONFIG_DIR / "oracle-url"
    approved = oracle_url()
    config.write_private(path, approved)
    assert config.read_database_url(settings) == approved
    path.write_text(oracle_url(host="unapproved.invalid"))
    with pytest.raises(
        config.ConfigurationError, match="approved account, host and port"
    ):
        config.read_database_url(settings)
    path.write_text(approved)
    path.chmod(0o644)
    with pytest.raises(config.ConfigurationError, match="mode 600"):
        config.read_database_url(settings)


def test_validate_url_accepts_all_approved_hosts_and_case_insensitive_account():
    settings = config.Settings(prefix="ae_deadbeef")
    for host in APPROVED_HOSTS:
        parsed = config.validate_url(
            oracle_url(host=host, username="agno_test_user"), settings
        )
        assert parsed.drivername == "oracle+oracledb"
        assert parsed.host == host
        assert parsed.port == settings.expected_port
        assert parsed.username.upper() == settings.expected_user
        assert parsed.password == "offline-password-never-sent"
        assert parsed.query["service_name"] == settings.expected_container


def test_validate_url_rejects_every_incomplete_or_unapproved_component():
    settings = config.Settings(prefix="ae_deadbeef")
    invalid = [
        oracle_url(**change)
        for change in (
            {"drivername": "oracle"},
            {"drivername": "postgresql"},
            {"username": None},
            {"username": ""},
            {"username": "OTHER_USER"},
            {"password": None},
            {"password": ""},
            {"host": None},
            {"host": "unapproved.invalid"},
            {"host": "localhost.evil.invalid"},
            {"port": None},
            {"port": 1521},
            {"port": 1551},
            {"query": {}},
            {"query": {"service_name": ""}},
        )
    ]
    invalid.extend(
        ("not a URL", "", "oracle+oracledb://u:p@localhost:invalid/?service_name=x")
    )
    for value in invalid:
        with pytest.raises(
            config.ConfigurationError, match="approved account, host and port"
        ) as error:
            config.validate_url(value, settings)
        assert "offline-password-never-sent" not in str(error.value)
        if value:
            assert value not in str(error.value)
    # The approval guard uses the supplied deployment's account and port.
    custom = config.Settings(
        prefix="ae_deadbeef", expected_user="CUSTOM_USER", expected_port=1552
    )
    config.validate_url(oracle_url(username="custom_user", port=1552), custom)
    with pytest.raises(config.ConfigurationError):
        config.validate_url(oracle_url(), custom)


def test_provider_ready_requires_model_key_and_no_reason(ready_provider):
    assert ready_provider.ready is True
    unavailable = (
        config.Provider(),
        config.Provider(api_key=SYNTHETIC_KEY),
        config.Provider(settings=ready_provider.settings),
        config.Provider(settings=ready_provider.settings, api_key=""),
        config.Provider(
            settings=ready_provider.settings, api_key=SYNTHETIC_KEY, reason="invalid"
        ),
    )
    assert all(provider.ready is False for provider in unavailable)


def test_provider_repr_and_status_do_not_expose_key_or_claim_live_test(ready_provider):
    assert SYNTHETIC_KEY not in repr(ready_provider)
    status = ready_provider.status()
    assert SYNTHETIC_KEY not in json.dumps(status)
    assert "api_key" not in status
    assert status["ready"] is True
    assert status["model_id"] == ready_provider.settings.model_id
    assert status["embedding_model"] == config.DEMO_EMBEDDING_MODEL
    assert status["live_model_tested"] is False
    assert config.Provider(reason="missing").status()["ready"] is False


def test_model_settings_accept_valid_ids_dimensions_and_https_endpoints():
    for dimensions in (1, 64, 1536, 4096):
        settings = config.ModelSettings(model_id="m" * 160, dimensions=dimensions)
        assert settings.validate() is settings
    for base_url in (
        None,
        "https://provider.invalid",
        "https://provider.invalid:8443/v1/",
    ):
        settings = config.ModelSettings(
            model_id="offline-responses-model", base_url=base_url
        )
        assert settings.validate() is settings


def test_model_settings_reject_invalid_model_embedding_and_dimension_values():
    for model_id in (None, "", "m" * 161, "two models", "model\nname", "model\tname"):
        with pytest.raises(config.ConfigurationError, match="valid provider model ID"):
            config.ModelSettings(model_id=model_id).validate()
    for dimensions in (-1, 0, 4097):
        with pytest.raises(
            config.ConfigurationError, match="dimensions between 1 and 4096"
        ):
            config.ModelSettings(
                model_id="offline-model", dimensions=dimensions
            ).validate()
    with pytest.raises(config.ConfigurationError, match="Choose an embedding model"):
        config.ModelSettings(model_id="offline-model", embedding_model="").validate()


def test_model_settings_reject_unsafe_or_incomplete_custom_base_urls():
    for base_url in (
        "http://provider.invalid/v1",
        "ftp://provider.invalid/v1",
        "provider.invalid/v1",
        "https://",
        "https:///v1",
        "https://user@provider.invalid/v1",
        "https://user:secret@provider.invalid/v1",
        "https://provider.invalid/v1?key=secret",
        "https://provider.invalid/v1#fragment",
    ):
        with pytest.raises(config.ConfigurationError, match="must use HTTPS") as error:
            config.ModelSettings(model_id="offline-model", base_url=base_url).validate()
        assert "secret" not in str(error.value)


def test_load_provider_without_model_config_has_explicit_static_fallback_reason():
    provider = config.load_provider()
    assert provider.ready is False
    assert provider.api_key is None
    assert "No live model configured" in provider.reason
    assert "static fallback" in provider.reason


def test_load_provider_reads_valid_persisted_settings_and_private_key(ready_provider):
    write_provider_files(ready_provider.settings)
    provider = config.load_provider()
    assert provider.ready is True
    assert provider.settings == ready_provider.settings
    assert provider.api_key == SYNTHETIC_KEY
    assert provider.reason is None
    assert provider.status()["live_model_tested"] is False


def test_load_provider_returns_safe_unavailable_status_for_invalid_config_and_key():
    write_provider_files(key=None)
    provider = config.load_provider()
    assert not provider.ready
    assert provider.api_key is None
    model_file = config.CONFIG_DIR / "model.json"
    valid_model = model_file.read_text()
    for payload in (
        "{",
        "[]",
        "{}",
        '{"model_id":"bad model"}',
        '{"model_id":"offline-model","dimensions":0}',
        '{"model_id":"offline-model","base_url":"http://provider.invalid"}',
    ):
        model_file.write_text(payload)
        provider = config.load_provider()
        assert not provider.ready
        assert provider.api_key is None
        assert "missing or invalid" in provider.reason
    model_file.write_text(valid_model)
    key_file = config.CONFIG_DIR / "openai-api-key"
    config.write_private(key_file, SYNTHETIC_KEY)
    for value, mode in ((SYNTHETIC_KEY, 0o644), ("", 0o600), ("sk two", 0o600)):
        key_file.write_text(value)
        key_file.chmod(mode)
        provider = config.load_provider()
        assert not provider.ready
        assert provider.api_key is None
        assert "missing or invalid" in provider.reason
        assert SYNTHETIC_KEY not in repr(provider)


def test_resolve_mode_keeps_explicit_static_with_or_without_live_config(ready_provider):
    for provider in (ready_provider, config.Provider(reason="missing")):
        result = config.resolve_mode("static", provider)
        assert result == {
            "requested_mode": "static",
            "effective_mode": "static",
            "fallback": False,
            "notice": "Static simulation—no LLM called.",
        }


def test_resolve_mode_selects_live_only_for_configured_provider(ready_provider):
    result = config.resolve_mode("live", ready_provider)
    assert result["requested_mode"] == result["effective_mode"] == "live"
    assert result["fallback"] is False
    assert "Live provider selected" in result["notice"]
    assert "charges" in result["notice"]


def test_resolve_mode_falls_back_with_notice_even_when_live_was_requested(
    ready_provider,
):
    unavailable = (
        config.load_provider(),
        config.Provider(settings=ready_provider.settings),
        config.Provider(api_key=SYNTHETIC_KEY),
        config.Provider(
            settings=ready_provider.settings,
            api_key=SYNTHETIC_KEY,
            reason="invalid config",
        ),
    )
    for provider in unavailable:
        result = config.resolve_mode("live", provider)
        assert result == {
            "requested_mode": "live",
            "effective_mode": "static",
            "fallback": True,
            "notice": "Static fallback—no LLM called.",
        }
        assert SYNTHETIC_KEY not in json.dumps(result)


def test_resolve_mode_rejects_unknown_requested_modes(ready_provider):
    for requested in ("auto", "LIVE", "", None):
        with pytest.raises(
            config.ConfigurationError, match="Choose static or live mode"
        ):
            config.resolve_mode(requested, ready_provider)


def test_configured_runtime_errors_propagate_instead_of_becoming_static(
    monkeypatch, ready_provider
):
    from e2e_oracle_demo.runtime import Runtime

    # Exercise real chat orchestration without calling Runtime.__init__ or any DB.
    runtime = object.__new__(Runtime)
    runtime.provider = ready_provider
    runtime._lock = RLock()
    runtime._session_locks = {}
    selected_modes = []

    def fail_run(*args, **kwargs):
        raise RuntimeError("synthetic configured provider failure")

    def fake_components(mode):
        selected_modes.append(mode)
        return {"agent": SimpleNamespace(run=fail_run)}

    monkeypatch.setattr(runtime, "components", fake_components)
    original_context = models.STATIC_REQUEST.get()
    with pytest.raises(RuntimeError, match="synthetic configured provider failure"):
        runtime.chat(
            {
                "mode": "live",
                "user_id": "alice",
                "component": "agent",
                "message": "offline failure",
                "action": "add_numbers",
            }
        )
    assert selected_modes == ["live"]
    assert runtime.mode("live")["effective_mode"] == "live"
    assert models.STATIC_REQUEST.get() is original_context


def test_static_model_uses_latest_user_and_bounded_agno_history():
    model = models.StaticModel(role_name="Worker")
    messages = [Message(role="user", content="excluded old context")]
    for index in range(1, 5):
        messages.extend(
            (
                Message(role="user", content=f"history {index} " + "x" * 200),
                Message(role="assistant", content="assistant-only context"),
            )
        )
    # A previous turn's tool result must not complete this turn's tool action.
    messages.append(
        Message(role="tool", content="previous tool result", tool_call_id="call_old")
    )
    messages.append(Message(role="user", content="current task"))
    response = model.invoke(messages)
    assert_static(response)
    assert "Worker received: current task" in response.content
    assert "deterministic test response, not AI reasoning" in response.content
    assert "Earlier user context supplied by Agno" in response.content
    assert "history 1" not in response.content
    assert "excluded old context" not in response.content
    assert "assistant-only context" not in response.content
    assert "previous tool result" not in response.content
    for index in (2, 3, 4):
        assert (f"history {index} " + "x" * 200)[:180] in response.content
    assert "x" * 181 not in response.content
    assert "static" in model.id
    assert "not an LLM" in model.name
    assert "simulation" in model.provider


def test_static_model_handles_empty_and_system_only_messages():
    for messages in ([], [Message(role="system", content="system instructions")]):
        response = models.StaticModel().invoke(messages)
        assert_static(response)
        assert "Assistant received: " in response.content
        assert "system instructions" not in response.content


def test_explicit_context_action_emits_agno_tool_json_and_stops_after_result():
    model = models.StaticModel()
    arguments = {"a": 2, "b": 3}
    # An old tool result precedes the newest user request and cannot suppress it.
    messages = [
        Message(role="user", content="previous task"),
        Message(role="tool", content="old result", tool_call_id="call_old"),
        Message(role="user", content="perform the selected action"),
    ]
    token = models.STATIC_REQUEST.set({"action": "add_numbers", "arguments": arguments})
    try:
        for tools in ([tool_spec()], [tool_spec()["function"]]):
            response = model.invoke(messages, tools=tools)
            assert isinstance(response, ModelResponse)
            assert response.role == "assistant"
            assert len(response.tool_calls) == 1
            call = response.tool_calls[0]
            assert re.fullmatch(r"call_[a-f0-9]{32}", call["id"])
            assert call["type"] == "function"
            assert call["function"]["name"] == "add_numbers"
            assert isinstance(call["function"]["arguments"], str)
            assert json.loads(call["function"]["arguments"]) == arguments
            followup = messages + [
                Message(role="assistant", tool_calls=response.tool_calls),
                Message(role="tool", content='{"result": 5}', tool_call_id=call["id"]),
            ]
            completed = model.invoke(followup, tools=tools)
            assert_static(completed)
            assert 'Tool returned:\n{"result": 5}' in completed.content
            assert "old result" not in completed.content
    finally:
        models.STATIC_REQUEST.reset(token)


def test_static_model_does_not_infer_tools_from_user_text_or_unavailable_action():
    messages = [Message(role="user", content="add_numbers: add 2 and 3")]
    for request, tools in (
        (None, [tool_spec()]),
        ({"action": "auto"}, [tool_spec()]),
        ({"action": ""}, [tool_spec()]),
        ({"action": "add_numbers", "arguments": {"a": 2, "b": 3}}, []),
        ({"action": "not_registered"}, [tool_spec()]),
    ):
        token = models.STATIC_REQUEST.set(request)
        try:
            assert_static(models.StaticModel().invoke(messages, tools=tools))
        finally:
            models.STATIC_REQUEST.reset(token)


def test_static_coordinator_delegates_to_worker_then_reviewer_then_finishes():
    model = models.StaticModel(role_name="Coordinator")
    assert model.role_name == "Coordinator"
    tools = [
        {
            "type": "function",
            "function": {
                "name": "delegate_task_to_member",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "member_id": {"type": "string"},
                        "task": {"type": "string"},
                    },
                    "required": ["member_id", "task"],
                },
            },
        }
    ]
    messages = [Message(role="user", content="prepare a report")]
    response = model.invoke(messages, tools=tools)
    assert response.role == "assistant"
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "delegate_task_to_member"
    assert json.loads(call["function"]["arguments"]) == {
        "member_id": "worker",
        "task": "prepare a report",
    }
    messages.extend(
        [
            Message(role="assistant", tool_calls=response.tool_calls),
            Message(role="tool", content="worker draft", tool_call_id=call["id"]),
        ]
    )
    review = model.invoke(messages, tools=tools)
    assert review.role == "assistant"
    assert len(review.tool_calls) == 1
    review_call = review.tool_calls[0]
    assert review_call["id"] != call["id"]
    assert review_call["type"] == "function"
    assert review_call["function"]["name"] == "delegate_task_to_member"
    assert json.loads(review_call["function"]["arguments"]) == {
        "member_id": "reviewer",
        "task": "worker draft",
    }
    messages.extend(
        [
            Message(role="assistant", tool_calls=review.tool_calls),
            Message(
                role="tool", content="reviewer final", tool_call_id=review_call["id"]
            ),
        ]
    )
    completed = model.invoke(messages, tools=tools)
    assert_static(completed)
    assert completed.content.endswith("worker draft\nreviewer final")


def test_static_stream_chunks_reassemble_exact_response():
    model = models.StaticModel()
    messages = [Message(role="user", content="a long user message " * 12)]
    expected = model.invoke(messages)
    chunks = list(model.invoke_stream(messages))
    assert len(chunks) > 1
    assert all(
        isinstance(chunk, ModelResponse) and chunk.role == "assistant"
        for chunk in chunks
    )
    assert all(
        0 < len(chunk.content) <= 36 and not chunk.tool_calls for chunk in chunks
    )
    assert "".join(chunk.content for chunk in chunks) == expected.content


def test_static_stream_emits_one_complete_tool_call_not_text_fragments():
    token = models.STATIC_REQUEST.set(
        {"action": "add_numbers", "arguments": {"a": 1, "b": 4}}
    )
    try:
        chunks = list(
            models.StaticModel().invoke_stream(
                [Message(role="user", content="add")],
                tools=[tool_spec()],
            )
        )
    finally:
        models.STATIC_REQUEST.reset(token)
    assert len(chunks) == 1
    assert chunks[0].role == "assistant"
    assert len(chunks[0].tool_calls) == 1
    assert json.loads(chunks[0].tool_calls[0]["function"]["arguments"]) == {
        "a": 1,
        "b": 4,
    }


def test_async_static_invoke_and_stream_support_text_and_explicit_tools():
    async def exercise():
        model = models.StaticModel()
        messages = [Message(role="user", content="async request")]
        expected = model.invoke(messages).content
        response = await model.ainvoke(messages)
        assert_static(response)
        assert response.content == expected
        chunks = [chunk async for chunk in model.ainvoke_stream(messages)]
        assert "".join(chunk.content for chunk in chunks) == expected
        token = models.STATIC_REQUEST.set(
            {"action": "add_numbers", "arguments": {"a": 7, "b": 8}}
        )
        try:
            response = await model.ainvoke(messages, tools=[tool_spec()])
            assert json.loads(response.tool_calls[0]["function"]["arguments"]) == {
                "a": 7,
                "b": 8,
            }
            chunks = [
                chunk
                async for chunk in model.ainvoke_stream(messages, tools=[tool_spec()])
            ]
            assert len(chunks) == 1
            assert chunks[0].tool_calls[0]["function"]["name"] == "add_numbers"
            assert json.loads(chunks[0].tool_calls[0]["function"]["arguments"]) == {
                "a": 7,
                "b": 8,
            }
            result = Message(
                role="tool", content="15", tool_call_id=response.tool_calls[0]["id"]
            )
            completed = await model.ainvoke(messages + [result], tools=[tool_spec()])
            assert_static(completed)
            assert "15" in completed.content
        finally:
            models.STATIC_REQUEST.reset(token)

    asyncio.run(exercise())


def test_static_embedder_is_deterministic_64_dimensional_and_normalized():
    embedder = models.StaticEmbedder()
    vector = embedder.get_embedding("Oracle tool testing Oracle")
    assert embedder.dimensions == 64
    assert embedder.enable_batch is False
    assert_unit_vector(vector)
    assert vector == models.StaticEmbedder().get_embedding("Oracle tool testing Oracle")
    assert vector == embedder.get_embedding("ORACLE, tool! testing oracle.")
    assert vector == embedder.get_embedding("testing Oracle Oracle tool")
    assert vector != embedder.get_embedding("a completely different document")
    # Returned lists must not share mutable state between calls.
    vector[0] = 99.0
    assert_unit_vector(embedder.get_embedding("Oracle tool testing Oracle"))


def test_static_embedder_empty_whitespace_and_punctuation_are_nonzero_unit_vectors():
    embedder = models.StaticEmbedder()
    for text in ("", " \t\n", "!?.,;—"):
        vector = embedder.get_embedding(text)
        assert_unit_vector(vector)
        assert vector == [1.0] + [0.0] * 63


def test_static_embedder_sync_usage_and_async_interfaces_agree():
    async def exercise():
        embedder = models.StaticEmbedder()
        for text in ("", "Oracle offline embedding"):
            expected = embedder.get_embedding(text)
            vector, usage = embedder.get_embedding_and_usage(text)
            assert vector == expected
            assert usage == {"total_tokens": 0}
            assert await embedder.async_get_embedding(text) == expected
            async_vector, async_usage = await embedder.async_get_embedding_and_usage(
                text
            )
            assert async_vector == expected
            assert async_usage == usage
            assert_unit_vector(async_vector)

    asyncio.run(exercise())


def test_live_factories_build_real_agno_connectors_without_initializing_clients(
    monkeypatch,
    ready_provider,
):
    import openai
    from agno.models.openai import OpenAIResponses

    def deny_client_initialization(self, *args, **kwargs):
        pytest.fail(
            "Live connector construction initialized an OpenAI client", pytrace=False
        )

    # Patch the class itself, including aliases already imported inside Agno.
    monkeypatch.setattr(openai.OpenAI, "__init__", deny_client_initialization)
    monkeypatch.setattr(openai.AsyncOpenAI, "__init__", deny_client_initialization)
    factory = models.build_live_model
    for unavailable in (
        config.Provider(),
        config.Provider(settings=ready_provider.settings),
    ):
        with pytest.raises(ValueError, match="Live provider is not configured"):
            factory(unavailable)
    connector = factory(ready_provider)
    assert isinstance(connector, OpenAIResponses)
    assert connector.api_key == SYNTHETIC_KEY
    assert str(connector.base_url).rstrip("/") == ready_provider.settings.base_url
    client_fields = {
        name: value
        for name, value in vars(connector).items()
        if name
        in {
            "client",
            "async_client",
            "openai_client",
            "async_openai_client",
            "_client",
            "_async_client",
        }
    }
    assert client_fields, "Inspect at least one real Agno client field"
    assert all(value is None for value in client_fields.values())
    assert connector.id == ready_provider.settings.model_id
    assert connector.store is False
    assert connector.max_output_tokens == ready_provider.settings.max_output_tokens
    assert connector.timeout == 60
    assert connector.max_retries == 0


@pytest.mark.parametrize("configured", [False, True])
def test_live_documents_use_static_embeddings_without_api_clients(
    configured, monkeypatch, ready_provider
):
    from agno.knowledge.embedder.openai import OpenAIEmbedder

    def deny_embedding_api(*args, **kwargs):
        pytest.fail("Static embeddings must not construct a provider embedder")

    monkeypatch.setattr(OpenAIEmbedder, "__init__", deny_embedding_api)
    provider = ready_provider if configured else config.Provider()
    embedder = models.build_live_embedder(provider)
    assert isinstance(embedder, models.StaticEmbedder)
    assert embedder.id == config.DEMO_EMBEDDING_MODEL
    assert embedder.dimensions == config.DEMO_EMBEDDING_DIMENSIONS
    assert embedder.get_embedding(
        "same local document"
    ) == models.StaticEmbedder().get_embedding("same local document")
    assert provider.embedding_status()["synthetic"] is True
    assert provider.embedding_status()["ready"] is True


@pytest.mark.parametrize("limit", [1, 1024, 4096, 16384])
def test_live_output_budget_is_validated_and_forwarded_without_a_request(limit):
    settings = config.ModelSettings(
        model_id="openai.gpt-oss-20b",
        base_url="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/v1",
        max_output_tokens=limit,
    ).validate()
    provider = config.Provider(settings, SYNTHETIC_KEY)
    connector = models.build_live_model(provider)
    assert connector.id == settings.model_id
    assert connector.max_output_tokens == limit
    assert connector.store is False
    assert connector.max_retries == 0
    assert provider.status()["max_output_tokens"] == limit


@pytest.mark.parametrize("limit", [0, -1, 16385, True, 1.5, "1024", None])
def test_invalid_output_budgets_are_rejected(limit):
    with pytest.raises(config.ConfigurationError, match="output token limit"):
        config.ModelSettings(
            model_id="offline-model", max_output_tokens=limit
        ).validate()


def test_oci_live_chat_uses_static_embeddings():
    settings = config.ModelSettings(
        model_id="openai.gpt-oss-20b",
        base_url="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/v1",
    ).validate()
    provider = config.Provider(settings, SYNTHETIC_KEY)
    assert provider.ready
    assert provider.status()["provider"] == "OCI Generative AI / Responses"
    assert provider.embeddings_ready
    assert provider.embedding_status()["ready"] is True
    assert provider.embedding_status()["model_id"] == config.DEMO_EMBEDDING_MODEL
    assert provider.embedding_status()["synthetic"] is True
    assert isinstance(models.build_live_embedder(provider), models.StaticEmbedder)


def test_oci_responses_endpoint_requires_the_openai_compatible_suffix():
    settings = config.ModelSettings(
        model_id="openai.gpt-oss-20b",
        base_url="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions",
    )
    with pytest.raises(config.ConfigurationError, match="/20231130/actions/v1"):
        settings.validate()


class FakeRuntime:
    """Only the app-facing contract; no Agno run, driver, or provider calls."""

    def __init__(self):
        self.provider = config.Provider(reason="Offline fixture has no model key.")
        self.chat_calls = []
        self.close_calls = 0
        self.failure = None
        self.last_result = None

    def mode(self, requested):
        return config.resolve_mode(requested, self.provider)

    def status(self):
        return {
            "database": {"connected": False, "offline_fixture": True},
            "model": self.provider.status(),
            "static": {"label": "[STATIC: no LLM] offline fixture"},
        }

    def chat(self, body, on_chunk=None):
        self.chat_calls.append(dict(body))
        if self.failure is not None:
            raise self.failure
        provenance = self.mode(body["mode"])
        content = (
            "[STATIC: no LLM] offline fake response"
            if provenance["effective_mode"] == "static"
            else "Synthetic live-path fixture response; no provider invoked."
        )
        if on_chunk is not None:
            on_chunk(content[:20])
            on_chunk(content[20:])
        self.last_result = {
            **provenance,
            "content": content,
            "session_id": "offline-session",
            "run_id": "offline-run",
            "status": "completed",
            "tools": [],
            "requirements": [],
            "evidence": {
                "offline_fixture": True,
                "simulated": provenance["effective_mode"] == "static",
            },
        }
        return self.last_result

    def close(self):
        self.close_calls += 1


@pytest.fixture
def app_client(monkeypatch):
    from fastapi.testclient import TestClient
    from e2e_oracle_demo import app as app_module

    def deny_real_runtime(*args, **kwargs):
        pytest.fail("Injected app attempted to construct a real runtime", pytrace=False)

    monkeypatch.setattr(app_module, "Runtime", deny_real_runtime)
    runtime = FakeRuntime()
    app = app_module.create_app(runtime=runtime, factory=deny_real_runtime)
    assert not hasattr(app.state, "runtime")
    with TestClient(
        app,
        base_url="http://testserver",
        headers={"Origin": "http://testserver"},
        raise_server_exceptions=False,
    ) as client:
        assert app.state.runtime is runtime
        yield client, runtime
        assert runtime.close_calls == 0
    assert runtime.close_calls == 1


def sse_records(response):
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    records = []
    for record in response.text.split("\n\n"):
        if record.strip():
            assert record.startswith("data: ")
            records.append(json.loads(record.removeprefix("data: ")))
    assert records
    return records


def test_injected_app_lifespan_status_and_security_headers(app_client):
    client, runtime = app_client
    for path in ("/api/status", "/health"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.json() == runtime.status()
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert not runtime.chat_calls


def test_injected_app_json_chat_preserves_static_fallback_and_configured_live(
    app_client, ready_provider
):
    client, runtime = app_client
    for requested in ("static", "live"):
        response = client.post(
            "/api/chat",
            json={
                "message": "offline request",
                "mode": requested,
                "user_id": "bob",
                "action": "add_numbers",
                "arguments": {"a": 2, "b": 3},
            },
        )
        assert response.status_code == 200
        assert response.json() == runtime.last_result
        assert response.json()["requested_mode"] == requested
        assert response.json()["effective_mode"] == "static"
        assert response.json()["fallback"] is (requested == "live")
        assert response.json()["content"].startswith("[STATIC: no LLM]")
        assert "no LLM called" in response.json()["notice"]
        assert runtime.chat_calls[-1] == {
            "user_id": "bob",
            "mode": requested,
            "component": "agent",
            "session_id": None,
            "message": "offline request",
            "stream": False,
            "action": "add_numbers",
            "arguments": {"a": 2, "b": 3},
        }
    runtime.provider = ready_provider
    response = client.post(
        "/api/chat", json={"message": "fake live request", "mode": "live"}
    )
    assert response.status_code == 200
    assert response.json() == runtime.last_result
    assert response.json()["effective_mode"] == "live"
    assert response.json()["fallback"] is False
    assert response.json()["evidence"]["simulated"] is False


def test_injected_app_sse_has_content_text_and_flat_final_result(app_client):
    client, runtime = app_client
    with client.stream(
        "POST",
        "/api/chat",
        json={
            "message": "stream offline",
            "mode": "live",
            "stream": True,
        },
    ) as response:
        response.read()
        records = sse_records(response)
    assert [record["type"] for record in records] == ["content", "content", "result"]
    assert all(set(record) == {"type", "text"} for record in records[:-1])
    assert (
        "".join(record["text"] for record in records[:-1])
        == runtime.last_result["content"]
    )
    assert records[-1] == {"type": "result", **runtime.last_result}
    assert "result" not in records[-1]
    assert records[-1]["requested_mode"] == "live"
    assert records[-1]["effective_mode"] == "static"
    assert records[-1]["fallback"] is True
    assert "Static fallback" in records[-1]["notice"]
    assert runtime.chat_calls[-1]["stream"] is True


def test_injected_app_rejects_untrusted_origins_hosts_and_invalid_requests(app_client):
    client, runtime = app_client
    response = client.post(
        "/api/chat",
        json={"message": "blocked"},
        headers={"Origin": "https://other.invalid"},
    )
    assert response.status_code == 403
    response = client.get(
        "/api/status",
        headers={"Host": "other.invalid", "Origin": "http://other.invalid"},
    )
    assert response.status_code == 400
    for overrides in (
        {"user_id": "mallory"},
        {"action": "invalid-action"},
        {"mode": "automatic"},
        {"component": "invalid-component"},
        {"message": ""},
        {"message": "x" * 12001},
        {"session_id": "x" * 161},
        {"arguments": {"value": "x" * 20001}},
    ):
        response = client.post(
            "/api/chat", json={"message": "offline request", **overrides}
        )
        assert response.status_code == 422
    for headers in ({}, {"Content-Type": "text/plain"}):
        response = client.post(
            "/api/chat", content='{"message":"offline request"}', headers=headers
        )
        assert response.status_code == 415
    response = client.post(
        "/api/chat",
        json={"message": "oversized request"},
        headers={"Content-Length": "100001"},
    )
    assert response.status_code == 413
    assert not runtime.chat_calls


def test_injected_app_reports_configured_live_errors_without_static_fallback(
    app_client, ready_provider
):
    client, runtime = app_client
    runtime.provider = ready_provider
    runtime.failure = RuntimeError("synthetic provider failure " + SYNTHETIC_KEY)
    response = client.post(
        "/api/chat", json={"message": "fail offline", "mode": "live"}
    )
    assert response.status_code == 500
    assert response.json()["detail"]["type"] == "RuntimeError"
    assert "hidden" in response.json()["detail"]["message"]
    assert SYNTHETIC_KEY not in response.text
    with client.stream(
        "POST",
        "/api/chat",
        json={
            "message": "fail offline stream",
            "mode": "live",
            "stream": True,
        },
    ) as response:
        response.read()
        records = sse_records(response)
    assert len(records) == 1
    assert records[0]["type"] == "error"
    assert records[0]["detail"]["type"] == "RuntimeError"
    assert SYNTHETIC_KEY not in json.dumps(records)
    assert runtime.last_result is None
    assert runtime.mode("live")["effective_mode"] == "live"
    assert len(runtime.chat_calls) == 2
    assert all(body["mode"] == "live" for body in runtime.chat_calls)


def iam_settings(**overrides):
    values = {
        "compartment_id": "ocid1.compartment.oc1..offlinecompartment",
        "config_file": "/offline/oci/config",
    }
    values.update(overrides)
    return config.OciIamSettings(**values)


@pytest.mark.parametrize("component", ["agent", "team"])
def test_stream_only_emits_selected_component_content(component):
    from e2e_oracle_demo.runtime import Runtime

    runtime = object.__new__(Runtime)
    runtime.provider = config.Provider(
        config.ModelSettings(model_id="offline-model"), SYNTHETIC_KEY
    )
    runtime._lock = RLock()
    runtime._session_locks = {}
    event = "TeamRunContent" if component == "team" else "RunContent"
    other_event = "RunContent" if component == "team" else "TeamRunContent"
    output = SimpleNamespace(
        run_id="parent", status="COMPLETED", content="Final answer", to_dict=lambda: {}
    )

    class Stream:
        def run(self, *args, **kwargs):
            yield SimpleNamespace(
                event=other_event, content="Member draft", run_id="child"
            )
            yield SimpleNamespace(event=event, content="Final ", run_id="parent")
            yield SimpleNamespace(
                event=other_event, content="Member review", run_id="child"
            )
            yield SimpleNamespace(event=event, content="answer", run_id="parent")
            yield output

    runtime.components = lambda mode: {component: Stream()}
    chunks = []
    result = runtime.chat(
        {
            "mode": "live",
            "user_id": "alice",
            "component": component,
            "message": "stream check",
        },
        on_chunk=chunks.append,
    )
    assert chunks == ["Final ", "answer"]
    assert "".join(chunks) == result["content"]
    assert result["status"] == "COMPLETED"


@pytest.mark.parametrize("model_id", ["openai.gpt-oss-20b", "openai.gpt-oss-120b"])
def test_oci_gpt_oss_replays_conversation_and_tool_results_without_reasoning(model_id):
    from copy import deepcopy
    from e2e_oracle_demo.iam import OciIamResponses

    reasoning = {
        "id": "rs_offline",
        "type": "reasoning",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": "synthetic offline text"}],
    }
    messages = [
        Message(role="user", content="My marker is cedar. Add 17 and 25."),
        Message(
            role="assistant",
            provider_data={"reasoning_output": reasoning},
            tool_calls=[
                {
                    "id": "fc_offline",
                    "call_id": "call_offline",
                    "type": "function",
                    "function": {"name": "add_numbers", "arguments": '{"a":17,"b":25}'},
                }
            ],
        ),
        Message(role="tool", tool_call_id="fc_offline", content="42"),
        Message(role="assistant", content="The total is 42."),
        Message(role="user", content="What was my marker?"),
    ]
    before = deepcopy(messages[1].provider_data)
    wire = OciIamResponses(id=model_id, store=False)._format_messages(messages)
    assert not any(
        (item.get("type") if isinstance(item, dict) else item.type) == "reasoning"
        for item in wire
    )
    assert {"role": "user", "content": "What was my marker?"} in wire
    assert {"role": "assistant", "content": "The total is 42."} in wire
    call = next(
        item
        for item in wire
        if isinstance(item, dict) and item.get("type") == "function_call"
    )
    output = next(
        item
        for item in wire
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    )
    assert call["call_id"] == output["call_id"] == "call_offline"
    assert json.loads(call["arguments"]) == {"a": 17, "b": 25}
    assert output["output"] == "42"
    assert messages[1].provider_data == before


@pytest.mark.parametrize(
    "component,event", [("agent", "RunError"), ("team", "TeamRunError")]
)
def test_stream_provider_error_is_reported_before_missing_final_output(
    component, event
):
    from e2e_oracle_demo.runtime import DemoError, Runtime

    runtime = object.__new__(Runtime)
    runtime.provider = config.Provider(
        config.ModelSettings(model_id="offline-model"), SYNTHETIC_KEY
    )
    runtime._lock = RLock()
    runtime._session_locks = {}

    class FailedStream:
        def run(self, *args, **kwargs):
            assert kwargs["stream"] is True
            yield SimpleNamespace(
                event="RunContent" if component == "agent" else "TeamRunContent",
                content="partial",
            )
            yield SimpleNamespace(
                event=event, run_id="offline-run", content=SYNTHETIC_KEY
            )

    runtime.components = lambda mode: {component: FailedStream()}
    chunks = []
    with pytest.raises(DemoError, match="failed during streaming") as raised:
        runtime.chat(
            {
                "user_id": "alice",
                "mode": "live",
                "component": component,
                "message": "offline",
            },
            on_chunk=chunks.append,
        )
    assert chunks == ["partial"]
    assert "final saved run" not in str(raised.value)
    assert SYNTHETIC_KEY not in str(raised.value)
    assert models.STATIC_REQUEST.get() is None


def test_iam_configuration_bypasses_existing_token_files(monkeypatch):
    settings = iam_settings().validate()
    config.write_private(
        config.CONFIG_DIR / "oci-iam.json", json.dumps(asdict(settings))
    )
    (config.CONFIG_DIR / "model.json").write_text("invalid token configuration")
    monkeypatch.setattr(config, "find_spec", lambda _: object())

    def no_token_read(*args, **kwargs):
        pytest.fail("IAM must not read the old API-token file")

    monkeypatch.setattr(config, "read_private_value", no_token_read)
    provider = config.load_provider()
    assert provider.ready and provider.api_key is None
    assert provider.iam == settings
    assert provider.status()["auth_mode"] == "oci_iam"
    assert provider.status()["region"] == "us-chicago-1"
    assert provider.status()["live_model_tested"] is False
    assert settings.compartment_id not in repr(provider)
    assert settings.config_file not in json.dumps(provider.status())
    assert models.build_live_embedder(provider).dimensions == 64


@pytest.mark.parametrize(
    "fault", ["invalid-json", "public-permissions", "symlink", "missing-package"]
)
def test_invalid_iam_settings_do_not_select_an_existing_token(monkeypatch, fault):
    write_provider_files()
    path = config.CONFIG_DIR / "oci-iam.json"
    config.write_private(path, json.dumps(asdict(iam_settings())))
    monkeypatch.setattr(
        config, "find_spec", lambda _: None if fault == "missing-package" else object()
    )
    if fault == "invalid-json":
        path.write_text("{")
    elif fault == "public-permissions":
        path.chmod(0o644)
    elif fault == "symlink":
        path.unlink()
        path.symlink_to(config.CONFIG_DIR / "absent-settings")
    provider = config.load_provider()
    assert not provider.ready
    assert provider.api_key is None
    assert "OCI IAM" in provider.reason


@pytest.mark.parametrize(
    "overrides",
    [
        {"region": "us-phoenix-1"},
        {"compartment_id": "not-an-ocid"},
        {"profile_name": "bad\nprofile"},
        {"config_file": "relative/config"},
        {"max_output_tokens": True},
        {"max_output_tokens": 0},
    ],
)
def test_iam_rejects_invalid_configuration(overrides):
    with pytest.raises(config.ConfigurationError):
        iam_settings(**overrides).validate()


def test_iam_model_is_lazy_and_caches_both_client_paths(monkeypatch):
    from e2e_oracle_demo import iam

    calls = []

    def build(settings, *, asynchronous=False):
        calls.append(asynchronous)
        return SimpleNamespace(is_closed=lambda: False)

    monkeypatch.setattr(iam, "build_signed_client", build)
    settings = iam_settings()
    provider = config.Provider(settings=settings.model_settings(), iam=settings)
    model = models.build_live_model(provider)
    assert isinstance(model, iam.OciIamResponses)
    assert calls == []
    assert model.get_client() is model.get_client()
    assert model.get_async_client() is model.get_async_client()
    assert calls == [False, True]
    assert model.max_output_tokens == 1024
    assert model.store is False and model.max_retries == 0


@pytest.mark.parametrize("as_dictionary", [False, True])
def test_iam_tool_wire_format_preserves_local_approval_policy(as_dictionary):
    from copy import deepcopy
    from agno.tools.function import Function
    from e2e_oracle_demo.iam import OciIamResponses

    function = Function(
        name="publish_report",
        requires_confirmation=True,
        parameters={
            "type": "object",
            "properties": {"requires_confirmation": {"type": "boolean"}},
        },
    )
    tool = (
        {"type": "function", "function": function.to_dict()}
        if as_dictionary
        else function
    )
    before = deepcopy(tool) if as_dictionary else function.to_dict()
    wire = OciIamResponses(id="offline")._format_tool_params([], [tool])[0]
    assert (
        not {"requires_confirmation", "external_execution", "approval_type"}
        & wire.keys()
    )
    assert wire["parameters"]["properties"]["requires_confirmation"] == {
        "type": "boolean"
    }
    assert (tool if as_dictionary else function.to_dict()) == before
    assert function.requires_confirmation is True


def test_iam_requested_confirmation_still_pauses_agno_before_execution():
    from agno.agent import Agent
    from agno.tools.function import Function
    from openai.types.responses import Response
    from e2e_oracle_demo.iam import OciIamResponses

    executed = []
    sent = []

    def publish_report():
        executed.append(True)
        return "published"

    def respond(**kwargs):
        sent.append(kwargs)
        return Response.model_validate(
            {
                "id": "resp_offline",
                "object": "response",
                "status": "completed",
                "created_at": 0,
                "model": "offline",
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_offline",
                        "call_id": "call_offline",
                        "name": "publish_report",
                        "arguments": "{}",
                    }
                ],
            }
        )

    model = OciIamResponses(id="offline", store=False)
    model.client = SimpleNamespace(
        is_closed=lambda: False, responses=SimpleNamespace(create=respond)
    )
    function = Function.from_callable(publish_report)
    function.requires_confirmation = True
    result = Agent(model=model, tools=[function], telemetry=False).run(
        "Publish the report"
    )
    assert str(result.status.value).upper() == "PAUSED"
    assert result.requirements and executed == []
    assert "requires_confirmation" not in sent[0]["tools"][0]
    assert function.requires_confirmation is True


@pytest.mark.parametrize("asynchronous", [False, True])
def test_oci_sdk_signs_chicago_responses_with_a_synthetic_key(
    monkeypatch, tmp_path, asynchronous
):
    import base64
    import hashlib
    import httpx
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from e2e_oracle_demo.iam import build_signed_client

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    key_file = tmp_path / "synthetic-private-key.pem"
    key_file.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    key_file.chmod(0o600)
    profile = tmp_path / "synthetic-oci-config"
    profile.write_text(
        "[OFFLINE]\nuser=ocid1.user.oc1..offline\n"
        "tenancy=ocid1.tenancy.oc1..offline\nregion=us-ashburn-1\n"
        "fingerprint=" + ":".join(["00"] * 16) + "\nkey_file=" + str(key_file) + "\n"
    )
    profile.chmod(0o600)
    settings = iam_settings(config_file=str(profile), profile_name="OFFLINE")
    received = []

    def respond(request):
        received.append(request)
        assert (
            request.url.host
            == "inference.generativeai.us-chicago-1.oci.oraclecloud.com"
        )
        assert request.url.path == "/20231130/actions/v1/responses"
        assert request.headers["authorization"].startswith("Signature ")
        assert "Bearer" not in request.headers["authorization"]
        assert request.headers["CompartmentId"] == settings.compartment_id
        assert (
            request.headers["x-content-sha256"]
            == base64.b64encode(hashlib.sha256(request.content).digest()).decode()
        )
        return httpx.Response(
            200,
            json={
                "id": "resp_offline",
                "object": "response",
                "status": "completed",
                "model": settings.model_id,
                "created_at": 0,
                "output": [
                    {
                        "type": "message",
                        "id": "msg_offline",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": "OK", "annotations": []}
                        ],
                    }
                ],
            },
        )

    def sync_send(self, request):
        return respond(request)

    async def async_send(self, request):
        return respond(request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", sync_send)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", async_send)
    kwargs = {
        "model": settings.model_id,
        "input": "Reply OK.",
        "store": False,
        "max_output_tokens": 1024,
    }
    if asynchronous:

        async def exercise():
            async with build_signed_client(settings, asynchronous=True) as client:
                response = await client.responses.create(**kwargs)
                assert response.output_text == "OK"

        asyncio.run(exercise())
    else:
        with build_signed_client(settings) as client:
            assert client.responses.create(**kwargs).output_text == "OK"
    assert len(received) == 1
