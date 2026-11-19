"""Regression tests for memory provider selection during AIAgent init."""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_dependency_activation(_hermetic_environment):
    # pytest already owns its interpreter. Launch-time PM activation probes beside
    # the checkout (independent of HERMES_HOME), which may be inside a real profile.
    # Keep the real agent/config imports without selecting another dependency env.
    from pm import environments

    activate_dependencies = environments.activate_dependencies
    with patch.object(environments, "activate_dependencies"):
        yield
    # hermes_bootstrap imports this function by value; don't leave a mock cached
    # for later tests that exercise the actual bootstrap.
    if bootstrap := sys.modules.get("hermes_bootstrap"):
        bootstrap.activate_dependencies = activate_dependencies


def test_process_bootstrap_fresh_import_in_isolated_home(monkeypatch, tmp_path):
    """A cold import must work under the home guard, even for a checkout inside a profile."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import agent

    monkeypatch.setattr(agent, "process_bootstrap", None, raising=False)
    for name in ("agent.process_bootstrap", "hermes_bootstrap"):
        # Record absent entries too, so the cold imports cannot leak into later tests.
        monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.delitem(sys.modules, name)

    bootstrap = importlib.import_module("agent.process_bootstrap")

    from hermes_constants import get_hermes_home

    assert get_hermes_home() == tmp_path
    assert callable(bootstrap.OpenAI)


class RecordingMemoryProvider:
    name = "recording"

    def __init__(self):
        self.init_kwargs = None
        self.init_session_id = None

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.init_session_id = session_id
        self.init_kwargs = dict(kwargs)

    def get_tool_schemas(self):
        return []

    def shutdown(self):
        pass


def test_shutdown_memory_provider_is_idempotent():
    from unittest.mock import MagicMock

    from run_agent import AIAgent

    manager = MagicMock()
    agent = object.__new__(AIAgent)
    agent._memory_manager = manager
    agent.context_compressor = None
    agent.session_id = "session-1"

    agent.shutdown_memory_provider([{"role": "user", "content": "one"}])
    agent.shutdown_memory_provider([{"role": "user", "content": "two"}])

    manager.on_session_end.assert_called_once()
    manager.shutdown_all.assert_called_once()


def test_blank_memory_provider_stays_opt_out():
    """Blank memory.provider loads no external provider and never rewrites the config."""
    cfg = {"memory": {"provider": ""}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("hermes_cli.config.save_config") as save_config,
        patch("plugins.memory.load_memory_provider") as load_memory_provider,
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
        )

    assert agent._memory_manager is None
    load_memory_provider.assert_not_called()
    save_config.assert_not_called()


def test_close_shuts_down_memory_provider():
    from unittest.mock import MagicMock

    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent._memory_manager = MagicMock()
    agent.context_compressor = None
    agent.session_id = ""
    agent._session_messages = []

    agent.close()

    agent._memory_manager.shutdown_all.assert_called_once()


def test_aiagent_forwards_user_id_alt_to_memory_provider():
    provider = RecordingMemoryProvider()
    cfg = {"memory": {"provider": "recording"}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            session_id="sess-alt",
            platform="feishu",
            user_id="open-id",
            user_id_alt="union-id",
        )

    assert agent._memory_manager is not None
    assert provider.init_session_id == "sess-alt"
    assert provider.init_kwargs["user_id"] == "open-id"
    assert provider.init_kwargs["user_id_alt"] == "union-id"
    assert provider.init_kwargs["platform"] == "feishu"
    assert "warning_callback" not in provider.init_kwargs
    assert "status_callback" not in provider.init_kwargs


class CoreShadowProvider:
    """Provider that tries to register tools shadowing built-in core tools."""

    name = "core-shadow"

    def get_tool_schemas(self):
        return [
            {"name": "clarify", "description": "shadows built-in clarify"},
            {"name": "delegate_task", "description": "shadows built-in delegate"},
            {"name": "honcho_search", "description": "legit memory tool"},
        ]


def test_core_tool_names_rejected_from_memory_routing_table():
    """Memory tools shadowing core tool names are rejected at registration (#40466).

    Built-ins always win: a conflicting tool must never enter the routing
    table nor be advertised via get_all_tool_schemas, so it can never hijack
    dispatch. The non-conflicting tool is preserved.
    """
    from agent.memory_manager import MemoryManager

    mm = MemoryManager()
    mm.add_provider(CoreShadowProvider())

    # Reserved names never enter the routing table
    assert not mm.has_tool("clarify")
    assert not mm.has_tool("delegate_task")
    assert "clarify" not in mm._tool_to_provider
    assert "delegate_task" not in mm._tool_to_provider

    # Non-conflicting tool survives
    assert mm.has_tool("honcho_search")
    assert "honcho_search" in mm._tool_to_provider

    # Manager never advertises a schema it would refuse to route
    schema_names = {s.get("name") for s in mm.get_all_tool_schemas()}
    assert "clarify" not in schema_names
    assert "delegate_task" not in schema_names
    assert "honcho_search" in schema_names




def test_aiagent_reuses_handed_in_memory_manager_without_reinitializing():
    """A caller that rebuilds the agent per turn (gateway api_server) hands back the session's manager:
    the provider keeps its state — no second load, no second initialize (#120116)."""
    provider = RecordingMemoryProvider()
    cfg = {"memory": {"provider": "recording"}, "agent": {}}
    common = dict(
        api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
        skip_context_files=True, skip_memory=False, session_id="sess-api", platform="api_server",
    )
    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=provider) as load_memory_provider,
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        first = AIAgent(**common)
        manager = first._memory_manager
        assert manager is not None and load_memory_provider.call_count == 1
        provider.init_session_id = None  # a re-initialize would set it again

        second = AIAgent(memory_manager=manager, **common)

    assert second._memory_manager is manager
    assert load_memory_provider.call_count == 1
    assert provider.init_session_id is None


@pytest.fixture
def configured_memory_agent(tmp_path, monkeypatch):
    """Exercise the real config loader and agent initialization in an isolated home."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = RecordingMemoryProvider()
    with (
        patch("plugins.memory.load_memory_provider", return_value=provider) as load_provider,
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        def build(timeout_yaml="", manager=None):
            (tmp_path / "config.yaml").write_text(
                'memory:\n  provider: recording\n' + timeout_yaml,
                encoding="utf-8",
            )
            return AIAgent(
                api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
                quiet_mode=True, skip_context_files=True, skip_memory=False,
                session_id="timeout-test", memory_manager=manager,
            )

        yield build, provider, load_provider


@pytest.mark.parametrize("value", ["0.125", "17", '"2.5"'])
def test_memory_prefetch_timeout_from_config(configured_memory_agent, value):
    build, provider, _ = configured_memory_agent
    agent = build(f"  external_prefetch_timeout: {value}\n")
    assert provider.init_session_id == "timeout-test"
    assert agent._memory_manager._external_prefetch_timeout == float(value.strip('"'))


@pytest.mark.parametrize("value", ["bad", ".nan", ".inf", "-.inf", "0", "-1", "null", "[]", "{}"])
def test_memory_prefetch_invalid_config_falls_back(configured_memory_agent, caplog, value):
    build, provider, _ = configured_memory_agent
    agent = build(f"  external_prefetch_timeout: {value}\n")
    assert provider.init_session_id == "timeout-test"
    assert agent._memory_manager._external_prefetch_timeout == 8.0
    assert any(
        record.levelname == "WARNING" and "memory.external_prefetch_timeout" in record.message
        and "8.0" in record.message for record in caplog.records
    )


def test_memory_prefetch_default_config(configured_memory_agent):
    from hermes_cli.config import load_config_readonly

    build, _, _ = configured_memory_agent
    agent = build()
    assert load_config_readonly()["memory"]["external_prefetch_timeout"] == 8.0
    assert agent._memory_manager._external_prefetch_timeout == 8.0


@pytest.mark.parametrize("replacement", ["42", "bad"])
def test_memory_prefetch_reused_manager_keeps_timeout(configured_memory_agent, caplog, replacement):
    build, provider, load_provider = configured_memory_agent
    first = build("  external_prefetch_timeout: 0.25\n")
    manager = first._memory_manager
    provider.init_session_id = None
    second = build(f"  external_prefetch_timeout: {replacement}\n", manager=manager)
    assert second._memory_manager is manager
    assert manager._external_prefetch_timeout == 0.25
    assert load_provider.call_count == 1
    assert provider.init_session_id is None
    assert not any("memory.external_prefetch_timeout" in record.message for record in caplog.records)
