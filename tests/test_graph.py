"""Tests for src/graph.py"""
import pytest
from unittest.mock import MagicMock, patch
from pydantic import ValidationError
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END


# ---------------------------------------------------------------------------
# 1. get_llm() provider selection
# ---------------------------------------------------------------------------

class TestGetLlm:
    def test_groq_provider(self):
        with patch("src.graph.ChatGroq") as mock_groq, \
             patch("src.graph.ChatAnthropic"), \
             patch("src.graph.ChatGoogleGenerativeAI"):
            from src.graph import get_llm
            get_llm("groq")
            mock_groq.assert_called_once()
            call_kwargs = mock_groq.call_args.kwargs
            assert call_kwargs["model"] == "llama-3.3-70b-versatile"

    def test_gemini_provider(self):
        with patch("src.graph.ChatGoogleGenerativeAI") as mock_gemini, \
             patch("src.graph.ChatAnthropic"), \
             patch("src.graph.ChatGroq"):
            from src.graph import get_llm
            get_llm("gemini")
            mock_gemini.assert_called_once()
            call_kwargs = mock_gemini.call_args.kwargs
            assert call_kwargs["model"] == "gemini-2.0-flash"

    def test_claude_haiku_provider(self):
        with patch("src.graph.ChatAnthropic") as mock_anthropic, \
             patch("src.graph.ChatGoogleGenerativeAI"), \
             patch("src.graph.ChatGroq"):
            from src.graph import get_llm
            get_llm("claude-haiku")
            mock_anthropic.assert_called_once()
            call_kwargs = mock_anthropic.call_args.kwargs
            assert call_kwargs["model"] == "claude-haiku-3-5-20251001"

    def test_claude_sonnet_provider(self):
        with patch("src.graph.ChatAnthropic") as mock_anthropic, \
             patch("src.graph.ChatGoogleGenerativeAI"), \
             patch("src.graph.ChatGroq"):
            from src.graph import get_llm
            get_llm("claude-sonnet")
            mock_anthropic.assert_called_once()
            call_kwargs = mock_anthropic.call_args.kwargs
            assert call_kwargs["model"] == "claude-sonnet-4-5-20251014"


# ---------------------------------------------------------------------------
# 2. _extract_response()
# ---------------------------------------------------------------------------

class TestExtractResponse:
    def setup_method(self):
        from src.graph import _extract_response
        self.extract = _extract_response

    def test_dict_with_messages(self):
        msg = MagicMock()
        msg.content = "hello"
        result = self.extract({"messages": [msg]})
        assert result == "hello"

    def test_dict_with_output_key(self):
        result = self.extract({"output": "direct output"})
        assert result == "direct output"

    def test_plain_string(self):
        result = self.extract("plain string")
        assert result == "plain string"

    def test_list_of_strings(self):
        result = self.extract(["foo", "bar", "baz"])
        assert result == "foo bar baz"

    def test_dict_with_empty_messages(self):
        result = self.extract({"messages": []})
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 3. process_query() with mocked agent
# ---------------------------------------------------------------------------

class TestProcessQuery:
    def test_returns_tuple_with_usage(self):
        fake_message = MagicMock()
        fake_message.content = "You spent $360.11 on food."
        fake_message.usage_metadata = {"input_tokens": 100, "output_tokens": 20}

        fake_result = {"messages": [fake_message]}
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = fake_result

        with patch("src.graph._build_agent", return_value=mock_agent):
            from src.graph import process_query
            response_text, usage = process_query(
                "How much did I spend on food?", provider="groq"
            )

        assert isinstance(response_text, str)
        assert isinstance(usage, dict)
        assert response_text == "You spent $360.11 on food."
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 20

    def test_raises_runtime_error_on_agent_failure(self):
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = Exception("boom")

        with patch("src.graph._build_agent", return_value=mock_agent):
            from src.graph import process_query
            with pytest.raises(RuntimeError):
                process_query("test query", provider="groq")


# ---------------------------------------------------------------------------
# 6. Config validation — missing API key
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_missing_groq_key_raises(self):
        from src.config import Settings
        with pytest.raises(ValidationError):
            Settings(model_provider="groq", groq_api_key="")


# ---------------------------------------------------------------------------
# 7. _extract_response edge cases
# ---------------------------------------------------------------------------

class TestExtractResponseEdgeCases:
    def setup_method(self):
        from src.graph import _extract_response
        self.extract = _extract_response

    def test_empty_dict(self):
        result = self.extract({})
        assert isinstance(result, str)

    def test_message_with_none_content(self):
        msg = MagicMock()
        msg.content = None
        result = self.extract({"messages": [msg]})
        assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# 8. _should_continue routing
# ---------------------------------------------------------------------------

class TestShouldContinue:
    def setup_method(self):
        from src.graph import _should_continue
        self.fn = _should_continue

    def test_ai_message_no_tool_calls_returns_end(self):
        state = {"messages": [AIMessage(content="done", tool_calls=[])]}
        assert self.fn(state) == END

    def test_ai_message_with_tool_calls_returns_tools(self):
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "python_analyst", "id": "1", "args": {}}],
        )
        state = {"messages": [msg]}
        assert self.fn(state) == "tools"

    def test_tool_message_prediction_complete_returns_end(self):
        msg = ToolMessage(content="PREDICTION COMPLETE: ...", tool_call_id="1")
        state = {"messages": [msg]}
        assert self.fn(state) == END

    def test_tool_message_no_prediction_returns_agent(self):
        msg = ToolMessage(content="You spent $360.11", tool_call_id="1")
        state = {"messages": [msg]}
        assert self.fn(state) == "agent"
