"""
Tests for Fireworks AI Responses API transformation.

Validates FireworksAIResponsesAPIConfig: provider identity, endpoint URL,
auth, supported params, param passthrough, request/response transformation,
and ProviderConfigManager registration.

Source: litellm/llms/fireworks_ai/responses/transformation.py
Ref:    https://docs.fireworks.ai/api-reference/post-responses
"""

import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.fireworks_ai.responses.transformation import (
    FireworksAIResponsesAPIConfig,
)
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

MODEL = "fireworks_ai/accounts/fireworks/models/glm-5p2"


class TestFireworksResponsesTransformation:
    def test_provider_identity(self):
        config = FireworksAIResponsesAPIConfig()
        assert config.custom_llm_provider == LlmProviders.FIREWORKS_AI

    def test_get_complete_url_default(self):
        config = FireworksAIResponsesAPIConfig()
        assert (
            config.get_complete_url(None, {})
            == "https://api.fireworks.ai/inference/v1/responses"
        )

    def test_get_complete_url_custom_base(self):
        config = FireworksAIResponsesAPIConfig()
        assert (
            config.get_complete_url("https://custom.fireworks.ai/inference/v1", {})
            == "https://custom.fireworks.ai/inference/v1/responses"
        )

    def test_get_complete_url_strips_trailing_slash(self):
        config = FireworksAIResponsesAPIConfig()
        assert (
            config.get_complete_url("https://api.fireworks.ai/inference/v1/", {})
            == "https://api.fireworks.ai/inference/v1/responses"
        )

    def test_get_complete_url_env_override(self, monkeypatch):
        monkeypatch.setenv(
            "FIREWORKS_API_BASE", "https://env.fireworks.ai/inference/v1"
        )
        config = FireworksAIResponsesAPIConfig()
        assert (
            config.get_complete_url(None, {})
            == "https://env.fireworks.ai/inference/v1/responses"
        )

    def test_supported_openai_params(self):
        config = FireworksAIResponsesAPIConfig()
        supported = config.get_supported_openai_params(MODEL)
        expected = [
            "max_output_tokens",
            "max_tool_calls",
            "metadata",
            "parallel_tool_calls",
            "previous_response_id",
            "reasoning",
            "store",
            "stream",
            "temperature",
            "text",
            "tool_choice",
            "tools",
            "top_p",
            "truncation",
            "user",
            "instructions",
        ]
        for param in expected:
            assert param in supported, f"Missing supported param: {param}"

    def test_map_openai_params_passthrough(self):
        config = FireworksAIResponsesAPIConfig()
        params = ResponsesAPIOptionalRequestParams(
            temperature=0.7,
            top_p=0.9,
            max_output_tokens=1024,
            store=False,
            tool_choice="auto",
        )
        result = config.map_openai_params(
            response_api_optional_params=params, model=MODEL, drop_params=False
        )
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["max_output_tokens"] == 1024
        assert result["store"] is False
        assert result["tool_choice"] == "auto"

    def test_function_tool_passthrough(self):
        config = FireworksAIResponsesAPIConfig()
        params = ResponsesAPIOptionalRequestParams(
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather",
                        "parameters": {"type": "object"},
                    },
                }
            ]
        )
        result = config.map_openai_params(
            response_api_optional_params=params, model=MODEL, drop_params=False
        )
        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["function"]["name"] == "get_weather"

    def test_previous_response_id_passthrough(self):
        config = FireworksAIResponsesAPIConfig()
        params = ResponsesAPIOptionalRequestParams(previous_response_id="resp_abc")
        result = config.map_openai_params(
            response_api_optional_params=params, model=MODEL, drop_params=False
        )
        assert result["previous_response_id"] == "resp_abc"

    def test_validate_environment_sets_bearer(self):
        config = FireworksAIResponsesAPIConfig()
        from litellm.types.router import GenericLiteLLMParams

        headers = config.validate_environment(
            headers={},
            model=MODEL,
            litellm_params=GenericLiteLLMParams(api_key="fw-test-key"),
        )
        assert headers["Authorization"] == "Bearer fw-test-key"
        assert headers["Content-Type"] == "application/json"

    def test_validate_environment_missing_key_raises(self):
        config = FireworksAIResponsesAPIConfig()
        from litellm.types.router import GenericLiteLLMParams

        with pytest.raises(ValueError, match="FIREWORKS_API_KEY"):
            config.validate_environment(
                headers={}, model=MODEL, litellm_params=GenericLiteLLMParams()
            )

    def test_provider_config_registration(self):
        """ProviderConfigManager returns FireworksAIResponsesAPIConfig for fireworks_ai."""
        config = ProviderConfigManager.get_provider_responses_api_config(
            model=MODEL,
            provider=LlmProviders.FIREWORKS_AI,
        )
        assert config is not None
        assert isinstance(config, FireworksAIResponsesAPIConfig)
        assert config.custom_llm_provider == LlmProviders.FIREWORKS_AI

    def test_provider_config_registration_string_provider(self):
        """String provider name also resolves (path used by litellm.responses)."""
        config = ProviderConfigManager.get_provider_responses_api_config(
            model=MODEL,
            provider="fireworks_ai",
        )
        assert config is not None
        assert isinstance(config, FireworksAIResponsesAPIConfig)

    def test_litellm_attribute_export(self):
        import litellm

        assert litellm.FireworksAIResponsesAPIConfig is FireworksAIResponsesAPIConfig

    def test_supports_native_file_search_false(self):
        config = FireworksAIResponsesAPIConfig()
        assert config.supports_native_file_search() is False

    def test_supports_native_websocket_false(self):
        config = FireworksAIResponsesAPIConfig()
        assert config.supports_native_websocket() is False

    def test_transform_request_basic(self):
        config = FireworksAIResponsesAPIConfig()
        data = config.transform_responses_api_request(
            model=MODEL,
            input="Hello, world!",
            response_api_optional_request_params={"temperature": 0.5},
            litellm_params={},
            headers={},
        )
        assert data["model"] == MODEL
        assert data["input"] == "Hello, world!"
        assert data["temperature"] == 0.5

    def test_transform_response_success(self):
        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObj,
        )

        config = FireworksAIResponsesAPIConfig()
        body = {
            "id": "resp_123",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": MODEL,
            "output": [
                {
                    "type": "message",
                    "id": "msg_123",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "Hi!", "annotations": []}
                    ],
                }
            ],
            "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        }
        raw_response = httpx.Response(
            status_code=200,
            json=body,
            request=httpx.Request(
                "POST", "https://api.fireworks.ai/inference/v1/responses"
            ),
        )
        logging_obj = LiteLLMLoggingObj(
            model=MODEL,
            messages=[],
            stream=False,
            call_type="responses",
            start_time=None,
            litellm_call_id="test",
            function_id="test",
        )
        response = config.transform_response_api_response(
            model=MODEL, raw_response=raw_response, logging_obj=logging_obj
        )
        assert response.id == "resp_123"
        assert response.status == "completed"
