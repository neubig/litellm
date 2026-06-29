"""
Fireworks AI Responses API — OpenAI-compatible.

Fireworks exposes a Responses API at ``POST /v1/responses`` that mirrors the
OpenAI Responses API surface (``model``, ``input``, ``instructions``,
``previous_response_id``, ``reasoning``, ``tools``, ``tool_choice``,
``parallel_tool_calls``, ``max_output_tokens``, ``max_tool_calls``,
``temperature``, ``top_p``, ``text``, ``truncation``, ``store``, ``stream``,
``metadata``, ``user``). This config inherits the OpenAI transformation logic
and only overrides the provider identity, auth, endpoint URL, and the
supported-parameter list.

Ref: https://docs.fireworks.ai/api-reference/post-responses
     https://docs.fireworks.ai/guides/response-api
"""

from typing import Any, Dict, List, Optional, Union

import litellm
from litellm.llms.fireworks_ai.common_utils import FireworksAIMixin
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

# Default base URL for Fireworks inference, matching the chat-completions path
# in litellm/llms/fireworks_ai/chat/transformation.py. Override with
# FIREWORKS_API_BASE or the per-call api_base.
FIREWORKS_API_BASE = "https://api.fireworks.ai/inference/v1"


class FireworksAIResponsesAPIConfig(OpenAIResponsesAPIConfig, FireworksAIMixin):
    """Configuration for Fireworks AI's Responses API.

    Inherits the OpenAI Responses request/response/streaming transformations
    and Fireworks' shared auth key resolution (FireworksAIMixin).
    """

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.FIREWORKS_AI

    def get_supported_openai_params(self, model: str) -> list:
        """Supported parameters for Fireworks' /v1/responses endpoint.

        Ref: https://docs.fireworks.ai/api-reference/post-responses
        """
        return [
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

    def map_openai_params(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        """Fireworks accepts OpenAI Responses params directly; no remapping needed."""
        return dict(response_api_optional_params)

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: Optional[GenericLiteLLMParams],
    ) -> dict:
        """Attach the Fireworks bearer token.

        Reuses FireworksAIMixin._get_api_key so the same env-var precedence
        (FIREWORKS_API_KEY / FIREWORKS_AI_API_KEY / FIREWORKSAI_API_KEY /
        FIREWORKS_AI_TOKEN) used for chat completions applies here.
        """
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = self._get_api_key(litellm_params.api_key)
        if api_key is None:
            raise ValueError(
                "FIREWORKS_API_KEY is not set. Set it (or pass api_key) to use "
                "the Fireworks Responses API."
            )
        headers.setdefault("Content-Type", "application/json")
        headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def get_complete_url(self, api_base: Optional[str], litellm_params: dict) -> str:
        """Build the full Fireworks /responses URL.

        Precedence: per-call api_base > FIREWORKS_API_BASE env >
        https://api.fireworks.ai/inference/v1.
        """
        api_base = (
            api_base
            or get_secret_str("FIREWORKS_API_BASE")
            or FIREWORKS_API_BASE
        )
        api_base = api_base.rstrip("/")
        # The chat path already includes /v1 in the default base; if a caller
        # supplies a bare host (no /v1), keep /responses at the root to match
        # the documented endpoint (POST /v1/responses with server base
        # https://api.fireworks.ai/inference).
        return f"{api_base}/responses"

    def supports_native_file_search(self) -> bool:
        """Fireworks Responses API does not provide OpenAI file_search."""
        return False

    def supports_native_websocket(self) -> bool:
        return False
