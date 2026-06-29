# feat(fireworks_ai): add native Responses API support

## Summary

Fireworks AI exposes an OpenAI-compatible Responses API at `POST /v1/responses`
([docs](https://docs.fireworks.ai/api-reference/post-responses),
[guide](https://docs.fireworks.ai/guides/response-api)), but LiteLLM had no
`ResponsesAPIConfig` registered for the `fireworks_ai` provider. As a result,
`litellm.responses("fireworks_ai/...", ...)` fell through to the
`responses_api_provider_config is None` branch in
`litellm/responses/main.py`, which silently emulates the Responses API on top
of `/chat/completions` — adding translation overhead and forfeiting any
Responses-API-native benefits (conversational continuation via
`previous_response_id`, MCP tool calls, server-side `store`, SSE streaming).

This PR adds a `FireworksAIResponsesAPIConfig` that subclasses
`OpenAIResponsesAPIConfig` (Fireworks' Responses API is OpenAI-compatible) and
registers it in `ProviderConfigManager`, so `litellm.responses(...)` now routes
to the native Fireworks `/v1/responses` endpoint.

## Motivation

When an OpenHands software-agent-sdk profile uses a Fireworks model (e.g.
`fireworks_ai/accounts/fireworks/models/glm-5p2`) and the SDK's
`LLM.uses_responses_api()` gate is enabled, the SDK calls
`litellm.responses(...)`. Without a Fireworks `ResponsesAPIConfig`, LiteLLM
takes the fallback path (`litellm_completion_transformation_handler`), which
translates the Responses-shaped request into a `/chat/completions` call and
reshapes the response back. That path works but is pure overhead: no native
`previous_response_id`, no native MCP tool support, and double
request/response transformation. With this change, the native Responses path
is used.

## Changes

- **`litellm/llms/fireworks_ai/responses/transformation.py`** (new): Adds
  `FireworksAIResponsesAPIConfig(OpenAIResponsesAPIConfig, FireworksAIMixin)`.
  - `custom_llm_provider` → `LlmProviders.FIREWORKS_AI`
  - `get_complete_url` → `{api_base}/responses`, default
    `https://api.fireworks.ai/inference/v1/responses` (honors `api_base` arg,
    `FIREWORKS_API_BASE` env, trailing-slash stripping — matching the chat
    path's base URL logic).
  - `validate_environment` → reuses `FireworksAIMixin._get_api_key` so the
    same env-var precedence (`FIREWORKS_API_KEY` / `FIREWORKS_AI_API_KEY` /
    `FIREWORKSAI_API_KEY` / `FIREWORKS_AI_TOKEN`) used for chat completions
    applies; raises `ValueError` if no key is resolvable.
  - `get_supported_openai_params` → the Fireworks-supported subset per the
    OpenAPI spec: `max_output_tokens`, `max_tool_calls`, `metadata`,
    `parallel_tool_calls`, `previous_response_id`, `reasoning`, `store`,
    `stream`, `temperature`, `text`, `tool_choice`, `tools`, `top_p`,
    `truncation`, `user`, `instructions`.
  - `map_openai_params` → passthrough (Fireworks accepts OpenAI Responses
    params directly).
  - `supports_native_file_search()` / `supports_native_websocket()` → `False`
    (Fireworks Responses has no OpenAI file_search or native WebSocket).
- **`litellm/utils.py`**: Register
  `litellm.FireworksAIResponsesAPIConfig()` in
  `ProviderConfigManager._get_python_responses_api_config` for
  `LlmProviders.FIREWORKS_AI`, alongside the existing `PERPLEXITY`,
  `DATABRICKS`, `OPENROUTER`, etc. entries.
- **`litellm/__init__.py`** + **`litellm/_lazy_imports_registry.py`**: Export
  `FireworksAIResponsesAPIConfig` through the lazy-import registry (both the
  `LLM_CONFIG_NAMES` tuple and the `_LLM_CONFIGS_IMPORT_MAP`) so
  `litellm.FireworksAIResponsesAPIConfig` resolves.
- **`tests/test_litellm/llms/fireworks_ai/responses/test_fireworks_ai_responses_transformation.py`**
  (new): 18 tests covering provider identity, URL construction (default /
  custom / trailing-slash / env override), supported params, param
  passthrough, function-tool passthrough, `previous_response_id` passthrough,
  auth (bearer set + missing-key error), `ProviderConfigManager` registration
  (both enum and string provider), `litellm` attribute export, file-search /
  websocket flags, request transformation, and response transformation.

## How to use

```python
import litellm

resp = litellm.responses(
    model="fireworks_ai/accounts/fireworks/models/glm-5p2",
    input="What is the capital of France?",
    api_key="<FIREWORKS_API_KEY>",
    reasoning={"effort": "high"},
    store=False,
)
```

Previously this hit `/chat/completions` under the hood; now it hits
`https://api.fireworks.ai/inference/v1/responses` directly.

## Backward compatibility

- `litellm.completion("fireworks_ai/...", ...)` is unchanged — chat
  completions still use `FireworksAIConfig`.
- Existing callers of `litellm.responses("fireworks_ai/...", ...)` that
  relied on the chat-completions emulation fallback will now use the native
  endpoint. Behavior should be a strict superset (same OpenAI Responses
  schema in/out), but callers that depended on emulation quirks should
  verify. No public API signatures change.

## Test plan

- [x] `uv run pytest tests/test_litellm/llms/fireworks_ai/responses/ -q`
      → 18 passed
- [x] `uv run pytest tests/test_litellm/llms/fireworks_ai/ -q`
      → 81 passed (no regressions in chat/rerank)
- [ ] Manual: `litellm.responses(...)` against a live Fireworks account for a
      reasoning model (e.g. `glm-5p2`) with `stream=True` and a function tool.
- [ ] Manual: `previous_response_id` conversational continuation per the
      [Fireworks cookbook](https://github.com/fw-ai/cookbook/blob/main/learn/response-api/fireworks_previous_response_cookbook.ipynb).

## Notes for reviewers

- The Fireworks Responses API schema was taken from the official OpenAPI spec
  at https://docs.fireworks.ai/api-reference/post-responses.md (see
  `CreateResponse` properties). `instructions` is supported (unlike XAI), so
  no param is dropped.
- `should_fake_stream` is left at the base default (`False`) because Fireworks
  supports native SSE streaming (`stream: true`).
- The `tests/llm_translation/test_fireworks_ai_translation.py` suite has
  pre-existing errors on this machine (`fake OpenAI endpoint ... did not
  become healthy within 30.0s`) that reproduce on the base branch without
  this change; they are environment-related and not affected by this PR.

---

_Created by an AI agent (OpenHands) on behalf of Graham Neubig._
