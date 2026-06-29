## Relevant issues

<!-- e.g., "Fixes #000" -->

N/A — no existing issue. Context: the `fireworks_ai` provider had no
`ResponsesAPIConfig` registered, so `litellm.responses("fireworks_ai/...", ...)`
silently fell back to emulating the Responses API on top of `/chat/completions`
(`litellm/responses/main.py` `responses_api_provider_config is None` branch).

## Linear ticket

N/A

## Pre-Submission checklist

**Please complete all items before asking a LiteLLM maintainer to review your PR**

- [x] I have added meaningful tests
- [ ] My PR passes all CI/CD checks (e.g., lint, format, unit tests)
- [x] My PR's scope is as isolated as possible; it only solves 1 specific problem
- [ ] I have requested a Greptile review by commenting `@greptileai` and received a **Confidence Score of at least 4/5** before requesting a maintainer review

## Delays in PR merge?

If you're seeing a delay in your PR being merged, ping the LiteLLM Team on [Slack (#pr-review)](https://join.slack.com/t/litellmossslack/shared_invite/zt-3o7nkuyfr-p_kbNJj8taRfXGgQI1~YyA).

## Screenshots / Proof of Fix

Fireworks AI exposes an OpenAI-compatible Responses API at
`POST /v1/responses` ([docs](https://docs.fireworks.ai/api-reference/post-responses),
[guide](https://docs.fireworks.ai/guides/response-api)). Before this PR,
LiteLLM had no `ResponsesAPIConfig` for `fireworks_ai`, so
`litellm.responses(...)` routed to the chat-completions emulation fallback.
After this PR it routes to the native `/v1/responses` endpoint.

**Before / after — provider config resolution:**

```bash
$ uv run python -c "
from litellm.utils import ProviderConfigManager
from litellm import get_llm_provider
model='fireworks_ai/accounts/fireworks/models/glm-5p2'
_, provider, _, _ = get_llm_provider(model=model)
cfg = ProviderConfigManager.get_provider_responses_api_config(model=model, provider=provider)
print('config is None (falls back to chat):', cfg is None)
"
config is None (falls back to chat): False
```

(On `origin/litellm_internal_staging` without this PR, the same command prints
`config is None (falls back to chat): True`.)

**URL + supported params resolved by the new config:**

```bash
$ uv run python -c "
import litellm
from litellm.utils import ProviderConfigManager
from litellm.types.utils import LlmProviders
cfg = ProviderConfigManager.get_provider_responses_api_config(
    model='fireworks_ai/accounts/fireworks/models/glm-5p2',
    provider=LlmProviders.FIREWORKS_AI,
)
print('class:', type(cfg).__name__)
print('url  :', cfg.get_complete_url(None, {}))
print('params:', sorted(cfg.get_supported_openai_params('fireworks_ai/accounts/fireworks/models/glm-5p2')))
print('litellm.FireworksAIResponsesAPIConfig:', litellm.FireworksAIResponsesAPIConfig)
"
class: FireworksAIResponsesAPIConfig
url  : https://api.fireworks.ai/inference/v1/responses
params: ['instructions', 'max_output_tokens', 'max_tool_calls', 'metadata', 'parallel_tool_calls', 'previous_response_id', 'reasoning', 'store', 'stream', 'temperature', 'text', 'tool_choice', 'tools', 'top_p', 'truncation', 'user']
litellm.FireworksAIResponsesAPIConfig: <class 'litellm.llms.fireworks_ai.responses.transformation.FireworksAIResponsesAPIConfig'>
```

**Tests:**

```bash
$ uv run pytest tests/test_litellm/llms/fireworks_ai/responses/ -q
..................                                                       [100%]
18 passed in 0.18s

$ uv run pytest tests/test_litellm/llms/fireworks_ai/ -q
........................................................................ [ 88%]
.........                                                                [100%]
81 passed in 4.01s
```

**End-to-end (live Fireworks account, real $):**

```bash
$ uv run python -c "
import litellm, os
r = litellm.responses(
    model='fireworks_ai/accounts/fireworks/models/glm-5p2',
    input='Reply with the single word: pong',
    api_key=os.environ['FIREWORKS_API_KEY'],
    store=False,
)
print('id    :', r.id)
print('status:', r.status)
print('text  :', r.output_text)
print('usage :', r.usage)
"
```
<!-- TODO(graham): run the above against a live Fireworks account and paste the
     real output here before requesting review. The unit tests above do not
     exercise the network path. -->

## Type

<!-- Select the type of Pull Request -->
<!-- Keep only the necessary ones -->

🆕 New Feature

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
- **`litellm/utils.py`**: Register `litellm.FireworksAIResponsesAPIConfig()`
  in `ProviderConfigManager._get_python_responses_api_config` for
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

**Backward compatibility:** `litellm.completion("fireworks_ai/...", ...)` is
unchanged (chat completions still use `FireworksAIConfig`). Existing callers
of `litellm.responses("fireworks_ai/...", ...)` that relied on the
chat-completions emulation fallback will now use the native endpoint — a
strict superset of the OpenAI Responses schema, with no public API signature
changes.

**Notes for reviewers:**

- The Fireworks Responses API schema was taken from the official OpenAPI spec
  at https://docs.fireworks.ai/api-reference/post-responses.md (see the
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
