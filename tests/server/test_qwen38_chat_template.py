"""Unit tests for `qwen38_fixed.jinja` and `--chat-template` CLI / `load_tokenizer` override.

Verifies:
1. 100% byte-identity against stock `nvidia/Qwen3.8-Flash-Next-NVFP4` template across standard
   single-turn, multi-turn, vision, tool-call, `enable_thinking=False`, and `reasoning_effort`
   (`low`, `medium`, `xhigh`, and default `xhigh`).
2. `probe_effort_profile` sees `validates=True`, `supported={'low', 'medium', 'xhigh'}`, `default='xhigh'`.
3. `probe_inline_system` upgrades from `'fold'` (on stock template) to `'preserve'` (on fixed template).
4. All 5 targeted fixes: consecutive leading `system`/`developer` merge (`"\\n\\n"`), mid-turn
   `system`/`developer` preservation, `[system, assistant]` / compacted tool-loop without user query,
   tool-first conversation opening `<|im_start|>user`, and embedded `<think>...</think>` extraction
   without mangling prose mentioning `<think>`.
5. `--chat-template` argument validation and `load_tokenizer(..., chat_template=...)`.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import jinja2
import jinja2.ext
import pytest
from freetoken.server.args import parse_args
from freetoken.tokenizer.effort import probe_effort_profile
from freetoken.tokenizer.inline_system import probe_inline_system
from freetoken.utils.hf import _read_chat_template_override

FIXED_TEMPLATE_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python"
    / "freetoken"
    / "tokenizer"
    / "templates"
    / "qwen38_fixed.jinja"
)

STOCK_NVIDIA_TEMPLATE = """{%- set image_count = namespace(value=0) %}
{%- set video_count = namespace(value=0) %}
{%- macro render_content(content, do_vision_count, is_system_content=false) %}
    {%- if content is string %}
        {{- content }}
    {%- elif content is iterable and content is not mapping %}
        {%- for item in content %}
            {%- if 'image' in item or 'image_url' in item or item.type == 'image' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain images.') }}
                {%- endif %}
                {%- if do_vision_count %}
                    {%- set image_count.value = image_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}
                    {{- 'Picture ' ~ image_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
            {%- elif 'video' in item or item.type == 'video' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain videos.') }}
                {%- endif %}
                {%- if do_vision_count %}
                    {%- set video_count.value = video_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}
                    {{- 'Video ' ~ video_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|video_pad|><|vision_end|>' }}
            {%- elif 'text' in item %}
                {{- item.text }}
            {%- else %}
                {{- raise_exception('Unexpected item type in content.') }}
            {%- endif %}
        {%- endfor %}
    {%- elif content is none or content is undefined %}
        {{- '' }}
    {%- else %}
        {{- raise_exception('Unexpected content type.') }}
    {%- endif %}
{%- endmacro %}
{%- if not messages %}
    {{- raise_exception('No messages provided.') }}
{%- endif %}
{%- set reasoning_instructions = '' %}
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh (default), medium, and low.') }}
    {%- endif %}
    {%- if resolved_reasoning_effort == 'xhigh' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.' %}
    {%- elif resolved_reasoning_effort == 'low' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.' %}
    {%- endif %}
{%- endif %}
{%- if tools and tools is iterable and tools is not mapping %}
    {{- '<|im_start|>system\\n' }}
    {%- if reasoning_instructions %}
        {{- reasoning_instructions + '\\n\\n' }}
    {%- endif %}
    {{- "# Tools\\n\\nYou have access to the following functions:\\n\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>" }}
    {{- '\\n\\nIf you choose to call a function ONLY reply in the following format with NO suffix:\\n\\n<tool_call>\\n<function=example_function_name>\\n<parameter=example_parameter_1>\\nvalue_1\\n</parameter>\\n<parameter=example_parameter_2>\\nThis is the value for the second parameter\\nthat can span\\nmultiple lines\\n</parameter>\\n</function>\\n</tool_call>\\n\\n<IMPORTANT>\\nReminder:\\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\\n- Required parameters MUST be specified\\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\\n</IMPORTANT>' }}
    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {%- if content %}
            {{- '\\n\\n' + content }}
        {%- endif %}
    {%- endif %}
    {{- '<|im_end|>\\n' }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {%- if content %}
            {{- '<|im_start|>system\\n' + (reasoning_instructions + '\\n\\n' if reasoning_instructions else '')  + content + '<|im_end|>\\n' }}
        {%- elif reasoning_instructions %}
            {{- '<|im_start|>system\\n' + reasoning_instructions + '<|im_end|>\\n' }}
        {%- endif %}
    {%- elif reasoning_instructions %}
        {{- '<|im_start|>system\\n' + reasoning_instructions + '<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" %}
        {%- set content = render_content(message.content, false)|trim %}
        {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}
            {%- set ns.multi_step_tool = false %}
            {%- set ns.last_query_index = index %}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if ns.multi_step_tool %}
    {{- raise_exception('No user query found in messages.') }}
{%- endif %}
{%- for message in messages %}
    {%- set content = render_content(message.content, true)|trim %}
    {%- if message.role == "system" %}
        {%- if not loop.first %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- endif %}
        {%- set reasoning_content = reasoning_content|trim %}
        {%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- endif %}
        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
            {%- for tool_call in message.tool_calls %}
                {%- if tool_call.function is defined %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {%- if loop.first %}
                    {%- if content|trim %}
                        {{- '\\n\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}
                    {%- else %}
                        {{- '<tool_call>\\n<function=' + tool_call.name + '>\\n' }}
                    {%- endif %}
                {%- else %}
                    {{- '\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}
                {%- endif %}
                {%- if tool_call.arguments is defined and tool_call.arguments != '' %}
                    {%- for args_name, args_value in tool_call.arguments|items %}
                        {{- '<parameter=' + args_name + '>\\n' }}
                        {%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}
                        {{- args_value }}
                        {{- '\\n</parameter>\\n' }}
                    {%- endfor %}
                {%- endif %}
                {{- '</function>\\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.previtem and loop.previtem.role != "tool" %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- content }}
        {{- '\\n</tool_response>' }}
        {%- if not loop.last and loop.nextitem.role != "tool" %}
            {{- '<|im_end|>\\n' }}
        {%- elif loop.last %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- else %}
        {{- raise_exception('Unexpected message role.') }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- else %}
        {{- '<think>\\n' }}
    {%- endif %}
{%- endif %}
"""


def _compile_jinja(template_str: str):
    def _raise_exception(msg: str) -> str:
        raise jinja2.TemplateError(msg)

    env = jinja2.Environment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[jinja2.ext.loopcontrols],
    )
    env.filters["tojson"] = lambda x: json.dumps(x, ensure_ascii=False)
    env.globals["raise_exception"] = _raise_exception
    return env.from_string(template_str)


def _render(
    tpl: jinja2.Template,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    add_generation_prompt: bool = True,
    **kwargs: Any,
) -> str:
    return tpl.render(
        messages=messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )


class _SpecialTokenStubTokenizer:
    """Tokenizer stub distinguishing <|im_start|> (500) and <|im_end|> (501) as special tokens."""

    all_special_ids = (500, 501)

    def get_added_vocab(self) -> dict[str, int]:
        return {"<|im_start|>": 500, "<|im_end|>": 501}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        out: list[int] = []
        i = 0
        while i < len(text):
            if text.startswith("<|im_start|>", i):
                out.append(500)
                i += len("<|im_start|>")
            elif text.startswith("<|im_end|>", i):
                out.append(501)
                i += len("<|im_end|>")
            else:
                out.append(ord(text[i]))
                i += 1
        return out

    def decode(self, ids: list[int]) -> str:
        parts: list[str] = []
        for tid in ids:
            if tid == 500:
                parts.append("<|im_start|>")
            elif tid == 501:
                parts.append("<|im_end|>")
            else:
                parts.append(chr(tid))
        return "".join(parts)


def test_byte_identical_to_stock_across_standard_inputs():
    stock = _compile_jinja(STOCK_NVIDIA_TEMPLATE)
    fixed = _compile_jinja(FIXED_TEMPLATE_PATH.read_text(encoding="utf-8"))

    sample_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]
    conversations = [
        [{"role": "user", "content": "Hello!"}],
        [{"role": "system", "content": ""}, {"role": "user", "content": "Empty sys"}],
        [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Hi"}],
        [
            {"role": "system", "content": "Sys"},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is this?"},
                ],
            },
        ],
        [
            {"role": "user", "content": "Weather in Montreal?"},
            {
                "role": "assistant",
                "reasoning_content": "Let me check.",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Montreal"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": '{"temp": -5}'},
            {
                "role": "assistant",
                "reasoning_content": "Got temp.",
                "content": "It is -5C.",
            },
            {"role": "user", "content": "Thanks!"},
        ],
    ]
    for msgs in conversations:
        for tools in (None, sample_tools):
            for kw in (
                {},
                {"reasoning_effort": "low"},
                {"reasoning_effort": "medium"},
                {"reasoning_effort": "xhigh"},
                {"enable_thinking": False},
                {"preserve_thinking": False},
            ):
                assert _render(fixed, msgs, tools=tools, **kw) == _render(
                    stock, msgs, tools=tools, **kw
                )


def test_retains_xhigh_default_and_validates_effort_profile():
    fixed = _compile_jinja(FIXED_TEMPLATE_PATH.read_text(encoding="utf-8"))
    profile = probe_effort_profile(
        lambda ctk, tools: _render(
            fixed, [{"role": "user", "content": "ping"}], tools=tools, **ctk
        )
    )
    assert profile.validates is True
    assert profile.default == "xhigh"
    assert profile.supported == frozenset({"low", "medium", "xhigh"})


def test_upgrades_inline_system_probe_from_fold_to_preserve():
    stock = _compile_jinja(STOCK_NVIDIA_TEMPLATE)
    fixed = _compile_jinja(FIXED_TEMPLATE_PATH.read_text(encoding="utf-8"))
    tok = _SpecialTokenStubTokenizer()

    stock_mode, _ = probe_inline_system(lambda msgs: _render(stock, msgs), tok)
    fixed_mode, reason = probe_inline_system(lambda msgs: _render(fixed, msgs), tok)
    assert stock_mode == "fold"
    assert fixed_mode == "preserve", reason


def test_all_five_qwen38_template_fixes():
    fixed = _compile_jinja(FIXED_TEMPLATE_PATH.read_text(encoding="utf-8"))

    # 1. Consecutive leading system + developer messages merge with "\n\n"
    merged = _render(
        fixed,
        [
            {"role": "system", "content": "Alpha"},
            {"role": "developer", "content": "Beta"},
            {"role": "system", "content": "  "},
            {"role": "system", "content": "Gamma"},
            {"role": "user", "content": "Question"},
        ],
    )
    single = _render(
        fixed,
        [
            {"role": "system", "content": "Alpha\n\nBeta\n\nGamma"},
            {"role": "user", "content": "Question"},
        ],
    )
    assert merged == single

    # 2. Mid-turn system / developer message renders in place
    mid = _render(
        fixed,
        [
            {"role": "system", "content": "Head"},
            {"role": "user", "content": "Turn 1"},
            {"role": "system", "content": "Reminder"},
        ],
    )
    assert "<|im_start|>system\nReminder<|im_end|>\n" in mid

    # 3. [system, assistant] prefill without plain user query does not raise
    prefill = _render(
        fixed,
        [
            {"role": "system", "content": "Head"},
            {"role": "assistant", "content": "Partial:"},
        ],
        add_generation_prompt=False,
    )
    assert "Partial:" in prefill

    # 4. Tool-first truncated history (with or without leading system) opens <|im_start|>user
    tool_first = _render(
        fixed,
        [
            {"role": "tool", "content": "result 1"},
            {"role": "user", "content": "Next?"},
        ],
        enable_thinking=False,
    )
    assert tool_first.startswith(
        "<|im_start|>user\n<tool_response>\nresult 1\n</tool_response><|im_end|>\n"
    )
    sys_then_tool = _render(
        fixed,
        [
            {"role": "system", "content": "Sys"},
            {"role": "tool", "content": "result 1"},
            {"role": "tool", "content": "result 2"},
        ],
        enable_thinking=False,
    )
    assert (
        "<|im_start|>user\n<tool_response>\nresult 1\n</tool_response>\n<tool_response>\nresult 2\n</tool_response><|im_end|>\n"
        in sys_then_tool
    )

    # 5. Embedded <think>...</think> in assistant.content is extracted without double tags
    #    (even when SDK serializes reasoning_content=""), while <think> mentioned mid-sentence
    #    in prose is untouched; string tool arguments don't crash.
    deduped = _render(
        fixed,
        [
            {"role": "user", "content": "Q1"},
            {
                "role": "assistant",
                "reasoning_content": "",
                "content": "<think>\nmy thought\n</think>\n\nFinal answer",
            },
            {"role": "user", "content": "Q2"},
        ],
    )
    assert (
        "<|im_start|>assistant\n<think>\nmy thought\n</think>\n\nFinal answer<|im_end|>\n"
        in deduped
    )
    assert "<think>\n\n</think>\n\n<think>" not in deduped

    prose_with_tags = _render(
        fixed,
        [
            {"role": "user", "content": "Explain tags"},
            {"role": "assistant", "content": "Qwen uses `<think>...</think>` blocks."},
            {"role": "user", "content": "Thanks"},
        ],
    )
    assert "Qwen uses `<think>...</think>` blocks." in prose_with_tags

    str_tool_args = _render(
        fixed,
        [
            {"role": "user", "content": "Call f"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "f", "arguments": '{"raw": 1}'}}],
            },
            {"role": "tool", "content": "ok"},
        ],
    )
    assert '<parameter=arguments>\n{"raw": 1}\n</parameter>' in str_tool_args


def test_chat_template_cli_arg_and_loader(tmp_path):
    custom = tmp_path / "custom.jinja"
    custom.write_text("{{- messages[0].content }}", encoding="utf-8")
    assert _read_chat_template_override(str(custom)) == "{{- messages[0].content }}"

    args, _ = parse_args(
        [
            "--model",
            str(tmp_path),
            "--chat-template",
            str(custom),
            "--dtype",
            "bfloat16",
        ]
    )
    assert args.chat_template == str(custom.resolve())

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--model",
                str(tmp_path),
                "--chat-template",
                str(tmp_path / "missing.jinja"),
                "--dtype",
                "bfloat16",
            ]
        )
