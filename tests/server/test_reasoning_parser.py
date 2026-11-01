"""Character-boundary contracts for every reasoning detector and template mode."""
import random
from types import SimpleNamespace

import pytest

from freetoken.server.generation import GenSpec, _make_reasoning_parser
from freetoken.server.reasoning_parser import BaseReasoningParser, ReasoningParser


def _qwen_call(name, value="rm -rf build"):
    return (
        f"<tool_call>\n<function={name}>\n<parameter=command>\n{value}\n</parameter>\n"
        "</function>\n</tool_call>"
    )


_PAD = ("More reasoning about the user request that goes on for a while. " * 12).strip()
_BASH = _qwen_call("bash")
_EXAMPLE = _qwen_call("example_function_name", "value_1")
_LS = _qwen_call("bash", "ls")
_WEATHER = _qwen_call("get_weather", "Paris")
_OPEN_QUOTE = "<tool_call><function=bash><parameter=command>rm -rf /\nOK instead"
_TWO_PARAM_OPEN_QUOTE = (
    "<tool_call><function=bash><parameter=command>rm</parameter><parameter=x>y\nInstead"
)
QWEN_TOOL_MARKER_IDS = [
    ("A-prose-quote-far-close",
     f"I must wrap the call in a <tool_call> block.\n{_PAD}\n</think>\n\nAnswer.", ["bash"],
     f"I must wrap the call in a <tool_call> block.\n{_PAD}", "Answer."),
    ("B-prose-quote-near-close",
     "I must wrap the call in a <tool_call> block.\nOK.\n</think>\n\nAnswer.", ["bash"],
     "I must wrap the call in a <tool_call> block.\nOK.", "Answer."),
    ("C-real-looking-quote-far-close",
     f"The format is:\n{_BASH}\nNot yet.\n{_PAD}\n</think>\n\nI will ask.", ["bash"],
     f"The format is:\n{_BASH}\nNot yet.\n{_PAD}", "I will ask."),
    ("C2-quote-right-before-close",
     f"The format is:\n{_BASH}\n</think>\n\nI will ask.", ["bash"],
     f"The format is:\n{_BASH}", "I will ask."),
    ("C3-two-quotes-far-close",
     f"Format:\n{_BASH}\n{_BASH}\n{_PAD}\n</think>\nAsk.", ["bash"],
     f"Format:\n{_BASH}\n{_BASH}\n{_PAD}", "Ask."),
    ("C4-close-inside-the-arguments",
     f"Edit.\n{_qwen_call('bash', '</think>')}", ["bash"],
     "Edit.\n<tool_call>\n<function=bash>\n<parameter=command>",
     "</parameter>\n</function>\n</tool_call>"),
    ("D-omitted-close-real-call", f"I will list files.\n{_BASH}", ["bash"],
     "I will list files.", _BASH),
    ("D2-no-tool-list", f"I will list files.\n{_BASH}", None, f"I will list files.\n{_BASH}", ""),
    ("D2b-empty-tool-list", f"I will list files.\n{_BASH}", [], f"I will list files.\n{_BASH}", ""),
    ("D3-parallel-calls", f"Two.\n{_BASH}\n{_BASH}\n", ["bash"], "Two.", f"{_BASH}\n{_BASH}\n"),
    ("D4-closer-dropped", "Go.\n<tool_call><function=bash></function>", ["bash"],
     "Go.", "<tool_call><function=bash></function>"),
    ("D5-closer-text-in-arguments",
     f"Go.\n{_qwen_call('bash', 'echo </tool_call> done')}", ["bash"],
     f"Go.\n{_qwen_call('bash', 'echo </tool_call> done')}", ""),
    ("D6-quote-then-text-then-real-call", f"Like:\n{_BASH}\nSo.\n{_BASH}", ["bash"],
     f"Like:\n{_BASH}\nSo.", _BASH),
    ("E-template-example-unknown-name", f"Format:\n{_EXAMPLE}\n{_PAD}\n</think>\nAsk.", ["bash"],
     f"Format:\n{_EXAMPLE}\n{_PAD}", "Ask."),
    ("E2-unknown-name-at-eos", f"Format:\n{_EXAMPLE}", ["bash"], f"Format:\n{_EXAMPLE}", ""),
    ("F-text-after-the-call-no-close", f"Go.\n{_BASH}\nDone.", ["bash"], f"Go.\n{_BASH}\nDone.", ""),
    ("G-cut-inside-the-call", "Go.\n<tool_call>\n<function=bash>\n<parameter=command>\nrm",
     ["bash"], "Go.\n<tool_call>\n<function=bash>\n<parameter=command>\nrm", ""),
    ("H-marker-then-prose", "a <tool_call> b <tool_call>", ["bash"], "a <tool_call> b <tool_call>", ""),
    ("I-marker-then-json", 'a <tool_call>\n{"name": "bash"}\n</tool_call>', ["bash"],
     'a <tool_call>\n{"name": "bash"}\n</tool_call>', ""),
    ("J-open-value-swallows-real-call", f"Never:\n{_OPEN_QUOTE}\n{_LS}", ["bash"],
     f"Never:\n{_OPEN_QUOTE}", _LS),
    ("J2-open-value-other-real-call", f"Never:\n{_OPEN_QUOTE}\n{_WEATHER}", ["bash", "get_weather"],
     f"Never:\n{_OPEN_QUOTE}", _WEATHER),
    ("J3-second-param-open", f"Never:\n{_TWO_PARAM_OPEN_QUOTE}\n{_LS}", ["bash"],
     f"Never:\n{_TWO_PARAM_OPEN_QUOTE}", _LS),
    ("J4-open-value-runs-into-param-tag",
     "N:\n<tool_call><function=bash><parameter=command>rm -rf /\n<parameter=command>ls</parameter></function>",
     ["bash"],
     "N:\n<tool_call><function=bash><parameter=command>rm -rf /\n<parameter=command>ls</parameter></function>",
     ""),
    ("K-param-tag-without-gt",
     "F:\n<tool_call>\n<function=bash>\n<parameter=command</parameter>\n</function>\n</tool_call>",
     ["bash"],
     "F:\n<tool_call>\n<function=bash>\n<parameter=command</parameter>\n</function>\n</tool_call>",
     ""),
    ("L-one-unknown-name-in-a-run", f"Go.\n{_BASH}\n{_EXAMPLE}", ["bash"], f"Go.\n{_BASH}\n{_EXAMPLE}", ""),
    ("M-function-tag-in-value", f"Go.\n{_qwen_call('bash', '<function=x>')}", ["bash"],
     f"Go.\n{_qwen_call('bash', '<function=x>')}", ""),
]
QWEN_TOOL_MARKER_CASES = [case[1:] for case in QWEN_TOOL_MARKER_IDS]


def _chunkings(text):
    rng = random.Random(7)
    chunkings = [[text], list(text)]
    for _ in range(20):
        parts, i = [], 0
        while i < len(text):
            n = rng.randint(1, 80)
            parts.append(text[i:i + n])
            i += n
        chunkings.append(parts)
    return chunkings


class _ReasoningContract:
    name: str
    opener: str
    closer: str
    content_opener = ""

    def parser(self):
        return ReasoningParser(self.name, force_reasoning=False)

    def stream(self, text):
        parser = self.parser()
        parts = [parser.parse_stream_chunk(char) for char in text]
        parts.append(parser.flush())
        assert parser.flush() == ("", "")
        return tuple("".join(pair[i] for pair in parts) for i in (0, 1))

    def test_complete_reasoning(self):
        text = self.opener + "consider" + self.closer + self.content_opener + "answer"
        assert self.parser().parse_non_stream(text) == ("consider", "answer")

    def test_every_character_boundary(self):
        text = self.opener + "consider" + self.closer + self.content_opener + "answer"
        assert self.stream(text) == ("consider", "answer")

    def test_truncated_reasoning_flush(self):
        # Reasoning detectors expose flush(), not recover_truncated_call(). EOS
        # inside an open reasoning body must retain it in the reasoning channel.
        assert self.stream(self.opener + "partial thought") == ("partial thought", "")


class TestDeepSeekV32ReasoningParser(_ReasoningContract):
    name = "deepseekv32"
    opener = "<think>"
    closer = "</think>"


class TestThinkReasoningParser(_ReasoningContract):
    name = "glm"
    opener = "<think>"
    closer = "</think>"


class TestQwen3ReasoningParser(_ReasoningContract):
    name = "qwen3"
    opener = "<think>"
    closer = "</think>"

    def test_markers_after_the_first_close_are_literal(self):
        text = "plan</think>\n\nSTART = \"<think>\"\nEND = \"</think>\"\n"
        expected = ("plan", 'START = "<think>"\nEND = "</think>"')
        parser = ReasoningParser(self.name, force_reasoning=True)
        assert parser.parse_non_stream(text) == expected
        parser = ReasoningParser(self.name, force_reasoning=True)
        parts = [parser.parse_stream_chunk(char) for char in text] + [parser.flush()]
        assert tuple("".join(pair[i] for pair in parts) for i in (0, 1)) == expected

    @pytest.mark.parametrize(
        ("text", "tools", "reasoning", "content"),
        QWEN_TOOL_MARKER_CASES,
        ids=[case[0] for case in QWEN_TOOL_MARKER_IDS],
    )
    def test_tool_marker_in_reasoning(self, text, tools, reasoning, content):
        def parser():
            return ReasoningParser(self.name, force_reasoning=True, tool_names=tools)

        expected = (reasoning, content)
        assert parser().parse_non_stream(text, "stop") == expected
        for chunks in _chunkings(text):
            stream = parser()
            out = [stream.parse_stream_chunk(chunk) for chunk in chunks] + [stream.flush("stop")]
            assert tuple("".join(pair[i] for pair in out) for i in (0, 1)) == expected

    @pytest.mark.parametrize("finish_reason", ["length", None])
    def test_tool_run_needs_a_normal_stop(self, finish_reason):
        text = f"The format is:\n{_BASH}"
        expected = (text, "")
        parser = ReasoningParser(self.name, force_reasoning=True, tool_names=["bash"])
        assert parser.parse_non_stream(text, finish_reason) == expected
        for chunks in _chunkings(text):
            stream = ReasoningParser(self.name, force_reasoning=True, tool_names=["bash"])
            out = [stream.parse_stream_chunk(c) for c in chunks] + [stream.flush(finish_reason)]
            assert tuple("".join(pair[i] for pair in out) for i in (0, 1)) == expected

    @pytest.mark.parametrize(
        ("tools", "held", "breaker"),
        [
            (None, "", "<tool_call>"),
            (["bash"], "<tool_call>\n", "<function=other>"),
            (["bash"], "<tool_call>\n<function=bash>\n<parameter=command>\nrm", "\n<tool_call>x"),
            (["bash"], "<tool_call>\n<function=bash>\n<parameter=command>\nrm", "\n<function=x"),
            (["bash"], "<tool_call>\n<function=bash>\n", "<parameter=a b"),
            (["bash"], _BASH + "\n", "Not yet."),
        ],
        ids=["no-tools", "unknown-name", "tool-call-in-value", "function-in-value", "bad-param-tag",
             "text-after-run"],
    )
    def test_broken_run_goes_to_reasoning_when_seen(self, tools, held, breaker):
        parser = ReasoningParser(self.name, force_reasoning=True, tool_names=tools)
        assert parser.parse_stream_chunk("plan\n" + held) == ("plan", "")
        assert parser.parse_stream_chunk(breaker) == ("\n" + held + breaker, "")

    def test_long_real_call_is_not_released_before_the_generation_ends(self):
        # Past the old 512-char hold: a quote this long must still be reclaimable.
        call = _qwen_call("bash", "x" * 4096)
        parser = ReasoningParser(self.name, force_reasoning=True, tool_names=["bash"])
        assert parser.parse_stream_chunk("plan\n" + call) == ("plan", "")
        assert parser.flush("stop") == ("", call)


class TestMiniMaxM3ReasoningParser(_ReasoningContract):
    name = "minimax_m3"
    opener = "<mm:think>"
    closer = "</mm:think>"


class TestGemmaThoughtReasoningParser(_ReasoningContract):
    name = "gemma4"
    opener = "<|channel>thought\n"
    closer = "<channel|>"


class TestGptOssHarmonyReasoningParser(_ReasoningContract):
    name = "gpt_oss"
    opener = "<|channel|>analysis<|message|>"
    closer = "<|end|>"
    content_opener = "<|start|>assistant<|channel|>final<|message|>"


class TestMuseGlimmerReasoningParser(_ReasoningContract):
    name = "muse_glimmer"
    opener = "<|start|>assistant to=self<|message|>"
    closer = "<|eom|>"
    content_opener = "<|start|>assistant to=user<|message|>"


def test_contract_classes_cover_every_reasoning_detector():
    covered = {type(cls().parser().detector) for cls in _ReasoningContract.__subclasses__()}
    detectors = set()
    pending = [BaseReasoningParser]
    while pending:
        for sub in pending.pop().__subclasses__():
            detectors.add(sub)
            pending.append(sub)
    assert covered == detectors
    assert covered == set(ReasoningParser.ReasoningParserEnum.values())


@pytest.mark.parametrize("tools", [None, [{"type": "function", "function": {"name": "read"}}]])
@pytest.mark.parametrize(("name", "kwargs", "forced"), [
    ("qwen3", {}, True),
    ("qwen3", {"enable_thinking": True}, True),
    ("qwen3", {"enable_thinking": False}, False),
    ("glm", {}, True),
    ("glm", {"enable_thinking": True}, True),
    ("glm", {"enable_thinking": False}, False),
    ("gemma4", {}, False),
    ("gemma4", {"thinking_mode": "thinking"}, True),
    ("gemma4", {"enable_thinking": True}, True),
    ("gemma4", {"thinking": True}, True),
    ("gemma4", {"thinking_mode": "chat", "enable_thinking": False}, False),
    ("minimax_m3", {}, False),
    ("minimax_m3", {"thinking_mode": "adaptive"}, False),
    ("minimax_m3", {"thinking_mode": "disabled"}, False),
    ("minimax_m3", {"thinking_mode": "enabled"}, True),
])
def test_generation_force_reasoning_matches_template(name, kwargs, forced, tools):
    spec = GenSpec(messages=[], sampling_params=None, chat_template_kwargs=kwargs,
                   template_tools=tools, parser_tools=tools)
    parser = _make_reasoning_parser(spec, SimpleNamespace(config=SimpleNamespace(reasoning_parser=name)))
    # Observe routing, not an internal flag: tools must not turn a disabled
    # template's ordinary response into invisible reasoning.
    assert parser.parse_non_stream("probe") == (("probe", "") if forced else ("", "probe"))
