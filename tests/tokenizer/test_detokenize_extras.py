"""Per-request detokenizer options: include_stop_str_in_output, skip_special_tokens, and
the reasoning token count reported on the finished reply."""

from __future__ import annotations

from freetoken.message import DetokenizeMsg
from freetoken.tokenizer.detokenize import DetokenizeManager

EOS, SPECIAL, THINK_END, THINK_START = 0, 99, 7, 8


class _Tok:
    """Token ids decode to letters; 99 is a special marker, 7 / 8 the reasoning end /
    start tags."""

    eos_token_id = EOS
    unk_token_id = -1

    def decode(self, ids, skip_special_tokens=False):
        out = []
        for t in ids:
            if t == SPECIAL:
                if not skip_special_tokens:
                    out.append("<s>")
            elif t == THINK_END:
                out.append("</think>")
            elif t == THINK_START:
                out.append("<think>")
            else:
                out.append(chr(ord("a") + t))
        return "".join(out)

    def batch_decode(self, ids_list, skip_special_tokens=False):
        return [self.decode(ids, skip_special_tokens) for ids in ids_list]


def _run(manager, uid, tokens, *, finished_kwargs=None, **msg_kwargs):
    text = ""
    meta = 0
    for i, t in enumerate(tokens):
        last = i == len(tokens) - 1
        kwargs = dict(msg_kwargs)
        if last and finished_kwargs:
            kwargs.update(finished_kwargs)
        strs, counts, _ = manager.detokenize_with_meta(
            [DetokenizeMsg(uid=uid, next_token=t, finished=last, **kwargs)]
        )
        text += strs[0]
        meta = counts[0]
    return text, meta


def test_stop_string_is_trimmed_by_default_and_kept_on_request():
    manager = DetokenizeManager(_Tok(), frozenset({EOS}))
    trimmed, _ = _run(
        manager, 1, [1, 2, 3], stop_strs=["cd"], finished_kwargs={"matched_stop": "cd"}
    )
    assert trimmed == "b"
    kept, _ = _run(
        manager,
        2,
        [1, 2, 3],
        stop_strs=["cd"],
        keep_stop_str=True,
        finished_kwargs={"matched_stop": "cd"},
    )
    assert kept == "bcd"


def test_skip_special_tokens_is_per_request():
    manager = DetokenizeManager(_Tok(), frozenset({EOS}))
    shown, _ = _run(manager, 1, [1, SPECIAL, 2, EOS])
    hidden, _ = _run(manager, 2, [1, SPECIAL, 2, EOS], skip_special_tokens=True)
    assert shown == "b<s>c"
    assert hidden == "bc"


def test_reasoning_tokens_count_up_to_the_end_tag():
    manager = DetokenizeManager(_Tok(), frozenset({EOS}), think_end_id=THINK_END)
    text, reasoning = _run(manager, 1, [1, 2, THINK_END, 3, 4, EOS])
    assert text == "bc</think>de"
    assert reasoning == 3  # b, c and the tag itself
    _, none = _run(manager, 2, [1, 2, EOS])
    assert none == 0


def test_reasoning_tokens_when_generation_stops_inside_the_block():
    # The prompt pre-opened <think> (qwen3, thinking on) and the output never closed it:
    # every completion token was reasoning, whether max_tokens or EOS ended it.
    manager = DetokenizeManager(
        _Tok(), frozenset({EOS}), think_end_id=THINK_END, think_start_id=THINK_START
    )
    _, by_length = _run(manager, 1, [1, 2, 3], starts_in_reasoning=True)
    assert by_length == 3
    _, by_eos = _run(manager, 2, [1, 2, EOS], starts_in_reasoning=True)
    assert by_eos == 3
    # the model opened the block itself
    _, self_opened = _run(manager, 3, [THINK_START, 1, 2])
    assert self_opened == 3
    # thinking off: no block, no reasoning
    _, off = _run(manager, 4, [1, 2, 3])
    assert off == 0
    # a closed block still counts up to the end tag
    _, closed = _run(manager, 5, [1, THINK_END, 2, 3], starts_in_reasoning=True)
    assert closed == 2


def test_a_batch_with_one_uid_twice_advances_its_state_in_order():
    # each message must see the decode state the previous one left, and the reasoning
    # counts must stay aligned with the texts
    manager = DetokenizeManager(_Tok(), frozenset({EOS}), think_end_id=THINK_END)
    msgs = [
        DetokenizeMsg(uid=1, next_token=1, finished=False),
        DetokenizeMsg(uid=1, next_token=THINK_END, finished=False),
        DetokenizeMsg(uid=2, next_token=3, finished=False),
        DetokenizeMsg(uid=1, next_token=2, finished=True),
    ]
    texts, counts, offsets = manager.detokenize_with_meta(msgs)
    assert texts == ["b", "</think>", "d", "c"]
    assert counts == [0, 0, 0, 2]
    assert offsets == [0, 1, 0, 9]
    assert manager.detokenize([DetokenizeMsg(uid=2, next_token=4, finished=True)]) == ["e"]


def test_text_offset_is_where_the_token_starts_despite_holdback():
    # "c" is held back as a possible start of the stop "cx", then sent with "d": its
    # offset is still where it starts, not how much text had been sent at its ack.
    manager = DetokenizeManager(_Tok(), frozenset({EOS}))
    sent, offsets = [], []
    for i, t in enumerate([1, 2, 3, EOS]):
        strs, _, offs = manager.detokenize_with_meta(
            [DetokenizeMsg(uid=1, next_token=t, finished=i == 3, stop_strs=["cx"])]
        )
        sent.append(strs[0])
        offsets.append(offs[0])
    assert sent == ["b", "", "cd", ""]
    assert offsets == [0, 1, 2, 3]


PARTIAL, COMPLETES = 50, 51


class _ByteTok(_Tok):
    """50 is the first half of a two-token character: alone it decodes to U+FFFD."""

    def decode(self, ids, skip_special_tokens=False):
        out = []
        for i, t in enumerate(ids):
            if t == PARTIAL:
                out.append("" if i + 1 < len(ids) and ids[i + 1] == COMPLETES else "�")
            elif t == COMPLETES:
                out.append("é" if i and ids[i - 1] == PARTIAL else "�")
            else:
                out.append(super().decode([t], skip_special_tokens))
        return "".join(out)


def _run_logprobs(manager, tokens, *, finished_kwargs=None, **msg_kwargs):
    """(text, entry token ids) per reply, each token's entry carrying its own id."""
    replies = []
    for i, t in enumerate(tokens):
        last = i == len(tokens) - 1
        kwargs = dict(msg_kwargs)
        if last and finished_kwargs:
            kwargs.update(finished_kwargs)
        texts, _, released = manager.detokenize_with_logprobs(
            [DetokenizeMsg(uid=1, next_token=t, finished=last, **kwargs)], [{"token_id": t}]
        )
        replies.append(
            (texts[0], [(e["token_id"], bool(e.get("stop_trimmed"))) for e in released[0]])
        )
    return replies


def test_logprob_entries_of_stop_trimmed_tokens_are_marked():
    manager = DetokenizeManager(_Tok(), frozenset({EOS}))
    assert _run_logprobs(
        manager, [1, 2, 3], stop_strs=["cd"], finished_kwargs={"matched_stop": "cd"}
    ) == [("b", [(1, False)]), ("", []), ("", [(2, True), (3, True)])]


def test_logprob_entry_of_a_token_a_stop_string_cuts_inside_carries_its_visible_length():
    # "b</think>c" with stop "k>c": "</thin" of the </think> token stays visible.
    manager = DetokenizeManager(_Tok(), frozenset({EOS}))
    final = None
    for i, t in enumerate([1, THINK_END, 2]):
        last = i == 2
        _, _, released = manager.detokenize_with_logprobs(
            [DetokenizeMsg(uid=1, next_token=t, finished=last, stop_strs=["k>c"],
                           matched_stop="k>c" if last else None)],
            [{"token_id": t}],
        )
        final = released[0]
    # each entry also carries where its token starts in the text ("b" is 1 char, "</think>" 8)
    assert final == [
        {"token_id": THINK_END, "text_offset": 1, "stop_visible": 6},
        {"token_id": 2, "text_offset": 9, "stop_trimmed": True},
    ]


def test_logprob_entry_waits_for_its_text_when_a_stop_prefix_is_released():
    # "b" is held as a prefix of "bc"; the second "b" releases the first, but its own
    # text is held, so its entry must not ride that reply -- the stop then trims it.
    manager = DetokenizeManager(_Tok(), frozenset({EOS}))
    assert _run_logprobs(
        manager, [1, 1, 2], stop_strs=["bc"], finished_kwargs={"matched_stop": "bc"}
    ) == [("", []), ("b", [(1, False)]), ("", [(1, True), (2, True)])]


def test_logprob_entries_kept_with_the_stop_string_or_without_a_match():
    manager = DetokenizeManager(_Tok(), frozenset({EOS}))
    kept = _run_logprobs(
        manager, [1, 2, 3], stop_strs=["cd"], keep_stop_str=True,
        finished_kwargs={"matched_stop": "cd"},
    )
    assert kept[-1] == ("cd", [(2, False), (3, False)])
    # finished by length while "c" was held as a stop prefix: nothing was trimmed
    by_length = _run_logprobs(manager, [1, 2], stop_strs=["cd"])
    assert by_length == [("b", [(1, False)]), ("c", [(2, False)])]
    # no stop strings: every entry rides its own reply, as before
    plain = _run_logprobs(manager, [1, 2, EOS], finished_kwargs={"finish_reason": "stop"})
    assert plain == [("b", [(1, False)]), ("c", [(2, False)]), ("", [(EOS, False)])]


def test_logprob_entry_of_a_partial_character_rides_the_reply_that_completes_it():
    manager = DetokenizeManager(_ByteTok(), frozenset({EOS}))
    assert _run_logprobs(manager, [1, PARTIAL, COMPLETES, 2], stop_strs=["zz"]) == [
        ("b", [(1, False)]),
        ("", []),
        ("é", [(PARTIAL, False), (COMPLETES, False)]),
        ("c", [(2, False)]),
    ]


def test_logprob_entries_released_together_keep_their_own_text_offsets():
    # "c" is held as a prefix of "cx" and released with "d": that reply carries both
    # entries, each stamped with where its own token starts, not one per-reply offset.
    manager = DetokenizeManager(_Tok(), frozenset({EOS}))
    replies = []
    for i, t in enumerate([1, 2, 3, EOS]):
        texts, _, released = manager.detokenize_with_logprobs(
            [DetokenizeMsg(uid=1, next_token=t, finished=i == 3, stop_strs=["cx"])],
            [{"token_id": t}],
        )
        replies.append((texts[0], [(e["token_id"], e["text_offset"]) for e in released[0]]))
    assert replies == [("b", [(1, 0)]), ("", []), ("cd", [(2, 1), (3, 2)]), ("", [(EOS, 3)])]
