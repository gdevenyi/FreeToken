"""The coalesced-frame decoder (#466) must accept a message larger than msgpack's
Unpacker default buffer (100 MiB), which unpackb never limited."""

import msgpack

from freetoken.utils.mp import _CoalescedUnpacker


def test_a_message_over_100_mib_decodes():
    payload = {"pixels": b"x" * (101 * 1024 * 1024)}
    unpacker = _CoalescedUnpacker()
    unpacker.feed(msgpack.packb(payload))
    assert len(unpacker) == 1 and len(unpacker.take()["pixels"]) == len(payload["pixels"])
