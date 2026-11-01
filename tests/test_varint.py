import random

import pytest

from paperlens.text.varint import (
    VarintError, encode_uvarint, decode_uvarint, encode_postings, decode_postings,
)


@pytest.mark.parametrize("n,size", [(0, 1), (1, 1), (127, 1), (128, 2), (16383, 2),
                                    (16384, 3), (2**32 - 1, 5), (2**63, 10), (2**64 - 1, 10)])
def test_known_sizes_roundtrip(n, size):
    b = encode_uvarint(n)
    assert len(b) == size and decode_uvarint(b) == (n, size)


def test_known_bytes():
    assert encode_uvarint(300) == bytes([0xAC, 0x02])      # classic LEB128 example


def test_random_roundtrip_many():
    rng = random.Random(1)
    for _ in range(5000):
        n = rng.getrandbits(rng.randint(0, 64))
        assert decode_uvarint(encode_uvarint(n))[0] == n


def test_rejects_bad_input():
    with pytest.raises(VarintError):
        encode_uvarint(-1)
    with pytest.raises(VarintError):
        encode_uvarint(2**64)
    with pytest.raises(VarintError):
        decode_uvarint(b"")
    with pytest.raises(VarintError):
        decode_uvarint(b"\x80")                              # continuation, then end
    with pytest.raises(VarintError):
        decode_uvarint(b"\x80" * 11)                         # overlong
    with pytest.raises(VarintError):
        decode_uvarint(b"\xff" * 9 + b"\x7f")                # 70 bits: exceeds u64


def test_postings_roundtrip_random():
    rng = random.Random(2)
    for _ in range(300):
        docs = sorted(rng.sample(range(0, 100000), rng.randint(0, 200)))
        pairs = [(d, rng.randint(1, 500)) for d in docs]
        blob = encode_postings(pairs)
        got, pos = decode_postings(blob, 0, len(pairs))
        assert got == pairs and pos == len(blob)


def test_postings_decode_from_offset_and_back_to_back():
    a, b = [(0, 1), (5, 2)], [(2, 9)]
    blob = b"XX" + encode_postings(a) + encode_postings(b)
    got_a, p = decode_postings(blob, 2, 2)
    got_b, p2 = decode_postings(blob, p, 1)
    assert (got_a, got_b, p2) == (a, b, len(blob))


def test_postings_validation():
    for bad in ([(3, 1), (3, 1)], [(5, 1), (2, 1)], [(0, 0)], [(-1, 1)]):
        with pytest.raises(VarintError):
            encode_postings(bad)


def test_truncated_postings_always_detected():
    blob = encode_postings([(i * 7, i + 1) for i in range(50)])
    for cut in range(len(blob)):
        with pytest.raises(VarintError):
            decode_postings(blob[:cut], 0, 50)


def test_zero_gap_is_corrupt():
    with pytest.raises(VarintError):
        decode_postings(bytes([0, 1]), 0, 1)
