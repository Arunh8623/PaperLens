import json
import os

import pytest

from paperlens.store.manifest import (
    Manifest, ManifestError, encode, decode, read_manifest, write_manifest,
    MANIFEST_NAME, TMP_NAME,
)


def sample():
    return Manifest(3, 120, ("seg-000001", "seg-000002"), 0, 1)


def test_missing_manifest_means_new_index(tmp_path):
    m = read_manifest(tmp_path)
    assert m == Manifest() and m.generation == 0 and m.segments == ()


def test_write_read_roundtrip(tmp_path):
    write_manifest(tmp_path, sample())
    assert read_manifest(tmp_path) == sample()
    assert not (tmp_path / TMP_NAME).exists()  # tmp was renamed away


def test_next_bumps_generation():
    n = sample().next(last_flushed_seq=200, segments=["seg-000003"])
    assert n.generation == 4 and n.last_flushed_seq == 200 and n.segments == ("seg-000003",)


def test_overwrite_replaces_whole_file(tmp_path):
    write_manifest(tmp_path, sample())
    write_manifest(tmp_path, sample().next(last_flushed_seq=999))
    assert read_manifest(tmp_path).last_flushed_seq == 999


def test_every_single_byte_flip_is_detected_or_harmless():
    """Flip each byte: decode must either raise ManifestError or return the
    SAME manifest (e.g. a flip inside insignificant whitespace). It must never
    return a different manifest silently."""
    raw = encode(sample())
    for i in range(len(raw)):
        bad = bytearray(raw)
        bad[i] ^= 0x01
        try:
            got = decode(bytes(bad))
        except ManifestError:
            continue
        assert got == sample(), f"silent change at byte {i}"


def test_truncated_and_garbage_rejected():
    raw = encode(sample())
    for cut in range(0, len(raw) - 1):
        with pytest.raises(ManifestError):
            decode(raw[:cut])
    for junk in (b"", b"null", b"[]", b"{}", b'{"body":1,"crc32":2}', b"\xff\xfe"):
        with pytest.raises(ManifestError):
            decode(junk)


def test_wrong_schema_rejected():
    from paperlens.store import manifest as mf
    body = sample().body(); body["schema"] = 99
    import zlib
    crc = zlib.crc32(mf._canon(body)) & 0xFFFFFFFF
    with pytest.raises(ManifestError, match="schema"):
        decode(json.dumps({"body": body, "crc32": crc}).encode())


def test_corrupt_file_on_disk_raises(tmp_path):
    write_manifest(tmp_path, sample())
    p = tmp_path / MANIFEST_NAME
    p.write_bytes(p.read_bytes().replace(b"120", b"121"))
    with pytest.raises(ManifestError, match="checksum"):
        read_manifest(tmp_path)


def test_stale_tmp_is_ignored_and_overwritten(tmp_path):
    (tmp_path / TMP_NAME).write_bytes(b"half written garbage")
    assert read_manifest(tmp_path) == Manifest()      # tmp never read
    write_manifest(tmp_path, sample())
    assert read_manifest(tmp_path) == sample()


def test_crash_between_tmp_and_replace_keeps_old(tmp_path):
    """Simulate dying after step 2: new bytes only in MANIFEST.tmp."""
    write_manifest(tmp_path, sample())
    (tmp_path / TMP_NAME).write_bytes(encode(sample().next(last_flushed_seq=500)))
    assert read_manifest(tmp_path).last_flushed_seq == 120
