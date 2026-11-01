r"""Check this machine (written for Windows, runs anywhere) and write a report to send back.

    python scripts\windows_check.py

It does three things and writes ALL of it to windows-check-report.txt:
  1. PROBES: tiny experiments on how your OS/filesystem behaves in the exact
     situations our crash-safety code relies on (renaming a folder that has an
     open file, replacing an open file, deleting a memory-mapped file, locking,
     newline handling, clock tick). They only OBSERVE; nothing is judged here.
  2. TESTS: the full pytest suite, summarised (counts + every failing test name).
  3. ENVIRONMENT: Python, OS, CPU count, free disk space, git commit.

Why: the code was written and tested on Linux. Windows differs in exactly these
places, so what the probes print tells us which assumptions hold on YOUR machine.
Send windows-check-report.txt back (it contains no personal data beyond folder
names under the temp directory).
"""
from __future__ import annotations

import mmap
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORT = os.path.join(ROOT, "windows-check-report.txt")
LINES: list = []


def say(msg: str = "") -> None:
    print(msg)
    LINES.append(msg)


def attempt(label: str, fn) -> None:
    try:
        r = fn()
        say(f"  [ok]    {label}" + (f" -> {r}" if r is not None else ""))
    except BaseException as e:                        # report, never crash the check
        say(f"  [error] {label} -> {type(e).__name__}: {e}")


def probes(tmp: str) -> None:
    say("== 1. PROBES (how this OS behaves) ==")

    def dir_fsync():
        fd = os.open(tmp, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    attempt("open + fsync a DIRECTORY", dir_fsync)

    def rename_dir_with_open_file():
        a, b = os.path.join(tmp, "dirA"), os.path.join(tmp, "dirB")
        os.makedirs(a)
        with open(os.path.join(a, "f"), "wb") as f:
            f.write(b"x")
            os.rename(a, b)                           # file still open here
        return "renamed while a file inside was open"
    attempt("rename a folder while a file inside is open", rename_dir_with_open_file)

    def replace_over_open_file():
        dst, src = os.path.join(tmp, "dst"), os.path.join(tmp, "src")
        open(dst, "wb").write(b"old")
        open(src, "wb").write(b"new")
        with open(dst, "rb"):
            os.replace(src, dst)
        return "os.replace over a file that is open for reading worked"
    attempt("os.replace over a destination that is open", replace_over_open_file)

    def delete_mapped_file():
        p = os.path.join(tmp, "mapped")
        open(p, "wb").write(b"0123456789")
        f = open(p, "rb")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            os.remove(p)
            return "deleted a file while it was memory-mapped"
        finally:
            mm.close()
            f.close()
            if os.path.exists(p):
                os.remove(p)
    attempt("delete a file while it is memory-mapped", delete_mapped_file)

    def two_locks():
        p = os.path.join(tmp, "LOCK")
        fd1 = os.open(p, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0))
        fd2 = os.open(p, os.O_RDWR | getattr(os, "O_BINARY", 0))
        try:
            if os.name == "nt":
                import msvcrt
                os.lseek(fd1, 0, 0)
                msvcrt.locking(fd1, msvcrt.LK_NBLCK, 1)
                try:
                    os.lseek(fd2, 0, 0)
                    msvcrt.locking(fd2, msvcrt.LK_NBLCK, 1)
                    return "SECOND lock SUCCEEDED (this would break our single-writer lock!)"
                except OSError:
                    return "second lock refused, as intended"
                finally:
                    os.lseek(fd1, 0, 0)
                    msvcrt.locking(fd1, msvcrt.LK_UNLCK, 1)
            import fcntl
            fcntl.flock(fd1, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                fcntl.flock(fd2, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return "SECOND lock SUCCEEDED (this would break our single-writer lock!)"
            except OSError:
                return "second lock refused, as intended"
        finally:
            os.close(fd1)
            os.close(fd2)
    attempt("two lock attempts on the same file in one process", two_locks)

    def newline_bytes():
        p = os.path.join(tmp, "nl")
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | getattr(os, "O_BINARY", 0))
        os.write(fd, b"a\nb\n")
        os.close(fd)
        got = open(p, "rb").read()
        return "bytes preserved" if got == b"a\nb\n" else f"CORRUPTED: {got!r}"
    attempt("binary write keeps 0x0A as 0x0A (needs O_BINARY)", newline_bytes)

    def clock():
        ticks = set()
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.2:
            ticks.add(time.monotonic())
        ds = sorted(b - a for a, b in zip(sorted(ticks), sorted(ticks)[1:]))
        t = time.perf_counter()
        time.sleep(0.01)
        slept = (time.perf_counter() - t) * 1000
        return f"smallest monotonic step {ds[0] * 1000:.3f} ms; sleep(10 ms) took {slept:.1f} ms"
    attempt("clock tick / sleep accuracy", clock)

    def long_path():
        p = tmp
        while len(p) < 200:
            p = os.path.join(p, "d" * 40)
        os.makedirs(p)
        open(os.path.join(p, "f"), "wb").write(b"x")
        return f"created a path of {len(p)} characters"
    attempt("path longer than 200 characters", long_path)


def tests() -> None:
    say("")
    say("== 2. TESTS (full pytest suite) ==")
    junit = os.path.join(tempfile.gettempdir(), "paperlens-windows-junit.xml")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q",
                        f"--junitxml={junit}"], cwd=ROOT, capture_output=True, text=True)
    secs = time.time() - t0
    passed = failed = skipped = 0
    bad: list = []
    if os.path.exists(junit):
        for case in ET.parse(junit).getroot().iter("testcase"):
            name = f"{case.get('classname')}::{case.get('name')}"
            f = case.find("failure")
            e = case.find("error")
            if f is not None or e is not None:
                failed += 1
                node = f if f is not None else e
                bad.append((name, (node.get("message") or "")[:300]))
            elif case.find("skipped") is not None:
                skipped += 1
            else:
                passed += 1
    say(f"  passed {passed}, failed {failed}, skipped {skipped}, in {secs:.0f} s (pytest exit code {p.returncode})")
    for name, msg in bad:
        say(f"  FAILED {name}\n         {msg}")
    if not os.path.exists(junit):
        say("  pytest did not produce a result file. Output tail:")
        say("\n".join("    " + ln for ln in (p.stdout + p.stderr).splitlines()[-25:]))
    elif failed:
        say("  --- last lines of pytest output ---")
        say("\n".join("    " + ln for ln in p.stdout.splitlines()[-30:]))


def environment() -> None:
    say("")
    say("== 3. ENVIRONMENT ==")
    say(f"  python    {platform.python_version()} ({platform.python_implementation()}) at {sys.executable}")
    say(f"  platform  {platform.platform()}")
    say(f"  cpus      {os.cpu_count()}")
    try:
        du = shutil.disk_usage(tempfile.gettempdir())
        say(f"  temp disk {du.free / 1e9:.1f} GB free of {du.total / 1e9:.1f} GB")
    except OSError as e:
        say(f"  temp disk unknown ({e})")
    try:
        g = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        say(f"  git       {g.stdout.strip() or 'not a git checkout'}")
    except OSError:
        say("  git       not installed")


def main() -> int:
    say(f"PaperLens environment check - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    tmp = tempfile.mkdtemp(prefix="paperlens-probe-")
    try:
        probes(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    tests()
    environment()
    with open(REPORT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nReport written to: {REPORT}\nSend that file back.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
