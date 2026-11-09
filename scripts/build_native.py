"""Build the optional C++ accelerator and copy it next to the Python package.

    python scripts/build_native.py            # Release build
    python scripts/build_native.py --clean    # delete build/native first

Needs: a C++17 compiler (Linux: g++ or clang; Windows: Visual Studio Build Tools with "Desktop development with C++"),
CMake, and pybind11:   python -m pip install pybind11 cmake
The project works without it (pure-Python fallback, with a warning). After building, check with:
    python -c "import paperlens.native as n; print(n.status())"
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    a = ap.parse_args()
    build = os.path.join(ROOT, "build", "native")
    out = os.path.join(ROOT, "build", "native-out")
    if a.clean:
        shutil.rmtree(build, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    try:
        import pybind11
        pb = pybind11.get_cmake_dir()
    except ImportError:
        sys.exit("pybind11 is missing: python -m pip install pybind11 cmake")
    cmake = shutil.which("cmake")
    if cmake is None:
        sys.exit("cmake not found: python -m pip install cmake   (or install CMake from cmake.org)")
    run([cmake, "-S", os.path.join(ROOT, "native"), "-B", build, f"-Dpybind11_DIR={pb}",
         f"-DPython_EXECUTABLE={sys.executable}", f"-DPython3_EXECUTABLE={sys.executable}",
         f"-DPYTHON_EXECUTABLE={sys.executable}",
         f"-DPAPERLENS_OUT_DIR={out}", "-DCMAKE_BUILD_TYPE=Release"])
    run([cmake, "--build", build, "--config", "Release"])
    built = glob.glob(os.path.join(out, "_native*"))
    if not built:
        sys.exit("build finished but no _native module was produced")
    dest = os.path.join(ROOT, "src", "paperlens")
    for old in glob.glob(os.path.join(dest, "_native*")):
        os.remove(old)
    for f in built:
        shutil.copy2(f, dest)
        print("installed", os.path.join(dest, os.path.basename(f)))
    print("check: python -c \"import paperlens.native as n; print(n.status())\"")


if __name__ == "__main__":
    main()
