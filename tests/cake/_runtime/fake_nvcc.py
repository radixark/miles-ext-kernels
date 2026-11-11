"""A fake nvcc: ``--version`` reports the toolkit it was written for, and a compile writes its ``-o`` file.

Every compile appends its arguments to ``calls.jsonl`` next to the script (:func:`calls` reads them). A file ``fail``
next to it makes every compile fail with ``error: synthetic failure`` on stderr.
"""

import json
import sys
from pathlib import Path

SCRIPT = """#!@PYTHON@
import json
import sys
from pathlib import Path

if sys.argv[1:] == ["--version"]:
    print("nvcc: NVIDIA (R) Cuda compiler driver")
    print("Cuda compilation tools, release @MAJOR@.@MINOR@, V@FULL@")
    print("Build cuda_@MAJOR@.@MINOR@.r@MAJOR@.@MINOR@/compiler.0_0")
    sys.exit(0)
with open(Path(__file__).with_name("calls.jsonl"), "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
if Path(__file__).with_name("fail").exists():
    sys.stderr.write("error: synthetic failure\\n")
    sys.exit(2)
Path(sys.argv[sys.argv.index("-o") + 1]).write_bytes(b"synthetic library")
"""


def write(path: Path, full: str) -> Path:
    """The fake nvcc of toolkit ``full`` (``"13.4.92"``) at ``path``."""
    major, minor, _ = full.split(".")
    path.parent.mkdir(parents=True, exist_ok=True)
    script = SCRIPT.replace("@PYTHON@", sys.executable).replace("@FULL@", full)
    path.write_text(script.replace("@MAJOR@", major).replace("@MINOR@", minor))
    path.chmod(0o755)
    return path


def calls(nvcc: Path) -> list[list[str]]:
    """The arguments of every compile ``nvcc`` ran, in order."""
    log = nvcc.with_name("calls.jsonl")
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
