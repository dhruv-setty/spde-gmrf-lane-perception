"""Static footprint of the forward filter for a Cortex-M4F target.

Cross-compiles the portable filter and the generated model, then reads section sizes from the
object files and the stack frame of each function from the generated assembly. It measures
size only; it does not run the code and gives no cycle or latency figure.

Requires pyelftools and a Thumb cross compiler. The default is zig cc (clang); set CC_M4 to
use another, for example "arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfloat-abi=hard".

Usage (from firmware/):  python tools/footprint.py
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

from elftools.elf.elffile import ELFFile

HERE = Path(__file__).resolve().parents[1]
CC = shlex.split(os.environ.get("CC_M4", "zig cc -target thumb-freestanding-eabihf -mcpu=cortex_m4"))
FLAGS = ["-Os", "-std=c99", "-I", str(HERE / "include")]
SOURCES = ["src/hmm_filter.c", "src/hmm_fc_s8_ref.c", "src/hmm_exp_lut.c"]
MODEL_TU = ('#include "hmm_filter.h"\n#include "hmm_model_generated.h"\n'
            "const hmm_model_t *const hmm_model_ptr = &hmm_gen_model;\nhmm_state_t hmm_state;\n")


def sections(obj: Path) -> dict:
    out = {"text": 0, "rodata": 0, "data": 0, "bss": 0}
    with obj.open("rb") as fh:
        for s in ELFFile(fh).iter_sections():
            for k in out:
                if s.name.startswith("." + k) and s["sh_flags"] & 2:
                    out[k] += s["sh_size"]
    return out


def frames(asm: str) -> dict:
    """Bytes pushed plus bytes reserved by 'sub sp' at the head of each function."""
    res, fn = {}, None
    for line in asm.splitlines():
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):", line)
        if m and not m.group(1).startswith((".", "$")):
            fn = m.group(1)
            res.setdefault(fn, 0)
            continue
        if fn is None:
            continue
        m = re.search(r"\bpush(?:\.w)?\s*\{([^}]*)\}", line)
        if m:
            res[fn] += 4 * len(m.group(1).split(","))
        m = re.search(r"\bsub(?:\.w)?\s+sp,\s*(?:sp,\s*)?#(\d+)", line)
        if m:
            res[fn] += int(m.group(1))
    return res


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "model.c").write_text(MODEL_TU)
        units = [HERE / s for s in SOURCES] + [td / "model.c"]
        total = {"text": 0, "rodata": 0, "data": 0, "bss": 0}
        fr = {}
        print("| Unit | text | rodata | data | bss |\n|---|---|---|---|---|")
        for u in units:
            obj = td / (u.stem + ".o")
            subprocess.run(CC + FLAGS + ["-c", str(u), "-o", str(obj)], check=True, cwd=HERE)
            s = sections(obj)
            for k in total:
                total[k] += s[k]
            print(f"| {u.name} | {s['text']} | {s['rodata']} | {s['data']} | {s['bss']} |")
            if u.suffix == ".c" and u.parent.name == "src":
                asm = td / (u.stem + ".s")
                subprocess.run(CC + FLAGS + ["-S", str(u), "-o", str(asm)], check=True, cwd=HERE)
                fr.update(frames(asm.read_text()))
        print(f"| total | {total['text']} | {total['rodata']} | {total['data']} | {total['bss']} |")
        print("\n| Function | Frame bytes |\n|---|---|")
        for k, v in sorted(fr.items()):
            print(f"| {k} | {v} |")


if __name__ == "__main__":
    main()
