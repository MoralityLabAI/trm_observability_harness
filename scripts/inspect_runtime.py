from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    report = {
        "nvidia_smi": None,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "repo_paths": {
            "prime_environments": str(Path(r"C:/projects/prime-environments").resolve()),
            "gptstoryworld": str(Path(r"C:/projects/GPTStoryworld").resolve()),
            "tesseract": str(Path(r"C:/projects/Tesseract").resolve()),
        },
        "candidate_model_dirs": [
            "C:/models/Qwopus-27B-EXL2",
            "C:/Qwopus/models",
            "D:/AI_Models/Qwopus",
            "D:/Download",
        ],
    }

    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free,memory.used,driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip()
            report["nvidia_smi"] = out
        except Exception as exc:  # noqa: BLE001
            report["nvidia_smi"] = f"error: {exc}"

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
