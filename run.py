"""Create the local environment, install dependencies, and start Streamlit."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import venv


BASE_DIR = Path(__file__).resolve().parent


def main():
    if sys.version_info[:2] != (3, 12):
        print("Please use Python 3.12 (64-bit) to run this project.", file=sys.stderr)
        return 1

    env_dir = BASE_DIR / ".venv"
    python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    requirements = BASE_DIR / "requirements.txt"
    marker = env_dir / ".requirements.sha256"
    signature = hashlib.sha256(requirements.read_bytes()).hexdigest()

    try:
        if not python.is_file():
            print("Creating .venv...", flush=True)
            venv.EnvBuilder(with_pip=True).create(env_dir)
            # An interrupted/removed environment must reinstall its dependencies.
            if marker.exists():
                marker.unlink()

        installed = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
        if installed != signature:
            print("Installing dependencies (the first run may take a while)...", flush=True)
            subprocess.run(
                [str(python), "-m", "pip", "install", "-r", str(requirements)],
                cwd=BASE_DIR,
                check=True,
            )
            subprocess.run([str(python), "-m", "pip", "check"], cwd=BASE_DIR, check=True)
            marker.write_text(signature, encoding="utf-8")

        print("Starting RAG Chatbot. Press Ctrl+C to stop.", flush=True)
        return subprocess.call(
            [str(python), "-m", "streamlit", "run", str(BASE_DIR / "streamlit_app.py"), *sys.argv[1:]],
            cwd=BASE_DIR,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Could not start the app: {error}", file=sys.stderr)
        print("Check your internet connection and Python 3.12 installation, then try again.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
