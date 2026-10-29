"""The `hf` CLI, run through Python. Use this where Windows Smart App Control blocks hf.exe.

    python scripts/hf.py auth whoami
    python scripts/hf.py upload <user>/<space> . . --repo-type space
"""
from huggingface_hub.cli.hf import main

if __name__ == "__main__":
    main()
