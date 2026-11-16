"""Atalho para rodar o chat da raiz do projeto: python chat.py --versao v1"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from chat import main  # noqa: E402  (src/chat.py)

if __name__ == "__main__":
    main()
