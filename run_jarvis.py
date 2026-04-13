#!/usr/bin/env python3
"""Root-level runner for J.A.R.V.I.S."""

from dotenv import load_dotenv

load_dotenv()  # Load ANTHROPIC_API_KEY from .env before anything else

from jarvis.main import main  # noqa: E402

if __name__ == "__main__":
    main()
