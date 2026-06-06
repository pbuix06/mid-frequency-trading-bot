#!/usr/bin/env python
"""Entry point for the research/paper-trading automation CLI. NO live trading.

Examples:
    python scripts/research_cli.py status
    python scripts/research_cli.py registry
    python scripts/research_cli.py validate-data --days 30
    python scripts/research_cli.py paper-run --strategy crypto_momentum_24h --days 60
    python scripts/research_cli.py paper-all --days 60
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from mft.automation.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
