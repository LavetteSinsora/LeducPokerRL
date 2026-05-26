"""Start the PokerRL dashboard server."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dashboard.server import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PokerRL server")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    args = parser.parse_args()
    run(port=args.port)
