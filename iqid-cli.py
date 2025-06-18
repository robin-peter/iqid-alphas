#!/usr/bin/env python3
"""
IQID-Alphas CLI - Standalone Executable

Advanced batch processing command-line interface for the IQID-Alphas project.
This script provides a convenient way to run the CLI without needing to use
the module syntax.

Usage:
    ./iqid-cli.py process --data /path/to/data --config configs/simple.json
    ./iqid-cli.py discover --data /path/to/data
    ./iqid-cli.py config --create configs/my_config.json
"""

import sys
from pathlib import Path

# Add the package to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the CLI
from iqid_alphas.cli import main

if __name__ == '__main__':
    main()
