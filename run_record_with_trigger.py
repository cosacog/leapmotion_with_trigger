#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Launcher script for record_with_trigger.py

This allows running the recording application from the project root directory.
The actual implementation is in src/record_with_trigger.py
"""

import sys
import os

# Add src directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

# Change working directory to project root (for data directory)
os.chdir(project_root)

# Run the main module
if __name__ == "__main__":
    # Import after path is set
    import src.record_with_trigger as main_module

    # The main module has its own __main__ block that will execute
    # when imported, but we need to execute it explicitly
    exec(open(os.path.join(src_dir, 'record_with_trigger.py')).read())
