#!/usr/bin/env python3
"""Convenience entrypoint script for FireRedAudio Studio WebUI."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fireredaudio_mlx.webui.__main__ import main

if __name__ == "__main__":
    main()
