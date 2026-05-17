# -*- coding: utf-8 -*-
"""
Created on Fri May 15 22:53:03 2026

@author: dkuch
"""


import logging
import sys
from pathlib import Path

# Make src/ importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
print(str(Path(__file__).resolve().parents[1] / "src"))

import utils_manifold_clustering