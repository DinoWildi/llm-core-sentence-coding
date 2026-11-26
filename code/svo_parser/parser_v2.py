import pandas as pd
import numpy as np
import ollama
import re
import json
import argparse

from src.io import read_tabular
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Literal

def parse_args():
    parser = argparse.ArgumentParser(
        description="Rewrite sentences in a structured SVO list."
    )

    # Main arguments: input, model, output, local/cluster
    parser.add_argument("-i", "--input", type = str, required = True,
                        help = "Path to input. Input file needs to be .csv")
    parser.add_argument("-m", "--model", type = str, required = True,
                        help = "Model name following Ollama naming standards (see www.ollama.com/models)")
    parser.add_argument("-o", "--output", type = Path, default = None,
                        help = "Define an output path. Defaults to svo_<modelname>.json.")
    parser.add_argument("-l", "--local", action = "store_true",
                        help = "Script defaults to HPC behaviour. Use -l for local inference.")

    # Additional model options that are usually not changed
    parser.add_argument("-t", "--thinking", action = "store_true",
                            help = "Turns on thinking. Disabled by default for SVO parsing.")
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--temperature", type = float, default = 0)
    parser.add_argument("--max-sent", type = int, default = None,
                        help = "Process only the first N sentences for testing")

    return parser.parse_args()