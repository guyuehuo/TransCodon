from typing import Dict, List, Tuple
import pandas as pd
from CAI import CAI, relative_adaptiveness
from tqdm import tqdm
import os

def get_GC_content(dna: str, lower: bool = False) -> float:
    """
    Calculate the GC content of a DNA sequence.

    Args:
        dna (str): The DNA sequence.
        lower (bool): If True, converts DNA sequence to lowercase before calculation.

    Returns:
        float: The GC content as a percentage.
    """
    if lower:
        dna = dna.lower()
    return (dna.count("G") + dna.count("C")) / len(dna) * 100


