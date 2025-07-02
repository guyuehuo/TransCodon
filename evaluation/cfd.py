
from typing import Dict, List, Tuple
import pickle
import pandas as pd
from CAI import CAI, relative_adaptiveness
from tqdm import tqdm
import os

def get_cfd(
    dna: str,
    codon_frequencies: Dict[str, Tuple[List[str], List[float]]],
    threshold: float = 0.3,
) -> float:
    """
    Calculate the codon frequency distribution (CFD) metric for a DNA sequence.

    Args:
        dna (str): The DNA sequence.
        codon_frequencies (Dict[str, Tuple[List[str], List[float]]]): Codon
            frequency distribution per amino acid.
        threshold (float): Frequency threshold for counting rare codons.

    Returns:
        float: The CFD metric as a percentage.
    """
    # Get a dictionary mapping each codon to its normalized frequency
    codon2frequency = {
        codon: freq / max(frequencies)
        for amino, (codons, frequencies) in codon_frequencies.items()
        for codon, freq in zip(codons, frequencies)
    }
    #print("codon2frequency",codon2frequency)

    cfd = 0

    # Iterate through the DNA sequence in steps of 3 to process each codon
    for i in range(0, len(dna), 3):
        codon = dna[i : i + 3]
        codon_frequency = codon2frequency[codon]

        if codon_frequency < threshold:
            cfd += 1

    return cfd / (len(dna) / 3) * 100



def save_to_file(data: Dict[str, dict], filename: str) -> None:
    """
    Save the dictionary to a file using pickle.

    Args:
        data (Dict[str, dict]): The data to save.
        filename (str): The name of the file to save the data.
    """
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data saved to {filename}")

def calculate_and_save_cfd(dataset: pd.DataFrame, organisms: str, cfd_weights: Dict[str, dict], output_file: str) -> None:
    """
    Calculate the CAI for each DNA sequence in the dataset and save the results to a CSV file.

    Args:
        dataset (pd.DataFrame): Dataset containing DNA sequence information.
        organisms (List[str]): List of organism names.
        csi_weights (Dict[str, dict]): The CSI weights for each organism.
        output_file (str): The path to save the output CSV file.
    """
    cfd_values = []

    # Iterate through each row in the dataset
    for _, row in tqdm(dataset.iterrows(), total=dataset.shape[0], desc="Calculating cfd", unit="row"):
        organism = organisms
        #dna = row['natural_dna']
        dna = row['prediction_dna']
        
        # Get the CSI weights for the current organism
        if organism in cfd_weights:
            weights = cfd_weights[organism]
            cfd_value = get_cfd(dna, weights)
            cfd_values.append(cfd_value)
        else:
            cfd_values.append(None)  # If organism is not found, append None

    # Add the CAI values to the dataset
    dataset['cfd'] = cfd_values

    # Save the updated dataset with CAI values to a new CSV file
    dataset.to_csv(output_file, index=False)
    print(f"cfd values saved to {output_file}")


def load_from_file(filename: str):
    """
    Load a dictionary from a pickle file.

    Args:
        filename (str): The name of the pickle file.

    Returns:
        dict: The data loaded from the file.
    """
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data
