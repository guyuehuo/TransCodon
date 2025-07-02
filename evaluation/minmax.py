
from typing import Dict, List, Tuple
import pickle
import pandas as pd
from CAI import CAI, relative_adaptiveness
from tqdm import tqdm
import os

def get_min_max_percentage(
    dna: str,
    codon_frequencies: Dict[str, Tuple[List[str], List[float]]],
    window_size: int = 18,
) -> List[float]:
    """
    Calculate the %MinMax metric for a DNA sequence.

    Args:
        dna (str): The DNA sequence.
        codon_frequencies (Dict[str, Tuple[List[str], List[float]]]): Codon
            frequency distribution per amino acid.
        window_size (int): Size of the window to calculate %MinMax.

    Returns:
        List[float]: List of %MinMax values for the sequence.

    Credit: https://github.com/chowington/minmax
    """
    # Get a dictionary mapping each codon to its respective amino acid
    codon2amino = {
        codon: amino
        for amino, (codons, frequencies) in codon_frequencies.items()
        for codon in codons
    }

    min_max_values = []
    codons = [dna[i : i + 3] for i in range(0, len(dna), 3)]  # Split DNA into codons

    # Iterate through the DNA sequence using the specified window size
    for i in range(len(codons) - window_size + 1):
        codon_window = codons[i : i + window_size]  # Codons in the current window

        Actual = 0.0  # Average of the actual codon frequencies
        Max = 0.0  # Average of the min codon frequencies
        Min = 0.0  # Average of the max codon frequencies
        Avg = 0.0  # Average of the averages of all frequencies for each amino acid

        # Sum the frequencies for codons in the current window
        for codon in codon_window:
            aminoacid = codon2amino[codon]
            frequencies = codon_frequencies[aminoacid][1]
            codon_index = codon_frequencies[aminoacid][0].index(codon)
            codon_frequency = codon_frequencies[aminoacid][1][codon_index]

            Actual += codon_frequency
            Max += max(frequencies)
            Min += min(frequencies)
            Avg += sum(frequencies) / len(frequencies)

        # Divide by the window size to get the averages
        Actual = Actual / window_size
        Max = Max / window_size
        Min = Min / window_size
        Avg = Avg / window_size

        # Calculate %MinMax
        percentMax = ((Actual - Avg) / (Max - Avg)) * 100
        percentMin = ((Avg - Actual) / (Avg - Min)) * 100

        # Append the appropriate %MinMax value
        if percentMax >= 0:
            min_max_values.append(percentMax)
        else:
            min_max_values.append(-percentMin)

    # Populate the last floor(window_size / 2) entries of min_max_values with None
    for i in range(int(window_size / 2)):
        min_max_values.append(None)

    return min_max_values


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

def calculate_and_save_minmax(dataset: pd.DataFrame, organisms: str, minmax_weights: Dict[str, dict], output_file: str) -> None:
    """
    Calculate the CAI for each DNA sequence in the dataset and save the results to a CSV file.

    Args:
        dataset (pd.DataFrame): Dataset containing DNA sequence information.
        organisms (List[str]): List of organism names.
        csi_weights (Dict[str, dict]): The CSI weights for each organism.
        output_file (str): The path to save the output CSV file.
    """
    minmax_values = []

    # Iterate through each row in the dataset
    for _, row in tqdm(dataset.iterrows(), total=dataset.shape[0], desc="Calculating minmax", unit="row"):
        organism = organisms
        #dna = row['natural_dna']
        dna = row['prediction_dna']
        
        # Get the CSI weights for the current organism
        if organism in minmax_weights:
            weights = minmax_weights[organism]
            minmax_value = get_min_max_percentage(dna, weights)
            minmax_values.append(minmax_value)
        else:
            minmax_values.append(None)  # If organism is not found, append None

    # Add the CAI values to the dataset
    dataset['minmax'] = minmax_values

    # Save the updated dataset with CAI values to a new CSV file
    dataset.to_csv(output_file, index=False)
    print(f"minmax values saved to {output_file}")


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

