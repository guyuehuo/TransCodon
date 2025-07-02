from typing import Dict, List, Tuple
import os
import pandas as pd
from CAI import CAI, relative_adaptiveness
from tqdm import tqdm
import pickle

def get_CSI_weights(sequences: List[str]) -> Dict[str, float]:
    """
    Calculate the Codon Similarity Index (CSI) weights for a list of DNA sequences.

    Args:
        sequences (List[str]): List of DNA sequences.

    Returns:
        dict: The CSI weights.
    """
    #reference_sequences = CAI.get_reference_sequences()  # This function gets the default reference sequences
    #return relative_adaptiveness(sequences=sequences, reference_sequences=reference_sequences)
    print("序列数量",len(sequences))
    return relative_adaptiveness(sequences=sequences)


def get_CSI_value(dna: str, weights: Dict[str, float]) -> float:
    """
    Calculate the Codon Similarity Index (CSI) for a DNA sequence.

    Args:
        dna (str): The DNA sequence.
        weights (dict): The CSI weights from get_CSI_weights.

    Returns:
        float: The CSI value.
    """
    return CAI(dna, weights)


def get_organism_to_CSI_weights(
    dataset: pd.DataFrame, organisms: List[str]
) -> Dict[str, dict]:
    """
    Calculate the Codon Similarity Index (CSI) weights for a list of organisms.

    Args:
        dataset (pd.DataFrame): Dataset containing organism and DNA sequence info.
        organisms (List[str]): List of organism names.

    Returns:
        Dict[str, dict]: A dictionary mapping each organism to its CSI weights.
    """
    organism2weights = {}

    # Iterate through each organism to calculate its CSI weights
    for organism in tqdm(organisms, desc="Calculating CSI Weights: ", unit="Organism"):
        print("organism",organism)
        organism_data = dataset.loc[dataset["organism"] == organism]
        #print("organism_data",organism_data)
        sequences = organism_data["dna"].to_list()
        #print("sequences",sequences)
        weights = get_CSI_weights(sequences)
        organism2weights[organism] = weights

    return organism2weights
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

def calculate_and_save_cai(dataset: pd.DataFrame, organisms: str, csi_weights: Dict[str, dict], output_file: str) -> None:
    """
    Calculate the CAI for each DNA sequence in the dataset and save the results to a CSV file.

    Args:
        dataset (pd.DataFrame): Dataset containing DNA sequence information.
        organisms (List[str]): List of organism names.
        csi_weights (Dict[str, dict]): The CSI weights for each organism.
        output_file (str): The path to save the output CSV file.
    """
    cai_values = []

    # Iterate through each row in the dataset
    for _, row in tqdm(dataset.iterrows(), total=dataset.shape[0], desc="Calculating CAI", unit="row"):
        organism = organisms
        #dna = row['natural_dna']
        dna = row['prediction_dna']
        
        # Get the CSI weights for the current organism
        if organism in csi_weights:
            weights = csi_weights[organism]
            cai_value = get_CSI_value(dna, weights)
            cai_values.append(cai_value)
        else:
            cai_values.append(None)  # If organism is not found, append None

    # Add the CAI values to the dataset
    dataset['CAI'] = cai_values

    # Save the updated dataset with CAI values to a new CSV file
    dataset.to_csv(output_file, index=False)
    print(f"CAI values saved to {output_file}")

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
