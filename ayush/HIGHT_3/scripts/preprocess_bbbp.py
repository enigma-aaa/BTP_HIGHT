
import os
import pandas as pd
import pickle
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from llava.datasets.smiles2graph import smiles2graph, construct_instruct_question

def main():
    # Define paths
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'MoleculeNet_data', 'bbbp')
    raw_file = os.path.join(data_dir, 'raw', 'BBBP.csv')
    processed_dir = os.path.join(data_dir, 'processed')
    output_file = os.path.join(processed_dir, 'instruct-random-train.pkl')

    # Create processed directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)

    # Read the raw CSV file
    df = pd.read_csv(raw_file)

    # Randomly sample 1500 records
    if len(df) > 1500:
        df = df.sample(n=1500, random_state=42)

    # Process the data
    processed_data = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing BBBP data"):
        smiles = row['smiles']
        label = row['p_np']

        try:
            # Convert SMILES to graph
            graph = smiles2graph(smiles)

            # Construct question and answer
            question = construct_instruct_question()
            answer = str(label)

            # Create final data dictionary
            data_dict = {
                "graph": graph,
                "instruction": question,
                "SMILES": smiles,
                "label": label
            }

            processed_data.append(data_dict)
        except Exception as e:
            print(f"Error processing SMILES: {smiles}, error: {e}")

    # Save the processed data as a pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)

    print(f"Successfully processed and saved data to {output_file}")

if __name__ == '__main__':
    main()
