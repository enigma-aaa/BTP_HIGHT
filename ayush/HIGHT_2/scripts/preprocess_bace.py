import os
import pandas as pd
import pickle
from tqdm import tqdm
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from llava.datasets.smiles2graph import smiles2graph, construct_instruct_question

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"])
    args = parser.parse_args()

    # Define paths
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'MoleculeNet_data', 'bace')
    raw_file = os.path.join(data_dir, 'raw', 'bace.csv')
    processed_dir = os.path.join(data_dir, 'processed')
    output_file = os.path.join(processed_dir, f'instruct-random-{args.split}.pkl')

    # Create processed directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)

    # Read the raw CSV file
    df = pd.read_csv(raw_file)

    # Filter dataframe based on split
    df = df[df['Model'] == args.split.capitalize()]

    # Process the data
    processed_data = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing BACE {args.split} data"):
        smiles = row['mol']
        label = row['Class']

        try:
            # Convert SMILES to graph
            graph = smiles2graph(smiles)
            graph['num_part'] = [graph['num_nodes']]

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