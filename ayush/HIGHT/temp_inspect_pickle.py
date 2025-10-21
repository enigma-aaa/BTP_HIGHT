import pickle

file_path = '/mnt/data1/gtoken/ayush/HIGHT/data/MoleculeNet_data/bace/processed/instruct-random-train.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Print the first few items to inspect their structure
for i, item in enumerate(data):
    print()
    if i >= 2: # Print only first 3 items
        break
