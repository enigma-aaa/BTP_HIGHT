import pickle

file_path = '/mnt/hdd/gtoken/data1/ayush/HIGHT_3/data/MoleculeNet_data/bace/processed/instruct-random-train.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Print the first few items to inspect their structure
for i, item in enumerate(data):
    print()
    if i >= 2: # Print only first 3 items
        break
