
import json
import os

def create_test_set(file_path, save_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    test_data = [entry for entry in data if entry.get('metadata', {}).get('split') == 'test']

    with open(save_path, 'w') as f:
        json.dump(test_data, f, indent=4)

    print(f"Successfully extracted {len(test_data)} records from 'test' split and saved to {save_path}")

base_path = '/mnt/hdd/gtoken/data1/ayush/HIGHT_4/data/Molecule-oriented_Instructions/full/'
save_base_path = '/mnt/hdd/gtoken/data1/ayush/HIGHT_4/data/Molecule-oriented_Instructions/'

full_path = os.path.join(base_path, 'property_prediction.json')
save_path = os.path.join(save_base_path, 'property_prediction-test.json')
create_test_set(full_path, save_path)
