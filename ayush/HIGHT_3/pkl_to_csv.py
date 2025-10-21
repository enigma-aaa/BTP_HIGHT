import pandas as pd
import pickle

# Load object
with open("/mnt/hdd/gtoken/data1/ayush/HIGHT_3/hipubchem/HIGHT/hi_data_dict_lap_fgprompt.pkl", "rb") as f:
    data = pickle.load(f)

# If it's a dict
if isinstance(data, dict):
    df = pd.DataFrame.from_dict(data)

# If it's a list of dicts
elif isinstance(data, list):
    df = pd.DataFrame(data)

else:
    raise ValueError("Pickle file does not contain a directly convertible structure")

# Save as CSV
df.head(10000).to_csv("hi_pub_chem_10000.csv", index=False)
