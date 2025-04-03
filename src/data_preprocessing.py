import pandas as pd
import os

save_path = "../data/processed/"
os.makedirs(save_path, exist_ok=True)

# Load dataset
df = pd.read_csv("../data/raw/drugs_side_effects_drugs_com.csv")

# Fill missing values
df.fillna("Unknown", inplace=True)

# Convert text to lowercase
df = df.apply(lambda x: x.astype(str).str.lower())

# Save cleaned dataset
df.to_csv(os.path.join(save_path, "cleaned_data.csv"), index=False)

print("✅ Data Preprocessing Complete")
