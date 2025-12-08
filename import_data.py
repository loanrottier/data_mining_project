from scipy.io import arff
import pandas as pd
import numpy as np


data, meta = arff.loadarff("data\WBC_withoutdupl_v01.arff")
df = pd.DataFrame(data)

# cleaning byte strings to normal strings
for col in df.select_dtypes([object]):
    df[col] = df[col].str.decode("utf-8")

print(df.head())
print(f"shape {df.shape}")


print("Columns available :", df.columns.tolist())


target_col = df.columns[-1]
print(f"La colonne cible identifiée est : '{target_col}'")
print("Valeurs uniques :", df[target_col].unique())

# transformr outlier into 0: normal, 1: anomaly
y = np.where(df[target_col] == "yes", 1, 0)
X = df.drop(columns=[target_col]).values
