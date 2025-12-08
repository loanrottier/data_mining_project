import numpy as np
from depth.model import DepthEucl
from import_data import load_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# ----------------------------------------------------------
# Without train test split
# ----------------------------------------------------------
X, y = load_data()
model = DepthEucl().load_dataset(X)
# get depth score on test set
scores = model.projection(X, NRandom=1000)
depth_scores = np.array(scores[0])

# get some informations about depth scores in order to choose a threshold
s_scores = pd.Series(depth_scores)
print(s_scores.describe(percentiles=[0.01, 0.05, 0.10, 0.25]))

# since we know we have arround 4% of outliers in the dataset, we can choose a threshold with 5% percentile
threshold = s_scores.quantile(0.05)
y_pred = (depth_scores < threshold).astype(int)
# evaluate the results

print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))

# with a threshold of 0.000192, we detect onlu 1 outliers over 10 and we have 11 false positives

# try a loop with some thresholds
# thresholds = s_scores.quantile(
#     [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
# )
# for t in thresholds:
#     y_pred = (depth_scores < t).astype(int)
#     print(f"Threshold: {t}")
#     print(confusion_matrix(y, y_pred))
#     print(classification_report(y, y_pred))


# The best threshold seems to be 0.00103 with 2 outliers detected over 10 and 14 false positives
# However the results stay bad


# simplicial volume depth
model = DepthEucl().load_dataset(X)

# get depth
scores = model.simplicialVolume(X, exact=False, k=10)
print(scores)
depth_scores = np.array(scores[0])
# get some informations about depth scores in order to choose a threshold
s_scores = pd.Series(depth_scores)
print(s_scores.describe(percentiles=[0.01, 0.05, 0.10, 0.25]))

threshold = s_scores.quantile(0.1)
print(threshold, " chosen threshold")
y_pred = (depth_scores < threshold).astype(int)
# evaluate the results
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))
