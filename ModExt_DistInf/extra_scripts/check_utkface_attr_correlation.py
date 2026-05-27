"""
Compute pairwise correlations among UTKFace attributes (age, gender, race)
so we can pick a classification target that is least confounded with the
chosen sensitive attribute.

Age is binarised at 30 (the experiment default) before computing correlations.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from amulet.datasets import load_utkface

data = load_utkface(
    path=Path(__file__).parent.parent / "data" / "utkface",
    target="age",
    attribute_1="gender",
    attribute_2="race",
    age_bins=[30],
)

assert data.y_train is not None
assert data.y_test is not None
assert data.z_train is not None
assert data.z_test is not None

age = np.concatenate([data.y_train, data.y_test]).astype(float)
z = np.concatenate([data.z_train, data.z_test], axis=0)
gender = z[:, 0].astype(float)
race = z[:, 1].astype(float)

attrs = {"age": age, "gender": gender, "race": race}
names = list(attrs.keys())

rows = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        corr = float(np.corrcoef(attrs[a], attrs[b])[0, 1])
        rows.append({"attr_1": a, "attr_2": b, "corr": corr, "|corr|": abs(corr)})

results = pd.DataFrame(rows).sort_values("|corr|", ascending=False)
print("Pairwise correlations among UTKFace attributes (|corr| descending):\n")
print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
