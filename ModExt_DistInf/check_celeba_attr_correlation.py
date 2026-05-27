"""
Rank all CelebA attributes by their correlation with the Male attribute
so we can pick a classification target that is least gender-confounded.
"""

import numpy as np
import pandas as pd
from pathlib import Path

attrs_path = Path(__file__).parent / "data" / "celeba" / "list_attr_celeba.txt"
attr_df = pd.read_csv(attrs_path, sep=r"\s+", header=1, index_col=0)
attr_df = (attr_df + 1) // 2  # convert -1/1 → 0/1

male = attr_df["Male"].to_numpy(dtype=float)
targets = [c for c in attr_df.columns if c != "Male"]

rows = []
for col in targets:
    y = attr_df[col].to_numpy(dtype=float)
    corr = float(np.corrcoef(male, y)[0, 1])
    prevalence = float(y.mean())
    rows.append({"attribute": col, "corr_with_male": corr, "prevalence": prevalence})

results = pd.DataFrame(rows).sort_values("corr_with_male", key=abs)
print("Attributes ranked by |correlation| with Male (lowest = least confounded):\n")
print(results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
