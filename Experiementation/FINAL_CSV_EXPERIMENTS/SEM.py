import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_ind

import pandas as pd
FILES = {
    "Baseline": "FINAL_results_NoDPD_lspan_trained_1.csv",
    "ML": "FINAL_results_DPD_NN_lspan_trained_1.csv",
    "MP": "FINAL_results_DPD_MP_BER_vs_Spans.csv",
    "WH": "FINAL_results_DPD_WH_BER_vs_Lspan.csv",
}

BASE_DIR = Path.cwd()

# -----------------------------
# Load data
# -----------------------------
dfs = {}

for case, file in FILES.items():
    df = pd.read_csv(file)

    dfs[case] = df

# -----------------------------
# Compute mean + SD
# -----------------------------
summaries = []