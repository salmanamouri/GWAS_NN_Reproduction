import pandas as pd
import numpy as np


def load_gametes_dataset(genotype_csv_path, phenotype_csv_path):
    """
    Simple adapter for GAMETES-generated CSVs.
    Assumes:
    - genotype CSV: rows=samples, cols=SNPs
    - phenotype CSV: one phenotype column
    """
    X = pd.read_csv(genotype_csv_path).values.astype(np.float32)
    y = pd.read_csv(phenotype_csv_path).iloc[:, 0].values.astype(np.float32)
    return X, y