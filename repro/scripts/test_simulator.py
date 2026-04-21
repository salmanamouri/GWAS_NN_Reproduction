import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repro.simulators.complex_simulator import ComplexSimulator

#executed a data generation pipeline in memory
sim = ComplexSimulator()
X, y, G = sim.generate()

pd.DataFrame(X).to_csv("X.csv", index=False)
pd.DataFrame(y).to_csv("y.csv", index=False)

print("Saved X.csv and y.csv")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("G shape:", G.shape)