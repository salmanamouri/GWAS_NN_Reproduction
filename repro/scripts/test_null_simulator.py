from repro.simulators.null_simulator import NullSimulator

sim = NullSimulator()
X, y, G = sim.generate()

print("X shape:", X.shape)
print("y shape:", y.shape) #NO true interactions inside y
print("G shape:", G.shape)