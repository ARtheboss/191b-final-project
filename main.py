import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# 1. PARAMETERS
N = 2  # Cities
num_qubits = N**2
A = 5.0  # Penalty for breaking rules
B = 1.0   # Weight for road distance

# Distance Matrix (4x4)
D = np.array([
    [0, 10, 5, 8],
    [10, 0, 7, 3],
    [5, 7, 0, 4],
    [8, 3, 4, 0]
])

def get_idx(city, step):
    return city * N + step

# 2. OPTIMIZED OPERATOR BUILDER
print(f"Initializing {num_qubits}-qubit identity...")
eye_all = qeye([2] * num_qubits)

def make_op(single_qubit_op, idx):
    """Efficiently places a gate on qubit 'idx' among 'num_qubits'."""
    op_list = [qeye(2)] * num_qubits
    op_list[idx] = single_qubit_op
    return tensor(op_list)

def q_op(idx):
    """Maps qubit state to binary 0 or 1: (I - Z)/2."""
    return 0.5 * (eye_all - make_op(sigmaz(), idx))

# 3. HAMILTONIAN CONSTRUCTION
print("Building TSP Hamiltonian (this may take a minute)...")
H = 0

# Constraint 1: Each city visited exactly once
for i in range(N):
    count_visits = sum(q_op(get_idx(i, t)) for t in range(N))
    H += A * (count_visits - eye_all)**2

# Constraint 2: Only one city per time step
for t in range(N):
    count_cities = sum(q_op(get_idx(i, t)) for i in range(N))
    H += A * (count_cities - eye_all)**2

# Cost: Sum of distances
for i in range(N):
    for j in range(N):
        if i != j:
            for t in range(N):
                t_next = (t + 1) % N
                # If we are in city i at t and city j at t+1, add distance
                H += B * D[i, j] * (q_op(get_idx(i, t)) * q_op(get_idx(j, t_next)))

# 4. DISSIPATIVE COOLING
print("Setting up dissipative jump operators...")
gamma = 1.0
# We apply a lowering operator (sigmam) to every qubit to drain energy
c_ops = [np.sqrt(gamma) * make_op(sigmam(), i) for i in range(num_qubits)]

# 5. RUN SIMULATION
# Start in a 'hot' state (equal superposition of all paths)
# Create a single '+' state: (|0> + |1>) / sqrt(2)
plus = (basis(2, 0) + basis(2, 1)).unit()

# Create the initial state as a tensor product of 16 '+' states
psi0 = tensor([plus] * num_qubits)
times = np.linspace(0, 150, 300)

print("Running Monte Carlo simulation...")
# ntraj=1 provides a single 'pathway' to the solution to save memory
result = mcsolve(H, psi0, times, c_ops, ntraj=1)

# 6. EXTRACT SOLUTION
final_ket = result.states[-1]
# Find the bitstring with the highest probability
probs = np.abs(final_ket.full())**2
best_idx = np.argmax(probs)
binary_sol = format(best_idx, f'0{num_qubits}b')

print("\n--- RESULTS ---")
print(f"Winning Bitstring: {binary_sol}")
for t in range(N):
    for i in range(N):
        # Qubit index logic for time t and city i
        if binary_sol[get_idx(i, t)] == '1':
            print(f"Step {t}: Visit City {i}")