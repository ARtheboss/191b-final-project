import numpy as np
from qutip import *

# 1. HARD-CODED PARAMETERS (No variables to avoid leakage)
N = 2  
num_qubits = N**2 # Fixed at 4 for N=2

# 2x2 Distance Matrix
D = np.array([[0, 5], [20, 0]])

A = np.max(D) * N
B = 1.0   
gamma = 0.5
beta = 5.0 

def get_idx(city, step):
    return (city * N) + step # Should be 0, 1, 2, or 3

# 2. CLEAN OPERATOR BUILDER
def make_op(single_qubit_op, idx):
    # Ensure we never create a tensor larger than 4 qubits
    op_list = [qeye(2)] * 4 
    op_list[idx] = single_qubit_op
    return tensor(op_list)

eye_all = tensor([qeye(2)] * 4)

def q_op(idx):
    return 0.5 * (eye_all - make_op(sigmaz(), idx))

# 3. HAMILTONIAN
H = 0
H += A * (eye_all - q_op(get_idx(0, 0)))
for i in range(N):
    H += A * (sum(q_op(get_idx(i, t)) for t in range(N)) - eye_all)**2
for t in range(N):
    H += A * (sum(q_op(get_idx(i, t)) for i in range(N)) - eye_all)**2
for i in range(N):
    for j in range(N):
        if i != j:
            for t in range(N):
                H += B * D[i, j] * (q_op(get_idx(i, t)) * q_op(get_idx(j, (t+1)%N)))

# 4. GLOBAL DISSIPATOR
energies, states = H.eigenstates(eigvals=5)
c_ops = []

for i in range(len(energies)):
    for j in range(len(energies)):
        delta_E = energies[j] - energies[i]
        
        # We only want transitions from HIGH energy (i) to LOW energy (j)
        if delta_E < -0.1:
            # Drop the exponential suppression for downward jumps
            # or use a thermal factor if you want to model a specific temp.
            # Here, we just give it a uniform cooling rate.
            strength = np.sqrt(gamma) 
            
            # Create a SEPARATE operator for every transition to avoid interference
            jump_op = strength * (states[j] * states[i].dag())
            c_ops.append(jump_op)

# 5. SIMULATION
plus = (basis(2, 0) + basis(2, 1)).unit()
psi0 = tensor([plus] * 4)
times = np.linspace(0, 100, 500)

# Create a list of the 4 operators we want to monitor
# (City 0 Step 0, City 0 Step 1, City 1 Step 0, City 1 Step 1)
monitor_ops = [q_op(i) for i in range(num_qubits)]

result = mesolve(H, psi0, times, c_ops, monitor_ops)

print("\n--- FINAL PROBABILITIES ---")
for i in range(num_qubits):
    final_prob = result.expect[i][-1]
    print(f"Qubit {i} (City {i//N}, Step {i%N}): {final_prob:.4f}")