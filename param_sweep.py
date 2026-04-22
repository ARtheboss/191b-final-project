import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# 1. PARAMETERS
N = 2  # Cities
num_qubits = N**2
B = 1.0  # Weight for road distance

A_vals = np.linspace(2.0, 10.0, 5)
gamma_vals = np.linspace(0.5, 1.5, 5)
beta_vals = np.linspace(0.5, 5.0, 5)

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


def one_city_per_timestep(binary_sol: str) -> bool:
    """True if each time step has exactly one city marked '1'."""
    for t in range(N):
        n_at_t = sum(1 for i in range(N) if binary_sol[get_idx(i, t)] == "1")
        if n_at_t != 1:
            return False
    return True


one_city_matches = []

for A in A_vals:
    for gamma in gamma_vals:
        for beta in beta_vals:
            print(f"Checking A={A}, gamma={gamma}, beta={beta}")

            # 3. HAMILTONIAN CONSTRUCTION
            print("Building TSP Hamiltonian (this may take a minute)...")
            H = 0

            # Constraint 1: Each city visited exactly once
            for i in range(N):
                count_visits = sum(q_op(get_idx(i, t)) for t in range(N))
                H += A * (count_visits - eye_all) ** 2

            # Constraint 2: Only one city per time step
            for t in range(N):
                count_cities = sum(q_op(get_idx(i, t)) for i in range(N))
                H += A * (count_cities - eye_all) ** 2

            # Cost: Sum of distances
            for i in range(N):
                for j in range(N):
                    if i != j:
                        for t in range(N):
                            t_next = (t + 1) % N
                            H += B * D[i, j] * (
                                q_op(get_idx(i, t)) * q_op(get_idx(j, t_next))
                            )

            # 4. DISSIPATIVE COOLING
            print("Setting up dissipative jump operators...")
            energies, evecs = H.eigenstates()

            L_global = 0
            for i in range(len(energies)):
                for j in range(len(energies)):
                    delta_E = energies[j] - energies[i]
                    if delta_E < 0:
                        strength = np.exp(-beta * abs(delta_E))
                        jump = evecs[j] * evecs[i].dag()
                        L_global += strength * jump

            c_ops = [np.sqrt(gamma) * L_global]

            # 5. RUN SIMULATION
            plus = (basis(2, 0) + basis(2, 1)).unit()
            psi0 = tensor([plus] * num_qubits)
            times = np.linspace(0, 150, 300)

            print("Running Monte Carlo simulation...")
            result = mcsolve(H, psi0, times, c_ops, ntraj=1)

            # 6. EXTRACT SOLUTION
            final_ket = result.states[-1]
            probs = np.abs(final_ket.full()) ** 2
            best_idx = np.argmax(probs)
            binary_sol = format(best_idx, f"0{num_qubits}b")

            print("\n--- RESULTS ---")
            print(f"Winning Bitstring: {binary_sol} (A={A}, gamma={gamma}, beta={beta})")
            if one_city_per_timestep(binary_sol):
                one_city_matches.append((float(A), float(gamma), float(beta), binary_sol))
                print(
                    "  -> Dominant state satisfies: exactly one city per time step."
                )
            for t in range(N):
                for i in range(N):
                    if binary_sol[get_idx(i, t)] == "1":
                        print(f"Step {t}: Visit City {i}")

print("\n=== SWEEP SUMMARY (one city per time step) ===")
if one_city_matches:
    print(
        f"Found {len(one_city_matches)} parameter set(s) whose dominant bitstring "
        "has exactly one city visited per time step:"
    )
    for A_m, g_m, b_m, bits in one_city_matches:
        print(f"  A={A_m}, gamma={g_m}, beta={b_m}, bitstring={bits}")
else:
    print(
        "No parameter combination produced a dominant bitstring with exactly "
        "one city per time step."
    )
