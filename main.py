import numpy as np
import qiskit
from qiskit.quantum_info import SparsePauliOp
import qiskit_aer
import matplotlib.pyplot as plt

# 2 runs of 10 shots and 10000 shots
n = [10,10000]
# n = [10000]
t_max = 4.0
t = np.arange(0.1, t_max, 0.1) # time steps for the evolution,  t=[0.1,0.2,0.3,...]

# ==================================
# Part 1
# ==================================
def construct_circuits_from_matrix():
    u_hat = lambda t_i: qiskit.circuit.library.PauliEvolutionGate(SparsePauliOp("X"), t_i) # the evolution operator for the pauli-x gate (sparsepauliop("X"))
    list_of_u_hat = [u_hat(t_i) for t_i in t] # list of evolution operators for each time step

    circuits = [None] * len(t) # list of quantum circuits to hold the evolution operators
    for i, u_gate in enumerate(list_of_u_hat):
        circuits[i] = qiskit.QuantumCircuit(1) # initialize
        circuits[i].append(u_gate, [0]) # append the evolution operator to the circuit
        circuits[i].measure_all() # measure the qubit at the end of the circuit
        circuits[i].name= f"{t[i]:.1f}"  # name the circuit with the corresponding time step
    return circuits

# ==================================
# Part 2
# ==================================
def construct_circuits_from_gates():
    circuits = [None] * len(t) # list of quantum circuits to hold the evolution operators
    for i, t_i in enumerate(t):
        circuits[i] = qiskit.QuantumCircuit(1) # initialize
        circuits[i].h(0)
        circuits[i].rz(2 * t_i, 0) # apply the rotation gate for the evolution operator
        circuits[i].h(0)
        circuits[i].measure_all() # measure the qubit at the end of the circuit
        circuits[i].name= f"{t[i]:.1f}"  # name the circuit with the corresponding time step
    return circuits

# ==================================
# Testing and plotting the results
# ==================================

part = int(input("Construct the circuits from (1) matrix or (2) gates? Enter 1 or 2: "))
if part == 1:
    circuits = construct_circuits_from_matrix()
elif part == 2:
    circuits = construct_circuits_from_gates()

# Instantiate the backend once outside the loop for efficiency
backend = qiskit_aer.AerSimulator()

transpiled = [qiskit.transpile(circuit, backend) for circuit in circuits] # transpile the circuits once for the backend

results=[]
for shots in n:
    count_list = []
    for circuit in transpiled:
        # run and get counts
        counts = backend.run(circuit, shots=shots).result().get_counts()
        count_list.append(counts)
    results.append(count_list)

    cols = 5
    rows = int(np.ceil(len(t) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.4 * rows), sharey=True)
    axes = np.atleast_1d(axes).ravel() # configure subplots

    for axis, t_i, counts in zip(axes, t, count_list):
        heights = [counts.get(k, 0) for k in ["0", "1"]] # get the counts for each measurement outcome
        axis.bar(["0", "1"], heights)
        # plot the theoretical outcomes as dashed lines
        axis.plot([-0.4, 0.4], [shots * np.cos(t_i) ** 2] * 2, "k--", lw=1)
        axis.plot([0.6, 1.4], [shots * np.sin(t_i) ** 2] * 2, "k--", lw=1)
        axis.set_title(f"t = {t_i:.1f}", fontsize=10)
        axis.set_ylim(0, shots)

    for axis in axes[len(t):]:
        axis.axis("off") # turn off axes for empty subplots

    fig.suptitle(f"Measurement counts vs. t   ({shots} shots per circuit; "
                 f"dashed = theory)", fontsize=13)
    fig.supxlabel("Measurement outcome")
    fig.supylabel("Frequency")
    fig.tight_layout()


    plt.show()

# plot the measurement of P(1) for t for both shot counts against the theoretical value \sin^2(t)

plt.figure()
for shots in n:
    p1 = [c.get("1", 0) / shots for c in results[n.index(shots)]]
    plt.plot(t, p1, label=f"{shots} shots")
plt.plot(t, [np.sin(ti) ** 2 for ti in t], "k--", label=r"Theoretical ($\sin^2(t)$)")
plt.xlabel("Time")
plt.ylabel("P(1)")
plt.title(r"Measured $P(1)$ under $\hat{U}(t)=e^{-i\sigma_x t}$")
plt.legend()
plt.show()

