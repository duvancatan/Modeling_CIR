import numpy as np
import matplotlib.pyplot as plt

# --- Style settings ---
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'text.usetex': False})

def generate_intervention_comparison():
    # 1. Parameters
    T = 100.0 # Long horizon to approximate infinity
    dt = 0.1
    steps = int(T / dt)
    s0 = 0.99
    mu = eta = 0.3
    theta = kappa = 0.5
    a = 0.5
    sigma_j = 1.5
    alpha_interv = 0.2 # Intervention strength
    N = 50000

    # 2. Calibration
    var_j = (sigma_j**2 * mu * (a - mu)) / (2 * theta + a * sigma_j**2)
    sigma_cir = np.sqrt((var_j * 2 * kappa) / eta)

    # 3. Path Simulation with Intervention
    h_jacobi = np.zeros(N)
    h_cir = np.zeros(N)
    pj = np.full(N, mu)
    pc = np.full(N, eta)

    time_vec = np.linspace(0, T, steps)
    phi = np.exp(-alpha_interv * time_vec)

    print(f"Simulating final epidemic size with alpha={alpha_interv}...")
    for i in range(steps):
        dW = np.random.normal(0, np.sqrt(dt), N)

        # Paths
        dj = theta * (mu - pj) * dt + sigma_j * np.sqrt(np.maximum(pj * (a - pj), 0)) * dW
        pj = np.clip(pj + dj, 0, a)

        dc = kappa * (eta - pc) * dt + sigma_cir * np.sqrt(np.maximum(pc, 0)) * dW
        pc = np.maximum(pc + dc, 0)

        # Modulated integration: beta_t = phi(t) * P_t
        h_jacobi += (phi[i] * pj) * dt
        h_cir += (phi[i] * pc) * dt

    # 4. Final Sizes
    size_j = 1 - 1 / (1 + ((1/s0) - 1) * np.exp(h_jacobi))
    size_c = 1 - 1 / (1 + ((1/s0) - 1) * np.exp(h_cir))

    # 5. Calculate Metrics for the Table
    print("\n--- Risk Metrics ---")
    print(f"Jacobi Mean: {np.mean(size_j):.3f}, 95th Perc: {np.percentile(size_j, 95):.3f}, Max: {np.max(size_j):.3f}")
    print(f"CIR Mean: {np.mean(size_c):.3f}, 95th Perc: {np.percentile(size_c, 95):.3f}, Max: {np.max(size_c):.3f}")

    # 6. Plotting
    plt.figure(figsize=(10, 7))
    plt.hist(size_j, bins=100, density=True, alpha=0.5, color='blue', label='Jacobi (Bounded)')
    plt.hist(size_c, bins=100, density=True, alpha=0.5, color='red', label='CIR (Unbounded)')

    plt.title(r'Final Epidemic Size $1-S_\infty$ ($\alpha=0.2$, $\sigma_J=1.5$)')
    plt.xlabel('Proportion of Infected Individuals')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('Intervention1.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    generate_intervention_comparison()