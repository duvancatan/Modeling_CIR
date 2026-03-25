import numpy as np
import matplotlib.pyplot as plt

# --- Style settings for high visibility ---
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 12,
    'text.usetex': False
})

def generate_early_stage_divergence():
    # 1. Parameters
    T = 10.0
    dt = 0.05  # Smaller step for better path resolution
    steps = int(T / dt)
    s0 = 0.99
    mu = eta = 0.3
    theta = kappa = 0.5
    a = 0.5
    sigma_j = 1.5
    N = 20000  # Number of Monte Carlo paths

    # 2. Calibration
    var_j = (sigma_j**2 * mu * (a - mu)) / (2 * theta + a * sigma_j**2)
    sigma_cir = np.sqrt((var_j * 2 * kappa) / eta)

    # 3. Path Simulation
    h_jacobi = np.zeros(N)
    h_cir = np.zeros(N)
    pj = np.full(N, mu)
    pc = np.full(N, eta)

    print(f"Simulating early stage divergence (T={T})...")
    for i in range(steps):
        dW = np.random.normal(0, np.sqrt(dt), N)

        # Jacobi path
        dj = theta * (mu - pj) * dt + sigma_j * np.sqrt(np.maximum(pj * (a - pj), 0)) * dW
        pj = np.clip(pj + dj, 1e-7, a - 1e-7)

        # CIR path
        dc = kappa * (eta - pc) * dt + sigma_cir * np.sqrt(np.maximum(pc, 0)) * dW
        pc = np.maximum(pc + dc, 1e-7)

        h_jacobi += pj * dt
        h_cir += pc * dt

    # 4. Calculate Infected Fraction 1 - S_T
    inf_jacobi = 1 - 1 / (1 + ((1/s0) - 1) * np.exp(h_jacobi))
    inf_cir = 1 - 1 / (1 + ((1/s0) - 1) * np.exp(h_cir))

    # 5. Plotting
    plt.figure(figsize=(10, 7))

    # Histogram of the infected fraction
    plt.hist(inf_jacobi, bins=80, range=(0, 1), density=True, alpha=0.5, color='blue', label='Jacobi (Bounded)')
    plt.hist(inf_cir, bins=80, range=(0, 1), density=True, alpha=0.5, color='red', label='CIR (Unbounded)')

    plt.title(f'Infected Fraction Distribution at $T={T}$ ($\sigma_J=1.5$)')
    plt.xlabel('Infected Fraction ($1 - S_T$)')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('sub1.3.png', dpi=300)
    print("Figure 'sub1.3.png' saved successfully.")
    plt.show()

if __name__ == "__main__":
    generate_early_stage_divergence()