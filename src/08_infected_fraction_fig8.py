# Libraries
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Settings
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 12,
    'text.usetex': False
})

def generate_early_stage_divergence():
    # Detect path: Up one level from 'src' to the project root
    base_path = Path(__file__).parent.parent
    output_dir = base_path / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parameters
    T = 10.0
    dt = 0.05  
    steps = int(T / dt)
    s0 = 0.99
    mu = eta = 0.3
    theta = kappa = 0.5
    a = 0.5
    sigma_j = 1.5
    N = 20000  # Number of Monte Carlo paths

    # Calibration
    var_j = (sigma_j**2 * mu * (a - mu)) / (2 * theta + a * sigma_j**2)
    sigma_cir = np.sqrt((var_j * 2 * kappa) / eta)

    # Path Simulation
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

    # Calculate Infected Fraction 1 - S_T
    inf_jacobi = 1 - 1 / (1 + ((1/s0) - 1) * np.exp(h_jacobi))
    inf_cir = 1 - 1 / (1 + ((1/s0) - 1) * np.exp(h_cir))

    # Plotting
    plt.figure(figsize=(10, 7))
    plt.hist(inf_jacobi, bins=80, range=(0, 1), density=True, alpha=0.5, color='blue', label='Jacobi (Bounded)')
    plt.hist(inf_cir, bins=80, range=(0, 1), density=True, alpha=0.5, color='red', label='CIR (Unbounded)')

    plt.title(f'Infected Fraction Distribution at $T={T}$ ($\sigma_J=1.5$)')
    plt.xlabel('Infected Fraction ($1 - S_T$)')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save using the automatic path in Modeling_CIR/figures
    plt.savefig(output_dir / '08_infected_fraction_fig8.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    generate_early_stage_divergence()