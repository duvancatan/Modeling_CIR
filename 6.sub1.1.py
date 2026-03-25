import numpy as np
import matplotlib.pyplot as plt

# --- GLOBAL STYLE SETTINGS (English & High Visibility) ---
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': False  # Disables external TeX to prevent errors
})

def generate_scenario_1_low_volatility():
    # 1. Parameters from the paper
    T = 15.0
    s0 = 0.99
    mu = 0.3
    eta = 0.3
    theta = 0.5
    kappa = 0.5
    a = 0.5
    sigma_j = 0.2
    N = 100000  # Number of Monte Carlo paths

    # 2. Calibration: Matching Variances
    # Var_Jacobi = (sigma_j^2 * mu * (a - mu)) / (2 * theta + a * sigma_j^2)
    var_j = (sigma_j**2 * mu * (a - mu)) / (2 * theta + a * sigma_j**2)

    # Solve for sigma_cir: var_j = (sigma_cir^2 * eta) / (2 * kappa)
    sigma_cir = np.sqrt((var_j * 2 * kappa) / eta)

    print(f"Calibration Results:")
    print(f"Target Variance: {var_j:.6f}")
    print(f"Calibrated Sigma CIR: {sigma_cir:.6f}")

    # 3. Stationary Distributions (Long-term behavior at T=15)
    # Jacobi follows a scaled Beta distribution
    alpha_j = 2 * theta * mu / (sigma_j**2 * a)
    beta_j = 2 * theta * (a - mu) / (sigma_j**2 * a)
    p_jacobi = np.random.beta(alpha_j, beta_j, N) * a

    # CIR follows a Gamma distribution
    shape_c = 2 * kappa * eta / sigma_cir**2
    scale_c = sigma_cir**2 / (2 * kappa)
    p_cir = np.random.gamma(shape_c, scale_c, N)

    # 4. Final Epidemic Size (1 - S_T)
    # Using the exact solution: S_T = [1 + (1/s0 - 1) * exp(H_T)]^-1
    # For a simple comparison, we assume H_T approx P_T * T
    # (assuming the process has reached its stationary regime)
    h_jacobi = p_jacobi * T
    h_cir = p_cir * T

    inf_jacobi = 1 - 1 / (1 + ((1/s0) - 1) * np.exp(h_jacobi))
    inf_cir = 1 - 1 / (1 + ((1/s0) - 1) * np.exp(h_cir))

    # 5. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left Panel: Infection Intensity
    ax1.hist(p_jacobi, bins=100, density=True, alpha=0.5, color='blue', label='Jacobi')
    ax1.hist(p_cir, bins=100, density=True, alpha=0.5, color='red', label='CIR')
    ax1.set_title(f'Intensity $P_T$ at $T={T}$')
    ax1.set_xlabel('Intensity Value')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(alpha=0.2)

    # Right Panel: Epidemic Size
    ax2.hist(inf_jacobi, bins=100, density=True, alpha=0.5, color='blue', label='Jacobi')
    ax2.hist(inf_cir, bins=100, density=True, alpha=0.5, color='red', label='CIR')
    ax2.set_title(f'Infected Fraction $1-S_T$ at $T={T}$')
    ax2.set_xlabel('Proportion')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig('sub1.1.png', dpi=300)
    print("Figure 'sub1.1.png' generated successfully.")
    plt.show()

if __name__ == "__main__":
    generate_scenario_1_low_volatility()
    