# Libraries
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Settings
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 19,
    'axes.labelsize': 17,
    'legend.fontsize': 15,
    'lines.linewidth': 1.6
})

def simular_jacobi_intervencion():
    # Detect path: Up one level from 'src' to the project root
    base_path = Path(__file__).parent.parent
    output_dir = base_path / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Simulation parameters
    T, dt = 15.0, 5e-3
    steps = int(T / dt)
    t = np.linspace(0, T, steps)
    N_sim = 50
    alpha = 0.2  # Intervention decline rate

    # # Jacobi process parameters
    theta, mu, sigma, a = 2.0, 0.4, 0.3, 1.0
    p0, s0 = 0.5, 0.99

    Pt = np.zeros((steps, N_sim))
    St = np.zeros((steps, N_sim))
    Pt[0, :] = p0
    St[0, :] = s0

    # Intervention function phi(t)
    phi = np.exp(-alpha * t)

    for i in range(1, steps):
        dW = np.random.normal(0, np.sqrt(dt), N_sim)
        # Evolution of P_t (Jacobi)
        drift = theta * (mu - Pt[i-1, :]) * dt
        diff = sigma * np.sqrt(np.maximum(Pt[i-1, :] * (a - Pt[i-1, :]), 0)) * dW
        Pt[i, :] = Pt[i-1, :] + drift + diff

        # Effective rate beta_t = phi(t) * P_t
        # Note: We calculate the integral of beta_s ds up to time i
        beta_path = phi[:i+1, None] * Pt[:i+1, :]
        Ht = np.trapz(beta_path, dx=dt, axis=0)

        St[i, :] = 1 / (1 + ((1/s0) - 1) * np.exp(Ht))
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for j in range(N_sim):
        ax1.plot(t, phi * Pt[:, j], color='seagreen', alpha=0.4)
    ax1.set_ylabel(r'Effective Rate $\beta_t = \varphi(t)P_t^J$', fontweight='bold')
    ax1.set_title(r'Modulated Transmission Intensity ($\alpha=0.2$)', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.5)

    for j in range(N_sim):
        ax2.plot(t, 1 - St[:, j], color='crimson', alpha=0.4)
    ax2.set_ylabel(r'Infected Fraction $1-S_t$', fontweight='bold')
    ax2.set_xlabel('Time $t$', fontweight='bold')
    ax2.set_title('Epidemic Control under Intervention', pad=15)
    ax2.set_ylim(0, 0.4) # Adjust limit to see braking details
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save using the automatic path in Modeling_CIR/figures
    plt.savefig(output_dir / '02_jacobi_convergence_fig2.png', dpi=300)
    plt.show()

simular_jacobi_intervencion()