# Libraries
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Settings
plt.rcParams.update({
    'font.size': 16,             
    'axes.titlesize': 20,       
    'axes.labelsize': 18,       
    'xtick.labelsize': 14,     
    'ytick.labelsize': 14,     
    'legend.fontsize': 16,      
    'figure.titlesize': 22,     
    'lines.linewidth': 1.5      
})

def simular_y_graficar_jacobi():
    # Detect path: Up one level from 'src' to the project root
    base_path = Path(__file__).parent.parent
    output_dir = base_path / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parameters
    T = 15.0
    dt = 5e-3
    steps = int(T / dt)
    t = np.linspace(0, T, steps)
    N_sim = 50
    theta, mu, sigma, a = 2.0, 0.4, 0.3, 1.0  
    p0, s0 = 0.5, 0.99

    # Arrays for storing results
    Pt = np.zeros((steps, N_sim))
    St = np.zeros((steps, N_sim))
    Pt[0, :] = p0
    St[0, :] = s0

    # Euler-Maruyama simulation
    for i in range(1, steps):
        dW = np.random.normal(0, np.sqrt(dt), N_sim)
        # Jacobi Process
        drift = theta * (mu - Pt[i-1, :]) * dt
        diffusion = sigma * np.sqrt(np.maximum(Pt[i-1, :] * (a - Pt[i-1, :]), 0)) * dW
        Pt[i, :] = Pt[i-1, :] + drift + diffusion

        # SI equation (using the exact solution from the paper for greater accuracy)
        # H_t is the integral of β_r dr. Here β_r = P_r (φ=1)
        # We calculate the cumulative integral using the trapezoidal rule
        integral_P = np.trapz(Pt[:i+1, :], dx=dt, axis=0)
        St[i, :] = 1 / (1 + ((1/s0) - 1) * np.exp(integral_P))
     # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for j in range(N_sim):
        ax1.plot(t, Pt[:, j], color='steelblue', alpha=0.4)
    ax1.axhline(y=a, color='red', linestyle='--', label=f'Upper bound $a={a}$')
    ax1.set_ylabel(r'Intensity $P_t^J$', fontweight='bold')
    ax1.set_title('Stochastic Transmission Intensity', pad=20)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)

    for j in range(N_sim):
        ax2.plot(t, 1 - St[:, j], color='darkorange', alpha=0.4)
    ax2.set_ylabel(r'Infected Fraction $1-S_t$', fontweight='bold')
    ax2.set_xlabel('Time $t$', fontweight='bold')
    ax2.set_title('Epidemic Evolution (Infected Population)', pad=20)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # Save using the automatic path in Modeling_CIR/figures
    plt.savefig(output_dir / '01_jacobi_iteration_fig1.png', dpi=300)
    plt.show()

simular_y_graficar_jacobi()