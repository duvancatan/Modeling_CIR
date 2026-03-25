import numpy as np
import matplotlib.pyplot as plt

# Estilo para el paper
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 19,
    'axes.labelsize': 17,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 15,
    'lines.linewidth': 1.5
})

def simular_cir_baseline():
    # Parámetros
    T, dt = 15.0, 5e-3
    steps = int(T / dt)
    t = np.linspace(0, T, steps)
    N_sim = 50
    kappa, eta, sigma = 2.0, 0.4, 0.3
    p0, s0 = 0.5, 0.99

    Pt = np.zeros((steps, N_sim))
    St = np.zeros((steps, N_sim))
    Pt[0, :] = p0
    St[0, :] = s0

    # Simulación Euler-Maruyama para CIR
    for i in range(1, steps):
        dW = np.random.normal(0, np.sqrt(dt), N_sim)
        # Proceso CIR: dP = kappa*(eta - P)*dt + sigma*sqrt(P)*dW
        drift = kappa * (eta - Pt[i-1, :]) * dt
        # Usamos abs o maximum para evitar problemas numéricos con la raíz
        diff = sigma * np.sqrt(np.maximum(Pt[i-1, :], 0)) * dW
        Pt[i, :] = Pt[i-1, :] + drift + diff

        # Integral de P_t para la solución exacta del SI
        Ht = np.trapz(Pt[:i+1, :], dx=dt, axis=0)
        St[i, :] = 1 / (1 + ((1/s0) - 1) * np.exp(Ht))

    # --- FIGURA ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Panel Superior: P_t^C
    for j in range(N_sim):
        ax1.plot(t, Pt[:, j], color='royalblue', alpha=0.4)
    ax1.set_ylabel(r'Intensity $P_t^C$', fontweight='bold')
    ax1.set_title('CIR Transmission Intensity (Non-intervention)', pad=15)
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Panel Inferior: 1 - S_t
    for j in range(N_sim):
        ax2.plot(t, 1 - St[:, j], color='darkred', alpha=0.4)
    ax2.set_ylabel(r'Infected Fraction $1-S_t$', fontweight='bold')
    ax2.set_xlabel('Time $t$', fontweight='bold')
    ax2.set_title('Epidemic Saturation', pad=15)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig('CIR1.png', dpi=300)
    print("Figura 'CIR1.png' guardada.")
    plt.show()

simular_cir_baseline()