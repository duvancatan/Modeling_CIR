import numpy as np
import matplotlib.pyplot as plt

# Estilo para el paper
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 19,
    'axes.labelsize': 17,
    'legend.fontsize': 15,
    'lines.linewidth': 1.6
})

def simular_cir_intervencion():
    # Parámetros
    T, dt = 15.0, 5e-3
    steps = int(T / dt)
    t = np.linspace(0, T, steps)
    N_sim = 50
    alpha = 0.2
    kappa, eta, sigma = 2.0, 0.4, 0.3
    p0, s0 = 0.5, 0.99

    Pt = np.zeros((steps, N_sim))
    St = np.zeros((steps, N_sim))
    Pt[0, :] = p0
    St[0, :] = s0

    phi = np.exp(-alpha * t)

    for i in range(1, steps):
        dW = np.random.normal(0, np.sqrt(dt), N_sim)
        # Trayectoria del CIR
        drift = kappa * (eta - Pt[i-1, :]) * dt
        diff = sigma * np.sqrt(np.maximum(Pt[i-1, :], 0)) * dW
        Pt[i, :] = Pt[i-1, :] + drift + diff

        # Integral acumulada de beta_s = phi(s) * P_s
        beta_path = phi[:i+1, None] * Pt[:i+1, :]
        Ht = np.trapz(beta_path, dx=dt, axis=0)
        St[i, :] = 1 / (1 + ((1/s0) - 1) * np.exp(Ht))

    # --- GRÁFICA ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Panel Superior: beta_t = phi * P_t
    for j in range(N_sim):
        ax1.plot(t, phi * Pt[:, j], color='mediumpurple', alpha=0.4)
    ax1.set_ylabel(r'Rate $\beta_t = \varphi(t)P_t^C$', fontweight='bold')
    ax1.set_title(r'CIR Modulated Rate under Intervention ($\alpha=0.2$)', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Panel Inferior: 1 - S_t
    for j in range(N_sim):
        ax2.plot(t, 1 - St[:, j], color='darkorange', alpha=0.4)
    ax2.set_ylabel(r'Infected Fraction $1-S_t$', fontweight='bold')
    ax2.set_xlabel('Time $t$', fontweight='bold')
    ax2.set_title('Stabilization of the Epidemic Size', pad=15)
    ax2.set_ylim(0, 0.4) # Escala ajustada para ver el efecto de frenado
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('CIR2.png', dpi=300)
    print("Figura 'CIR2.png' generada.")
    plt.show()

simular_cir_intervencion()