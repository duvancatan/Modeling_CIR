import numpy as np
import matplotlib.pyplot as plt

# Estilo estándar para el paper
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 19,
    'axes.labelsize': 17,
    'legend.fontsize': 15,
    'lines.linewidth': 1.6
})

def simular_jacobi_intervencion():
    # Parámetros de simulación
    T, dt = 15.0, 5e-3
    steps = int(T / dt)
    t = np.linspace(0, T, steps)
    N_sim = 50
    alpha = 0.2  # Tasa de decaimiento de la intervención

    # Parámetros del proceso Jacobi
    theta, mu, sigma, a = 2.0, 0.4, 0.3, 1.0
    p0, s0 = 0.5, 0.99

    Pt = np.zeros((steps, N_sim))
    St = np.zeros((steps, N_sim))
    Pt[0, :] = p0
    St[0, :] = s0

    # Función de intervención phi(t)
    phi = np.exp(-alpha * t)

    for i in range(1, steps):
        dW = np.random.normal(0, np.sqrt(dt), N_sim)
        # Evolución de P_t (Jacobi)
        drift = theta * (mu - Pt[i-1, :]) * dt
        diff = sigma * np.sqrt(np.maximum(Pt[i-1, :] * (a - Pt[i-1, :]), 0)) * dW
        Pt[i, :] = Pt[i-1, :] + drift + diff

        # Tasa efectiva beta_t = phi(t) * P_t
        # Nota: Calculamos la integral de beta_s ds hasta el tiempo i
        beta_path = phi[:i+1, None] * Pt[:i+1, :]
        Ht = np.trapz(beta_path, dx=dt, axis=0)

        St[i, :] = 1 / (1 + ((1/s0) - 1) * np.exp(Ht))

    # --- GRÁFICA ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Panel Superior: Tasa Efectiva beta_t
    for j in range(N_sim):
        ax1.plot(t, phi * Pt[:, j], color='seagreen', alpha=0.4)
    ax1.set_ylabel(r'Effective Rate $\beta_t = \varphi(t)P_t^J$', fontweight='bold')
    ax1.set_title(r'Modulated Transmission Intensity ($\alpha=0.2$)', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Panel Inferior: 1 - S_t
    for j in range(N_sim):
        ax2.plot(t, 1 - St[:, j], color='crimson', alpha=0.4)
    ax2.set_ylabel(r'Infected Fraction $1-S_t$', fontweight='bold')
    ax2.set_xlabel('Time $t$', fontweight='bold')
    ax2.set_title('Epidemic Control under Intervention', pad=15)
    ax2.set_ylim(0, 0.4) # Ajustamos el límite para ver el detalle del frenado
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('Jacobi-exp.png', dpi=300)
    print("Figura 'Jacobi-exp.png' generada con éxito.")
    plt.show()

simular_jacobi_intervencion()