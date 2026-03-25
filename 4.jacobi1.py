import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE ESTILO (A pedido del compañero) ---
plt.rcParams.update({
    'font.size': 16,            # Tamaño de fuente general más grande
    'axes.titlesize': 20,       # Títulos de los ejes
    'axes.labelsize': 18,       # Etiquetas (X e Y)
    'xtick.labelsize': 14,      # Números en el eje X
    'ytick.labelsize': 14,      # Números en el eje Y
    'legend.fontsize': 16,      # Tamaño de la leyenda
    'figure.titlesize': 22,     # Título principal
    'lines.linewidth': 1.5      # Grosor de las líneas de trayectoria
})

def simular_y_graficar_jacobi():
    # Parámetros (según tu texto de la sección 3)
    T = 15.0
    dt = 5e-3
    steps = int(T / dt)
    t = np.linspace(0, T, steps)
    N_sim = 50
    theta, mu, sigma, a = 2.0, 0.4, 0.3, 1.0  # a=1.0 como ejemplo
    p0, s0 = 0.5, 0.99

    # Matrices para guardar resultados
    Pt = np.zeros((steps, N_sim))
    St = np.zeros((steps, N_sim))
    Pt[0, :] = p0
    St[0, :] = s0

    # Simulación Euler-Maruyama
    for i in range(1, steps):
        dW = np.random.normal(0, np.sqrt(dt), N_sim)
        # Proceso de Jacobi
        drift = theta * (mu - Pt[i-1, :]) * dt
        diffusion = sigma * np.sqrt(np.maximum(Pt[i-1, :] * (a - Pt[i-1, :]), 0)) * dW
        Pt[i, :] = Pt[i-1, :] + drift + diffusion

        # Ecuación SI (usando la solución exacta del paper para mayor precisión)
        # H_t es la integral de beta_r dr. Aquí beta_r = P_r (phi=1)
        # Calculamos la integral acumulada usando la regla del trapecio
        integral_P = np.trapz(Pt[:i+1, :], dx=dt, axis=0)
        St[i, :] = 1 / (1 + ((1/s0) - 1) * np.exp(integral_P))

    # --- CREACIÓN DE LA FIGURA ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Panel Superior: P_t
    for j in range(N_sim):
        ax1.plot(t, Pt[:, j], color='steelblue', alpha=0.4)
    ax1.axhline(y=a, color='red', linestyle='--', label=f'Upper bound $a={a}$')
    ax1.set_ylabel(r'Intensity $P_t^J$', fontweight='bold')
    ax1.set_title('Stochastic Transmission Intensity', pad=20)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Panel Inferior: 1 - S_t
    for j in range(N_sim):
        ax2.plot(t, 1 - St[:, j], color='darkorange', alpha=0.4)
    ax2.set_ylabel(r'Infected Fraction $1-S_t$', fontweight='bold')
    ax2.set_xlabel('Time $t$', fontweight='bold')
    ax2.set_title('Epidemic Evolution (Infected Population)', pad=20)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()

    # Guardar en alta resolución
    plt.savefig('Jacobi_Baseline_Corrected.png', dpi=300, bbox_inches='tight')
    print("Figura guardada como 'Jacobi_Baseline_Corrected.png'")
    plt.show()

# Ejecutar
simular_y_graficar_jacobi()