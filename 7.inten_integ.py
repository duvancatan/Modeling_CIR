import numpy as np
import matplotlib.pyplot as plt

# Estilo para el paper
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'text.usetex': False})

def generate_high_volatility_ht_precise():
    # 1. Parámetros
    T = 100.0
    dt = 0.1  # Paso de tiempo para la integración
    steps = int(T / dt)
    mu = eta = 0.3
    theta = kappa = 0.5
    a = 0.5
    sigma_j = 1.5
    N = 10000  # Reducimos N a 10k para que la simulación de trayectorias sea rápida

    # 2. Calibración de sigma_cir
    var_j = (sigma_j**2 * mu * (a - mu)) / (2 * theta + a * sigma_j**2)
    sigma_cir = np.sqrt((var_j * 2 * kappa) / eta)

    # 3. Simulación de Trayectorias (Euler-Maruyama)
    h_jacobi = np.zeros(N)
    h_cir = np.zeros(N)

    # Condiciones iniciales en la media
    pj = np.full(N, mu)
    pc = np.full(N, eta)

    print(f"Simulating {N} paths... please wait.")
    for i in range(steps):
        dW = np.random.normal(0, np.sqrt(dt), N)

        # Jacobi
        dj = theta * (mu - pj) * dt + sigma_j * np.sqrt(np.maximum(pj * (a - pj), 0)) * dW
        pj = np.clip(pj + dj, 0, a) # Forzamos el soporte [0, a]

        # CIR
        dc = kappa * (eta - pc) * dt + sigma_cir * np.sqrt(np.maximum(pc, 0)) * dW
        pc = np.maximum(pc + dc, 0) # Forzamos positividad

        # Acumulamos la integral (H_T)
        h_jacobi += pj * dt
        h_cir += pc * dt

    # 4. Gráfica
    plt.figure(figsize=(10, 7))
    plt.hist(h_jacobi, bins=80, density=True, alpha=0.5, color='blue', label='Jacobi (Bounded)')
    plt.hist(h_cir, bins=80, density=True, alpha=0.5, color='red', label='CIR (Unbounded)')

    plt.axvline(x=a*T, color='black', linestyle='--', label=r'Structural Limit $a \cdot T$')

    plt.title(f'Integrated Intensity $H_{{100}}$ Distribution ($\sigma_J={sigma_j}$)')
    plt.xlabel('Cumulative Intensity ($H_T$)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)

    # El zoom es clave: mostramos un poco más allá del límite de Jacobi
    plt.xlim(mu*T - 15, a*T + 15)

    plt.tight_layout()
    plt.savefig('AccumulateHT.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    generate_high_volatility_ht_precise()