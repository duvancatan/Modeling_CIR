
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# --- STYLE SETTINGS FOR THE PAPER ---
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 11,
    'lines.linewidth': 2.5,
    'text.usetex': False  # Disabling external TeX to avoid system dependencies
})

def plot_chernoff_transition_english():
    # 1. CIR Model Parameters
    kappa = 2.0
    sigma = 0.3
    eta = 0.4
    p0 = 0.5
    t = 1.0

    # Singularity lambda_c (where the MGF blows up)
    lambda_c = (kappa**2) / (2 * sigma**2)

    # Expected value of H_1 (Reference for the trivial bound)
    mean_H = eta*t + (p0 - eta)*(1 - np.exp(-kappa*t))/kappa

    # 2. M values for analysis (The transition zoom)
    Ms = [0.3, 0.5, 0.55, 0.6, 0.65, 0.7]

    # Lambda range (from near 0 to near lambda_c)
    lambdas = np.linspace(0.001, lambda_c - 0.05, 500)

    # 3. Log-MGF function: Lambda_t(lambda) = A(t, lam) + B(t, lam)*p0
    def get_log_mgf(lam_val):
        gamma = np.sqrt(kappa**2 - 2 * sigma**2 * lam_val)
        exp_gt = np.exp(gamma * t)
        den = (gamma + kappa) * (exp_gt - 1) + 2 * gamma
        B = (2 * lam_val * (exp_gt - 1)) / den
        term_ln = (2 * gamma * np.exp((gamma + kappa) * t / 2)) / den
        A = (2 * kappa * eta / sigma**2) * np.log(term_ln)
        return A + B * p0

    log_mgf_vals = get_log_mgf(lambdas)

    # 4. Figure Creation
    plt.figure(figsize=(12, 8))

    # Color map for the different M curves
    colors = cm.plasma(np.linspace(0, 0.8, len(Ms)))

    for i, M in enumerate(Ms):
        f_lambda = np.exp(-lambdas * M + log_mgf_vals)

        if M <= mean_H:
            # Trivial case: M < mean, bound is >= 1
            plt.plot(lambdas, f_lambda,
                     label=f'M = {M} (Trivial: bound $\geq$ 1)',
                     color='gray', linestyle='--', alpha=0.6)
        else:
            # Optimal case: M > mean
            min_idx = np.argmin(f_lambda)
            min_f = f_lambda[min_idx]
            opt_lambda = lambdas[min_idx]

            # Label with the probability bound value
            label_text = r'M = ' + f'{M}' + r' (Prob. Bound $\approx$ ' + f'{min_f:.3f}' + r')'

            plt.plot(lambdas, f_lambda, label=label_text, color=colors[i])

            # Mark the minimum (optimal lambda*)
            plt.plot(opt_lambda, min_f, 'ko', markersize=6, zorder=5)

    # 5. Aesthetic details
    plt.axvline(x=lambda_c, color='red', linestyle=':', lw=2,
                label=r'Singularity $\lambda_c \approx$ ' + f'{lambda_c:.2f}')

    plt.yscale('log')
    plt.xlabel(r'Chernoff Parameter $\lambda$', fontsize=16)
    plt.ylabel(r'Risk Bound $f(\lambda)$', fontsize=16)
    plt.title(r'Probability Bounds $\mathbb{P}(H_t \geq M)$ via Chernoff (CIR Process)', pad=20)

    plt.legend(loc='lower left', frameon=True, shadow=True)
    plt.grid(True, axis='both', which="both", ls="-", alpha=0.2)

    plt.ylim(1e-2, 1.5)
    plt.xlim(0, lambda_c + 0.5)

    plt.tight_layout()
    plt.savefig('Chernoff_Transition_English.png', dpi=300)
    print("Figure successfully saved as 'Chernoff_Transition_English.png'")
    plt.show()

if __name__ == "__main__":
    plot_chernoff_transition_english()