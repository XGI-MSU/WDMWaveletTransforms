import matplotlib.pyplot as plt
import numpy as np

from WDMWaveletTransforms.modified_gaussian import phi_eval, phihat_eval

if __name__ == '__main__':
    # ============================================================
    # Compute phi and phihat
    # ============================================================
    x_plot = np.linspace(-5, 5, 3000)
    y_plot = np.linspace(-5, 5, 3000)

    phi = phi_eval(x_plot)
    phihat = phihat_eval(y_plot)

    print('max |Im phi|    =', np.max(np.abs(np.imag(phi))))
    print('max |Im phihat| =', np.max(np.abs(np.imag(phihat))))

    # ============================================================
    # Partition of unity check using phi
    #
    # Check:
    #   sum_k |phi(x + k)|^2 = 1
    # ============================================================

    x0 = np.linspace(0, 1, 2000, endpoint=False)
    partition_phi = np.zeros_like(x0, dtype=float)

    Kshift = 40

    for k in range(-Kshift, Kshift + 1):
        partition_phi += np.abs(phi_eval(x0 + k)) ** 2

    print('\nPartition of unity using phi:')
    print('  min =', partition_phi.min())
    print('  max =', partition_phi.max())
    print('  max error =', np.max(np.abs(partition_phi - 1.0)))

    # ============================================================
    # Equation 2.3-style orthogonality diagnostics using phi
    #
    # C_j(theta) =
    #   sum_l conj(phi(theta + l)) phi(theta + l + 2j)
    #
    # Expected:
    #   C_j(theta) = delta_{j0}
    #
    # D_j(theta) =
    #   sum_l (-1)^l conj(phi(theta + l)) phi(theta + l + 2j)
    #
    # Expected:
    #   D_j(theta) = 0
    # ============================================================

    theta = np.linspace(0, 2, 1000, endpoint=False)
    L_orth = 40
    j_values = range(-4, 5)

    print('\nOrthogonality checks using phi:')
    print('j      max|C_j-delta|       max|D_j|')

    for j in j_values:
        Cj = np.zeros_like(theta, dtype=complex)
        Dj = np.zeros_like(theta, dtype=complex)

        for ell in range(-L_orth, L_orth + 1):
            f0 = phi_eval(theta + ell)
            f1 = phi_eval(theta + ell + 2 * j)
            f2 = phi_eval(theta - ell + 2 * j + 1)

            Cj += np.conj(f0) * f1
            Dj += ((-1.0) ** ell) * np.conj(f0) * f2

        delta = 1.0 if j == 0 else 0.0

        C_err = np.max(np.abs(Cj - delta))
        D_err = np.max(np.abs(Dj))

        print(f'{j:+d}    {C_err:14.6e}    {D_err:14.6e}')

    # ============================================================
    # Plot phi and phihat
    # ============================================================

    plt.figure(figsize=(8, 4))
    plt.plot(x_plot, np.real(phi))
    plt.axhline(0, linewidth=0.5)
    plt.xlim(-5, 5)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\phi(x)$')
    plt.title(r'$\phi(x)$')
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(8, 4))
    plt.plot(y_plot, np.real(phihat))
    plt.axhline(0, linewidth=0.5)
    plt.xlim(-5, 5)
    plt.xlabel(r'$y$')
    plt.ylabel(r'$\widehat{\phi}(y)$')
    plt.title(r'$\widehat{\phi}(y)$')
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(8, 4))
    plt.plot(x0, partition_phi)
    plt.axhline(1.0, linestyle='--', linewidth=1)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\sum_k |\phi(x+k)|^2$')
    plt.title(r'Partition of unity check using $\phi(x)$')
    plt.grid(True)
    plt.tight_layout()
    ep = 1e-9
    plt.ylim([1 - ep, 1 + ep])
    plt.show()

    # ============================================================
    # Plot selected C_j curves
    # ============================================================

    plt.figure(figsize=(8, 4))

    for j in [-2, -1, 0, 1, 2]:
        Cj = np.zeros_like(theta, dtype=complex)

        for ell in range(-L_orth, L_orth + 1):
            Cj += np.conj(phi_eval(theta + ell)) * phi_eval(theta + ell + 2 * j)

        plt.plot(theta, np.real(Cj), label=rf'$j={j}$')

    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$C_j(\theta)$')
    plt.title(r'Equation 2.3 orthogonality sums using $\phi$')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # ============================================================
    # Optional: alternating sums D_j
    # ============================================================

    plt.figure(figsize=(8, 4))

    for j in [-2, -1, 0, 1, 2]:
        Dj = np.zeros_like(theta, dtype=complex)

        for ell in range(-L_orth, L_orth + 1):
            Dj += ((-1.0) ** ell) * np.conj(phi_eval(theta + ell)) * phi_eval(theta + ell + 2 * j)

        plt.plot(theta, np.real(Dj), label=rf'$j={j}$')

    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$D_j(\theta)$')
    plt.title(r'Alternating orthogonality sums using $\phi$')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()
