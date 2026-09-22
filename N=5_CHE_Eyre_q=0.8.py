#!/usr/bin/env python3
###############################################################################
# First-order finite-difference solution of the 2D five-component Cahn--Hilliard
# equation using backward Euler, a fixed-point implicit solve, and a
# convex--concave splitting.
#
# This is the N=5 analogue of the supplied ternary program.  The numerical
# method, diagnostics, CSV output, stopping controls, and plotting cadence are
# retained.  The implementation uses two time levels rather than storing the
# entire 50,001-frame trajectory in RAM.
###############################################################################

"""
python N5_tCHE_Eyer_q=0.8.py \
  --steps 100000 \
  --dt 0.0015 \
  --tsallis 0.8 \
  --theta-critical 4.65 \
  --initial-dir N=5_initial_conditions \
  --output-dir N=5_thetac4.65how_q0.8 \
  --imax 1000 \
  --implicit-tol 1e-8 \
  --relaxation 0.25 > N5_tCHE_Eyre_q=0.8.txt
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np


NC = 5
COMPONENT_NAMES = ("red", "green", "blue", "yellow", "magenta")
COMPONENT_RGB = np.array(
    [
        [1.00, 0.00, 0.00],
        [0.00, 0.68, 0.12],
        [0.00, 0.30, 1.00],
        [1.00, 0.84, 0.00],
        [0.82, 0.00, 0.82],
    ],
    dtype=float,
)
DEFAULT_MEANS = np.array([0.10, 0.15, 0.20, 0.25, 0.30])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Five-component Tsallis Cahn--Hilliard simulation"
    )
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.0060)
    parser.add_argument("--pictures-every", type=int, default=100)
    parser.add_argument("--imax", type=int, default=500)
    parser.add_argument("--implicit-tol", type=float, default=1.0e-9)
    parser.add_argument(
        "--relaxation",
        type=float,
        default=0.5,
        help="Under-relaxation factor for the implicit Picard iteration (0 < omega <= 1).",
    )
    parser.add_argument("--tsallis", type=float, default=0.9)
    parser.add_argument(
        "--theta-critical",
        type=float,
        default=4.25,
        help=(
            "Interaction/quench strength theta_c. Use the same value for every "
            "q in a comparison sweep (recommended starting value: 4.25)."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("N=5_data"))
    parser.add_argument(
        "--initial-dir",
        type=Path,
        default=None,
        help="Directory containing initial_data0red.csv, ..., initial_data4magenta.csv",
    )
    parser.add_argument(
        "--generate-initial",
        action="store_true",
        help="Generate positive noisy initial data even if CSV files already exist",
    )
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def ted(q, x):
    if q <= 0.0:
        raise ValueError("The Tsallis parameter q must be positive.")
    if np.isclose(q, 1.0):
        return x * np.log(x)
    return x * (x ** (1.0 - q) - 1.0) / (1.0 - q)


def dted(q, x):
    if q <= 0.0:
        raise ValueError("The Tsallis parameter q must be positive.")
    if np.isclose(q, 1.0):
        return np.log(x) + 1.0
    return ((2.0 - q) * x ** (1.0 - q) - 1.0) / (1.0 - q)


def normalized_path_mobility(nc):
    """PSD path-graph mobility with nullspace span{(1,...,1)} and norm 1."""
    mobility = np.zeros((nc, nc), dtype=float)
    for j in range(nc - 1):
        mobility[j, j] += 1.0
        mobility[j + 1, j + 1] += 1.0
        mobility[j, j + 1] -= 1.0
        mobility[j + 1, j] -= 1.0
    return mobility / np.linalg.eigvalsh(mobility)[-1]


def laplacian_neumann(field, hx, hy):
    """Componentwise five-point Laplacian with zero-flux edge ghost values."""
    padded = np.pad(field, ((0, 0), (1, 1), (1, 1)), mode="edge")
    return (
        (padded[:, 2:, 1:-1] + padded[:, :-2, 1:-1] - 2.0 * field) / hx**2
        + (padded[:, 1:-1, 2:] + padded[:, 1:-1, :-2] - 2.0 * field) / hy**2
    )


def interaction_gradient(phi, theta_c):
    """Gradient of theta_c * sum_{i<j} phi_i phi_j."""
    return theta_c * (np.sum(phi, axis=0, keepdims=True) - phi)


def energy_parts(phi, q, theta, theta_c, gamma, hx, hy):
    h = hx * hy

    # This reproduces the gradient-energy quadrature used by the source code:
    # centered differences in the interior, one-sided differences on all four
    # boundaries, and the source code's additional four corner contributions.
    grad_sum = np.sum(
        ((phi[:, 2:, 1:-1] - phi[:, :-2, 1:-1]) / (2.0 * hx)) ** 2
    )
    grad_sum += np.sum(
        ((phi[:, 1:-1, 2:] - phi[:, 1:-1, :-2]) / (2.0 * hy)) ** 2
    )
    grad_sum += np.sum(((phi[:, :, -1] - phi[:, :, -2]) / hy) ** 2)
    grad_sum += np.sum(((phi[:, :, 1] - phi[:, :, 0]) / hy) ** 2)
    grad_sum += np.sum(((phi[:, -1, :] - phi[:, -2, :]) / hx) ** 2)
    grad_sum += np.sum(((phi[:, 1, :] - phi[:, 0, :]) / hx) ** 2)

    corners = ((0, 0, 1, 0, 0, 1), (-1, 0, -2, 0, -1, 1),
               (-1, -1, -2, -1, -1, -2), (0, -1, 1, -1, 0, -2))
    for x0, y0, x1, y1, x2, y2 in corners:
        grad_sum += np.sum(((phi[:, x1, y1] - phi[:, x0, y0]) / hx) ** 2)
        grad_sum += np.sum(((phi[:, x2, y2] - phi[:, x0, y0]) / hy) ** 2)

    e1 = 0.5 * gamma * h * grad_sum
    e2 = theta * h * np.sum(ted(q, phi))
    pair_sum = 0.5 * ((np.sum(phi, axis=0)) ** 2 - np.sum(phi**2, axis=0))
    e3 = theta_c * h * np.sum(pair_sum)
    return e1, e2, e3


def relative_error_l2(phi_new, phi_old, h):
    numerator = h * np.sum((phi_new - phi_old) ** 2)
    denominator = h * np.sum(phi_new**2)
    return numerator / denominator


def initial_filenames(directory):
    return [directory / f"initial_data{k}{name}.csv" for k, name in enumerate(COMPONENT_NAMES)]


def make_initial_data(grid, seed):
    rng = np.random.default_rng(seed)
    phi = DEFAULT_MEANS[:, None, None] + rng.uniform(-0.03, 0.03, (NC, grid, grid))
    phi = np.maximum(phi, 1.0e-8)
    phi /= np.sum(phi, axis=0, keepdims=True)
    return phi


def load_or_create_initial_data(directory, grid, seed, force_generate):
    directory.mkdir(parents=True, exist_ok=True)
    files = initial_filenames(directory)
    if force_generate or not all(path.exists() for path in files):
        phi = make_initial_data(grid, seed)
        for k, path in enumerate(files):
            np.savetxt(path, phi[k], delimiter=",")
        print("Generated five positive initial-data CSV files in", directory)
        return phi

    arrays = [np.loadtxt(path, delimiter=",") for path in files]
    phi = np.stack(arrays)
    if phi.shape != (NC, grid, grid):
        raise ValueError(
            f"Initial CSV shape is {phi.shape}; expected {(NC, grid, grid)}."
        )
    if np.any(phi <= 0.0):
        raise ValueError("Tsallis/logarithmic initial concentrations must be positive.")
    if not np.allclose(np.sum(phi, axis=0), 1.0, atol=1.0e-10, rtol=1.0e-10):
        raise ValueError("The five initial concentrations must sum to one at every pixel.")
    return phi


def prepare_output_tree(root):
    images = root / "images"
    component_dirs = []
    for k, name in enumerate(COMPONENT_NAMES):
        path = images / f"{k}{name}"
        path.mkdir(parents=True, exist_ok=True)
        component_dirs.append(path)
    composite_dir = images / "composite_actual"
    dominant_dir = images / "dominant_phase"
    panel_dir = images / "order_parameter_panel"
    for path in (composite_dir, dominant_dir, panel_dir):
        path.mkdir(parents=True, exist_ok=True)
    return component_dirs, composite_dir, dominant_dir, panel_dir


def concentration_rgb(phi):
    """Five-color concentration blend; uses every actual phi_i value."""
    weights = np.moveaxis(np.clip(phi, 0.0, None), 0, -1)
    weights /= np.maximum(np.sum(weights, axis=-1, keepdims=True), 1.0e-15)
    return np.clip(weights @ COMPONENT_RGB, 0.0, 1.0)


def save_order_parameter_plots(phi, frame, component_dirs, composite_dir,
                               dominant_dir, panel_dir):
    filename = f"phi0000{frame:05d}.png"
    cmaps = [
        LinearSegmentedColormap.from_list(f"phase{k}", [(1, 1, 1), rgb])
        for k, rgb in enumerate(COMPONENT_RGB)
    ]

    for k in range(NC):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(phi[k], cmap=cmaps[k], vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_axis_off()
        fig.savefig(component_dirs[k] / filename, bbox_inches="tight", pad_inches=0, dpi=200)
        plt.close(fig)

    # Correct replacement for the invalid five-channel imshow call: each RGB
    # pixel is the concentration-weighted blend of five fixed phase colors.
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(concentration_rgb(phi), interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(composite_dir / filename, bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close(fig)

    # A second, unambiguous categorical view records which phase is largest.
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.argmax(phi, axis=0), cmap=ListedColormap(COMPONENT_RGB),
              vmin=-0.5, vmax=NC - 0.5, interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(dominant_dir / filename, bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for k, ax in enumerate(axes.flat[:NC]):
        image = ax.imshow(phi[k], cmap=cmaps[k], vmin=0.0, vmax=1.0,
                          interpolation="nearest")
        ax.set_title(f"$\\phi_{k + 1}$ ({COMPONENT_NAMES[k]})")
        ax.set_axis_off()
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    axes.flat[-1].imshow(concentration_rgb(phi), interpolation="nearest")
    axes.flat[-1].set_title("actual five-phase concentration blend")
    axes.flat[-1].set_axis_off()
    fig.savefig(panel_dir / filename, bbox_inches="tight", dpi=200)
    plt.close(fig)


def save_diagnostic_plots(root, count, through, concentration_errors, energies,
                          extrema, mu_extrema, rel_error):
    suffix = f"{count:04d}.png"
    x0 = np.arange(through + 1)
    x1 = np.arange(1, through + 1)
    line_colors = COMPONENT_RGB

    fig, ax = plt.subplots()
    for k in range(NC):
        ax.plot(x0, concentration_errors[k, x0], color=line_colors[k], label=f"phi{k + 1}")
    ax.plot(x0, np.sum(concentration_errors[:, x0], axis=0), color="black", label="sum")
    ax.legend()
    fig.savefig(root / f"con_error{suffix}", bbox_inches="tight", dpi=180)
    plt.close(fig)

    energy_names = ("energy-e1", "energy-e2", "energy-e3", "energy-total")
    energy_colors = ("red", "green", "blue", "black")
    for series, name, color in zip(energies, energy_names, energy_colors):
        fig, ax = plt.subplots()
        ax.plot(x1, series[x1], color=color)
        fig.savefig(root / f"{name}{suffix}", bbox_inches="tight", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(x1, energies[3][x1] - energies[3][x1 - 1], color="black")
    fig.savefig(root / f"energy-difference{suffix}", bbox_inches="tight", dpi=180)
    plt.close(fig)

    for high, low, name in ((extrema[0], extrema[1], "max-min"),
                            (mu_extrema[0], mu_extrema[1], "mumax-mumin")):
        fig, ax = plt.subplots()
        ax.plot(x0, high[x0], color="blue", label="max")
        ax.plot(x0, low[x0], color="red", label="min")
        ax.legend()
        fig.savefig(root / f"{name}{suffix}", bbox_inches="tight", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(x1, rel_error[x1], color="black")
    fig.savefig(root / f"relative_error{suffix}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    if args.grid < 3:
        raise ValueError("--grid must be at least 3.")
    if args.steps < 1 or args.imax < 2 or args.pictures_every < 1:
        raise ValueError("steps and pictures-every must be positive; imax must exceed 1.")
    if not (0.0 < args.relaxation <= 1.0):
        raise ValueError("--relaxation must satisfy 0 < relaxation <= 1.")

    # Constants from the supplied program.
    theta = 1.0
    theta_c = args.theta_critical
    epsilon = 0.022360679774998
    kappa = 1.0 * epsilon**2
    gamma = 1.3 * epsilon**2
    pos_tol = 0.001
    lx = ly = 1.0
    hx = 2.0 * lx / args.grid
    hy = 2.0 * ly / args.grid
    h = hx * hy
    mobility = normalized_path_mobility(NC)

    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    initial_dir = args.initial_dir if args.initial_dir is not None else root
    component_dirs, composite_dir, dominant_dir, panel_dir = prepare_output_tree(root)

    print("Five-component normalized mobility matrix:\n", mobility)
    print("Mobility eigenvalues:", np.linalg.eigvalsh(mobility))
    print("Mobility row sums:", np.sum(mobility, axis=1))
    print("Tsallis q:", args.tsallis)
    print("Critical interaction theta_c:", theta_c)

    phi_old = load_or_create_initial_data(
        initial_dir, args.grid, args.seed, args.generate_initial
    )
    mu_old = (
        -gamma * laplacian_neumann(phi_old, hx, hy)
        + theta * dted(args.tsallis, phi_old)
        + interaction_gradient(phi_old, theta_c)
    )

    means = np.zeros((NC, args.steps + 1))
    concentration_errors = np.zeros((NC, args.steps + 1))
    means[:, 0] = np.mean(phi_old, axis=(1, 2))
    e1 = np.zeros(args.steps + 1)
    e2 = np.zeros(args.steps + 1)
    e3 = np.zeros(args.steps + 1)
    en = np.zeros(args.steps + 1)
    locmax = np.zeros(args.steps + 1)
    locmin = np.zeros(args.steps + 1)
    mumax = np.zeros(args.steps + 1)
    mumin = np.zeros(args.steps + 1)
    rel_e = np.zeros(args.steps + 1)

    e1[0], e2[0], e3[0] = energy_parts(
        phi_old, args.tsallis, theta, theta_c, gamma, hx, hy
    )
    en[0] = e1[0] + e2[0] + e3[0]
    locmax[0], locmin[0] = np.max(phi_old), np.min(phi_old)
    mumax[0], mumin[0] = np.max(mu_old), np.min(mu_old)

    energy_csv = root / "energy[n].csv"
    rel_csv = root / "rel_error[n].csv"
    with energy_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["e1", "delta_e1", "e2", "delta_e2", "e3", "delta_e3",
                         "total_energy", "delta_total_energy"])
        writer.writerow([e1[0], 0.0, e2[0], 0.0, e3[0], 0.0, en[0], 0.0])
    with rel_csv.open("w", newline="") as handle:
        csv.writer(handle).writerow(["relative_error_l2"])

    save_order_parameter_plots(
        phi_old, 0, component_dirs, composite_dir, dominant_dir, panel_dir
    )
    print("=" * 72)
    print("All pointwise sums = 1.0?", np.allclose(np.sum(phi_old, axis=0), 1.0))
    print("Average value of each INITIAL component:", *means[:, 0])
    print("Sum of concentrations:", np.sum(means[:, 0]))
    print("Minimum / maximum phi:", locmin[0], locmax[0])
    print("Minimum / maximum mu:", mumin[0], mumax[0])
    print("Initial energies:", e1[0], e2[0], e3[0], en[0])

    last_completed = 0
    stop_reason = "requested final step reached"
    for n in range(1, args.steps + 1):
        psi = phi_old.copy()
        mu_iter = mu_old.copy()
        concave_old = interaction_gradient(phi_old, theta_c)

        converged = False
        implicit_error = np.inf
        omega = args.relaxation
        for iteration in range(1, args.imax + 1):
            lap_mu = laplacian_neumann(mu_iter, hx, hy)

            # Undamped Picard candidate for the backward-Euler update.
            psi_raw = phi_old + args.dt * kappa * np.einsum(
                "ij,jxy->ixy", mobility, lap_mu
            )

            # Measure the residual of the actual fixed-point equation before
            # relaxation, then damp the update to remain robust near pure phases.
            implicit_error = np.max(np.abs(psi_raw - psi))
            psi_new = (1.0 - omega) * psi + omega * psi_raw

            if not np.all(np.isfinite(psi_new)) or np.any(psi_new <= 0.0):
                implicit_error = np.inf
                break

            mu_new = (
                -gamma * laplacian_neumann(psi_new, hx, hy)
                + theta * dted(args.tsallis, psi_new)
                + concave_old
            )
            psi, mu_iter = psi_new, mu_new

            if implicit_error < args.implicit_tol:
                converged = True
                break

        if not converged:
            stop_reason = (
                f"fixed-point failure at step {n}: IMAX={args.imax}, "
                f"last error={implicit_error:.6e}"
            )
            print(stop_reason)
            break

        phi_new = psi
        mu_new = mu_iter
        means[:, n] = np.mean(phi_new, axis=(1, 2))
        concentration_errors[:, n] = np.abs(means[:, n] - means[:, 0])
        e1[n], e2[n], e3[n] = energy_parts(
            phi_new, args.tsallis, theta, theta_c, gamma, hx, hy
        )
        en[n] = e1[n] + e2[n] + e3[n]
        locmax[n], locmin[n] = np.max(phi_new), np.min(phi_new)
        mumax[n], mumin[n] = np.max(mu_new), np.min(mu_new)
        rel_e[n] = relative_error_l2(phi_new, phi_old, h)

        with energy_csv.open("a", newline="") as handle:
            csv.writer(handle).writerow(
                [e1[n], e1[n] - e1[n - 1], e2[n], e2[n] - e2[n - 1],
                 e3[n], e3[n] - e3[n - 1], en[n], en[n] - en[n - 1]]
            )
        with rel_csv.open("a", newline="") as handle:
            csv.writer(handle).writerow([rel_e[n]])

        print("=" * 72)
        print(f"Finished iteration n = {n} of {args.steps}; fixed-point iterations = {iteration}")
        print("All pointwise sums = 1.0?", np.allclose(np.sum(phi_new, axis=0), 1.0))
        print("Average value of each component:", *means[:, n])
        print("Concentration errors:", *concentration_errors[:, n],
              "sum =", np.sum(concentration_errors[:, n]))
        print("Minimum / maximum phi:", locmin[n], locmax[n])
        print("Minimum / maximum mu:", mumin[n], mumax[n])
        print("Current energies:", e1[n], e2[n], e3[n], en[n])
        print("Energy slope:", en[n] - en[n - 1])
        print("Current relative error:", rel_e[n])

        last_completed = n
        if n % args.pictures_every == 0 or n == args.steps:
            save_order_parameter_plots(
                phi_new, n, component_dirs, composite_dir, dominant_dir, panel_dir
            )
            save_diagnostic_plots(
                root, 0, n, concentration_errors, (e1, e2, e3, en),
                (locmax, locmin), (mumax, mumin), rel_e
            )

        if en[n] - en[n - 1] > pos_tol:
            stop_reason = f"energy increased by more than {pos_tol} at step {n}"
            print(stop_reason)
            phi_old, mu_old = phi_new, mu_new
            break
        if math.isnan(en[n]):
            stop_reason = f"NaN energy at step {n}"
            print("Blow-up.", stop_reason)
            break

        phi_old, mu_old = phi_new, mu_new

    np.savez_compressed(
        root / "final_state.npz",
        phi=phi_old,
        mu=mu_old,
        step=last_completed,
        component_names=np.array(COMPONENT_NAMES),
        mobility=mobility,
        tsallis=args.tsallis,
        dt=args.dt,
        epsilon=epsilon,
        kappa=kappa,
        gamma=gamma,
        theta=theta,
        theta_c=theta_c,
    )
    with (root / "run_summary.txt").open("w") as handle:
        handle.write(f"last_completed_step = {last_completed}\n")
        handle.write(f"stop_reason = {stop_reason}\n")
        handle.write(f"final_total_energy = {en[last_completed]:.17g}\n")
        handle.write(f"final_component_means = {means[:, last_completed].tolist()}\n")
        handle.write(f"final_phi_min = {locmin[last_completed]:.17g}\n")
        handle.write(f"final_phi_max = {locmax[last_completed]:.17g}\n")
    print("=" * 72)
    print("Simulation finished:", stop_reason)
    print("Outputs written to", root)


if __name__ == "__main__":
    main()
