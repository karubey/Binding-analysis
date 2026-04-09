import numpy as np
import matplotlib.pyplot as plt
import pymbar
import os

# =========================
# PARAMETERS
# =========================
BASE_DIR = "data/"

FRAMES = 10000
REPS = 20
RESIDUES = 342

CUTOFF = 1000
TEMP = 300

UNBOUND_REPS = [0, 1, 5, 10, 12, 13, 14, 19]

STEP = 10
WINDOW_SIZE = 50

# =========================
# CONSTANTS
# =========================
kB = 1.3806503 * 6.0221415 / 4184.0
beta = 1 / (kB * TEMP)


# =========================
# DATA LOADING FUNCTIONS
# =========================
def read_data(file_path):
    with open(file_path, "r") as f:
        next(f)
        return np.array([list(map(float, line.split()[1:])) for line in f])


def moving_average(data, window):
    return np.convolve(data, np.ones(window) / window, mode="valid")


def load_sorting_indices(file_path):
    sortinds = np.zeros((FRAMES, REPS), dtype=int)
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            row = line.split()
            for j in range(REPS):
                sortinds[i, j] = int(row[j + 1])
    return sortinds


def sort_trajectory(data, sortinds):
    sorted_data = np.zeros_like(data)
    for i in range(len(data)):
        for j in range(len(data[0])):
            sorted_data[i, j] = data[i][sortinds[i, j] - 1]
    return sorted_data


# =========================
# MBAR ANALYSIS
# =========================
def load_energy_matrix():
    u_kn = np.zeros((REPS, REPS, FRAMES - CUTOFF))

    for a in range(REPS):
        for b in range(REPS):
            file_path = os.path.join(
                BASE_DIR,
                f"energies/rep_{a+1:03d}/energy_{b+1}.txt"
            )

            with open(file_path, "r") as f:
                energies = [float(x) for x in f]

            u_kn[a, b, :] = energies[CUTOFF:FRAMES]

    return u_kn


def run_mbar(u_kn):
    N_k = np.full(REPS, FRAMES - CUTOFF, dtype=int)
    u_kln = u_kn * beta

    mbar = pymbar.MBAR(u_kln, N_k, verbose=True)
    results = mbar.compute_free_energy_differences()
    weights = mbar.weights()

    return results, weights


# =========================
# MAIN WORKFLOW
# =========================
def main():

    # -------------------------
    # MBAR PART
    # -------------------------
    print("Loading energy data...")
    u_kn = load_energy_matrix()

    print("Running MBAR...")
    results, weights = run_mbar(u_kn)

    # -------------------------
    # STRUCTURAL DATA
    # -------------------------
    print("Loading structural data...")

    rmsd = read_data(os.path.join(BASE_DIR, "rmsd.dat"))
    tip = read_data(os.path.join(BASE_DIR, "flap_distance.dat"))

    sortinds = load_sorting_indices(
        os.path.join(BASE_DIR, "repind_sort.0912")
    )

    armsd = sort_trajectory(rmsd, sortinds)
    atip = sort_trajectory(tip, sortinds)

    np.savetxt(os.path.join(BASE_DIR, "armsd_sorted.dat"), armsd, fmt="%.4f")
    np.savetxt(os.path.join(BASE_DIR, "atip_sorted.dat"), atip, fmt="%.4f")

    # -------------------------
    # RMSD SUBPLOTS
    # -------------------------
    fig, axes = plt.subplots(4, 5, figsize=(20, 18), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        ax.plot(np.arange(FRAMES) / FRAMES, armsd[:, i], ".", markersize=0.7)
        ax.tick_params(direction="in", top=True, right=True)
        ax.minorticks_on()

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "rmsd_subplots.png"))
    plt.close()

    # -------------------------
    # RMSD vs FLAP DISTANCE
    # -------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.tab20(np.linspace(0, 1, REPS))

    for i in UNBOUND_REPS:
        rmsd_i = armsd[:5000:STEP, i]
        flap_i = atip[:5000:STEP, i]

        smooth_rmsd = moving_average(rmsd_i, WINDOW_SIZE)
        smooth_flap = moving_average(flap_i, WINDOW_SIZE)

        n = min(len(smooth_rmsd), len(smooth_flap))

        ax.plot(
            smooth_rmsd[:n],
            smooth_flap[:n],
            ".",
            color=colors[i],
            alpha=0.7,
            markersize=2
        )

    ax.set_xlabel("RMSD (Å)")
    ax.set_ylabel("Flap distance (Å)")
    ax.set_xlim(1, 60)
    ax.set_ylim(8, 24)

    ax.tick_params(direction="in", top=True, right=True)
    ax.minorticks_on()

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "flap_vs_rmsd.png"))
    plt.show()

    print("Analysis complete.")


if __name__ == "__main__":
    main()
