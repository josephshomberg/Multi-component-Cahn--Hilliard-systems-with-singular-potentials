# Multi-component Cahn–Hilliard simulations with Tsallis entropy

Python simulations accompanying **“Multi-component Cahn--Hilliard systems with singular potentials: numerical results and cascading phenomena”** by C. G. Gal, M. Grasselli, A. Poiatti, and J. L. Shomberg. The experiments compare ternary (`N=3`) and five-component (`N=5`) phase separation as the Tsallis parameter varies over `q = 0.8, 0.9, 1.0, 1.1, 1.2`.

The numerical method uses a first-order, semi-implicit convex–concave splitting: the discrete chemical potential treats the gradient and entropy contributions implicitly and the interaction contribution explicitly. The five-component program solves the nonlinear system by an under-relaxed Picard iteration. Its images and energy differences illustrate successive episodes of rapid energy dissipation; they do not by themselves establish convergence to a unique equilibrium.

## Files

| File | Purpose |
| --- | --- |
| `N=3_CHE_Eyre_q=0.8.py` | Original ternary simulation; the value of `tsallis` is edited inside the file for each run. |
| `N=5_CHE_Eyre_q=0.8.py` | Five-component simulator with command-line parameters, shared initial-data support, diagnostics, images, and CSV output. Despite its filename, `--tsallis` selects any supported positive `q`. |
| `Energy_plots_from_file.py` | Combines five `energy[n].csv` files into an energy plot. |
| `Energy_difference_plots_from_file.py` | Combines the same files into an energy-difference plot. |

## Requirements

Python 3 with NumPy and Matplotlib:

```bash
python -m pip install numpy matplotlib
```

Plots use Matplotlib's noninteractive `Agg` backend in the five-component simulator. A full run writes many figures and CSV rows; use a short run first to check the installation and output locations.

## Model and experiment settings

The concentration vector satisfies `u_i > 0` initially and `sum_i u_i = 1` at every grid point. In the five-component experiment the target mean composition is `(0.10, 0.15, 0.20, 0.25, 0.30)` for red, green, blue, yellow, and magenta, respectively. Independent uniform perturbations in `(-0.03, 0.03)` are added and the five fields are normalized pointwise. The resulting spatial means are *near*, not necessarily exactly equal to, the target vector.

The interaction matrix and mobility for `N=5` are

```text
A = -theta_c * (ones(5,5) - I_5)
alpha = [2 / (5 + sqrt(5))] *
        [[ 1, -1,  0,  0,  0],
         [-1,  2, -1,  0,  0],
         [ 0, -1,  2, -1,  0],
         [ 0,  0, -1,  2, -1],
         [ 0,  0,  0, -1,  1]]
```

The mobility is symmetric, positive semidefinite, has zero row sums, and has spectral norm one. The `N=5` implementation computes this normalized path-graph matrix rather than hard-coding it. For `q=1`, its entropy density is `u log(u)`; for other positive `q`, it uses `u (u^(1-q) - 1)/(1-q)`.

| Setting | `N=3` script | Paper's `N=5` run |
| --- | --- | --- |
| Domain and grid | `[-1,1]^2`, `64 × 64` arrays | `[-1,1]^2`, `64 × 64` arrays |
| Steps and time step | 50,000; `dt=0.006` | 100,000; `dt=0.0015` |
| Final simulated time | 300 | 150 |
| `theta`, `theta_c` | 1, 3.4 | 1, 4.65 |
| `gamma`, `kappa` | `1.3 epsilon^2`, `epsilon^2` | `1.3 epsilon^2`, `epsilon^2` |
| `epsilon` | 0.022360679774998 | 0.022360679774998 |
| Initial composition | approximately 25% red, 25% green, 50% blue, read from CSV | target 10%, 15%, 20%, 25%, 30%; generated or read from CSV |
| Nonlinear iteration | script-defined implicit iteration | Picard iteration with `omega=0.25`, `IMAX=1000`, tolerance `1e-8` for the paper run |

These are two different experiments. Differences in minimum depth or iteration number between `N=3` and `N=5` cannot be attributed solely to component count.

## Run the five-component experiment

The defaults in the `N=5` script are convenient for experimentation **but differ from the paper's settings**. Specify the parameters shown below to reproduce its intended configuration. Run from a directory containing the scripts:

```bash
mkdir -p 'N5_runs/q=0.8'
python "N=5_CHE_Eyre_q=0.8.py" \
  --steps 100000 --grid 64 --dt 0.0015 \
  --tsallis 0.8 --theta-critical 4.65 \
  --initial-dir N5_shared_initial \
  --output-dir 'N5_runs/q=0.8/data' \
  --imax 1000 --implicit-tol 1e-8 --relaxation 0.25 \
  > 'N5_runs/q=0.8/run.log'
```

On its first run the simulator creates the five initial-data CSVs in `N5_shared_initial` if they are absent. **Reuse that directory for every `q`** so the only varied parameter is the entropy parameter. Run the following sweep sequentially; the first run creates the common initial condition:

```bash
for q in 0.8 0.9 1.0 1.1 1.2; do
  mkdir -p "N5_runs/q=$q"
  python "N=5_CHE_Eyre_q=0.8.py" \
    --steps 100000 --grid 64 --dt 0.0015 \
    --tsallis "$q" --theta-critical 4.65 \
    --initial-dir N5_shared_initial \
    --output-dir "N5_runs/q=$q/data" \
    --imax 1000 --implicit-tol 1e-8 --relaxation 0.25 \
    > "N5_runs/q=$q/run.log"
done
```

For a quick check, substitute `--steps 2 --grid 8 --pictures-every 2` and use separate temporary initial and output directories. Do not reuse 8 × 8 initial CSVs for a 64 × 64 run.

The five-component output directory contains `energy[n].csv`, `rel_error[n].csv`, `final_state.npz`, `run_summary.txt`, and `images/`. Image subdirectories include the five component maps (`0red` through `4magenta`), `composite_actual` (an RGB blend of the **actual concentration values**), `dominant_phase` (largest component at each pixel), and `order_parameter_panel`. The energy CSV columns are `e1, delta_e1, e2, delta_e2, e3, delta_e3, total_energy, delta_total_energy`. In particular, columns 7 and 8 in ordinary one-based counting are total energy and its successive difference. Inspect `run_summary.txt` after every run: the simulator can stop early on solver failure or an energy increase greater than its configured tolerance.

## Run the original ternary script

The ternary script expects `data/initial_data0red.csv`, `data/initial_data1green.csv`, and `data/initial_data2blue.csv` in its **current working directory**. Each must be a `64 × 64` CSV field; the three positive fields must sum to one pointwise. The script does not generate these inputs on its active code path. It also expects its output folders to exist.

For each `q` run, use a separate working directory, copy the script and the **same three initial CSVs**, then set the `tsallis` constant near the top of that copy to the desired value. Before running, create the folders:

```bash
mkdir -p data images/0red images/1green images/2blue
python "N=3_CHE_Eyre_q=0.8.py" > run.log
```

It writes `data/energy[n].csv`, `data/rel_error[n].csv`, and images under `images/`. Unlike the `N=5` code, this legacy implementation stores full time histories in memory; its default 50,000-step run requires substantial RAM. It writes the energy CSV without a header.

## Combine the energy curves

Both plotting scripts expect these paths **relative to the directory from which they are run**:

```text
q=0.8/data/energy[n].csv
q=0.9/data/energy[n].csv
q=1.0/data/energy[n].csv
q=1.1/data/energy[n].csv
q=1.2/data/energy[n].csv
```

Put copies of the plotting scripts in the parent of those `q=...` directories, or adjust the five paths at the top of each script. `Energy_plots_from_file.py` reads zero-based CSV column `6` (total energy) and saves `Tsallis_energy.png`; `Energy_difference_plots_from_file.py` reads column `7` (successive energy change) and saves `Tsallis_energy_difference.png`.

**N=5 plotting adjustment:** the N=5 energy CSV has a header, but the supplied plotting scripts call `float()` on every row. Before plotting an N=5 sweep, change `row_begin = 0` to `row_begin = 2` in both plotting scripts. Their row counter starts at 1 and increments *before* testing the row, so this skips exactly the header. The difference plot also fixes its vertical range with `ax.set_ylim(-0.0000145, 0.0000005)`; remove or adjust that line if it clips a run. Its `row_end` variable is presently unused. N=3 CSVs have no header and retain `row_begin = 0`.

## Interpretation and citation

The figures compare five runs from a fixed initial realization within each component count. A negative local minimum in `delta_total_energy` records a period of especially rapid discrete energy decrease. The paper reports two pronounced episodes for `q=0.8` in the five-component case and qualitatively stronger cascading for values below one. The observed curves, images, conserved masses, and separation diagnostics should be read alongside the paper's analysis; these plots alone are not convergence proofs.

If you use these scripts or figures, cite the accompanying paper by Gal, Grasselli, Poiatti, and Shomberg. Replace this sentence with the journal citation and DOI when available.
