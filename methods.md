"""
    def los_probability :

    Compute the probability of a Line-of-Sight (LoS) connection between a
    drone-BS at altitude h (m) and a ground receiver at horizontal distance
    r (m).

    Formula (Eq. 1 in paper):
        P(LoS) = 1 / (1 + a * exp(−b * (θ_deg − a)))

    where θ_deg = (180/π) * arctan(h/r)  is the elevation angle in degrees.

    Parameters
    h : float  — drone altitude (m)
    r : float  — horizontal distance to receiver (m)
    a, b : float — environment-specific constants

    Returns
    float in [0, 1]
"""

"""
    def path_loss_db :

    Mean air-to-ground path loss in dB (Eq. 2 in paper).

        PL(dB) = 20·log10(4π·fc·d / c)
                 + P(LoS)·η_LoS  +  P(NLoS)·η_NLoS

    The first term is free-space path loss; the second term is the
    environment-dependent additional loss weighted by LoS/NLoS probability.

    Parameters
    h  : float — drone altitude (m)
    r  : float — horizontal distance (m)
    fc : float — carrier frequency (Hz)
    eta_los, eta_nlos : float — additional losses (dB)

    Returns
    float — path loss in dB
"""

""" 
    def path_loss_linear :

    Return path loss as a dimensionless linear ratio (not dB).
"""

 """
    def plot_figure2 :
    Figure 2: Path loss vs. drone altitude for two fixed horizontal distances
    r = 200 m and r = 500 m.  Carrier frequency = 2 GHz, urban environment.
"""


"""
    def estimate_num_drones :
    Estimate the initial number of drone-BSs needed based on capacity alone.

    Steps (Section II-B, Eq. 3–4):
        1. C_BS = B × η   (drone-BS capacity)
        2. N_UBS = floor(C_BS / R)   (max users per drone-BS)
        3. N_BS  = ceil(N_U / N_UBS) (minimum drone-BSs needed)

    Parameters
    n_users      : total number of users in the area
    bandwidth    : drone-BS bandwidth (Hz)
    spectral_eff : average spectral efficiency (bps/Hz)
    rate_target  : per-user target download rate (bps)

    Returns
    (n_bs, n_ubs) : (initial drone count, max users per drone)
"""

"""
    def generate_scenario1 :
    
    Scenario I — Non-uniform uniform distribution.

    Layout:
      • Left  half (x < 5000 m): 20 % of users, uniform
      • Right half (x ≥ 5000 m): 80 % of users, uniform

    Parameters
    n_users : total number of users
    area    : side length of the square area (m)
    seed    : random seed for reproducibility

    Returns
    users : ndarray of shape (n_users, 2)  — (x, y) positions in metres
"""

"""
    def generate_scenario2 :
    
    Scenario II — Gaussian centre + uniform remainder.

    Layout:
      • Central cluster: 40 % of users, Gaussian (μ = centre, σ = 1000 m)
      • Remaining area : 60 % of users, uniform across the full area

    Parameters
    n_users : total number of users
    area    : side length of the square area (m)
    seed    : random seed for reproducibility

    Returns
    users : ndarray of shape (n_users, 2)  — (x, y) positions in metres
"""

"""
    def compute_sinr_matrix :
    
    Compute the SINR (linear) for every (user, drone) pair.

    For user i served by drone j:
        SINR_ij = (P_tx / PL_ij) / (Σ_{k≠j} P_tx/PL_ik  + N₀)

    Parameters
    users  : (N_U, 2)  — user (x, y) positions (m)
    drones : (N_BS, 3) — drone (x, y, h) positions (m)
    p_tx   : transmit power per drone (W)

    Returns
    sinr : (N_U, N_BS) array of linear SINR values
"""

"""
    def assign_users_to_drones :
    
    Each user connects to the drone giving the highest SINR (best SINR policy).

    Parameters
    sinr_matrix : (N_U, N_BS) array of linear SINR

    Returns
    assignment : (N_U,) array of drone indices (0-based)
"""

"""
    def count_covered_users :
    
    Count users whose best SINR exceeds the threshold γ_th.

    A user is 'covered' if  max_j(SINR_ij) ≥ γ_th.

    Parameters
    sinr_matrix : (N_U, N_BS) linear SINR matrix
    threshold   : linear SINR threshold

    Returns
    int — number of covered users
"""

"""
    def compute_spectral_efficiency :
    
    Compute average spectral efficiency η̄ = 1 / E{1/η_i}
    where η_i = log2(1 + SINR_i)  for user i (Shannon capacity).

    This corresponds to the harmonic mean of the per-user spectral efficiency,
    as required by constraint (11) in the paper.

    Parameters
    sinr_matrix : (N_U, N_BS) linear SINR

    Returns
    float — harmonic mean spectral efficiency (bps/Hz)
"""

"""
    def compute_rho :
    
    Compute ρ_{j,k} = a_{j,k} / A_j  for each (drone j, subarea k) pair.

    ρ_{j,k} is the fraction of drone j's circular coverage footprint that
    falls inside subarea k.  Estimated by Monte-Carlo sampling.

    Parameters
    drone_xy        : (N_BS, 2) drone x-y positions
    subareas        : list of dicts, each with keys:
                        'x_min', 'x_max', 'y_min', 'y_max',
                        'density' (users/m²), 'area' (m²)
    coverage_radius : radius of each drone's ground footprint (m)

    Returns
    rho : (N_BS, N_subareas) array
"""

"""
    def check_capacity_constraint :
    Check Eq. (7): Σ_j  N_UBS · ρ_{j,k} ≥ D_k · S_k  for all k.

    This ensures that the combined capacity (in user-slots) projected onto
    each subarea is sufficient to serve the users residing there.

    Parameters
    rho      : (N_BS, N_subareas) overlap fractions
    n_ubs    : max users per drone (N_UBS)
    subareas : list of subarea dicts with 'density' and 'area' keys

    Returns
    bool — True if all subarea capacity constraints are satisfied
"""

"""
    def utility_U1 :
    U1 (Eq. 14) — Capacity violation penalty.

    U1 = Σ_k Σ_j { N_UBS · ρ_{j,k} − D_k·S_k }

    A positive U1 means capacity is NOT yet satisfied.
    U1 ≤ 0 means the capacity constraint is met for all subareas.

    The PSO minimises U1 in Phase 1.
"""

"""
    def utility_U3 :
    U3 (Eq. 16) — Spectral efficiency constraint (activated after ζ·N_U users covered).

    U3 = −N_U + (η_target − 1/E{1/η_i})

    Returns 0 if capacity constraint is violated.
    The PSO minimises U3, pushing the spectral efficiency above η_target.
"""

"""
    Particle Swarm Optimisation for 3D drone-BS placement.

    Each particle encodes the (x, y, h) positions of all N_BS drones
    as a flat vector of length 3·N_BS.

    The algorithm runs in three sequential phases (U1 → U2 → U3) as
    described in Section III of the paper.

    Parameters
    n_bs           : number of drone-BSs to place
    users          : (N_U, 2) user positions
    subareas       : list of subarea dicts
    n_ubs          : max users per drone
    area           : side length of the square area (m)
    h_min, h_max   : altitude bounds (m)
    n_particles    : swarm size (L)
    max_iter       : maximum PSO iterations
"""

"""def _decode : Reshape a flat particle vector into an (N_BS, 3) drone array"""

"""
    def _evaluate :
    Evaluate the utility function for a given particle and PSO phase.

    Phase 1 → U1  (capacity)
    Phase 2 → U2  (coverage)
    Phase 3 → U3  (spectral efficiency)
"""

""" def _clip_position : Clip particle positions to stay within the valid search space."""

"""
    def optimise :
    Run the PSO algorithm (Algorithm 1 in the paper).

    Phase transitions:
        • Start in Phase 1 (U1)
        • Switch to Phase 2 (U2) when global best U1 ≤ 0
        • Switch to Phase 3 (U3) when ≥ ζ·N_U users are covered

    Returns

    best_drones : (N_BS, 3) array of optimised drone positions
"""

"""
    def remove_redundant_drones :
    Iteratively remove drone-BSs that are redundant (Section III, last paragraph).

    Algorithm:
      1. For each drone, temporarily remove it and check if constraints hold.
      2. If multiple drones can be removed, remove the one that causes the
         fewest users to be disconnected (least harmful removal first).
      3. Repeat until no more redundant drones exist.

    Parameters
    drones          : (N_BS, 3) current drone positions
    users           : (N_U, 2) user positions
    subareas        : list of subarea dicts
    n_ubs           : max users per drone
    coverage_radius : drone footprint radius (m)

    Returns
    pruned_drones : (N_BS_final, 3) array after removing redundant drones
"""

"""
    def _voronoi_finite_polygons :
    Reconstruct infinite Voronoi regions into finite ones for plotting.
    Adapted from the standard scipy Voronoi plotting helper.
"""

"""
    def plot_2d_placement :
    Plot 2D map: user positions (blue dots), drone positions (coloured squares),
    and Voronoi tessellation (thin grey lines).  Reproduces Figs 3a and 6a.

    Parameters
    users      : (N_U, 2) user positions
    drones     : (N_BS, 3) final drone positions
    title      : plot title string
    filename   : output PNG file name
    boundary   : side length of the area (m)
    split_x    : x-coordinate used to colour-code drone-BSs by region
                 (None = all same colour)
"""

"""
    def plot_3d_placement :
    3D scatter plot of drone-BS positions. Reproduces Figs 3b and 6b.

    Parameters
    drones   : (N_BS, 3) drone (x, y, h) positions
    title    : plot title
    filename : output PNG file name
    split_x  : x boundary to colour-code two drone groups
"""

"""
    def plot_sinr_cdf : 
    Plot the empirical CDF of the per-user best SINR (dB). Reproduces Figs 4 and 7.

    Parameters
    sinr_matrix : (N_U, N_BS) linear SINR values
    title       : plot title
    filename    : output PNG file name
"""

"""
    def plot_convergence :
    Plot PSO utility function vs. iteration number. Reproduces Figs 5 and 8.

    Parameters
    history  : list of global-best utility values per iteration
    title    : plot title
    filename : output PNG file name
"""

"""
    def run_scenario :
    Full pipeline for one scenario:
      1. Estimate initial drone count
      2. Run PSO to find 3D placement
      3. Remove redundant drones
      4. Compute final metrics
      5. Generate all four plots for the scenario

    Parameters
    scenario_id : 1 or 2
    users       : (N_U, 2) user positions
    subareas    : list of subarea dicts describing user density regions
    verbose     : print progress messages

    Returns
    dict with keys: 'drones', 'n_covered', 'sinr_matrix', 'history'
"""

"""
    def main : 
    Main entry point. Runs all steps in order:
      0. Verify channel model (Fig. 2)
      1. Scenario I  — asymmetric uniform density (Figs 3–5)
      2. Scenario II — Gaussian centre + uniform  (Figs 6–8)
"""