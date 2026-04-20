import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import Voronoi, voronoi_plot_2d
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (needed for 3D projection)
import warnings
warnings.filterwarnings("ignore")

# SECTION 0 — GLOBAL SIMULATION PARAMETERS (Table I in paper)

#  Urban environment constants (LoS probability model)
A_ENV   = 9.61      # 'a' constant for urban environment
B_ENV   = 0.16      # 'b' constant for urban environment
ETA_LOS  = 1.0      # Additional loss (dB) for LoS links
ETA_NLOS = 20.0     # Additional losses (dB) for NLoS links

#  System / RF parameters
FC      = 2e9       # Carrier frequency (Hz)  → 2 GHz
C_LIGHT = 3e8       # Speed of light (m/s)
BW      = 20e6      # Bandwidth (Hz)           → 20 MHz
ETA_SE  = 1.7       # Target average spectral efficiency (bps/Hz)
R_TARGET = 1e6      # Target download rate per user (bps) → 1 Mbps
P_TX    = 5.0       # Transmit power per drone-BS (W)
SINR_TH_DB = -7.0   # Minimum SINR threshold (dB) → −7 dB
SINR_TH    = 10 ** (SINR_TH_DB / 10.0)   # Linear SINR threshold
ZETA    = 0.95      # Fraction of users that must be covered (95 %)

#  Area / user parameters
AREA_SIDE   = 10_000    # Side length of square area (m) → 100 km²
N_USERS     = 1_000     # Total number of users
H_MIN       = 50.0      # Minimum drone altitude (m)
H_MAX       = 600.0     # Maximum drone altitude (m)   (paper constraint)

#  Noise floor
# Thermal noise: N₀ = kTB  (k=1.38e-23, T=290 K, B=20 MHz) → ~8e-14 W
NOISE_W = 1.38e-23 * 290 * BW   # ~8.004e-14 W

#  PSO hyper-parameters
PSO_PARTICLES   = 50    # Number of particles (L)
PSO_MAX_ITER    = 500   # Maximum iterations
PSO_INERTIA     = 0.7   # Inertia weight (φ)
PSO_C1          = 1.5   # Personal learning coefficient
PSO_C2          = 1.5   # Global  learning coefficient
PSO_VEL_MAX_XY  = AREA_SIDE * 0.1   # Max velocity in x/y plane
PSO_VEL_MAX_H   = (H_MAX - H_MIN) * 0.1  # Max velocity in altitude


# SECTION 1 — AIR-TO-GROUND CHANNEL MODEL

def los_probability(h: float, r: float,
                    a: float = A_ENV, b: float = B_ENV) -> float:
    if r == 0:
        return 1.0   # directly below → pure LoS
    theta_rad = np.arctan(h / r)
    theta_deg = np.degrees(theta_rad)
    p_los = 1.0 / (1.0 + a * np.exp(-b * (theta_deg - a)))
    return float(np.clip(p_los, 0.0, 1.0))


def path_loss_db(h: float, r: float,
                 fc: float = FC,
                 eta_los: float = ETA_LOS,
                 eta_nlos: float = ETA_NLOS) -> float:
    d = np.sqrt(h**2 + r**2)          # 3-D Euclidean distance
    fspl = 20.0 * np.log10(4 * np.pi * fc * d / C_LIGHT)   # free-space PL
    p_los  = los_probability(h, r)
    p_nlos = 1.0 - p_los
    pl = fspl + p_los * eta_los + p_nlos * eta_nlos
    return pl

def path_loss_linear(h: float, r: float) -> float:
    pl_db = path_loss_db(h, r)
    return 10.0 ** (pl_db / 10.0)


# Reproduce Figure 2

def plot_figure2():
    altitudes = np.linspace(10, 1200, 400)
    pl_200 = [path_loss_db(h, 200) for h in altitudes]
    pl_500 = [path_loss_db(h, 500) for h in altitudes]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(altitudes, pl_200, 'b-',  lw=2, label='r = 200 m')
    ax.plot(altitudes, pl_500, 'r--', lw=2, label='r = 500 m')
    ax.set_xlabel('Altitude (meters)', fontsize=12)
    ax.set_ylabel('Path Loss (dB)',    fontsize=12)
    ax.set_title('Fig. 2 — Path Loss vs. Drone Altitude\n'
                 '(Urban environment, fc = 2 GHz)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    ax.set_xlim([0, 1200])
    ax.set_ylim([85, 120])
    plt.tight_layout()
    plt.savefig(f"Results/{'figure2_pathloss_vs_altitude.png'}", dpi=150)
    plt.show()
    print("[✓] Figure 2 saved → figure2_pathloss_vs_altitude.png")


# SECTION 2 — CAPACITY ESTIMATION (initial number of drone-BSs)

def estimate_num_drones(n_users: int = N_USERS,
                        bandwidth: float = BW,
                        spectral_eff: float = ETA_SE,
                        rate_target: float = R_TARGET) -> tuple:

    c_bs  = bandwidth * spectral_eff          # total drone-BS capacity (bps)
    n_ubs = int(np.floor(c_bs / rate_target)) # max users a single drone can serve
    n_bs  = int(np.ceil(n_users / n_ubs))     # number of drones needed
    print(f"  C_BS   = {c_bs/1e6:.1f} Mbps")
    print(f"  N_UBS  = {n_ubs} users/drone")
    print(f"  N_BS   = {n_bs} drones (initial estimate)")
    return n_bs, n_ubs


# SECTION 3 — USER DISTRIBUTIONS FOR BOTH SCENARIOS

def generate_scenario1(n_users: int = N_USERS,
                        area: float = AREA_SIDE,
                        seed: int = 42) -> np.ndarray:

    rng = np.random.default_rng(seed)
    n_left  = int(0.20 * n_users)
    n_right = n_users - n_left

    # Left half: x ∈ [0, area/2],  y ∈ [0, area]
    left  = rng.uniform([0, 0], [area/2, area], size=(n_left,  2))
    # Right half: x ∈ [area/2, area], y ∈ [0, area]
    right = rng.uniform([area/2, 0], [area, area], size=(n_right, 2))

    users = np.vstack([left, right])
    return users


def generate_scenario2(n_users: int = N_USERS,
                        area: float = AREA_SIDE,
                        seed: int = 42) -> np.ndarray:

    rng    = np.random.default_rng(seed)
    centre = area / 2.0
    std    = 1000.0          # standard deviation (m) — stated in paper
    n_gauss  = int(0.40 * n_users)
    n_uniform = n_users - n_gauss

    # Gaussian cluster, clipped to the area boundary
    gauss = rng.normal(loc=centre, scale=std, size=(n_gauss, 2))
    gauss = np.clip(gauss, 0, area)

    # Uniform over the full area
    unif  = rng.uniform(0, area, size=(n_uniform, 2))

    users = np.vstack([gauss, unif])
    return users


# SECTION 4 — SINR & COVERAGE UTILITIES

def compute_sinr_matrix(users: np.ndarray,
                        drones: np.ndarray,
                        p_tx: float = P_TX) -> np.ndarray:
    
    n_u   = len(users)
    n_bs  = len(drones)
    sinr  = np.zeros((n_u, n_bs))

    # Pre-compute received power matrix  S[i, j] = P_tx / PL_ij  (linear)
    S = np.zeros((n_u, n_bs))
    for j, (dx, dy, dh) in enumerate(drones):
        for i, (ux, uy) in enumerate(users):
            r = np.sqrt((ux - dx)**2 + (uy - dy)**2)
            pl = path_loss_linear(dh, r)
            S[i, j] = p_tx / pl

    # Total received power from ALL drones at each user
    total_power = S.sum(axis=1, keepdims=True)   # shape (N_U, 1)

    # SINR for each (user, drone) pair
    for j in range(n_bs):
        interference = total_power[:, 0] - S[:, j]   # remove serving drone
        sinr[:, j] = S[:, j] / (interference + NOISE_W)

    return sinr


def assign_users_to_drones(sinr_matrix: np.ndarray) -> np.ndarray:

    return np.argmax(sinr_matrix, axis=1)


def count_covered_users(sinr_matrix: np.ndarray,
                         threshold: float = SINR_TH) -> int:

    best_sinr = sinr_matrix.max(axis=1)
    return int(np.sum(best_sinr >= threshold))


def compute_spectral_efficiency(sinr_matrix: np.ndarray) -> float:

    best_sinr = sinr_matrix.max(axis=1)
    se_per_user = np.log2(1.0 + best_sinr)    # Shannon rate (bps/Hz) per user
    # Avoid division by zero for users with near-zero SE
    se_per_user = np.maximum(se_per_user, 1e-12)
    harmonic_mean_se = 1.0 / np.mean(1.0 / se_per_user)
    return harmonic_mean_se


# SECTION 5 — CAPACITY CONSTRAINT (subarea overlap)

def compute_rho(drone_xy: np.ndarray,
                subareas: list,
                coverage_radius: float) -> np.ndarray:
    n_bs = len(drone_xy)
    n_sa = len(subareas)
    rho  = np.zeros((n_bs, n_sa))
    n_mc = 500   # Monte-Carlo samples per drone

    for j, (dx, dy) in enumerate(drone_xy):
        # Sample points uniformly inside the circular footprint
        angles = np.random.uniform(0, 2*np.pi, n_mc)
        radii  = coverage_radius * np.sqrt(np.random.uniform(0, 1, n_mc))
        pts_x  = dx + radii * np.cos(angles)
        pts_y  = dy + radii * np.sin(angles)

        for k, sa in enumerate(subareas):
            inside_circle   = np.ones(n_mc, dtype=bool)   # all inside circle
            inside_subarea  = ((pts_x >= sa['x_min']) & (pts_x < sa['x_max']) &
                               (pts_y >= sa['y_min']) & (pts_y < sa['y_max']))
            overlap_count   = np.sum(inside_circle & inside_subarea)
            rho[j, k]       = overlap_count / n_mc

    return rho


def check_capacity_constraint(rho: np.ndarray,
                               n_ubs: int,
                               subareas: list) -> bool:

    for k, sa in enumerate(subareas):
        supply = np.sum(rho[:, k]) * n_ubs          # Σ_j N_UBS·ρ_{j,k}
        demand = sa['density'] * sa['area']          # D_k · S_k  (users)
        if supply < demand:
            return False
    return True


# SECTION 6 — PSO UTILITY FUNCTIONS (U1, U2, U3)

def utility_U1(drones: np.ndarray,
               subareas: list,
               n_ubs: int,
               coverage_radius: float) -> float:

    drone_xy = drones[:, :2]
    rho = compute_rho(drone_xy, subareas, coverage_radius)
    u1 = 0.0
    for k, sa in enumerate(subareas):
        supply = np.sum(rho[:, k]) * n_ubs
        demand = sa['density'] * sa['area']
        u1 += supply - demand
    return -u1   # negate: PSO minimises, so we want the most negative value


def utility_U2(drones: np.ndarray,
               users: np.ndarray,
               subareas: list,
               n_ubs: int,
               coverage_radius: float) -> float:
    drone_xy = drones[:, :2]
    rho = compute_rho(drone_xy, subareas, coverage_radius)
    if not check_capacity_constraint(rho, n_ubs, subareas):
        return 0.0   # penalise: capacity must hold first
    sinr = compute_sinr_matrix(users, drones)
    n_covered = count_covered_users(sinr)
    return float(-n_covered)   # minimising gives maximum coverage


def utility_U3(drones: np.ndarray,
               users: np.ndarray,
               subareas: list,
               n_ubs: int,
               coverage_radius: float,
               n_u: int = N_USERS,
               zeta: float = ZETA,
               eta_target: float = ETA_SE) -> float:
    drone_xy = drones[:, :2]
    rho = compute_rho(drone_xy, subareas, coverage_radius)
    if not check_capacity_constraint(rho, n_ubs, subareas):
        return 0.0
    sinr = compute_sinr_matrix(users, drones)
    n_covered = count_covered_users(sinr)
    if n_covered < zeta * n_u:
        return float(-n_covered)   # not enough coverage yet
    eta_actual = compute_spectral_efficiency(sinr)
    return float(-n_u + (eta_target - eta_actual))


# SECTION 7 — PARTICLE SWARM OPTIMISATION (Algorithm 1)

class PSO:

    def __init__(self, n_bs, users, subareas, n_ubs,
                 area=AREA_SIDE, h_min=H_MIN, h_max=H_MAX,
                 n_particles=PSO_PARTICLES, max_iter=PSO_MAX_ITER):

        self.n_bs         = n_bs
        self.users        = users
        self.subareas     = subareas
        self.n_ubs        = n_ubs
        self.n_u          = len(users)
        self.area         = area
        self.h_min        = h_min
        self.h_max        = h_max
        self.n_particles  = n_particles
        self.max_iter     = max_iter
        self.dim          = 3 * n_bs     # dimensionality of each particle

        # Estimate a reasonable coverage radius for rho computation
        # (equal-area partition heuristic)
        self.cov_radius = np.sqrt(area**2 / (np.pi * n_bs)) * 1.5

        #  Initialise particle positions uniformly inside the area 
        # Positions are stored as (x₁,y₁,h₁, x₂,y₂,h₂, ..., x_N,y_N,h_N)
        rng = np.random.default_rng(0)
        self.pos = np.zeros((n_particles, self.dim))
        for l in range(n_particles):
            for j in range(n_bs):
                self.pos[l, 3*j]   = rng.uniform(0, area)       # x
                self.pos[l, 3*j+1] = rng.uniform(0, area)       # y
                self.pos[l, 3*j+2] = rng.uniform(h_min, h_max)  # h

        #  Initialise velocities to zero─
        self.vel = np.zeros((n_particles, self.dim))

        #  Personal and global bests─
        self.pbest_pos = self.pos.copy()
        self.pbest_val = np.full(n_particles, np.inf)
        self.gbest_pos = self.pos[0].copy()
        self.gbest_val = np.inf

        #  History for convergence plots─
        self.history = []

    #
    def _decode(self, particle: np.ndarray) -> np.ndarray:
        return particle.reshape(self.n_bs, 3)

    def _evaluate(self, particle: np.ndarray, phase: int) -> float:
        drones = self._decode(particle)
        if phase == 1:
            return utility_U1(drones, self.subareas, self.n_ubs, self.cov_radius)
        elif phase == 2:
            return utility_U2(drones, self.users, self.subareas,
                              self.n_ubs, self.cov_radius)
        else:
            return utility_U3(drones, self.users, self.subareas,
                              self.n_ubs, self.cov_radius)

    def _clip_position(self, pos: np.ndarray) -> np.ndarray:
        pos = pos.copy()
        for j in range(self.n_bs):
            pos[3*j]   = np.clip(pos[3*j],   0, self.area)
            pos[3*j+1] = np.clip(pos[3*j+1], 0, self.area)
            pos[3*j+2] = np.clip(pos[3*j+2], self.h_min, self.h_max)
        return pos

    def optimise(self, verbose: bool = True) -> np.ndarray:
        phase = 1
        rng   = np.random.default_rng(1)

        # Evaluate initial population
        for l in range(self.n_particles):
            val = self._evaluate(self.pos[l], phase)
            self.pbest_val[l] = val
            if val < self.gbest_val:
                self.gbest_val = val
                self.gbest_pos = self.pos[l].copy()
        
        def display_val(gbest_val, current_phase):
            if current_phase == 1:
                return -gbest_val   # U1 stored as negative penalty → flip back
                                    # so Phase 1 shows POSITIVE values on plot
            else:
                return gbest_val    # U2/U3 are already negative user counts

        self.history.append(self.gbest_val)

        for t in range(self.max_iter):
            for l in range(self.n_particles):
                #  Velocity update (Eq. 17)
                phi1 = rng.uniform(0, 1, self.dim)
                phi2 = rng.uniform(0, 1, self.dim)
                self.vel[l] = (PSO_INERTIA * self.vel[l]
                               + PSO_C1 * phi1 * (self.pbest_pos[l] - self.pos[l])
                               + PSO_C2 * phi2 * (self.gbest_pos     - self.pos[l]))

                # Clip velocities to prevent explosion
                for j in range(self.n_bs):
                    self.vel[l, 3*j]   = np.clip(self.vel[l, 3*j],
                                                  -PSO_VEL_MAX_XY, PSO_VEL_MAX_XY)
                    self.vel[l, 3*j+1] = np.clip(self.vel[l, 3*j+1],
                                                  -PSO_VEL_MAX_XY, PSO_VEL_MAX_XY)
                    self.vel[l, 3*j+2] = np.clip(self.vel[l, 3*j+2],
                                                  -PSO_VEL_MAX_H,  PSO_VEL_MAX_H)

                #  Position update (Eq. 18)
                self.pos[l] = self._clip_position(self.pos[l] + self.vel[l])

                #  Evaluate new position─
                val = self._evaluate(self.pos[l], phase)

                # Update personal best
                if val < self.pbest_val[l]:
                    self.pbest_val[l] = val
                    self.pbest_pos[l] = self.pos[l].copy()

                # Update global best
                if val < self.gbest_val:
                    self.gbest_val = val
                    self.gbest_pos = self.pos[l].copy()

            #  Phase transition logic
            if phase == 1 and self.gbest_val <= 0:
                phase = 2
                self.gbest_val = np.inf
                self.pbest_val[:] = np.inf
                if verbose:
                    print(f"  [PSO iter {t:4d}] Phase 1→2 (capacity satisfied)")

            elif phase == 2:
                drones    = self._decode(self.gbest_pos)
                sinr_mat  = compute_sinr_matrix(self.users, drones)
                n_covered = count_covered_users(sinr_mat)
                if n_covered >= ZETA * self.n_u:
                    phase = 3
                    self.gbest_val = np.inf
                    self.pbest_val[:] = np.inf
                    if verbose:
                        print(f"  [PSO iter {t:4d}] Phase 2→3 "
                              f"({n_covered}/{self.n_u} users covered)")

            #  Convergence check─
            self.history.append(display_val(self.gbest_val, phase))
            if self.gbest_val <= -self.n_u:
                if verbose:
                    print(f"  [PSO iter {t:4d}] Converged! All users served.")
                break

            if verbose and (t + 1) % 50 == 0:
                print(f"  [PSO iter {t+1:4d}] phase={phase}  "
                      f"best_util={self.gbest_val:.1f}  "
                      f"(target={-self.n_u})")

        return self._decode(self.gbest_pos)


# SECTION 8 — REDUNDANT DRONE REMOVAL

def remove_redundant_drones(drones: np.ndarray,
                             users: np.ndarray,
                             subareas: list,
                             n_ubs: int,
                             coverage_radius: float,
                             verbose: bool = True) -> np.ndarray:

    active = list(range(len(drones)))   # indices of currently active drones

    while True:
        removable = []   # list of (drone_index, n_users_disconnected)

        for idx in active:
            # Temporarily remove drone `idx`
            candidate = [i for i in active if i != idx]
            cand_drones = drones[candidate]

            if len(cand_drones) == 0:
                continue

            # Check capacity constraint
            rho = compute_rho(cand_drones[:, :2], subareas, coverage_radius)
            cap_ok = check_capacity_constraint(rho, n_ubs, subareas)
            if not cap_ok:
                continue

            # Check coverage constraint
            sinr    = compute_sinr_matrix(users, cand_drones)
            n_cov   = count_covered_users(sinr)
            if n_cov < ZETA * len(users):
                continue

            # How many users are disconnected compared to current solution?
            full_sinr = compute_sinr_matrix(users, drones[active])
            n_cov_full = count_covered_users(full_sinr)
            disconnected = n_cov_full - n_cov

            removable.append((idx, disconnected))

        if not removable:
            break   # no more drones can be removed

        # Remove the drone with the smallest impact (fewest disconnected users)
        removable.sort(key=lambda x: x[1])
        remove_idx = removable[0][0]
        active.remove(remove_idx)
        if verbose:
            print(f"  Removed drone {remove_idx}  "
                  f"(disconnects {removable[0][1]} users)  "
                  f"→ {len(active)} drones remaining")

    return drones[active]


# SECTION 9 — PLOTTING HELPERS

def _voronoi_finite_polygons(vor, radius=None):
    if radius is None:
        radius = AREA_SIDE * 2

    new_vertices = []
    center = vor.points.mean(axis=0)
    ridge_points = vor.ridge_points
    ridge_vertices = vor.ridge_vertices

    # Map from point index → list of ridge vertices
    point_to_ridge = {i: [] for i in range(len(vor.points))}
    for (p1, p2), verts in zip(ridge_points, ridge_vertices):
        point_to_ridge[p1].append((p2, verts))
        point_to_ridge[p2].append((p1, verts))

    finite_regions = []
    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if -1 not in region:
            finite_regions.append(vor.vertices[region])
        else:
            # For infinite regions, append the far points
            ridges = point_to_ridge[i]
            new_region = [v for v in region if v != -1]
            for p2, verts in ridges:
                if -1 in verts:
                    t = vor.points[p2] - vor.points[i]
                    t /= np.linalg.norm(t)
                    n = np.array([-t[1], t[0]])
                    midpoint = vor.points[[i, p2]].mean(axis=0)
                    direction = np.sign(np.dot(midpoint - center, n)) * n
                    far_point = vor.vertices[[v for v in verts if v != -1][0]] \
                                + direction * radius
                    new_region.append(len(vor.vertices) + len(new_vertices))
                    new_vertices.append(far_point)
            all_verts = np.vstack([vor.vertices, new_vertices]) \
                        if new_vertices else vor.vertices
            # Sort vertices by angle around centroid
            pts = all_verts[new_region]
            c   = pts.mean(axis=0)
            angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
            sorted_pts = pts[np.argsort(angles)]
            finite_regions.append(sorted_pts)

    return finite_regions


def plot_2d_placement(users, drones, title, filename,
                      boundary=AREA_SIDE,
                      left_label='Left-region drones',
                      right_label='Right-region drones',
                      split_x=None):

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, boundary)
    ax.set_ylim(0, boundary)
    ax.set_aspect('equal')

    #  Voronoi tessellation
    if len(drones) >= 3:
        try:
            vor = Voronoi(drones[:, :2])
            for simplex in vor.ridge_vertices:
                simplex = np.asarray(simplex)
                if np.all(simplex >= 0):
                    ax.plot(vor.vertices[simplex, 0],
                            vor.vertices[simplex, 1],
                            'k-', lw=0.5, alpha=0.3)
        except Exception:
            pass

    #  Users─
    ax.scatter(users[:, 0], users[:, 1],
               s=4, c='steelblue', alpha=0.5, label='Users', zorder=2)

    #  Drones
    if split_x is not None:
        left_mask  = drones[:, 0] <  split_x
        right_mask = drones[:, 0] >= split_x
        ax.scatter(drones[left_mask,  0], drones[left_mask,  1],
                   s=80, marker='s', c='green', zorder=5, label=left_label)
        ax.scatter(drones[right_mask, 0], drones[right_mask, 1],
                   s=80, marker='s', c='red',   zorder=5, label=right_label)
    else:
        ax.scatter(drones[:, 0], drones[:, 1],
                   s=80, marker='s', c='red', zorder=5, label='Drone-BSs')

    ax.set_xlabel('X (meters)', fontsize=11)
    ax.set_ylabel('Y (meters)', fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"Results/{filename}", dpi=150)
    plt.show()
    print(f"[✓] Saved → {filename}")


def plot_3d_placement(drones, title, filename, split_x=None,
                      boundary=AREA_SIDE):

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')

    if split_x is not None:
        left  = drones[drones[:, 0] <  split_x]
        right = drones[drones[:, 0] >= split_x]
        ax.scatter(left[:, 0],  left[:, 1],  left[:, 2],
                   s=60, marker='s', c='green', label='Left region', depthshade=True)
        ax.scatter(right[:, 0], right[:, 1], right[:, 2],
                   s=60, marker='s', c='red',   label='Right region', depthshade=True)
        ax.legend(fontsize=9)
    else:
        c = cm.viridis((drones[:, 2] - H_MIN) / (H_MAX - H_MIN))
        ax.scatter(drones[:, 0], drones[:, 1], drones[:, 2],
                   s=60, marker='s', c=c, depthshade=True)

    ax.set_xlabel('X (meters)',    fontsize=9)
    ax.set_ylabel('Y (meters)',    fontsize=9)
    ax.set_zlabel('Altitude (m)',  fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_xlim([0, boundary])
    ax.set_ylim([0, boundary])
    ax.set_zlim([H_MIN, H_MAX])
    plt.tight_layout()
    plt.savefig(f"Results/{filename}", dpi=150)
    plt.show()
    print(f"[✓] Saved → {filename}")


def plot_sinr_cdf(sinr_matrix, title, filename):

    best_sinr_db = 10.0 * np.log10(sinr_matrix.max(axis=1))
    sorted_sinr  = np.sort(best_sinr_db)
    cdf          = np.arange(1, len(sorted_sinr) + 1) / len(sorted_sinr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sorted_sinr, cdf, 'b-', lw=2)
    ax.axvline(SINR_TH_DB, color='r', linestyle='--', label=f'γ_th = {SINR_TH_DB} dB')
    ax.set_xlabel('SINR (dB)',  fontsize=12)
    ax.set_ylabel('CDF',        fontsize=12)
    ax.set_title(title,         fontsize=12)
    ax.set_xlim([-10, 25])
    ax.set_ylim([0, 1])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"Results/{filename}", dpi=150)
    plt.show()
    print(f"[✓] Saved → {filename}")


def plot_convergence(history, title, filename):

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(len(history)), history, 'b-', lw=1.5)
    ax.axhline(-N_USERS * ZETA, color='r', linestyle='--', alpha=0.6,
               label=f'−ζ·N_U = {-int(N_USERS*ZETA)}')
    ax.axhline(-N_USERS, color='g', linestyle='--', alpha=0.6,
               label=f'−N_U = {-N_USERS}')
    ax.set_xlabel('Iterations', fontsize=12)
    ax.set_ylabel('Utility Function', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"Results/{filename}", dpi=150)
    plt.show()
    print(f"[✓] Saved → {filename}")


# SECTION 10 — MAIN PIPELINE (runs both scenarios end-to-end)

def run_scenario(scenario_id: int,
                 users: np.ndarray,
                 subareas: list,
                 verbose: bool = True) -> dict:

    print(f"\n{'='*60}")
    print(f"  SCENARIO {scenario_id}")
    print(f"{'='*60}")

    #  Step 1: Estimate initial drone count
    print("\n[Step 1] Capacity-based initial drone count estimate")
    n_bs, n_ubs = estimate_num_drones()
    cov_radius  = np.sqrt(AREA_SIDE**2 / (np.pi * n_bs)) * 1.5
    print(f"  Coverage radius estimate: {cov_radius:.0f} m")

    #  Step 2: Run PSO─
    print(f"\n[Step 2] Running PSO with {PSO_PARTICLES} particles, "
          f"up to {PSO_MAX_ITER} iterations...")
    pso = PSO(n_bs=n_bs, users=users, subareas=subareas, n_ubs=n_ubs)
    best_drones = pso.optimise(verbose=verbose)
    print(f"  PSO finished. Best utility: {pso.gbest_val:.2f}")

    #  Step 3: Remove redundant drones
    print(f"\n[Step 3] Removing redundant drones...")
    final_drones = remove_redundant_drones(
        best_drones, users, subareas, n_ubs, cov_radius, verbose=verbose)
    print(f"  Final drone count: {len(final_drones)}")

    #  Step 4: Compute final metrics─
    sinr_mat  = compute_sinr_matrix(users, final_drones)
    n_covered = count_covered_users(sinr_mat)
    avg_se    = compute_spectral_efficiency(sinr_mat)
    print(f"\n[Step 4] Final Metrics:")
    print(f"  Drones used      : {len(final_drones)}")
    print(f"  Users covered    : {n_covered}/{N_USERS} "
          f"({100*n_covered/N_USERS:.1f} %)")
    print(f"  Avg spectral eff : {avg_se:.3f} bps/Hz  (target ≥ {ETA_SE})")

    #  Step 5: Plots─
    print(f"\n[Step 5] Generating plots for Scenario {scenario_id}...")
    s = scenario_id

    split_x = AREA_SIDE / 2 if s == 1 else None
    lbl_l = 'Left-region drones' if s == 1 else 'Central-region drones'
    lbl_r = 'Right-region drones' if s == 1 else 'Outer-region drones'

    plot_2d_placement(
        users, final_drones,
        title=f'Fig. {2*s+1}a — Scenario {s}: User Distribution & 2D Drone Placement',
        filename=f'figure{2*s+1}a_scenario{s}_2d.png',
        split_x=split_x,
        left_label=lbl_l, right_label=lbl_r)

    plot_3d_placement(
        final_drones,
        title=f'Fig. {2*s+1}b — Scenario {s}: 3D Drone Positions',
        filename=f'figure{2*s+1}b_scenario{s}_3d.png',
        split_x=split_x)

    plot_sinr_cdf(
        sinr_mat,
        title=f'Fig. {2*s+2} — Scenario {s}: SINR CDF',
        filename=f'figure{2*s+2}_scenario{s}_sinr_cdf.png')

    plot_convergence(
        pso.history,
        title=f'Fig. {2*s+3} — Scenario {s}: PSO Convergence',
        filename=f'figure{2*s+3}_scenario{s}_convergence.png')

    return {
        'drones'      : final_drones,
        'n_covered'   : n_covered,
        'sinr_matrix' : sinr_mat,
        'history'     : pso.history,
    }


# SECTION 11 — ENTRY POINT

def main():

    print("=" * 60)
    print("  DRONE-BS 3D PLACEMENT — Paper Reproduction")
    print("=" * 60)

    #  Step 0: Channel model verification (Fig. 2)─
    print("\n[Fig. 2] Plotting path loss vs. altitude...")
    plot_figure2()

    #─
    # SCENARIO I
    # Area: 10 km × 10 km split into left (20 % users) and right (80 % users)
    #─
    users1 = generate_scenario1()
    half   = AREA_SIDE / 2.0

    # Subarea definitions: each dict describes one region with constant density
    total_area_half = half * AREA_SIDE       # area of each half (m²)
    subareas1 = [
        {
            'x_min'  : 0,
            'x_max'  : half,
            'y_min'  : 0,
            'y_max'  : AREA_SIDE,
            'density': int(0.20 * N_USERS) / total_area_half,
            'area'   : total_area_half,
        },
        {
            'x_min'  : half,
            'x_max'  : AREA_SIDE,
            'y_min'  : 0,
            'y_max'  : AREA_SIDE,
            'density': int(0.80 * N_USERS) / total_area_half,
            'area'   : total_area_half,
        },
    ]

    results1 = run_scenario(scenario_id=1, users=users1, subareas=subareas1)

    #─
    # SCENARIO II
    # Central Gaussian cluster (40 %) + uniform remainder (60 %)
    #─
    users2 = generate_scenario2()
    centre = AREA_SIDE / 2.0
    r_cluster = 2000.0    # approximate radius of the Gaussian cluster (m)
    cluster_area    = np.pi * r_cluster**2
    remaining_area  = AREA_SIDE**2 - cluster_area

    subareas2 = [
        {
            'x_min'  : centre - r_cluster,
            'x_max'  : centre + r_cluster,
            'y_min'  : centre - r_cluster,
            'y_max'  : centre + r_cluster,
            'density': int(0.40 * N_USERS) / cluster_area,
            'area'   : cluster_area,
        },
        {
            'x_min'  : 0,
            'x_max'  : AREA_SIDE,
            'y_min'  : 0,
            'y_max'  : AREA_SIDE,
            'density': int(0.60 * N_USERS) / AREA_SIDE**2,
            'area'   : AREA_SIDE**2,
        },
    ]

    results2 = run_scenario(scenario_id=2, users=users2, subareas=subareas2)

    #  Summary─
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  Scenario I  — {len(results1['drones'])} drones, "
          f"{results1['n_covered']}/{N_USERS} users covered "
          f"({100*results1['n_covered']/N_USERS:.1f} %)")
    print(f"  Scenario II — {len(results2['drones'])} drones, "
          f"{results2['n_covered']}/{N_USERS} users covered "
          f"({100*results2['n_covered']/N_USERS:.1f} %)")
    print("\nAll figures saved as PNG files in the current directory.")
    print("=" * 60)


if __name__ == "__main__":
    main()