from math import pi, sqrt

import openseespy.opensees as ops


# --- 1. Define CQC Function (From Provided Example) ---
def CQC(mu, lambdas, dmp, scalf):
    u = 0.0
    ne = len(lambdas)
    for i in range(ne):
        for j in range(ne):
            di = dmp[i]
            dj = dmp[j]
            bij = lambdas[i] / lambdas[j]
            rho = (8.0 * sqrt(di * dj) * (di + bij * dj) * (bij ** (3.0 / 2.0))) / (
                (1.0 - bij**2.0) ** 2.0
                + 4.0 * di * dj * bij * (1.0 + bij**2.0)
                + 4.0 * (di**2.0 + dj**2.0) * bij**2.0
            )
            u += scalf[i] * mu[i] * scalf[j] * mu[j] * rho
    return sqrt(u)


def SRSS(mu, scalf):
    u = 0.0
    for i in range(len(mu)):
        u += (scalf[i] * mu[i]) ** 2.0
    return sqrt(u)


def interpolate_sa(period, periods, spectral_acc):
    if period <= periods[0]:
        return spectral_acc[0]
    if period >= periods[-1]:
        return spectral_acc[-1]

    for i in range(1, len(periods)):
        if period <= periods[i]:
            t1, t2 = periods[i - 1], periods[i]
            s1, s2 = spectral_acc[i - 1], spectral_acc[i]
            ratio = (period - t1) / (t2 - t1)
            return s1 + ratio * (s2 - s1)
    return spectral_acc[-1]


def gb50011_spectrum_params(damping_ratio=0.05):
    xi = damping_ratio
    gamma = 0.9 + (0.05 - xi) / (0.3 + 6.0 * xi)
    eta1 = 0.02 + (0.05 - xi) / (4.0 + 32.0 * xi)
    eta2 = 1.0 + (0.05 - xi) / (0.08 + 1.6 * xi)

    # Common implementation constraints used with GB50011 spectrum shape factors.
    eta1 = max(0.0, eta1)
    eta2 = min(1.0, max(0.55, eta2))
    return gamma, eta1, eta2


def gb50011_alpha(T, alpha_max, Tg, damping_ratio=0.05):
    gamma, eta1, eta2 = gb50011_spectrum_params(damping_ratio)

    if T <= 0.1:
        return alpha_max * (0.45 + (eta2 - 0.45) * (T / 0.1))
    if T <= Tg:
        return eta2 * alpha_max
    if T <= 5.0 * Tg:
        return eta2 * alpha_max * ((Tg / T) ** gamma)
    if T <= 6.0:
        return alpha_max * (eta2 * (0.2**gamma) - eta1 * (T - 5.0 * Tg))
    return alpha_max * max(0.0, eta2 * (0.2**gamma) - eta1 * (6.0 - 5.0 * Tg))


def build_gb50011_spectrum(alpha_max, Tg, damping_ratio=0.05, g=9.81, t_max=6.0, dt=0.01):
    periods = []
    spectral_acc = []
    steps = int(t_max / dt)
    for i in range(steps + 1):
        t = i * dt
        alpha = gb50011_alpha(t, alpha_max, Tg, damping_ratio=damping_ratio)
        periods.append(t)
        spectral_acc.append(alpha * g)
    return periods, spectral_acc


def resolve_gb50011_design_params(
    intensity,
    design_acc_g,
    site_class,
    seismic_group,
    alpha_max_override=None,
    Tg_override=None,
):
    # GB50011 common design basic acceleration -> alpha_max mapping.
    alpha_max_map = {
        0.05: 0.04,
        0.10: 0.08,
        0.15: 0.12,
        0.20: 0.16,
        0.30: 0.24,
        0.40: 0.32,
    }

    # GB50011 site class/group -> characteristic period Tg (s).
    tg_map = {
        "I0": {1: 0.20, 2: 0.25, 3: 0.30},
        "I": {1: 0.25, 2: 0.30, 3: 0.35},
        "II": {1: 0.35, 2: 0.40, 3: 0.45},
        "III": {1: 0.45, 2: 0.55, 3: 0.65},
        "IV": {1: 0.65, 2: 0.75, 3: 0.90},
    }

    acc_key = round(float(design_acc_g), 2)
    if alpha_max_override is None:
        if acc_key not in alpha_max_map:
            raise ValueError(f"Unsupported design_acc_g={design_acc_g}. Please set alpha_max_override.")
        alpha_max = alpha_max_map[acc_key]
    else:
        alpha_max = float(alpha_max_override)

    site_key = str(site_class).upper()
    group_key = int(seismic_group)
    if Tg_override is None:
        if site_key not in tg_map or group_key not in tg_map[site_key]:
            raise ValueError(
                f"Unsupported site/group combination: site_class={site_class}, seismic_group={seismic_group}. "
                "Please set Tg_override."
            )
        Tg = tg_map[site_key][group_key]
    else:
        Tg = float(Tg_override)

    return alpha_max, Tg, acc_key, site_key, group_key, intensity


def combine_story_drifts(
    modal_drifts,
    lambdas,
    damping,
    scale_factors,
    method="CQC",
):
    method_upper = method.upper()
    if method_upper == "CQC":
        return [CQC(story_modal, lambdas, damping, scale_factors) for story_modal in modal_drifts]
    if method_upper == "SRSS":
        return [SRSS(story_modal, scale_factors) for story_modal in modal_drifts]
    raise ValueError(f"Unsupported combination method: {method}. Use 'CQC' or 'SRSS'.")


def run_manual_rsa(
    floor_nodes,
    floor_masses,
    story_height,
    direction,
    eigs,
    periods,
    spectral_acc,
    mass_active_directions=(1,),
):
    num_modes = len(eigs)
    num_stories = len(floor_nodes)

    modal_drifts = [[] for _ in range(num_stories)]

    for mode in range(1, num_modes + 1):
        lam = eigs[mode - 1]
        if lam <= 0:
            continue

        omega = sqrt(lam)
        period = 2.0 * pi / omega

        sa_t = interpolate_sa(period, periods, spectral_acc)
        sd_t = sa_t / lam

        # mode shape (single direction)
        mode_shape = [ops.nodeEigenvector(node_tag, mode, direction) for node_tag in floor_nodes]

        # participation factor
        numerator = sum(m * phi for m, phi in zip(floor_masses, mode_shape))
        denominator = sum(m * phi**2 for m, phi in zip(floor_masses, mode_shape))

        gamma = numerator / denominator if denominator != 0.0 else 0.0

        # modal displacement
        modal_displacements = [gamma * phi * sd_t for phi in mode_shape]

        # drift
        prev_disp = 0.0
        for j, disp in enumerate(modal_displacements):
            story_drift = (disp - prev_disp) / story_height
            modal_drifts[j].append(story_drift)
            prev_disp = disp

    return modal_drifts


def run_builtin_rsa_for_comparison(floor_nodes, story_height, direction, periods, spectral_acc, num_modes):
    num_stories = len(floor_nodes)
    modal_drifts = [[] for _ in range(num_stories)]

    for mode in range(1, num_modes + 1):
        # OpenSees updates the domain with one modal response at a time.
        # Results must be read immediately after the corresponding -mode call.
        ops.responseSpectrumAnalysis(direction, "-Tn", *periods, "-Sa", *spectral_acc, "-mode", mode)

        prev_disp = 0.0
        for j, node_tag in enumerate(floor_nodes):
            current_disp = ops.nodeDisp(node_tag, direction)

            story_drift = (current_disp - prev_disp) / story_height
            modal_drifts[j].append(story_drift)

            prev_disp = current_disp

    return modal_drifts


# --- 2. Build an 8-Story 3D Frame Model (1x1 Bay) ---
ops.wipe()
ops.model("basic", "-ndm", 3, "-ndf", 6)

# Parameters
num_stories_model = 8
story_height = 3.0  # meters
bay_x = 6.0  # meters
bay_y = 6.0  # meters
mass_per_floor = 50000.0  # kg
col_A = 0.20
beam_A = 0.15
E = 3.0e10
G = 1.2e10
col_Iy = 0.005
col_Iz = 0.005
beam_x_Iy = 0.004
beam_x_Iz = 0.006
beam_y_Iy = 0.006
beam_y_Iz = 0.004
col_J = 0.001
beam_J = 0.001

corner_xy = [
    (0.0, 0.0),
    (bay_x, 0.0),
    (bay_x, bay_y),
    (0.0, bay_y),
]


def corner_tag(level, corner_index):
    return level * 10 + corner_index + 1


def master_tag(level):
    return 1000 + level


floor_nodes = []
floor_masses = []

# Corner nodes for all levels
for level in range(0, num_stories_model + 1):
    z = level * story_height
    for c_idx, (x, y) in enumerate(corner_xy):
        ntag = corner_tag(level, c_idx)
        ops.node(ntag, x, y, z)
        if level == 0:
            ops.fix(ntag, 1, 1, 1, 1, 1, 1)

# Floor master nodes with lumped masses
for level in range(1, num_stories_model + 1):
    z = level * story_height
    mtag = master_tag(level)
    ops.node(mtag, 0.5 * bay_x, 0.5 * bay_y, z)
    ops.mass(mtag, mass_per_floor, mass_per_floor, 0.0, 0.0, 0.0, 0.0)
    ops.fix(mtag, 0, 0, 1, 1, 1, 1)

    for c_idx in range(4):
        ctag = corner_tag(level, c_idx)
        ops.equalDOF(mtag, ctag, 1, 2)

    floor_nodes.append(mtag)
    floor_masses.append(mass_per_floor)

# Geometric transformations
ops.geomTransf("Linear", 1, 1.0, 0.0, 0.0)  # columns
ops.geomTransf("Linear", 2, 0.0, 0.0, 1.0)  # X beams
ops.geomTransf("Linear", 3, 1.0, 0.0, 0.0)  # Y beams

ele_tag = 1

# Columns (4 lines)
for level in range(1, num_stories_model + 1):
    for c_idx in range(4):
        ni = corner_tag(level - 1, c_idx)
        nj = corner_tag(level, c_idx)
        ops.element(
            "elasticBeamColumn",
            ele_tag,
            ni,
            nj,
            col_A,
            E,
            G,
            col_J,
            col_Iy,
            col_Iz,
            1,
        )
        ele_tag += 1

# Floor beams in X and Y (1 bay each direction)
for level in range(1, num_stories_model + 1):
    n1 = corner_tag(level, 0)
    n2 = corner_tag(level, 1)
    n3 = corner_tag(level, 2)
    n4 = corner_tag(level, 3)

    # X direction beams
    ops.element("elasticBeamColumn", ele_tag, n1, n2, beam_A, E, G, beam_J, beam_x_Iy, beam_x_Iz, 2)
    ele_tag += 1
    ops.element("elasticBeamColumn", ele_tag, n4, n3, beam_A, E, G, beam_J, beam_x_Iy, beam_x_Iz, 2)
    ele_tag += 1

    # Y direction beams
    ops.element("elasticBeamColumn", ele_tag, n1, n4, beam_A, E, G, beam_J, beam_y_Iy, beam_y_Iz, 3)
    ele_tag += 1
    ops.element("elasticBeamColumn", ele_tag, n2, n3, beam_A, E, G, beam_J, beam_y_Iy, beam_y_Iz, 3)
    ele_tag += 1

# --- 3. Analysis Setup & Eigenvalues ---
# Required settings for static analysis used by the RSA command
ops.constraints("Transformation")
ops.numberer("RCM")
ops.system("BandGen")
ops.test("NormDispIncr", 1.0e-6, 10)
ops.algorithm("Linear")
ops.integrator("LoadControl", 0.0)
ops.analysis("Static")

num_modes = 6
try:
    eigs = ops.eigen("-genBandArpack", num_modes)
except Exception:
    eigs = ops.eigen("-fullGenLapack", num_modes)

if isinstance(eigs, (int, float)):
    eigs = [eigs]
else:
    eigs = list(eigs)

num_modes = len(eigs)
ops.modalProperties()

# --- 4. Response Spectrum Definition (GB50011-2010) ---
# Parameters defined by code tables:
# intensity + design_acc_g -> alpha_max; site_class + seismic_group -> Tg.
intensity = 7
design_acc_g = 0.10
site_class = "II"
seismic_group = 1

# Optional manual overrides (set to float to force a value).
alpha_max_override = None
Tg_override = None

alpha_max, Tg, acc_used, site_used, group_used, intensity_used = resolve_gb50011_design_params(
    intensity=intensity,
    design_acc_g=design_acc_g,
    site_class=site_class,
    seismic_group=seismic_group,
    alpha_max_override=alpha_max_override,
    Tg_override=Tg_override,
)

damping_ratio = 0.05

Tn, Sa = build_gb50011_spectrum(
    alpha_max=alpha_max,
    Tg=Tg,
    damping_ratio=damping_ratio,
)

print(
    "GB50011 spectrum params: "
    f"intensity={intensity_used}, design_acc={acc_used}g, site={site_used}, group={group_used}, "
    f"alpha_max={alpha_max:.3f}, Tg={Tg:.3f}s, damping={damping_ratio:.2f}"
)

# --- 5. RSA Mode-by-Mode Execution ---
dmp = [0.05] * num_modes
scalf = [1.0] * num_modes
combination_method = "CQC"

# --- 6. Post-Processing: CQC Combination (Both Directions) ---
for direction, label in [(1, "X"), (2, "Y")]:
    manual_modal_drifts = run_manual_rsa(
        floor_nodes=floor_nodes,
        floor_masses=floor_masses,
        story_height=story_height,
        direction=direction,
        eigs=eigs,
        periods=Tn,
        spectral_acc=Sa,
        mass_active_directions=(1, 2),
    )

    manual_cqc_drifts = combine_story_drifts(
        manual_modal_drifts,
        eigs,
        dmp,
        scalf,
        method=combination_method,
    )

    builtin_modal_drifts = run_builtin_rsa_for_comparison(
        floor_nodes=floor_nodes,
        story_height=story_height,
        direction=direction,
        periods=Tn,
        spectral_acc=Sa,
        num_modes=num_modes,
    )

    builtin_cqc_drifts = combine_story_drifts(
        builtin_modal_drifts,
        eigs,
        dmp,
        scalf,
        method=combination_method,
    )

    print(f"--- {label}-Direction {combination_method} Inter-Story Drifts: Manual RSA vs Built-in RSA ---")
    for j, (manual_drift, builtin_drift) in enumerate(zip(manual_cqc_drifts, builtin_cqc_drifts), start=1):
        abs_diff = abs(manual_drift - builtin_drift)
        rel_diff = (abs_diff / abs(builtin_drift) * 100.0) if builtin_drift != 0.0 else 0.0
        print(
            f"Story {j}: manual={manual_drift * 100:.3f}%, "
            f"builtin={builtin_drift * 100:.3f}%, "
            f"abs_diff={abs_diff * 100:.4f}%, rel_diff={rel_diff:.2f}%"
        )

ops.wipe()
