import math
from time import time

import openseespy.opensees as ops

from ..config import ModelConfig
from ..modal_combination import combine_story_drifts


def run_builtin_rsa(master_nodes: list[int], config: ModelConfig) -> dict[str, list[float]]:
    story_heights = config.get_story_heights()
    if len(story_heights) != len(master_nodes):
        raise ValueError(
            "Story profile count must match modeled story count. "
            f"profiles={len(story_heights)}, master_nodes={len(master_nodes)}"
        )

    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.test("NormDispIncr", 1.0e-6, 20)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 0.0)
    ops.analysis("Static")

    nreq = min(config.num_modes, len(master_nodes) * 2)
    print(f"Extracting {nreq} eigenvalues...")
    start = time()
    eigs = ops.eigen("-genBandArpack", nreq)
    end = time()
    print(f"Eigenvalue extraction completed in {end - start:.2f} seconds.")

    if isinstance(eigs, (int, float)):
        eigs = [float(eigs)]
    else:
        eigs = [float(x) for x in eigs]

    eigs = [x for x in eigs if x > 1e-12]
    num_modes = len(eigs)
    if num_modes == 0:
        raise RuntimeError("No valid eigenvalues were extracted.")

    ops.modalProperties()

    periods = [2.0 * math.pi / math.sqrt(lam) for lam in eigs]
    print(f"Modal periods (s): {[round(t, 4) for t in periods]}")

    damping = [config.seismic.damping_ratio] * num_modes
    scale_factors = [1.0] * num_modes
    result: dict[str, list[float]] = {}

    for direction, label in ((1, "X"), (2, "Y")):
        modal_drifts = [[] for _ in master_nodes]
        for mode in range(1, num_modes + 1):
            ops.responseSpectrumAnalysis(
                direction,
                "-Tn",
                *config.seismic.periods,
                "-Sa",
                *config.seismic.sa,
                "-mode",
                mode,
            )

            prev = 0.0
            for i, node in enumerate(master_nodes):
                d = ops.nodeDisp(node, direction)
                drift = (d - prev) / story_heights[i]
                modal_drifts[i].append(drift)
                prev = d

        combined = combine_story_drifts(
            modal_drifts,
            eigs,
            damping,
            scale_factors,
            method=config.seismic.combination_method,
        )
        result[label] = combined

    return result
