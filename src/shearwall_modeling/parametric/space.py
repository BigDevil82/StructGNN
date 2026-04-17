import math
import random
from typing import Any

from .types import ParametricModelParams, SamplingMethod

PARAM_SPACE: dict[str, list[Any]] = {
    "N": list(range(18, 34)),
    "tw_bot": [200, 250, 300, 350, 400],
    "tw_mid": [160, 180, 200, 250, 300],
    "tw_top": [160, 180, 200, 250],
    "hb_main": [400, 500, 550, 600, 650, 700],
    "bb_main": [200, 250, 300, 350],
    "hb_sec": [300, 400, 450, 500],
    "bb_sec": [200, 250, 300],
    "conc_bot": ["C30", "C35", "C40", "C45", "C50"],
    "intensity": [6.0, 7.0, 7.5, 8.0],
    "site_class": ["I0", "I", "II", "III", "IV"],
    "seismic_group": [1, 2, 3],
}


def sample_parametric_model_params(rng: random.Random | None = None) -> ParametricModelParams:
    sampler = rng or random.Random()
    raw = {key: sampler.choice(values) for key, values in PARAM_SPACE.items()}
    return normalize_params(raw)


def sample_parametric_model_params_batch(
    n: int,
    method: SamplingMethod = "lhs",
    seed: int = 42,
) -> list[ParametricModelParams]:
    if n <= 0:
        return []

    if method == "random":
        rng = random.Random(seed)
        return [sample_parametric_model_params(rng) for _ in range(n)]

    if method != "lhs":
        raise ValueError(f"Unsupported sampling_method={method}.")

    keys = list(PARAM_SPACE.keys())
    rng = random.Random(seed)
    idx_by_key: dict[str, list[int]] = {}
    for key in keys:
        levels = len(PARAM_SPACE[key])
        perm = list(range(n))
        rng.shuffle(perm)
        idxs: list[int] = []
        for i in range(n):
            u = (perm[i] + rng.random()) / n
            idx = int(math.floor(u * levels))
            idxs.append(min(levels - 1, max(0, idx)))
        idx_by_key[key] = idxs

    return [normalize_params({key: PARAM_SPACE[key][idx_by_key[key][i]] for key in keys}) for i in range(n)]


def normalize_params(raw: ParametricModelParams | dict[str, Any]) -> ParametricModelParams:
    if isinstance(raw, ParametricModelParams):
        p = raw
    else:
        p = ParametricModelParams(
            N=int(raw["N"]),
            tw_bot=int(raw["tw_bot"]),
            tw_mid=int(raw["tw_mid"]),
            tw_top=int(raw["tw_top"]),
            hb_main=int(raw["hb_main"]),
            bb_main=int(raw["bb_main"]),
            hb_sec=int(raw["hb_sec"]),
            bb_sec=int(raw["bb_sec"]),
            hs=int(raw.get("hs", 120)),
            conc_bot=_normalize_conc(str(raw["conc_bot"])),
            conc_mid=_normalize_conc(str(raw.get("conc_mid", raw["conc_bot"]))),
            conc_top=_normalize_conc(str(raw.get("conc_top", raw["conc_bot"]))),
            intensity=float(raw["intensity"]),
            site_class=str(raw["site_class"]).upper(),
            seismic_group=int(raw["seismic_group"]),
            h_story=float(raw.get("h_story", 2.9)),
        )

    grades = ["C30", "C35", "C40", "C45", "C50"]
    i_bot = grades.index(p.conc_bot)
    i_mid = i_bot
    i_top = i_bot
    tw_mid = min(p.tw_mid, p.tw_bot)
    tw_top = min(p.tw_top, tw_mid)

    return ParametricModelParams(
        N=p.N,
        tw_bot=p.tw_bot,
        tw_mid=tw_mid,
        tw_top=tw_top,
        hb_main=p.hb_main,
        bb_main=p.bb_main,
        hb_sec=p.hb_sec,
        bb_sec=p.bb_sec,
        hs=p.hs,
        conc_bot=grades[i_bot],
        conc_mid=grades[i_mid],
        conc_top=grades[i_top],
        intensity=p.intensity,
        site_class=p.site_class,
        seismic_group=p.seismic_group,
        h_story=p.h_story,
    )


def split_stories(total_stories: int) -> tuple[int, int, int]:
    if total_stories < 3:
        raise ValueError("N must be at least 3.")
    n1 = min(10, max(2, int(math.floor(0.3 * total_stories))))
    n2 = max(n1 + 1, int(math.floor(0.7 * total_stories)))
    n2 = min(total_stories - 1, n2)
    return n1, n2 - n1, total_stories - n2


def _normalize_conc(grade: str) -> str:
    g = grade.strip().upper()
    return g if g.startswith("C") else f"C{g}"
