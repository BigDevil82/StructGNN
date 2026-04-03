import math


def cqc(mu: list[float], lambdas: list[float], damping: list[float], scale_factors: list[float]) -> float:
    total = 0.0
    nm = len(mu)
    for i in range(nm):
        for j in range(nm):
            di = damping[i]
            dj = damping[j]
            bij = lambdas[i] / lambdas[j]
            rho = (8.0 * math.sqrt(di * dj) * (di + bij * dj) * (bij**1.5)) / (
                (1.0 - bij**2.0) ** 2.0
                + 4.0 * di * dj * bij * (1.0 + bij**2.0)
                + 4.0 * (di**2.0 + dj**2.0) * bij**2.0
            )
            total += scale_factors[i] * mu[i] * scale_factors[j] * mu[j] * rho
    return math.sqrt(max(0.0, total))


def srss(mu: list[float], scale_factors: list[float]) -> float:
    return math.sqrt(sum((scale_factors[i] * mu[i]) ** 2.0 for i in range(len(mu))))


def combine_story_drifts(
    modal_drifts: list[list[float]],
    lambdas: list[float],
    damping: list[float],
    scale_factors: list[float],
    method: str,
) -> list[float]:
    method_upper = method.upper()
    if method_upper == "CQC":
        return [cqc(story_modal, lambdas, damping, scale_factors) for story_modal in modal_drifts]
    if method_upper == "SRSS":
        return [srss(story_modal, scale_factors) for story_modal in modal_drifts]
    raise ValueError(f"Unsupported method {method}. Use CQC or SRSS.")
