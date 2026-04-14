import math

import numpy as np


def cqc(mu: list[float], lambdas: list[float], damping: list[float], scale_factors: list[float]) -> float:
    mu = np.asarray(mu)
    lam = np.asarray(lambdas)
    d = np.asarray(damping)
    sf = np.asarray(scale_factors)

    # 频率比矩阵 β[i,j] = λ[i] / λ[j]，shape (n, n)
    beta = lam[:, None] / lam[None, :]

    # 阻尼积矩阵，shape (n, n)
    di = d[:, None]  # 列向量广播
    dj = d[None, :]  # 行向量广播

    # CQC 相关系数矩阵 ρ
    numer = 8.0 * np.sqrt(di * dj) * (di + beta * dj) * beta**1.5
    denom = (1.0 - beta**2) ** 2 + 4.0 * di * dj * beta * (1.0 + beta**2) + 4.0 * (di**2 + dj**2) * beta**2
    rho = numer / denom  # shape (n, n)

    # 权重向量 w[i] = scale_factors[i] * mu[i]
    w = sf * mu

    # 二次型：total = wᵀ ρ w
    total = w @ rho @ w

    return float(np.sqrt(max(0.0, total)))


def srss(mu: list[float], scale_factors: list[float]) -> float:
    return float(np.sqrt(np.sum((np.asarray(scale_factors) * np.asarray(mu)) ** 2.0)))


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
