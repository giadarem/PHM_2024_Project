from enum import Enum

from scipy import stats
import numpy as np


class ProbabilityPDF(Enum):
    NORMAL = "norm"
    EXPONENTIAL = "expon"
    UNIFORM = "uniform"
    GAMMA = "gamma"
    BETA = "beta"
    LOG_NORMAL = "lognorm"
    CHI_SQUARED = "chi2"
    WEIBULL = "weibull_min"
    STUDENT_T = "t"
    F = "f"
    CAUCHY = "cauchy"
    LAPLACE = "laplace"
    RAYLEIGH = "rayleigh"
    PARETO = "pareto"
    GUMBEL = "gumbel_r"
    LOGISTIC = "logistic"
    ERLANG = "erlang"
    POWER_LAW = "powerlaw"
    NAKAGAMI = "nakagami"
    BETA_PRIME = "betaprime"

def build_distribution(probability: ProbabilityPDF, trq_margin: float):
    match probability:
        case ProbabilityPDF.UNIFORM:
            dist = stats.uniform(
                loc=trq_margin - 0.5,
                scale=1.0
            )
        case ProbabilityPDF.BETA:
            dist = stats.beta(
                a=1.5,
                b=1.5,
                loc=trq_margin - 0.6365,
                scale=1.273
            )
        case ProbabilityPDF.CAUCHY:
            dist = stats.cauchy(
                loc=trq_margin,
                scale=1 / np.pi
            )
        case ProbabilityPDF.CAUCHY:
            dist = stats.cauchy(
                loc=trq_margin,
                scale=1 / np.pi
            )

    return dist.pdf

def make_x_grid(center: float, width: float = 5.0, n: int = 2000) -> np.ndarray:
    return np.linspace(center - width, center + width, n)