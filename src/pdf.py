import numpy as np
from matplotlib import pyplot as plt

from probabilistic_rf_scoring import MODEL_PDFS

ALLOWED_PDFS = ("uniform", "beta", "norm", "cauchy")

FILTERED_MODEL_PDFS = {
    k: v for k, v in MODEL_PDFS.items()
    if k in ALLOWED_PDFS
}

def calculate_pdf_args (T_hat: float):
    T_hat = float(T_hat)
    return {
        "uniform": {"args": (), "loc": T_hat - 0.5,    "scale": 1.0},
        "beta":    {"args": (1.5, 1.5), "loc": T_hat - 0.6365, "scale": 1.273},
        "norm":    {"args": (), "loc": T_hat,          "scale": 1.0 / np.sqrt(2.0*np.pi)},
        "cauchy":  {"args": (), "loc": T_hat,          "scale": 1.0 / np.pi},
    }

def central_interval(dist, pdf_args, central=0.99):
    alpha = 1.0 - float(central)
    lo_p = alpha / 2.0
    hi_p = 1.0 - alpha / 2.0

    shape_args = tuple(pdf_args.get("args", ()))
    kwargs = {k: v for k, v in pdf_args.items() if k != "args"}

    q_low  = dist.ppf(lo_p, *shape_args, **kwargs)
    q_high = dist.ppf(hi_p, *shape_args, **kwargs)
    return q_low, q_high

def passes_rule(samples, dist, pdf_args, required_frac=0.99, central=0.99):
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return False, (np.nan, np.nan), 0.0

    q_low, q_high = central_interval(dist, pdf_args, central=central)
    if not (np.isfinite(q_low) and np.isfinite(q_high)) or not (q_low < q_high):
        return False, (q_low, q_high), 0.0

    frac_inside = float(np.mean((samples >= q_low) & (samples <= q_high)))
    return frac_inside >= required_frac, (float(q_low), float(q_high)), frac_inside

def choose_pdf(samples, T_hat, order=("uniform", "beta", "norm"), fallback="cauchy"):
    args_map =  calculate_pdf_args (T_hat)
    tested = []

    for name in order:
        dist = FILTERED_MODEL_PDFS[name]
        pdf_args = args_map[name]
        ok, (q_low, q_high), frac = passes_rule(samples, dist, pdf_args, required_frac=0.99, central=0.99)
        tested.append({"name": name, "passed": ok, "frac_inside": frac, "q_low": q_low, "q_high": q_high, "pdf_args": pdf_args})
        if ok:
            return name, pdf_args, tested

    # fallback Cauchy (sempre)
    return fallback, args_map[fallback], tested


def plot_curves(samples, T_hat, chosen_name, bins=40, title=None):
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]

    args_map = calculate_pdf_args(T_hat)

    if samples.size > 0:
        x_min = float(np.min(samples))
        x_max = float(np.max(samples))
    else:
        x_min, x_max = float(T_hat - 3), float(T_hat + 3)

    pad = 0.25 * (x_max - x_min + 1e-9)
    x_left, x_right = x_min - pad, x_max + pad

    for k in ("uniform", "beta"):
        loc = args_map[k]["loc"]
        scale = args_map[k]["scale"]
        x_left = min(x_left, loc - 0.1 * scale)
        x_right = max(x_right, loc + 1.1 * scale)

    x = np.linspace(x_left, x_right, 3000)

    plt.figure(figsize=(10, 5))

    # Istogramma campioni
    if samples.size > 0:
        plt.hist(
            samples,
            bins=bins,
            density=True,
            alpha=0.25,
            label="Samples"
        )

    # Curve PDF
    for name in ("uniform", "beta", "norm", "cauchy"):
        dist = FILTERED_MODEL_PDFS[name]
        pdf_args = args_map[name]
        shape_args = tuple(pdf_args.get("args", ()))
        kwargs = {k: v for k, v in pdf_args.items() if k != "args"}
        y = dist.pdf(x, *shape_args, **kwargs)

        lw = 2.8 if name == chosen_name else 1.6
        lbl = f"{name.upper()}" + (" (chosen)" if name == chosen_name else "")
        plt.plot(x, y, linewidth=lw, label=lbl)

    # Intervallo centrale 99% della PDF scelta
    chosen_dist = FILTERED_MODEL_PDFS[chosen_name]
    q_low, q_high = central_interval(chosen_dist, args_map[chosen_name], central=0.99)

    if np.isfinite(q_low):
        plt.axvline(q_low, linestyle="--", linewidth=1.3, color="black",
                    label="99% central interval" if q_low == q_low else None)
    if np.isfinite(q_high):
        plt.axvline(q_high, linestyle="--", linewidth=1.3, color="black")

    plt.title(title if title else f"T_hat = {T_hat:.4f} | chosen PDF = {chosen_name.upper()}")
    plt.xlabel("Value")
    plt.ylabel("Density")

    plt.legend(frameon=True)
    plt.tight_layout()
    plt.show()
