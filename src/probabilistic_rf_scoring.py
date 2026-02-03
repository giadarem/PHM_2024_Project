import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from scipy.integrate import simpson


MODEL_PDFS = {
    "norm": stats.norm,
    "expon": stats.expon,
    "uniform": stats.uniform,
    "gamma": stats.gamma,
    "beta": stats.beta,
    "lognorm": stats.lognorm,
    "chi2": stats.chi2,
    "weibull_min": stats.weibull_min,
    "t": stats.t,
    "f": stats.f,
    "cauchy": stats.cauchy,
    "laplace": stats.laplace,
    "rayleigh": stats.rayleigh,
    "pareto": stats.pareto,
    "gumbel_r": stats.gumbel_r,
    "logistic": stats.logistic,
    "erlang": stats.erlang,
    "powerlaw": stats.powerlaw,
    "nakagami": stats.nakagami,
    "betaprime": stats.betaprime,
}

def get_regression_score(pdf_type, pdf_args, true_target, min_scale=1e-6):
    model = MODEL_PDFS[pdf_type]

    shape_args = tuple(pdf_args.get("args", ()))
    kwargs = {k: v for k, v in pdf_args.items() if k != "args"}

    if not getattr(model, "shapes", None):
        shape_args = ()

    if "scale" in kwargs:
        kwargs["scale"] = float(max(kwargs["scale"], min_scale))

    score = float(model.pdf(true_target, *shape_args, **kwargs))

    x = np.linspace(-100, 100, 100000)
    y = model.pdf(x, *shape_args, **kwargs)

    area = float(simpson(y, x))
    if area > 1:
        score /= area

    y_max = float(np.max(y))
    if y_max > 1:
        score /= y_max

    return float(score)


def fit_scipy_distribution(dist, samples):
    """
    dist: oggetto scipy.stats (es stats.t)
    samples: array 1D
    """
    params = dist.fit(samples)
    n_shapes = len(dist.shapes.split()) if dist.shapes else 0
    shape_params = tuple(float(p) for p in params[:n_shapes])
    loc = float(params[n_shapes])
    scale = float(params[n_shapes + 1])
    pdf_args = {"loc": loc, "scale": scale, "args": shape_params}
    return pdf_args


def select_best_pdf_one_sample(
    rf_model,
    x0,
    y_true,
    trq_margin=False,
    trq_mesaured=None,
    candidates=None,
    min_sigma=1e-6,
    eps_target=1e-6,
    y_true_is_margin=False
):
    if candidates is None:
        candidates = ["norm"]

    tree_preds = rf_model.predict_trees(x0)[0].astype(float)
    if trq_margin:
        if trq_mesaured is None:
            raise ValueError("trq_mesaured è obbligatorio quando trq_margin=True")


        denom = np.maximum(tree_preds, eps_target)
        tree_preds = 100.0 * (float(trq_mesaured) - denom) / denom

        if y_true_is_margin:
            true_target = float(y_true)
        else:
            denom_true = max(float(y_true), eps_target)
            true_target = 100.0 * (float(trq_mesaured) - denom_true) / denom_true
    else:
        true_target = float(y_true)


    if np.std(tree_preds) < min_sigma:
        tree_preds = tree_preds + np.random.normal(0.0, min_sigma, size=tree_preds.shape)

    results = []

    for pdf_type in candidates:
        dist = MODEL_PDFS[pdf_type]
        try:
            pdf_args = fit_scipy_distribution(dist, tree_preds)

            score = get_regression_score(
                pdf_type=pdf_type,
                pdf_args=pdf_args,
                true_target=true_target
            )

            results.append({
                "pdf_type": pdf_type,
                "score": float(score),
                "pdf_args": pdf_args,
                "loc": float(pdf_args["loc"]),
                "scale": float(pdf_args["scale"]),
                "shape_params": tuple(pdf_args.get("args", ()))
            })

        except Exception as e:
            results.append({
                "pdf_type": pdf_type,
                "score": -np.inf,
                "error": str(e)
            })

    results_sorted = sorted(results, key=lambda d: d["score"], reverse=True)
    best = results_sorted[0]
    return best, results_sorted, tree_preds

def plot_best_pdf(best, tree_preds, y_true):
    pdf_type = best["pdf_type"]
    dist = MODEL_PDFS[pdf_type]

    shape_params = best.get("shape_params", ())
    loc = best.get("loc", 0.0)
    scale = best.get("scale", 1.0)

    mu = float(np.mean(tree_preds))
    sig = float(np.std(tree_preds))
    sig = max(sig, 1e-6)

    x = np.linspace(mu - 6*sig, mu + 6*sig, 1500)
    y = dist.pdf(x, *shape_params, loc=loc, scale=scale)

    plt.figure()
    plt.hist(tree_preds, bins=25, density=True, alpha=0.6, label="Predizioni alberi")
    plt.plot(x, y, linewidth=2, label=f"{pdf_type} (best) | score={best['score']:.3e}")
    plt.axvline(float(y_true), linestyle="--", linewidth=2, label="Valore vero y")
    plt.title(f"Best PDF: {pdf_type}")
    plt.xlabel("y (torque margin)")
    plt.ylabel("densità")
    plt.legend()
    plt.show()