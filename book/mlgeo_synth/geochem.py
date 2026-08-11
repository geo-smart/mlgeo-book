"""Synthetic geochemical / petrophysical sample tables.

Stands in for: whole-rock major-element analyses with lithology labels, the
kind of tabular data used for classification homework. Major-element oxides
are strongly correlated (they sum toward 100 wt%), which is exactly what
justifies the PCA step in the workflow; the planted correlation structure
makes PCA and feature-importance answers checkable against ground truth.

Breaks down: compositions are Gaussian around class means rather than drawn
from a closed compositional simplex; no analytical censoring or detection
limits are modeled.
"""

import numpy as np
import pandas as pd

# Class means loosely follow granite / basalt / andesite averages (wt% oxides,
# g/cm3 density, SI magnetic susceptibility). Imbalance is deliberate.
_CLASSES = {
    "granite": dict(w=0.55, sio2=72, al2o3=14, feo=2.5, mgo=0.8, cao=1.8,
                    na2o=3.5, k2o=4.2, density=2.65, chi=1e-4),
    "basalt": dict(w=0.35, sio2=49, al2o3=15, feo=10.0, mgo=7.5, cao=11.0,
                   na2o=2.5, k2o=0.6, density=2.95, chi=1e-3),
    "andesite": dict(w=0.10, sio2=59, al2o3=17, feo=6.5, mgo=3.5, cao=6.5,
                     na2o=3.8, k2o=1.8, density=2.80, chi=5e-4),
}

_OXIDES = ["sio2", "al2o3", "feo", "mgo", "cao", "na2o", "k2o"]


def geochem_table(n=10_000, label_noise=0.0, seed=0):
    """Sample table with 3 imbalanced lithology classes and correlated features.

    ``label_noise`` flips that fraction of labels uniformly at random — the
    training-data-curation knob used in the Chapter 4 model-development lab.

    Columns: SiO2..K2O (wt%), density_g_cm3, mag_susc_si, label.
    A latent "differentiation index" drives the oxide correlations within each
    class, so the first principal component has a known physical meaning.
    """
    rng = np.random.default_rng(seed)
    names = list(_CLASSES)
    weights = np.array([_CLASSES[c]["w"] for c in names])
    labels = rng.choice(names, size=n, p=weights / weights.sum())

    rows = np.zeros((n, len(_OXIDES) + 2))
    for i, lab in enumerate(labels):
        c = _CLASSES[lab]
        # Latent differentiation index correlates oxides within the class:
        # more evolved -> higher SiO2/K2O/Na2O, lower FeO/MgO/CaO.
        d = rng.normal(0, 1)
        signs = dict(sio2=+2.0, al2o3=-0.3, feo=-1.2, mgo=-1.0, cao=-1.0, na2o=+0.3, k2o=+0.5)
        for j, ox in enumerate(_OXIDES):
            rows[i, j] = c[ox] + signs[ox] * d + rng.normal(0, 0.6)
        rows[i, -2] = c["density"] - 0.02 * d + rng.normal(0, 0.03)
        rows[i, -1] = c["chi"] * np.exp(rng.normal(0, 0.5))

    df = pd.DataFrame(rows, columns=[o.upper() for o in _OXIDES] + ["density_g_cm3", "mag_susc_si"])
    df[[o.upper() for o in _OXIDES]] = df[[o.upper() for o in _OXIDES]].clip(lower=0.0)
    df["label"] = labels

    if label_noise > 0:
        flip = rng.random(n) < label_noise
        df.loc[flip, "label"] = rng.choice(names, size=int(flip.sum()))
    return df
