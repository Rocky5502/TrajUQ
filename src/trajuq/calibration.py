from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def standardized_logistic_pipeline(random_state: int = 2027) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    random_state=random_state,
                    max_iter=2000,
                ),
            ),
        ]
    )


def fit_isotonic_logistic_sensitivity(
    x: np.ndarray,
    y_error: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 2027,
) -> tuple[CalibratedClassifierCV, int]:
    """Fit leak-safe isotonic post-calibration of the primary logistic model.

    The base standardized logistic classifier is calibrated from out-of-fold
    decision scores via CalibratedClassifierCV(ensemble=False), then refit on
    all calibration examples for test prediction. No test labels are used.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y_error, dtype=int)
    if x.ndim == 1:
        x = x[:, None]
    if len(x) != len(y):
        raise ValueError("x and y_error must have equal length")
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) < 2:
        raise ValueError("Isotonic sensitivity requires both error classes.")
    folds = min(int(n_splits), int(counts.min()))
    if folds < 2:
        raise ValueError(
            "Isotonic sensitivity requires at least two examples in each class."
        )

    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )
    model = CalibratedClassifierCV(
        estimator=standardized_logistic_pipeline(random_state),
        method="isotonic",
        cv=splitter,
        ensemble=False,
    )
    model.fit(x, y)
    return model, folds


@dataclass
class RiskCalibrator:
    method: str = "logistic"
    random_state: int = 2027

    def __post_init__(self) -> None:
        if self.method == "logistic":
            self.model = standardized_logistic_pipeline(self.random_state)
        elif self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip")
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

    def fit(self, x: np.ndarray, y_error: np.ndarray) -> "RiskCalibrator":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y_error, dtype=int)
        if len(np.unique(y)) < 2:
            raise ValueError("Calibration requires both correct and incorrect examples.")
        if self.method == "isotonic":
            if x.ndim == 2:
                if x.shape[1] != 1:
                    raise ValueError("Isotonic calibration accepts exactly one risk feature.")
                x = x[:, 0]
            self.model.fit(x, y)
        else:
            if x.ndim == 1:
                x = x[:, None]
            self.model.fit(x, y)
        return self

    def predict_error_probability(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.method == "isotonic":
            if x.ndim == 2:
                x = x[:, 0]
            return np.asarray(self.model.predict(x), dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        return np.asarray(self.model.predict_proba(x)[:, 1], dtype=float)



def cross_fitted_error_probabilities(
    x: np.ndarray,
    y_error: np.ndarray,
    n_splits: int = 5,
    random_state: int = 2027,
) -> tuple[np.ndarray, int]:
    """Out-of-fold logistic error probabilities for policy selection.

    This lets a downstream threshold be selected on calibration examples
    without using predictions from a calibrator trained on those same examples.
    """

    from sklearn.model_selection import StratifiedKFold

    x = np.asarray(x, dtype=float)
    y = np.asarray(y_error, dtype=int)
    if x.ndim == 1:
        x = x[:, None]
    if len(x) != len(y):
        raise ValueError("x and y_error must have equal length")
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) < 2:
        raise ValueError("Cross-fitting requires both correct and incorrect examples.")
    folds = min(int(n_splits), int(counts.min()))
    if folds < 2:
        raise ValueError("Cross-fitting requires at least two examples in each class.")

    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )
    predictions = np.full(len(y), np.nan, dtype=float)
    for train_idx, valid_idx in splitter.split(x, y):
        model = RiskCalibrator(
            method="logistic",
            random_state=random_state,
        ).fit(x[train_idx], y[train_idx])
        predictions[valid_idx] = model.predict_error_probability(x[valid_idx])

    if not np.all(np.isfinite(predictions)):
        raise ValueError("Cross-fitted calibration produced non-finite probabilities.")
    return predictions, folds
