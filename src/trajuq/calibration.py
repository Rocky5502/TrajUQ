from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class RiskCalibrator:
    method: str = "logistic"
    random_state: int = 2027

    def __post_init__(self) -> None:
        if self.method == "logistic":
            self.model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            random_state=self.random_state,
                            max_iter=2000,
                        ),
                    ),
                ]
            )
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
