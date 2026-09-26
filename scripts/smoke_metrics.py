#!/usr/bin/env python
from __future__ import annotations

import numpy as np

from trajuq.calibration import RiskCalibrator
from trajuq.decision import DecisionPolicy
from trajuq.evaluation import calibration_metrics, error_detection_metrics, selective_curve


def main() -> None:
    # Synthetic values only; these are not experimental results.
    x = np.array([[0.10], [0.20], [0.35], [0.55], [0.70], [0.85], [0.90], [0.40]])
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0])

    calibrator = RiskCalibrator("logistic").fit(x, y)
    p = calibrator.predict_error_probability(x)

    print("error detection:", error_detection_metrics(y, x[:, 0]))
    print("calibration:", calibration_metrics(y, p))
    print("AURC:", selective_curve(y, p).aurc)

    policy = DecisionPolicy()
    print("actions:", [policy.decide(float(v)).value for v in p[:4]])


if __name__ == "__main__":
    main()
