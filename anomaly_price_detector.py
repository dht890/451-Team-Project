#anomaly_price_detector
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class IsolationForestResult:
    scored_rows: list[dict[str, Any]]
    summary: str
    warning: str | None = None


REQUIRED_COLUMNS = {"date", "total_price"}


def _validate_input(df: pd.DataFrame) -> tuple[bool, str | None]:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    return True, None


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.sort_values("date").reset_index(drop=True)
    data["total_price"] = pd.to_numeric(data["total_price"], errors="coerce")

    numeric_defaults = {
        "seats": 0,
        "usage": 0,
        "term_months": 0,
        "add_on_count": 0,
        "discount_expired": 0,
        "days_to_renewal": 0,
    }

    for col, default in numeric_defaults.items():
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(default)
        else:
            data[col] = default

    data["price_per_seat"] = np.where(
        data["seats"] > 0,
        data["total_price"] / data["seats"],
        data["total_price"],
    )

    data["pct_growth"] = (
        data["total_price"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    )
    data["seat_growth"] = (
        data["seats"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    )
    data["usage_growth"] = (
        data["usage"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    )

    data["growth_minus_seat_growth"] = data["pct_growth"] - data["seat_growth"]
    data["growth_minus_usage_growth"] = data["pct_growth"] - data["usage_growth"]

    return data


def _feature_matrix(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = [
        "total_price",
        "price_per_seat",
        "pct_growth",
        "seat_growth",
        "usage_growth",
        "growth_minus_seat_growth",
        "growth_minus_usage_growth",
        "term_months",
        "add_on_count",
        "discount_expired",
        "days_to_renewal",
    ]

    X = data[feature_cols].replace([np.inf, -np.inf], 0).fillna(0).copy()
    return X, feature_cols


def _top_feature_drivers(
    X: pd.DataFrame,
    feature_cols: list[str],
    row_index: int,
    top_k: int = 3,
) -> list[str]:
    medians = X.median()
    scales = X.std(ddof=0).replace(0, 1)

    row = X.loc[row_index, feature_cols]
    z_like = ((row - medians) / scales).abs().sort_values(ascending=False)
    return list(z_like.head(top_k).index)


def _explain_drivers(drivers: list[str]) -> str:
    explanations = {
        "total_price": "total spend is unusually high",
        "price_per_seat": "cost per seat is unusually high",
        "pct_growth": "price increased sharply relative to the previous period",
        "seat_growth": "seat count changed unusually",
        "usage_growth": "usage changed unusually",
        "growth_minus_seat_growth": "price growth is not aligned with seat growth",
        "growth_minus_usage_growth": "price growth is not aligned with usage growth",
        "term_months": "contract term length differs from the normal pattern",
        "add_on_count": "bundled add-ons may be affecting price",
        "discount_expired": "discount expiration may be contributing to the increase",
        "days_to_renewal": "timing near renewal may be influencing pricing",
    }
    return "; ".join(explanations.get(d, d) for d in drivers)


def analyze_price_history(
    df: pd.DataFrame,
    contamination: float = 0.15,
    random_state: int = 42,
) -> IsolationForestResult:
    valid, warning = _validate_input(df)
    if not valid:
        return IsolationForestResult(scored_rows=[], summary="", warning=warning)

    data = _prepare_features(df)
    X, feature_cols = _feature_matrix(data)

    if len(X) < 4:
        return IsolationForestResult(
            scored_rows=[],
            summary="Not enough pricing records to run Isolation Forest reliably.",
            warning="Provide at least 4 pricing events.",
        )

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X)

    raw_scores = model.score_samples(X)
    predictions = model.predict(X)

    data["anomaly_score"] = -raw_scores
    data["is_anomaly"] = predictions == -1

    top_drivers_all: list[list[str]] = []
    explanations_all: list[str] = []

    for idx in data.index:
        drivers = _top_feature_drivers(X, feature_cols, idx)
        top_drivers_all.append(drivers)
        explanations_all.append(_explain_drivers(drivers))

    data["top_drivers"] = top_drivers_all
    data["explanation"] = explanations_all

    anomaly_rows = data[data["is_anomaly"]].copy()

    if anomaly_rows.empty:
        summary = "No strong pricing anomalies detected in the provided history."
    else:
        lines: list[str] = []
        for _, row in anomaly_rows.iterrows():
            date_str = (
                row["date"].strftime("%Y-%m-%d")
                if pd.notnull(row["date"])
                else "unknown date"
            )
            lines.append(
                f"- {date_str}: anomaly score {row['anomaly_score']:.3f}; {row['explanation']}."
            )
        summary = "Detected pricing anomalies:\n" + "\n".join(lines)

    return IsolationForestResult(
        scored_rows=data.to_dict(orient="records"),
        summary=summary,
        warning=None,
    )