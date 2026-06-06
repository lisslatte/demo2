from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import ClassifierMixin
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


FEATURE_COLUMNS = [
    "Year","Quarter","Month","DayofMonth","DayOfWeek",
    "CRSDepHour",#"CRSArrHour",
    "is_peak_hour",
    "Distance","DistanceGroup",
    "Origin_freq","Dest_freq",
    "IS_WEEKEND",
    "IS_HOLIDAY","IS_NEAR_HOLIDAY","IS_PEAK_TRAVEL",
    "IS_SUMMER_BREAK","IS_WINTER_BREAK","IS_SPRING_BREAK","IS_SCHOOL_BREAK",
    "prev_delay",
    "traffic_level",
    #"Origin_delay_rate","Dest_delay_rate","Airline_delay_rate","Hour_delay_rate",
    "ROUTE_enc",
    #"airline_code",
    "Operating_Airline_9E","Operating_Airline_AA","Operating_Airline_AS","Operating_Airline_B6",
    "Operating_Airline_C5","Operating_Airline_DL","Operating_Airline_F9","Operating_Airline_G4",
    "Operating_Airline_G7","Operating_Airline_HA","Operating_Airline_MQ","Operating_Airline_NK",
    "Operating_Airline_OH","Operating_Airline_OO","Operating_Airline_PT","Operating_Airline_QX",
    "Operating_Airline_UA","Operating_Airline_WN","Operating_Airline_YV","Operating_Airline_YX",
    "Operating_Airline_ZW",
    "DEP_TIME_SLOT_Afternoon","DEP_TIME_SLOT_Evening","DEP_TIME_SLOT_Morning","DEP_TIME_SLOT_Night",
    "SEASON_Fall","SEASON_Spring","SEASON_Summer","SEASON_Winter",
    'temperature_2m', 'relative_humidity_2m', 
    #'dew_point_2m', 'apparent_temperature',
    'precipitation', 
    #'rain', 'snowfall', 
    'snow_depth', 'surface_pressure',
    'cloud_cover', 'wind_speed_10m', 'wind_gusts_10m',
    'wind_dir_sin', 'wind_dir_cos',
]



class FlightDelayFeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, feature_columns: list[str] | None = None):
        self.feature_columns = feature_columns or FEATURE_COLUMNS

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_df = X.copy()
        y_series = pd.Series(y).reset_index(drop=True)
        X_df = X_df.reset_index(drop=True)

        route_values = self._clean_route(X_df.get("ROUTE", pd.Series("", index=X_df.index)))
        unique_routes = pd.Index(route_values.dropna().unique())
        self.route_mapping_ = {route: idx + 1 for idx, route in enumerate(unique_routes)}

        self.global_delay_rate_ = float(y_series.mean())
        self.origin_delay_rate_ = y_series.groupby(X_df["Origin"]).mean().to_dict()
        self.dest_delay_rate_ = y_series.groupby(X_df["Dest"]).mean().to_dict()
        self.airline_delay_rate_ = y_series.groupby(X_df["Operating_Airline"]).mean().to_dict()
        self.hour_delay_rate_ = y_series.groupby(X_df["CRSDepHour"]).mean().to_dict()
        self.feature_names_out_ = list(self.feature_columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = X.copy()

        route_values = self._clean_route(X_df.get("ROUTE", pd.Series("", index=X_df.index)))
        X_df["ROUTE_enc"] = route_values.map(self.route_mapping_).fillna(0).astype(int)
        X_df["Origin_delay_rate"] = X_df["Origin"].map(self.origin_delay_rate_).fillna(self.global_delay_rate_)
        X_df["Dest_delay_rate"] = X_df["Dest"].map(self.dest_delay_rate_).fillna(self.global_delay_rate_)
        X_df["Airline_delay_rate"] = (
            X_df["Operating_Airline"].map(self.airline_delay_rate_).fillna(self.global_delay_rate_)
        )
        X_df["Hour_delay_rate"] = X_df["CRSDepHour"].map(self.hour_delay_rate_).fillna(self.global_delay_rate_)

        for column in self.feature_columns:
            if column not in X_df.columns:
                X_df[column] = 0

        feature_frame = X_df[self.feature_columns].copy()

        # Force a strictly numeric matrix so downstream sklearn estimators do not
        # receive nullable dtypes, object values, or +/-inf that later surface as NaN errors.
        for column in feature_frame.columns:
            if pd.api.types.is_bool_dtype(feature_frame[column]):
                feature_frame[column] = feature_frame[column].astype(int)
            else:
                feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")

        feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
        return feature_frame.fillna(0.0)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_, dtype=object)

    @staticmethod
    def _clean_route(route_series: pd.Series) -> pd.Series:
        return route_series.fillna("UNKNOWN").astype(str).str.strip()


def _build_model_pipelines(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "CatBoost": Pipeline(
            [
                ("features", FlightDelayFeatureBuilder(feature_columns=FEATURE_COLUMNS)),
                (
                    "model",
                    CatBoostClassifier(
                        iterations=200,
                        learning_rate=0.1,
                        depth=6,
                        verbose=0,
                        random_seed=random_state,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("features", FlightDelayFeatureBuilder(feature_columns=FEATURE_COLUMNS)),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.1,
                        scale_pos_weight=3,
                        random_state=random_state,
                        eval_metric="logloss",
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("features", FlightDelayFeatureBuilder(feature_columns=FEATURE_COLUMNS)),
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=3,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("features", FlightDelayFeatureBuilder(feature_columns=FEATURE_COLUMNS)),
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=100,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Logistic Regression": Pipeline(
            [
                ("features", FlightDelayFeatureBuilder(feature_columns=FEATURE_COLUMNS)),
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def _build_threshold_table(y_true: pd.Series, y_prob: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.arange(0.05, 0.951, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
        recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        specificity = 0.0 if (tn + fp) == 0 else tn / (tn + fp)
        false_positive_rate = 1.0 - specificity
        f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)

        rows.append(
            {
                "threshold": float(round(threshold, 2)),
                "precision": float(precision),
                "recall": float(recall),
                "specificity": float(specificity),
                "false_positive_rate": float(false_positive_rate),
                "f1": float(f1),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
        )

    return pd.DataFrame(rows)


def _pick_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    min_precision: float = 0.35,
    max_false_positive_rate: float = 0.25,
) -> tuple[dict[str, float], pd.DataFrame]:
    threshold_table = _build_threshold_table(y_true, y_prob)

    constrained_table = threshold_table[
        (threshold_table["precision"] >= min_precision)
        & (threshold_table["false_positive_rate"] <= max_false_positive_rate)
    ]

    if constrained_table.empty:
        constrained_table = threshold_table[threshold_table["precision"] >= min_precision]

    if constrained_table.empty:
        constrained_table = threshold_table

    best_row = constrained_table.sort_values(
        ["recall", "precision", "specificity", "f1", "threshold"],
        ascending=[False, False, False, False, False],
    ).iloc[0]

    return best_row.to_dict(), threshold_table


def _plot_precision_recall_curves(
    y_true: pd.Series,
    model_probabilities: dict[str, np.ndarray],
    results_df: pd.DataFrame,
    output_path: Path,
):
    plt.figure(figsize=(12, 7))
    for model_name, probabilities in model_probabilities.items():
        precisions, recalls, _ = precision_recall_curve(y_true, probabilities)
        best_f1 = results_df.loc[results_df["model_name"] == model_name, "f1"].iloc[0]
        plt.plot(recalls, precisions, linewidth=2, label=f"{model_name} (F1={best_f1:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves on CV2024")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()


def _plot_metric_bars(results_df: pd.DataFrame, output_path: Path):
    metrics_to_plot = ["roc_auc", "average_precision", "precision", "recall", "specificity", "f1"]
    chart_df = results_df.set_index("model_name")[metrics_to_plot]

    ax = chart_df.plot(kind="bar", figsize=(13, 7))
    ax.set_title("CV2024 Model Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()


def _plot_feature_importance(best_pipeline: Pipeline, model_name: str, output_path: Path):
    estimator = best_pipeline.named_steps["model"]
    feature_names = best_pipeline.named_steps["features"].get_feature_names_out()

    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        importances = np.abs(np.asarray(estimator.coef_)[0])
    else:
        return None

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
        .sort_values("importance", ascending=True)
    )

    plt.figure(figsize=(10, 7))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.title(f"Top Features for Best Model: {model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()
    return importance_df


def _plot_threshold_tradeoff(
    threshold_table: pd.DataFrame,
    model_name: str,
    chosen_threshold: float,
    output_path: Path,
):
    plt.figure(figsize=(12, 7))
    plt.plot(threshold_table["threshold"], threshold_table["recall"], label="Recall", linewidth=2)
    plt.plot(threshold_table["threshold"], threshold_table["precision"], label="Precision", linewidth=2)
    plt.plot(threshold_table["threshold"], threshold_table["specificity"], label="Specificity", linewidth=2)
    plt.axvline(chosen_threshold, color="black", linestyle="--", label=f"Chosen threshold={chosen_threshold:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Threshold Trade-off for {model_name}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()


def _refit_best_pipeline(
    model_name: str,
    best_result_row: pd.Series,
    df_train: pd.DataFrame,
    df_cv: pd.DataFrame,
    target_column: str,
    random_state: int,
) -> Pipeline:
    combined_df = pd.concat([df_train, df_cv], axis=0, ignore_index=True)
    combined_y = combined_df[target_column].copy()
    combined_X = combined_df.copy()

    pipelines = _build_model_pipelines(random_state=random_state)
    best_pipeline = pipelines[model_name]
    estimator = best_pipeline.named_steps["model"]

    if isinstance(estimator, RandomForestClassifier):
        estimator.set_params(
            n_estimators=200,
            min_samples_leaf=3,
            class_weight={0: 1, 1: 2},
        )
    elif isinstance(estimator, GradientBoostingClassifier):
        estimator.set_params(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=3,
        )
    elif isinstance(estimator, XGBClassifier):
        estimator.set_params(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            scale_pos_weight=4,
        )
    elif isinstance(estimator, CatBoostClassifier):
        estimator.set_params(
            iterations=300,
            depth=6,
            learning_rate=0.08,
            auto_class_weights="Balanced",
        )
    elif isinstance(estimator, LogisticRegression):
        estimator.set_params(
            C=0.8,
            class_weight={0: 1, 1: 2},
        )

    best_pipeline.fit(combined_X, combined_y)
    return best_pipeline


def train_and_select_best_model(
    df_train: pd.DataFrame,
    df_cv: pd.DataFrame,
    target_column: str = "DepDel15",
    output_dir: str | Path = "flight_delay_model_artifacts",
    min_precision: float = 0.38,
    max_false_positive_rate: float = 0.22,
    random_state: int = 42,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train = df_train.copy()
    y_train = df_train[target_column].copy()
    X_cv = df_cv.copy()
    y_cv = df_cv[target_column].copy()

    pipelines = _build_model_pipelines(random_state=random_state)
    results = []
    model_probabilities = {}
    trained_pipelines = {}
    threshold_tables = {}

    for model_name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_cv)[:, 1]
        threshold_metrics, threshold_table = _pick_threshold(
            y_cv,
            y_prob,
            min_precision=min_precision,
            max_false_positive_rate=max_false_positive_rate,
        )

        row = {
            "model_name": model_name,
            "roc_auc": float(roc_auc_score(y_cv, y_prob)),
            "average_precision": float(average_precision_score(y_cv, y_prob)),
            **threshold_metrics,
        }
        results.append(row)
        model_probabilities[model_name] = y_prob
        trained_pipelines[model_name] = pipeline
        threshold_tables[model_name] = threshold_table

    results_df = pd.DataFrame(results).sort_values(
        ["recall", "precision", "specificity", "f1", "roc_auc"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    best_model_name = results_df.loc[0, "model_name"]
    best_threshold = float(results_df.loc[0, "threshold"])
    best_pipeline = _refit_best_pipeline(
        model_name=best_model_name,
        best_result_row=results_df.loc[0],
        df_train=df_train,
        df_cv=df_cv,
        target_column=target_column,
        random_state=random_state,
    )

    comparison_path = output_dir / "model_comparison.csv"
    pr_curve_path = output_dir / "precision_recall_curves.png"
    metric_chart_path = output_dir / "model_metrics.png"
    importance_path = output_dir / "best_model_feature_importance.png"
    threshold_chart_path = output_dir / "best_model_threshold_tradeoff.png"
    threshold_table_path = output_dir / "best_model_thresholds.csv"
    pipeline_path = output_dir / "best_delay_model_pipeline.joblib"
    metadata_path = output_dir / "best_delay_model_metadata.json"

    results_df.to_csv(comparison_path, index=False)
    _plot_precision_recall_curves(y_cv, model_probabilities, results_df, pr_curve_path)
    _plot_metric_bars(results_df, metric_chart_path)
    importance_df = _plot_feature_importance(best_pipeline, best_model_name, importance_path)
    best_threshold_table = threshold_tables[best_model_name]
    best_threshold_table.to_csv(threshold_table_path, index=False)
    _plot_threshold_tradeoff(best_threshold_table, best_model_name, best_threshold, threshold_chart_path)

    artifact = {
        "model_name": best_model_name,
        "threshold": best_threshold,
        "target_column": target_column,
        "feature_columns": FEATURE_COLUMNS,
        "pipeline": best_pipeline,
    }
    joblib.dump(artifact, pipeline_path)

    metadata = {
        "best_model_name": best_model_name,
        "best_threshold": best_threshold,
        "target_column": target_column,
        "min_precision": min_precision,
        "max_false_positive_rate": max_false_positive_rate,
        "comparison_csv": str(comparison_path),
        "precision_recall_chart": str(pr_curve_path),
        "metric_chart": str(metric_chart_path),
        "feature_importance_chart": str(importance_path),
        "threshold_tradeoff_chart": str(threshold_chart_path),
        "threshold_table_csv": str(threshold_table_path),
        "pipeline_path": str(pipeline_path),
        "ranking": results_df.to_dict(orient="records"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Best model selected:", best_model_name)
    print(f"Saved pipeline: {pipeline_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved comparison table: {comparison_path}")

    return {
        "comparison": results_df,
        "best_model_name": best_model_name,
        "best_threshold": best_threshold,
        "pipeline_path": str(pipeline_path),
        "metadata_path": str(metadata_path),
        "comparison_path": str(comparison_path),
        "feature_importance": importance_df,
        "threshold_table_path": str(threshold_table_path),
    }


def load_saved_pipeline(pipeline_path: str | Path = "flight_delay_model_artifacts/best_delay_model_pipeline.joblib"):
    return joblib.load(pipeline_path)


def score_with_saved_pipeline(
    df_new: pd.DataFrame,
    pipeline_path: str | Path = "flight_delay_model_artifacts/best_delay_model_pipeline.joblib",
    output_csv: str | Path | None = None,
):
    artifact = load_saved_pipeline(pipeline_path)
    pipeline = artifact["pipeline"]
    threshold = float(artifact["threshold"])

    probabilities = pipeline.predict_proba(df_new)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    scored_df = df_new.copy()
    scored_df["delay_probability"] = probabilities
    scored_df["predicted_delay"] = predictions

    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        scored_df.to_csv(output_csv, index=False)
        print(f"Saved scored dataset: {output_csv}")

    return scored_df
