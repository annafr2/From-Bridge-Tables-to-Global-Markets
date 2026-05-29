"""
src/stage1_clustering/preprocessing.py
========================================
Full pre-processing pipeline for player feature clustering.

Pipeline (Dr. Rami's review, May 2026):
    1. Variance check       — flag low-variance features (cv < 0.10)
    2. Correlation analysis — flag highly correlated pairs (|r| > 0.7)
    3. Outlier detection    — Mahalanobis distance + IQR
    4. Scaling              — RobustScaler (less sensitive to outliers than StandardScaler)
    5. PCA                  — keep enough components to explain 80% variance

The output is a clean feature matrix ready for K-Means / GMM / HDBSCAN.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


# ── Defaults ──────────────────────────────────────────────────────────────────

MIN_CV_THRESHOLD: float = 0.10        # below this = low-variance feature
MAX_CORR_THRESHOLD: float = 0.70      # above this = redundant feature pair
PCA_TARGET_VARIANCE: float = 0.80     # keep components until 80% explained
MAHALANOBIS_ALPHA: float = 0.01       # p < 0.01 = outlier
RANDOM_STATE: int = 42


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class PreprocessingReport:
    """Diagnostic report from the preprocessing pipeline."""

    n_input_players: int
    n_output_players: int
    n_input_features: int
    n_output_features: int

    variance_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    low_variance_features: list[str] = field(default_factory=list)

    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    correlated_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    dropped_correlated: list[str] = field(default_factory=list)

    n_outliers_detected: int = 0
    outlier_method: str = "mahalanobis"

    final_features: list[str] = field(default_factory=list)
    pca_n_components: int = 0
    pca_explained_variance: list[float] = field(default_factory=list)
    pca_cumulative_variance: float = 0.0
    scaler_type: str = ""

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            "=" * 60,
            "PREPROCESSING REPORT",
            "=" * 60,
            f"Players:  {self.n_input_players} -> {self.n_output_players}  "
            f"(removed {self.n_input_players - self.n_output_players} outliers)",
            f"Features: {self.n_input_features} -> {self.n_output_features}  "
            f"(removed {self.n_input_features - self.n_output_features})",
            "",
            f"Low-variance features removed ({len(self.low_variance_features)}):",
        ]
        for f in self.low_variance_features:
            cv = self.variance_table.loc[f, "cv"]
            lines.append(f"  - {f:<25} cv = {cv:.3f}")

        lines.append("")
        lines.append(f"Correlated pairs (|r| > {MAX_CORR_THRESHOLD}):")
        if self.correlated_pairs:
            for a, b, r in self.correlated_pairs:
                lines.append(f"  - {a:<25} <-> {b:<25} r = {r:+.3f}")
        else:
            lines.append("  (none)")

        lines.append("")
        lines.append(f"Outliers removed: {self.n_outliers_detected} "
                     f"(method: {self.outlier_method})")
        lines.append("")
        lines.append(f"Scaler: {self.scaler_type}")
        lines.append(f"Final features used: {self.final_features}")
        lines.append(f"PCA components: {self.pca_n_components} "
                     f"(cumulative variance: {self.pca_cumulative_variance*100:.1f}%)")
        for i, v in enumerate(self.pca_explained_variance, 1):
            lines.append(f"  PC{i}: {v*100:.1f}%")
        lines.append("=" * 60)
        return "\n".join(lines)


# ── Step 1: Variance ──────────────────────────────────────────────────────────

def variance_analysis(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_cv: float = MIN_CV_THRESHOLD,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute coefficient of variation per feature.

    Args:
        df:           Input feature DataFrame.
        feature_cols: Columns to analyze.
        min_cv:       Threshold below which a feature is "low-variance" noise.

    Returns:
        (variance_table, low_variance_features) where low_variance_features
        is a list of column names with cv < min_cv.
    """
    stats = df[feature_cols].agg(["mean", "std", "min", "max"]).T
    stats["cv"] = stats["std"] / stats["mean"].replace(0, np.nan)
    stats["keep"] = stats["cv"] >= min_cv
    low_var = stats[~stats["keep"]].index.tolist()
    logger.info("Variance analysis: %d / %d features flagged as low-variance",
                len(low_var), len(feature_cols))
    return stats, low_var


# ── Step 2: Correlation ───────────────────────────────────────────────────────

def correlation_analysis(
    df: pd.DataFrame,
    feature_cols: list[str],
    max_corr: float = MAX_CORR_THRESHOLD,
) -> tuple[pd.DataFrame, list[tuple[str, str, float]], list[str]]:
    """Find highly correlated feature pairs and suggest which to drop.

    Strategy: for each correlated pair, drop the feature whose mean absolute
    correlation with all OTHER features is higher (i.e., the more redundant one).

    Args:
        df:           Input feature DataFrame.
        feature_cols: Columns to analyze.
        max_corr:     Threshold above which two features are "redundant".

    Returns:
        (corr_matrix, correlated_pairs, features_to_drop)
    """
    corr = df[feature_cols].corr().abs()
    pairs: list[tuple[str, str, float]] = []
    to_drop: set[str] = set()

    # Upper triangle only (avoid duplicates and diagonal)
    for i, col_a in enumerate(feature_cols):
        for col_b in feature_cols[i + 1:]:
            r = corr.loc[col_a, col_b]
            if r > max_corr:
                pairs.append((col_a, col_b, float(r)))
                # Drop whichever has higher mean correlation with everyone else
                mean_corr_a = corr[col_a].drop(col_a).mean()
                mean_corr_b = corr[col_b].drop(col_b).mean()
                to_drop.add(col_a if mean_corr_a > mean_corr_b else col_b)

    logger.info("Correlation analysis: %d highly correlated pairs, %d features to drop",
                len(pairs), len(to_drop))
    return corr, pairs, sorted(to_drop)


# ── Step 3: Outlier detection ─────────────────────────────────────────────────

def detect_outliers_mahalanobis(
    X: np.ndarray,
    alpha: float = MAHALANOBIS_ALPHA,
) -> np.ndarray:
    """Detect multivariate outliers using Mahalanobis distance.

    A point is an outlier if its squared Mahalanobis distance exceeds the
    chi-squared critical value with df = n_features.

    Args:
        X:     Feature matrix (n_samples, n_features).
        alpha: Significance level (e.g. 0.01 = top 1% outliers).

    Returns:
        Boolean mask of length n_samples, True where outlier.
    """
    cov = EmpiricalCovariance().fit(X)
    distances = cov.mahalanobis(X)
    threshold = chi2.ppf(1 - alpha, df=X.shape[1])
    outlier_mask = distances > threshold
    logger.info("Mahalanobis: %d outliers (threshold = chi2.ppf(%.3f, df=%d) = %.2f)",
                outlier_mask.sum(), 1 - alpha, X.shape[1], threshold)
    return outlier_mask


# ── Full pipeline ─────────────────────────────────────────────────────────────

def preprocess_full(
    features: pd.DataFrame,
    feature_cols: list[str],
    min_cv: float = MIN_CV_THRESHOLD,
    max_corr: float = MAX_CORR_THRESHOLD,
    target_variance: float = PCA_TARGET_VARIANCE,
    outlier_alpha: float = MAHALANOBIS_ALPHA,
    scaler_type: str = "robust",
    remove_outliers: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, PreprocessingReport]:
    """Run the full preprocessing pipeline.

    Pipeline:
        1. Remove low-variance features (cv < min_cv)
        2. Remove highly correlated features (|r| > max_corr)
        3. Detect and (optionally) remove multivariate outliers
        4. Scale (RobustScaler or StandardScaler)
        5. PCA — keep components until cumulative variance >= target_variance

    Args:
        features:        DataFrame from compute_player_features().
        feature_cols:    All candidate feature columns.
        min_cv:          Coefficient of variation threshold for low-variance filter.
        max_corr:        Correlation threshold for redundancy filter.
        target_variance: Target cumulative PCA variance (e.g., 0.80).
        outlier_alpha:   Mahalanobis significance level.
        scaler_type:     "robust" (less sensitive to outliers) or "standard".
        remove_outliers: If True, drop detected outliers from the output.

    Returns:
        (X_scaled, X_pca, kept_features_df, report)
        where:
          - X_scaled is the scaled feature matrix (n_clean, n_kept_features)
          - X_pca    is the PCA-reduced matrix (n_clean, n_pca_components)
          - kept_features_df is the original features DataFrame, subset to
            non-outlier rows, with index preserved
          - report is a PreprocessingReport with all diagnostics
    """
    logger.info("=== Preprocessing pipeline (Dr. Rami's review) ===")

    report = PreprocessingReport(
        n_input_players=len(features),
        n_output_players=len(features),
        n_input_features=len(feature_cols),
        n_output_features=len(feature_cols),
        scaler_type=scaler_type,
    )

    # ── Step 1: Variance ──────────────────────────────────────────────────────
    var_table, low_var = variance_analysis(features, feature_cols, min_cv)
    report.variance_table = var_table
    report.low_variance_features = low_var
    kept_features = [c for c in feature_cols if c not in low_var]
    logger.info("After variance filter: %d features", len(kept_features))

    # ── Step 2: Correlation ───────────────────────────────────────────────────
    corr_matrix, corr_pairs, corr_drops = correlation_analysis(
        features, kept_features, max_corr,
    )
    report.correlation_matrix = corr_matrix
    report.correlated_pairs = corr_pairs
    report.dropped_correlated = corr_drops
    kept_features = [c for c in kept_features if c not in corr_drops]
    logger.info("After correlation filter: %d features", len(kept_features))

    # ── Step 3: Outliers ──────────────────────────────────────────────────────
    X_raw = features[kept_features].fillna(0).values

    # Scale temporarily for fair outlier detection
    temp_scaler = StandardScaler()
    X_temp = temp_scaler.fit_transform(X_raw)
    outlier_mask = detect_outliers_mahalanobis(X_temp, alpha=outlier_alpha)
    report.n_outliers_detected = int(outlier_mask.sum())
    report.outlier_method = f"mahalanobis (alpha={outlier_alpha})"

    if remove_outliers:
        keep_mask = ~outlier_mask
        clean_df = features.loc[keep_mask].reset_index(drop=True)
        X_raw_clean = X_raw[keep_mask]
        logger.info("After outlier removal: %d players (was %d)",
                    len(clean_df), len(features))
    else:
        clean_df = features.copy()
        X_raw_clean = X_raw
        logger.info("Outlier removal disabled — keeping all %d players",
                    len(clean_df))
    report.n_output_players = len(clean_df)

    # ── Step 4: Final scaling ─────────────────────────────────────────────────
    if scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler_type {scaler_type!r}")

    X_scaled = scaler.fit_transform(X_raw_clean)
    logger.info("Scaled with %s (final shape %s)", scaler_type, X_scaled.shape)

    # ── Step 5: PCA — auto-select n_components ────────────────────────────────
    # First fit with all components to see the variance curve
    full_pca = PCA(n_components=len(kept_features), random_state=RANDOM_STATE)
    full_pca.fit(X_scaled)

    cumulative = np.cumsum(full_pca.explained_variance_ratio_)
    # Find first index where cumulative >= target
    n_components = int(np.searchsorted(cumulative, target_variance) + 1)
    n_components = min(n_components, len(kept_features))

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    report.final_features = kept_features
    report.n_output_features = len(kept_features)
    report.pca_n_components = n_components
    report.pca_explained_variance = [
        float(round(v, 4)) for v in pca.explained_variance_ratio_
    ]
    report.pca_cumulative_variance = float(round(
        np.sum(pca.explained_variance_ratio_), 4
    ))

    logger.info("PCA: %d components -> %.1f%% cumulative variance",
                n_components, report.pca_cumulative_variance * 100)

    return X_scaled, X_pca, clean_df, report
