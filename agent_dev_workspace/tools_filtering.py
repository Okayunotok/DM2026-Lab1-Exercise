from typing import List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.0) -> dict:
        """Filter terms by column variance across the document-term matrix.

        Computes term variance directly on the sparse matrix session.feature_matrix
        without converting to a dense array. Flags terms with variance strictly greater
        than threshold for keeping, matching sklearn's VarianceThreshold. Does not mutate
        session.feature_matrix. Passes a full DataFrame report to session.store_result
        for downstream visualization.

        Args:
            threshold: Minimum variance threshold; terms with variance strictly > threshold
                       are kept (default: 0.0).

        Returns:
            Dict containing result_id, threshold, total_terms, kept_count,
            removed_count, sample_kept_terms (up to 10 sample terms), and
            sample_removed_terms (up to 10 sample terms).
        """
        X = session.feature_matrix
        feature_names = session.feature_names

        if X is None or feature_names is None:
            return {"error": "No document-term matrix found. Please run build_dtm_tool first."}

        # Compute population variance directly on sparse matrix: E[X^2] - (E[X])^2
        # E[X] (mean per column across N documents)
        n_docs = X.shape[0]
        if n_docs == 0:
            return {"error": "Document-term matrix has zero documents."}

        means = np.asarray(X.mean(axis=0)).ravel()
        
        # E[X^2] (mean of squared elements per column)
        X_squared = X.copy()
        X_squared.data **= 2
        means_sq = np.asarray(X_squared.mean(axis=0)).ravel()

        variances = means_sq - (means ** 2)

        # Build full_report DataFrame: first column 'term', second column 'variance', sorted descending
        report_df = pd.DataFrame({
            "term": feature_names,
            "variance": variances
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        # Identify kept and removed terms based on threshold (strictly greater than threshold)
        kept_mask = variances > threshold
        kept_terms = [name for name, keep in zip(feature_names, kept_mask) if keep]
        removed_terms = [name for name, keep in zip(feature_names, kept_mask) if not keep]

        total_terms = len(feature_names)
        kept_count = len(kept_terms)
        removed_count = len(removed_terms)

        result_id = session.next_result_id("variance_filter")

        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "total_terms": total_terms,
            "kept_count": kept_count,
            "removed_count": removed_count,
            "sample_kept_terms": kept_terms[:10],
            "sample_removed_terms": removed_terms[:10],
        }

        # Store summary and full_report DataFrame for visualize_result_tool
        session.store_result(
            "variance_filter_tool",
            {"threshold": threshold},
            summary,
            full_report=report_df,
        )

        return summary

    @tool
    def pearson_filter_tool(target_class: str, threshold: float = 0.0) -> dict:
        """Compute Pearson correlation between term counts and a target class.

        Calculates linear correlation (pearson_r) for each term against a binary target
        indicator for target_class. Handles zero-variance terms by setting r = 0.0.

        Note on threshold:
        - Filtering terms based on absolute correlation (|pearson_r| > threshold) is a
          design choice for pipeline flexibility; TOOLS_PIPELINE_BREAKDOWN.md only specifies
          computing Pearson correlation.

        Args:
            target_class: The category name to correlate against (one-vs-rest binary target).
            threshold: Minimum absolute correlation threshold for keeping terms (default: 0.0).

        Returns:
            Dict containing result_id, target_class, threshold, total_terms, kept_count,
            removed_count, sample_kept_terms (up to 10 sample terms), and sample_removed_terms.
        """
        X = session.feature_matrix
        feature_names = session.feature_names
        labels = session.labels

        if X is None or feature_names is None:
            return {"error": "No document-term matrix found. Please run build_dtm_tool first."}

        if labels is None:
            return {"error": "No category labels found in session state."}

        # Create binary target indicator vector y (1 for target_class, 0 otherwise)
        y = (np.asarray(labels) == target_class).astype(float)
        if y.sum() == 0:
            return {"error": f"Target class '{target_class}' not found in document labels."}

        n_docs, n_terms = X.shape
        y_std = np.std(y, ddof=0)
        if y_std == 0:
            return {"error": f"Target indicator vector for '{target_class}' has zero variance."}

        # Convert matrix to CSC format for fast column slicing
        X_csc = X.tocsc()

        # Compute Pearson correlation column by column
        pearson_rs = []
        y_mean = np.mean(y)

        for col_idx in range(n_terms):
            x_col = X_csc.getcol(col_idx).toarray().ravel()
            x_std = np.std(x_col, ddof=0)
            if x_std == 0:
                pearson_rs.append(0.0)
            else:
                cov = np.mean((x_col - np.mean(x_col)) * (y - y_mean))
                r = cov / (x_std * y_std)
                pearson_rs.append(0.0 if np.isnan(r) else float(r))

        pearson_rs = np.array(pearson_rs)

        # Build full_report DataFrame: 1st column 'term', 2nd column 'pearson_r', sorted by |r| descending
        report_df = pd.DataFrame({
            "term": feature_names,
            "pearson_r": pearson_rs,
            "abs_r": np.abs(pearson_rs),
        }).sort_values(by="abs_r", ascending=False).drop(columns=["abs_r"]).reset_index(drop=True)

        kept_mask = np.abs(pearson_rs) > threshold
        kept_terms = [name for name, keep in zip(feature_names, kept_mask) if keep]
        removed_terms = [name for name, keep in zip(feature_names, kept_mask) if not keep]

        result_id = session.next_result_id("pearson_filter")

        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(feature_names),
            "kept_count": len(kept_terms),
            "removed_count": len(removed_terms),
            "sample_kept_terms": kept_terms[:10],
            "sample_removed_terms": removed_terms[:10],
        }

        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=report_df,
        )

        return summary

    @tool
    def spearman_filter_tool(target_class: str, threshold: float = 0.0) -> dict:
        """Compute Spearman rank correlation between term counts and a target class.

        Calculates rank correlation (spearman_r) for each term against a binary target
        indicator for target_class. Handles zero-variance terms by setting r = 0.0.

        Note on threshold:
        - Filtering terms based on absolute rank correlation (|spearman_r| > threshold) is a
          design choice for pipeline flexibility; TOOLS_PIPELINE_BREAKDOWN.md only specifies
          computing Spearman correlation.

        Args:
            target_class: The category name to correlate against (one-vs-rest binary target).
            threshold: Minimum absolute rank correlation threshold for keeping terms (default: 0.0).

        Returns:
            Dict containing result_id, target_class, threshold, total_terms, kept_count,
            removed_count, sample_kept_terms (up to 10 sample terms), and sample_removed_terms.
        """
        X = session.feature_matrix
        feature_names = session.feature_names
        labels = session.labels

        if X is None or feature_names is None:
            return {"error": "No document-term matrix found. Please run build_dtm_tool first."}

        if labels is None:
            return {"error": "No category labels found in session state."}

        # Create binary target indicator vector y (1 for target_class, 0 otherwise)
        y = (np.asarray(labels) == target_class).astype(float)
        if y.sum() == 0:
            return {"error": f"Target class '{target_class}' not found in document labels."}

        n_docs, n_terms = X.shape

        # Convert matrix to CSC format for fast column slicing
        X_csc = X.tocsc()

        spearman_rs = []
        for col_idx in range(n_terms):
            x_col = X_csc.getcol(col_idx).toarray().ravel()
            if np.std(x_col) == 0:
                spearman_rs.append(0.0)
            else:
                r, _ = stats.spearmanr(x_col, y)
                spearman_rs.append(0.0 if np.isnan(r) else float(r))

        spearman_rs = np.array(spearman_rs)

        # Build full_report DataFrame: 1st column 'term', 2nd column 'spearman_r', sorted by |r| descending
        report_df = pd.DataFrame({
            "term": feature_names,
            "spearman_r": spearman_rs,
            "abs_r": np.abs(spearman_rs),
        }).sort_values(by="abs_r", ascending=False).drop(columns=["abs_r"]).reset_index(drop=True)

        kept_mask = np.abs(spearman_rs) > threshold
        kept_terms = [name for name, keep in zip(feature_names, kept_mask) if keep]
        removed_terms = [name for name, keep in zip(feature_names, kept_mask) if not keep]

        result_id = session.next_result_id("spearman_filter")

        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(feature_names),
            "kept_count": len(kept_terms),
            "removed_count": len(removed_terms),
            "sample_kept_terms": kept_terms[:10],
            "sample_removed_terms": removed_terms[:10],
        }

        session.store_result(
            "spearman_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=report_df,
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
