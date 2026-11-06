from typing import List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def cosine_similarity_tool(doc1_idx: int, doc2_idx: int) -> dict:
        """Compute cosine similarity between two documents in the loaded dataset.

        Uses session.feature_matrix rows if available, or vectorizes document texts if needed.
        Handles zero-vector/empty documents by returning 0.0 similarity instead of NaN.
        Truncates returned document texts in summary to 100 characters.

        Args:
            doc1_idx: Row index of the first document.
            doc2_idx: Row index of the second document.

        Returns:
            Dict containing result_id, doc1_idx, doc2_idx, similarity score,
            doc1_text (truncated), and doc2_text (truncated).
        """
        df = session.dataframe
        if df is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        total_docs = len(df)
        if doc1_idx < 0 or doc1_idx >= total_docs:
            return {"error": f"doc1_idx {doc1_idx} out of range (0 to {total_docs - 1})."}
        if doc2_idx < 0 or doc2_idx >= total_docs:
            return {"error": f"doc2_idx {doc2_idx} out of range (0 to {total_docs - 1})."}

        # Extract document texts truncated to 100 characters
        text_col = "text" if "text" in df.columns else df.columns[0]
        raw_text1 = str(df.iloc[doc1_idx].get(text_col, ""))
        raw_text2 = str(df.iloc[doc2_idx].get(text_col, ""))
        doc1_text = raw_text1[:100]
        doc2_text = raw_text2[:100]

        # Compute cosine similarity using feature_matrix if available
        if session.feature_matrix is not None:
            vec1 = session.feature_matrix[doc1_idx]
            vec2 = session.feature_matrix[doc2_idx]
            sim_matrix = cosine_similarity(vec1, vec2)
            raw_sim = float(sim_matrix[0, 0])
        else:
            # Fallback to count vectorizer artifact if feature_matrix not built yet
            vectorizer = session.artifacts.get("count_vectorizer")
            if vectorizer is None:
                return {"error": "No feature_matrix or count_vectorizer found. Run build_dtm_tool first."}
            vecs = vectorizer.transform([raw_text1, raw_text2])
            sim_matrix = cosine_similarity(vecs[0], vecs[1])
            raw_sim = float(sim_matrix[0, 0])

        # Guard against NaN (e.g. empty document zero vector)
        similarity = 0.0 if np.isnan(raw_sim) else round(raw_sim, 4)

        result_id = session.next_result_id("cosine_similarity")

        summary = {
            "result_id": result_id,
            "doc1_idx": doc1_idx,
            "doc2_idx": doc2_idx,
            "similarity": similarity,
            "doc1_text": doc1_text,
            "doc2_text": doc2_text,
        }

        session.store_result(
            "cosine_similarity_tool",
            {"doc1_idx": doc1_idx, "doc2_idx": doc2_idx},
            summary,
        )

        return summary

    @tool
    def feature_correlation_matrix_tool(top_n: int = 20) -> dict:
        """Compute Pearson correlation matrix and heatmap over top-N terms by variance.

        Computes term variance efficiently directly on session.feature_matrix, selects top_n terms
        by variance, expands only those columns to dense arrays, computes Pearson correlation matrix,
        and sets session.pending_figure to the correlation heatmap.

        Args:
            top_n: Number of highest-variance terms to include (default: 20).

        Returns:
            Dict containing result_id, top_terms list, and correlation_matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found. Run build_dtm_tool first."}

        X = session.feature_matrix
        vocab = np.array(session.feature_names)
        n_features = X.shape[1]

        if n_features == 0:
            return {"error": "Feature matrix has no terms/columns."}

        # Efficient variance computation on sparse matrix without full dense expansion: Var = E[X^2] - (E[X])^2
        mean_X = np.asarray(X.mean(axis=0)).ravel()
        mean_X2 = np.asarray(X.power(2).mean(axis=0)).ravel()
        variances = mean_X2 - (mean_X ** 2)

        # Select indices of top_n terms by variance (descending)
        actual_k = min(top_n, n_features)
        top_indices = np.argsort(variances)[::-1][:actual_k]

        top_terms = vocab[top_indices].tolist()

        # Slice only the selected top_n columns and convert ONLY those to a dense array
        X_top_dense = X[:, top_indices].toarray()

        # Compute Pearson correlation matrix across top terms
        corr_matrix = np.corrcoef(X_top_dense, rowvar=False)

        # Ensure corr_matrix is a 2D numpy array (handles 1x1 case cleanly)
        if corr_matrix.ndim == 0:
            corr_matrix = np.array([[1.0]])

        # Handle any numerical NaNs in corr_matrix (e.g. constant terms)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        corr_matrix_rounded = np.round(corr_matrix, 4).tolist()

        # Build heatmap visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            xticklabels=top_terms,
            yticklabels=top_terms,
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            ax=ax,
        )
        ax.set_title(f"Feature Correlation Matrix (Top {actual_k} Terms by Variance)")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("feature_correlation_matrix")

        summary = {
            "result_id": result_id,
            "top_terms": top_terms,
            "correlation_matrix": corr_matrix_rounded,
        }

        session.store_result(
            "feature_correlation_matrix_tool",
            {"top_n": top_n},
            summary,
        )

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
