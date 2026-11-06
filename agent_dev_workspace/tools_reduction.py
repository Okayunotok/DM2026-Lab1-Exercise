from typing import Literal, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = 42,
    ) -> dict:
        """Reduce dimensionality of the document-term matrix using PCA, t-SNE, or UMAP.

        Computes 2D reduced coordinates from session.feature_matrix and generates a
        scatter plot colored by session.labels, setting session.pending_figure.
        Stores full coordinates in session.artifacts["reduced_coordinates"].

        Args:
            method: Reduction algorithm - 'pca', 'tsne', or 'umap' (default: 'pca').
            n_components: Target number of dimensions (default: 2).
            perplexity: Perplexity parameter for t-SNE (default: 30.0).
            n_neighbors: Number of neighbors for UMAP (default: 15).
            random_state: Seed for random number generator reproducibility (default: 42).

        Returns:
            Dict containing result_id, method, shape, explained_variance_ratio (PCA only),
            effective_perplexity (t-SNE), effective_n_neighbors (UMAP), and sample_coordinates.
        """
        X = session.feature_matrix
        labels = session.labels

        if X is None:
            return {"error": "No document-term matrix found. Please run build_dtm_tool first."}

        # Convert sparse matrix to dense array for dimensional reduction algorithms
        X_dense = X.toarray()
        n_docs = X_dense.shape[0]

        if n_docs == 0:
            return {"error": "Document-term matrix contains zero documents."}

        explained_variance_ratio = None
        effective_perplexity = None
        effective_n_neighbors = None
        method_lower = method.lower()

        if method_lower == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            coords = reducer.fit_transform(X_dense)
            explained_variance_ratio = [
                round(float(v), 4) for v in reducer.explained_variance_ratio_
            ]

        elif method_lower == "tsne":
            # Guard against perplexity >= n_samples error on small datasets
            effective_perplexity = min(perplexity, max(1.0, float(n_docs - 1)))
            reducer = TSNE(
                n_components=n_components,
                perplexity=effective_perplexity,
                random_state=random_state,
            )
            coords = reducer.fit_transform(X_dense)

        elif method_lower == "umap":
            try:
                import umap
            except ImportError:
                return {"error": "umap-learn package is not installed."}

            # Guard against n_neighbors > n_samples on small datasets
            effective_n_neighbors = min(n_neighbors, max(2, n_docs))
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=effective_n_neighbors,
                random_state=random_state,
            )
            coords = reducer.fit_transform(X_dense)

        else:
            return {"error": f"Unsupported reduction method '{method}'. Must be 'pca', 'tsne', or 'umap'."}

        # Store full reduced coordinates in session.artifacts
        session.artifacts["reduced_coordinates"] = coords

        # Build scatter plot colored by session.labels and set session.pending_figure
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_hue = labels if labels is not None else ["Document"] * n_docs

        sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1] if n_components > 1 else np.zeros(n_docs),
            hue=plot_hue,
            ax=ax,
            palette="Set2",
            s=60,
        )

        ax.set_title(f"{method.upper()} Dimensionality Reduction")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2" if n_components > 1 else "Zero")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()

        # Pass plot to session.pending_figure without calling plt.show()
        session.pending_figure = fig

        result_id = session.next_result_id("reduce_dimensions")

        # Format first 5 document coordinates as sample
        sample_coords = [
            [round(float(val), 4) for val in row] for row in coords[:5]
        ]

        summary = {
            "result_id": result_id,
            "method": method_lower,
            "shape": list(coords.shape),
            "sample_coordinates": sample_coords,
        }

        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        if effective_perplexity is not None:
            summary["effective_perplexity"] = effective_perplexity

        if effective_n_neighbors is not None:
            summary["effective_n_neighbors"] = effective_n_neighbors

        session.store_result(
            "reduce_dimensions_tool",
            {
                "method": method,
                "n_components": n_components,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors,
                "random_state": random_state,
            },
            summary,
        )

        return summary

    return [reduce_dimensions_tool]
