from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def build_dtm_tool(
        text_column: str = "text",
        ngram_range: List[int] = [1, 1],
        min_df: int = 1,
        stop_words: Optional[str] = None,
    ) -> dict:
        """Build a document-term matrix (DTM) from the loaded dataset.

        Computes count vectorizer features across documents and updates session state:
        stores session.feature_matrix, session.feature_names, and
        session.artifacts["count_vectorizer"]. Also sets category labels if present.

        Args:
            text_column: Column containing document text (default: 'text').
            ngram_range: Lower and upper boundary of range of n-values to extract (default: [1, 1]).
                         Typed as List[int] to ensure Gemini API schema compatibility.
            min_df: Minimum document frequency threshold (default: 1).
            stop_words: Optional stop words removal setting, e.g. 'english' (default: None).

        Returns:
            Dict containing result_id, vocabulary_size, n_documents, non_zero,
            total_elements, sparsity_pct, and a sample of the first 10 vocabulary terms.
        """
        df = session.dataframe
        if df is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        if text_column not in df.columns:
            return {"error": f"Column '{text_column}' not found in dataset."}

        # Guard check: Validate ngram_range structure and bounds
        if (
            not isinstance(ngram_range, (list, tuple))
            or len(ngram_range) != 2
            or not all(isinstance(n, int) and n > 0 for n in ngram_range)
            or ngram_range[0] > ngram_range[1]
        ):
            return {
                "error": (
                    f"Invalid ngram_range {ngram_range}. Must be a list of two positive "
                    "integers [min_n, max_n] where min_n <= max_n."
                )
            }

        # Convert list parameter to tuple for CountVectorizer
        ngram_tuple = tuple(ngram_range)

        vectorizer = CountVectorizer(
            ngram_range=ngram_tuple,
            min_df=min_df,
            stop_words=stop_words,
        )

        # Fit and transform dataset text
        texts = df[text_column].fillna("").astype(str)
        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session state attributes
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in df.columns:
            session.set_labels(df["category_name"].to_numpy())
        elif "category" in df.columns:
            session.set_labels(df["category"].to_numpy())

        # Compute matrix sparsity metrics matching spec field names
        n_documents, vocabulary_size = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = n_documents * vocabulary_size
        sparsity_pct = 100.0 * (1.0 - non_zero / total_elements) if total_elements > 0 else 0.0

        result_id = session.next_result_id("build_dtm")

        summary = {
            "result_id": result_id,
            "vocabulary_size": vocabulary_size,
            "n_documents": n_documents,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
            "sample_terms": feature_names[:10],
        }

        session.store_result(
            "build_dtm_tool",
            {
                "text_column": text_column,
                "ngram_range": ngram_range,
                "min_df": min_df,
                "stop_words": stop_words,
            },
            summary,
        )

        return summary

    @tool
    def term_frequency_tool(top_k: int = 20) -> dict:
        """Aggregates term frequencies across all documents in the document-term matrix.

        Sums columns of session.feature_matrix to calculate total occurrences for each term,
        sorts them in descending order of frequency (and alphabetically by term for tie-breaking),
        and returns the top-K terms summary as well as storing full_report for visualization.

        Args:
            top_k: Number of top most frequent terms to return in the summary (default: 20).

        Returns:
            Dict containing result_id, top_terms list, total_terms count, and total_occurrences.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found. Run build_dtm_tool first."}

        # Sum total counts per column (term) across all documents (rows)
        frequencies = np.asarray(session.feature_matrix.sum(axis=0)).ravel()

        # Build DataFrame for full report
        df_report = pd.DataFrame({
            "term": session.feature_names,
            "frequency": frequencies
        })

        # Sort by frequency descending, then by term ascending for deterministic tie-breaking
        df_report = df_report.sort_values(
            by=["frequency", "term"],
            ascending=[False, True]
        ).reset_index(drop=True)

        # Slice top-K for summary report
        df_top = df_report.head(top_k)
        top_terms = df_top.to_dict(orient="records")

        result_id = session.next_result_id("term_frequency")

        summary = {
            "result_id": result_id,
            "total_terms": len(session.feature_names),
            "total_occurrences": int(frequencies.sum()),
            "top_terms": top_terms,
        }

        # Store result with full_report for visualize_result_tool
        session.store_result(
            "term_frequency_tool",
            {"top_k": top_k},
            summary,
            full_report=df_report,
        )

        return summary

    return [build_dtm_tool, term_frequency_tool]
