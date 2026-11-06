from typing import Literal
import pandas as pd
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def check_missing_tool(text_column: str = "text") -> dict:
        """Check for missing or empty text values in the loaded dataset.

        Identifies null/NaN values, empty strings (""), and whitespace-only strings
        (e.g. "   ") in the specified text column without modifying session.dataframe.

        Note: Treating whitespace-only strings as missing values is a design choice
        for text mining (since whitespace produces no tokens), rather than an explicit
        requirement of TOOLS_PIPELINE_BREAKDOWN.md.

        Args:
            text_column: Name of the column containing document text (default: 'text').

        Returns:
            Dict containing summary of missing value counts, total rows, missing percentage,
            and the stored result_id.
        """
        df = session.dataframe
        if df is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        if text_column not in df.columns:
            return {"error": f"Column '{text_column}' not found in dataset."}

        series = df[text_column]

        # Identify missing values: NaN/null OR empty/whitespace-only strings
        is_missing = series.isna() | series.astype("string").str.strip().eq("").fillna(False)

        missing_indices = df.index[is_missing].tolist()
        missing_count = len(missing_indices)
        total_rows = len(df)
        missing_pct = (missing_count / total_rows * 100.0) if total_rows > 0 else 0.0

        result_id = session.next_result_id("check_missing")

        summary = {
            "result_id": result_id,
            "text_column": text_column,
            "total_rows": total_rows,
            "missing_count": missing_count,
            "missing_pct": round(missing_pct, 4),
            "missing_indices": missing_indices,
        }

        session.store_result("check_missing_tool", {"text_column": text_column}, summary)
        return summary

    @tool
    def check_duplicates_tool(text_column: str = "text", drop: bool = False) -> dict:
        """Check for (and optionally drop) duplicate documents in the dataset.

        Duplicates are checked only on the specified text column.

        Note on counting vs dropping:
        - duplicate_count reports the number of extra/redundant copies (using keep='first').
        - When drop=True, all copies of duplicated rows are removed (using keep=False),
          matching Master's drop_duplicates(keep=False, inplace=True) behavior.
        - Therefore, duplicate_count (extra copies) and rows_dropped (total rows removed
          with keep=False) are distinct values when duplicates exist.

        Args:
            text_column: Name of the column containing document text (default: 'text').
            drop: If True, drops all copies of duplicate rows from session.dataframe
                  and resets the DataFrame index (default: False).

        Returns:
            Dict containing result_id, duplicate_count, duplicate_indices,
            total_rows_before, total_rows_after, rows_dropped, and drop status.
        """
        df = session.dataframe
        if df is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        if text_column not in df.columns:
            return {"error": f"Column '{text_column}' not found in dataset."}

        # Count extra copies (keep='first') for reporting duplicate_count & indices
        dup_mask_first = df.duplicated(subset=[text_column], keep="first")
        duplicate_indices = df.index[dup_mask_first].tolist()
        duplicate_count = len(duplicate_indices)

        total_rows_before = len(df)
        rows_dropped = 0

        if drop and duplicate_count > 0:
            # Drop ALL copies of duplicate rows (keep=False) as specified
            dup_mask_all = df.duplicated(subset=[text_column], keep=False)
            df_cleaned = df[~dup_mask_all].reset_index(drop=True)
            rows_dropped = total_rows_before - len(df_cleaned)
            session.dataframe = df_cleaned

        total_rows_after = len(session.dataframe)

        result_id = session.next_result_id("check_duplicates")

        summary = {
            "result_id": result_id,
            "text_column": text_column,
            "drop": drop,
            "duplicate_count": duplicate_count,
            "duplicate_indices": duplicate_indices,
            "total_rows_before": total_rows_before,
            "total_rows_after": total_rows_after,
            "rows_dropped": rows_dropped,
        }

        session.store_result("check_duplicates_tool", {"text_column": text_column, "drop": drop}, summary)
        return summary

    @tool
    def inspect_data_tool(
        n: int = 5,
        mode: Literal["head", "tail"] = "head",
        text_column: str = "text",
    ) -> dict:
        """Inspect a sample of rows from the dataset text and category columns.

        Peeks at the head or tail rows of session.dataframe, extracting text and category_name
        columns, truncating text to 100 characters for compact presentation.

        Note on contract:
        - inspect_data_tool has no defined spec contract or fixture known answer; its design
          is custom tailored for dataset inspection.

        Args:
            n: Number of sample rows to inspect (default: 5).
            mode: Inspection slice mode - 'head' or 'tail' (default: 'head').
            text_column: Text column name to inspect (default: 'text').

        Returns:
            Dict containing result_id, mode, n, total_rows, columns, and sample_rows list.
        """
        df = session.dataframe
        if df is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        total_rows = len(df)
        column_names = df.columns.tolist()

        mode_lower = mode.lower()
        if mode_lower == "head":
            sliced_df = df.head(n)
        elif mode_lower == "tail":
            sliced_df = df.tail(n)
        else:
            return {"error": f"Invalid mode '{mode}'. Must be 'head' or 'tail'."}

        # Extract text and category_name columns
        cat_col = "category_name" if "category_name" in df.columns else ("category" if "category" in df.columns else None)

        sample_rows = []
        for idx, row in sliced_df.iterrows():
            raw_text = str(row.get(text_column, "")) if text_column in df.columns else ""
            truncated_text = raw_text[:100]
            cat_val = str(row.get(cat_col, "N/A")) if cat_col else "N/A"

            sample_rows.append({
                "index": int(idx),
                "text": truncated_text,
                "category_name": cat_val,
            })

        result_id = session.next_result_id("inspect_data")

        summary = {
            "result_id": result_id,
            "mode": mode_lower,
            "n": n,
            "total_rows": total_rows,
            "columns": column_names,
            "sample_rows": sample_rows,
        }

        session.store_result("inspect_data_tool", {"n": n, "mode": mode}, summary)
        return summary

    return [check_missing_tool, check_duplicates_tool, inspect_data_tool]
