import pandas as pd
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_check_missing_tool_known_answer_2():
    # 1. Initialize session and load premade load_dataset_tool
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # 2. Load the sample fixture CSV into session.dataframe
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Make a deep copy of the dataframe before invoking the tool
    df_before = session.dataframe.copy(deep=True)

    # 3. Instantiate check_missing_tool from tools_data using the same session
    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    # 4. Invoke check_missing_tool
    result = check_missing_tool.invoke({"text_column": "text"})

    # 5. Assert expected known answer #2 values
    assert result["total_rows"] == 8, f"Expected total_rows to be 8, got {result['total_rows']}"
    assert result["missing_count"] == 1, f"Expected missing_count to be 1, got {result['missing_count']}"
    assert result["missing_indices"] == [7], f"Expected missing_indices to be [7], got {result['missing_indices']}"
    assert pytest.approx(result["missing_pct"], abs=1e-3) == 12.5, f"Expected missing_pct to be 12.5, got {result['missing_pct']}"

    # 6. Confirm tool does not mutate session.dataframe (exact contents and structure)
    pd.testing.assert_frame_equal(session.dataframe, df_before)


def test_check_duplicates_tool_known_answer_3():
    # 1. Initialize fresh session and load premade load_dataset_tool
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # 2. Load the sample fixture CSV into session.dataframe
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Make a deep copy of the dataframe before invoking the tool
    df_before = session.dataframe.copy(deep=True)

    # 3. Instantiate check_duplicates_tool from tools_data using the same session
    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    # --- Part A: Test when drop=False ---
    result_no_drop = check_duplicates_tool.invoke({"text_column": "text", "drop": False})

    assert result_no_drop["duplicate_count"] == 1, f"Expected duplicate_count to be 1, got {result_no_drop['duplicate_count']}"
    assert result_no_drop["duplicate_indices"] == [6], f"Expected duplicate_indices to be [6], got {result_no_drop['duplicate_indices']}"
    assert result_no_drop["rows_dropped"] == 0, f"Expected rows_dropped to be 0 when drop=False, got {result_no_drop['rows_dropped']}"

    # Confirm data was not modified at all when drop=False
    pd.testing.assert_frame_equal(session.dataframe, df_before)

    # --- Part B: Test when drop=True ---
    result_drop = check_duplicates_tool.invoke({"text_column": "text", "drop": True})

    assert result_drop["duplicate_count"] == 1, f"Expected duplicate_count to be 1, got {result_drop['duplicate_count']}"
    assert result_drop["rows_dropped"] == 2, f"Expected rows_dropped to be 2, got {result_drop['rows_dropped']}"
    assert result_drop["total_rows_after"] == 6, f"Expected total_rows_after to be 6, got {result_drop['total_rows_after']}"
    assert len(session.dataframe) == 6, f"Expected session.dataframe length to be 6, got {len(session.dataframe)}"

    # Confirm 'always gamma delta' no longer appears in the remaining data (both copies removed)
    remaining_texts = session.dataframe["text"].tolist()
    assert "always gamma delta" not in remaining_texts, "'always gamma delta' should be completely removed when keep=False."

    # Confirm index was reset to RangeIndex(0, 6)
    assert list(session.dataframe.index) == list(range(6)), f"Expected index to be 0..5, got {list(session.dataframe.index)}"


def test_inspect_data_tool_head():
    # 1. Initialize fresh session and load premade load_dataset_tool
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. Instantiate inspect_data_tool
    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    # 3. Invoke inspect_data_tool with n=3 and mode="head"
    result = inspect_data_tool.invoke({"n": 3, "mode": "head"})

    # 4. Assert returned row count and metadata
    assert result["n"] == 3, f"Expected n to be 3, got {result['n']}"
    assert result["mode"] == "head", f"Expected mode to be 'head', got {result['mode']}"
    assert len(result["sample_rows"]) == 3, f"Expected 3 sample_rows, got {len(result['sample_rows'])}"

    # 5. Assert first row's text and category_name
    first_row = result["sample_rows"][0]
    assert first_row["text"] == "always alpha alpha alpha beta", f"Expected first row text 'always alpha alpha alpha beta', got {first_row['text']}"
    assert first_row["category_name"] == "catA", f"Expected first row category_name 'catA', got {first_row['category_name']}"


def test_describe_data_tool_known_answer_18():
    # 1. Initialize fresh session & build tools
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")

    # 2. Invoke describe_data_tool
    result = describe_data_tool.invoke({"text_column": "text"})

    # 3. Assert overall statistics matching Known Answer #18
    overall = result["overall_stats"]
    assert overall["count"] == 8, f"Expected count to be 8, got {overall['count']}"
    assert pytest.approx(overall["mean"], abs=1e-3) == 19.875, f"Expected overall mean 19.875, got {overall['mean']}"
    assert pytest.approx(overall["std"], abs=1e-3) == 9.4330, f"Expected overall std 9.4330, got {overall['std']}"

    # 4. Assert per-category statistics matching Known Answer #18
    per_cat = result["per_category_stats"]
    assert "catA" in per_cat and "catB" in per_cat, f"Expected categories 'catA' and 'catB', got {list(per_cat.keys())}"

    catA = per_cat["catA"]
    assert pytest.approx(catA["mean"], abs=1e-3) == 17.25, f"Expected catA mean 17.25, got {catA['mean']}"
    assert pytest.approx(catA["std"], abs=1e-3) == 12.5, f"Expected catA std 12.5, got {catA['std']}"

    catB = per_cat["catB"]
    assert pytest.approx(catB["mean"], abs=1e-3) == 22.5, f"Expected catB mean 22.5, got {catB['mean']}"
    assert pytest.approx(catB["std"], abs=1e-3) == 5.7446, f"Expected catB std 5.7446, got {catB['std']}"

    # 5. Assert session.pending_figure was set
    assert session.pending_figure is not None, "Expected session.pending_figure to be set to a matplotlib figure."

    # 6. Confirm text_length column was NOT permanently added to session.dataframe
    assert "text_length" not in session.dataframe.columns, "'text_length' should not be added to session.dataframe."
