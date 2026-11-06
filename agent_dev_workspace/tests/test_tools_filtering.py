import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool_known_answer_5():
    # 1. Initialize session and load premade load_dataset_tool
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. Build DTM matrix into session
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 3. Instantiate variance_filter_tool
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # 4. Invoke variance_filter_tool with threshold=0.15
    result = variance_filter_tool.invoke({"threshold": 0.15})

    # Assert kept and removed counts/lists
    assert result["kept_count"] == 4, f"Expected kept_count to be 4, got {result['kept_count']}"
    assert result["removed_count"] == 1, f"Expected removed_count to be 1, got {result['removed_count']}"
    assert sorted(result["sample_kept_terms"]) == ["alpha", "beta", "delta", "gamma"]
    assert result["sample_removed_terms"] == ["always"]

    # 5. Fetch full_report from session.results_store to verify detailed numbers and structure
    stored_entry = session.results_store[result["result_id"]]
    full_report = stored_entry["full_report"]

    # Verify column structure of full_report (1st column 'term', 2nd column 'variance')
    assert list(full_report.columns[:2]) == ["term", "variance"], f"Expected columns ['term', 'variance'], got {list(full_report.columns[:2])}"

    # Verify rows are sorted in descending order of variance
    variances_in_report = full_report["variance"].tolist()
    assert variances_in_report == sorted(variances_in_report, reverse=True), "full_report should be sorted in descending order of variance."

    # Map terms to variance values for precise checks
    term_to_var = dict(zip(full_report["term"], full_report["variance"]))

    # Verify exact population variance values matching Known Answer #5
    assert pytest.approx(term_to_var["alpha"], abs=1e-3) == 1.1875, f"Expected alpha variance to be 1.1875, got {term_to_var['alpha']}"
    assert pytest.approx(term_to_var["always"], abs=1e-3) == 0.109375, f"Expected always variance to be 0.109375, got {term_to_var['always']}"


def test_pearson_filter_tool_known_answer_6():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    result = pearson_filter_tool.invoke({"target_class": "catB"})

    stored_entry = session.results_store[result["result_id"]]
    full_report = stored_entry["full_report"]

    # Check full_report structure
    assert list(full_report.columns[:2]) == ["term", "pearson_r"]

    # Verify rows are sorted by |pearson_r| descending
    abs_r_values = full_report["pearson_r"].abs().tolist()
    assert abs_r_values == sorted(abs_r_values, reverse=True), "full_report should be sorted by |pearson_r| descending."

    term_to_r = dict(zip(full_report["term"], full_report["pearson_r"]))

    # Known Answer #6 assertions for Pearson r
    assert pytest.approx(term_to_r["delta"], abs=1e-3) == 1.0, f"Expected delta Pearson r to be 1.0, got {term_to_r['delta']}"
    assert pytest.approx(term_to_r["gamma"], abs=1e-3) == 0.8307, f"Expected gamma Pearson r to be 0.8307, got {term_to_r['gamma']}"
    assert pytest.approx(term_to_r["beta"], abs=1e-3) == -0.7746, f"Expected beta Pearson r to be -0.7746, got {term_to_r['beta']}"
    assert term_to_r["beta"] < 0, "beta correlation must retain its negative sign."


def test_spearman_filter_tool_known_answer_6():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    result = spearman_filter_tool.invoke({"target_class": "catB"})

    stored_entry = session.results_store[result["result_id"]]
    full_report = stored_entry["full_report"]

    # Check full_report structure
    assert list(full_report.columns[:2]) == ["term", "spearman_r"]

    # Verify rows are sorted by |spearman_r| descending
    abs_r_values = full_report["spearman_r"].abs().tolist()
    assert abs_r_values == sorted(abs_r_values, reverse=True), "full_report should be sorted by |spearman_r| descending."

    term_to_r = dict(zip(full_report["term"], full_report["spearman_r"]))

    # Known Answer #6 assertions for Spearman r
    assert pytest.approx(term_to_r["delta"], abs=1e-3) == 1.0, f"Expected delta Spearman r to be 1.0, got {term_to_r['delta']}"
    assert pytest.approx(term_to_r["gamma"], abs=1e-3) == 0.9363, f"Expected gamma Spearman r to be 0.9363, got {term_to_r['gamma']}"
    assert pytest.approx(term_to_r["beta"], abs=1e-3) == -0.7746, f"Expected beta Spearman r to be -0.7746, got {term_to_r['beta']}"
