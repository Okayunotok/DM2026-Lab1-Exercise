import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_pattern_tools


def test_mine_patterns_tool_variance_filtering_empty_answer_17():
    # 1. Initialize session and load premade load_dataset_tool
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    pattern_tools = make_pattern_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    # 2. Test variance filtering on catA -> 0 terms survive, returns clean summary with note
    result = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1,
    })

    assert result["pattern_count"] == 0, f"Expected pattern_count to be 0, got {result['pattern_count']}"
    assert result["patterns"] == [], f"Expected patterns to be empty list, got {result['patterns']}"
    assert result.get("note") == "no terms survived filtering", f"Expected note 'no terms survived filtering', got {result.get('note')}"


def test_mine_patterns_tool_tfidf_fpgrowth_known_answer_17():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    pattern_tools = make_pattern_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    # --- catA with tfidf + fpgrowth (min_sup=1) -> finds {alpha} with support 3 ---
    result_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1,
    })

    assert result_catA["pattern_count"] == 1, f"Expected 1 pattern for catA, got {result_catA['pattern_count']}"
    catA_pat = result_catA["patterns"][0]
    assert "alpha" in catA_pat["pattern"], f"Expected 'alpha' in pattern, got {catA_pat['pattern']}"
    assert catA_pat["support"] == 3, f"Expected support 3 for catA alpha, got {catA_pat['support']}"

    # --- catB with tfidf + fpgrowth (min_sup=1) -> finds {gamma} with support 4 ---
    result_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1,
    })

    assert result_catB["pattern_count"] == 1, f"Expected 1 pattern for catB, got {result_catB['pattern_count']}"
    catB_pat = result_catB["patterns"][0]
    assert "gamma" in catB_pat["pattern"], f"Expected 'gamma' in pattern, got {catB_pat['pattern']}"
    assert catB_pat["support"] == 4, f"Expected support 4 for catB gamma, got {catB_pat['support']}"


def test_mine_patterns_tool_tfidf_maxfpgrowth_known_answer_20():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    pattern_tools = make_pattern_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    # tfidf + maxfpgrowth (min_sup=1) -> 0 patterns returned without exception
    result_max = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1,
    })

    assert result_max["pattern_count"] == 0, f"Expected 0 patterns for maxfpgrowth, got {result_max['pattern_count']}"
    assert result_max["patterns"] == [], f"Expected empty patterns list for maxfpgrowth, got {result_max['patterns']}"
