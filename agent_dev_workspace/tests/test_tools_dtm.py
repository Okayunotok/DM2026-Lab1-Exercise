import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_known_answer_4():
    # 1. Initialize session and load dataset using premade load_dataset_tool
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. Instantiate build_dtm_tool from tools_dtm using the same session
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")

    # 3. Invoke build_dtm_tool with default settings
    result = build_dtm_tool.invoke({})

    # 4. Assert vocabulary size and exact vocabulary terms in session.feature_names
    expected_vocab = ["alpha", "always", "beta", "delta", "gamma"]
    assert result["vocabulary_size"] == 5, f"Expected vocabulary_size to be 5, got {result['vocabulary_size']}"
    assert session.feature_names == expected_vocab, f"Expected feature_names to be {expected_vocab}, got {session.feature_names}"

    # 5. Assert sparsity metrics matching Known Answer #4
    assert result["non_zero"] == 21, f"Expected non_zero to be 21, got {result['non_zero']}"
    assert result["total_elements"] == 40, f"Expected total_elements to be 40, got {result['total_elements']}"
    assert pytest.approx(result["sparsity_pct"], abs=1e-3) == 47.5, f"Expected sparsity_pct to be 47.5, got {result['sparsity_pct']}"

    # 6. Assert fitted CountVectorizer artifact is saved in session.artifacts
    assert "count_vectorizer" in session.artifacts, "'count_vectorizer' missing from session.artifacts."


def test_term_frequency_tool_known_answer_14():
    # 1. Each test creates its own fresh session & builds tools
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in tools if t.name == "term_frequency_tool")

    # Build document-term matrix first
    build_dtm_tool.invoke({})

    # 2. Invoke term_frequency_tool
    result = term_frequency_tool.invoke({"top_k": 20})

    # 3. Check summary frequencies matching Known Answer #14
    expected_frequencies = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7,
    }
    top_terms = result["top_terms"]
    actual_frequencies = {item["term"]: item["frequency"] for item in top_terms}
    assert actual_frequencies == expected_frequencies, f"Expected frequencies {expected_frequencies}, got {actual_frequencies}"

    # 4. Check that full_report was stored in session.results_store and tie-breaking is alphabetical
    res_id = result["result_id"]
    stored_result = session.results_store[res_id]
    full_report = stored_result["full_report"]

    # In full_report, terms with frequency 7 are "always" and "gamma".
    # Alphabetical tie-breaking requires "always" before "gamma".
    terms_order = full_report["term"].tolist()
    always_idx = terms_order.index("always")
    gamma_idx = terms_order.index("gamma")
    assert always_idx < gamma_idx, f"Expected 'always' before 'gamma' due to alphabetical tie-breaking, got order: {terms_order}"
