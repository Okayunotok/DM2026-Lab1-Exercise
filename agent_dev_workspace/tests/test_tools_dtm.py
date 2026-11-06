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
