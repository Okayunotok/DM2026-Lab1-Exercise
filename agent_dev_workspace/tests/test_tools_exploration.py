import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


def test_cosine_similarity_tool_known_answer_16():
    # 1. Initialize session and load premade load_dataset_tool
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. Build DTM matrix first
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 3. Instantiate cosine_similarity_tool
    exp_tools = make_exploration_tools(session)
    cosine_sim_tool = next(t for t in exp_tools if t.name == "cosine_similarity_tool")

    # 4. Check pair 0 vs 1 (both catA, near-identical text) -> expected 0.9847
    res_0_1 = cosine_sim_tool.invoke({"doc1_idx": 0, "doc2_idx": 1})
    assert pytest.approx(res_0_1["similarity"], abs=1e-3) == 0.9847, f"Expected 0 vs 1 similarity 0.9847, got {res_0_1['similarity']}"

    # 5. Check pair 0 vs 3 (catA vs catB, share only 'always') -> expected 0.0909
    res_0_3 = cosine_sim_tool.invoke({"doc1_idx": 0, "doc2_idx": 3})
    assert pytest.approx(res_0_3["similarity"], abs=1e-3) == 0.0909, f"Expected 0 vs 3 similarity 0.0909, got {res_0_3['similarity']}"

    # 6. Check pair 0 vs 7 (catA vs empty text row) -> expected exactly 0.0
    res_0_7 = cosine_sim_tool.invoke({"doc1_idx": 0, "doc2_idx": 7})
    assert res_0_7["similarity"] == 0.0, f"Expected 0 vs 7 similarity exactly 0.0, got {res_0_7['similarity']}"


def test_feature_correlation_matrix_tool_known_answer_19():
    # 1. Initialize session and load dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. Build DTM matrix first
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 3. Instantiate feature_correlation_matrix_tool
    exp_tools = make_exploration_tools(session)
    corr_tool = next(t for t in exp_tools if t.name == "feature_correlation_matrix_tool")

    # 4. Invoke feature_correlation_matrix_tool
    result = corr_tool.invoke({"top_n": 20})

    # 5. Assert top terms order by variance descending: alpha, gamma, delta, beta, always
    expected_top_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert result["top_terms"] == expected_top_terms, f"Expected top terms {expected_top_terms}, got {result['top_terms']}"

    # 6. Verify correlation matrix values matching Known Answer #19
    corr_matrix = np.array(result["correlation_matrix"])
    top_terms = result["top_terms"]

    # Check diagonal is 1.0
    for i in range(len(top_terms)):
        assert pytest.approx(corr_matrix[i, i], abs=1e-3) == 1.0, f"Diagonal cell [{i},{i}] should be 1.0"

    # Check correlation between alpha (index 0) and beta (index 3) is 0.8885
    idx_alpha = top_terms.index("alpha")
    idx_beta = top_terms.index("beta")
    assert pytest.approx(corr_matrix[idx_alpha, idx_beta], abs=1e-3) == 0.8885, f"Expected alpha/beta correlation 0.8885, got {corr_matrix[idx_alpha, idx_beta]}"

    # Check correlation between gamma (index 1) and delta (index 2) is 0.8307
    idx_gamma = top_terms.index("gamma")
    idx_delta = top_terms.index("delta")
    assert pytest.approx(corr_matrix[idx_gamma, idx_delta], abs=1e-3) == 0.8307, f"Expected gamma/delta correlation 0.8307, got {corr_matrix[idx_gamma, idx_delta]}"

    # 7. Assert session.pending_figure was set to matplotlib figure heatmap
    assert session.pending_figure is not None, "Expected session.pending_figure to be assigned to heatmap figure."
