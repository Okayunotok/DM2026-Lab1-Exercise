import matplotlib.pyplot as plt
import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def test_reduce_dimensions_tool_pca_known_answer_7():
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

    # 3. Instantiate reduce_dimensions_tool
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # 4. Invoke reduce_dimensions_tool with default method="pca"
    result = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})

    # Assert PCA explained_variance_ratio matching Known Answer #7
    expected_evr = [0.7532, 0.1965]
    assert np.allclose(result["explained_variance_ratio"], expected_evr, atol=1e-3), \
        f"Expected explained_variance_ratio close to {expected_evr}, got {result['explained_variance_ratio']}"

    # Assert 2D reduced coordinates for document 0 matching Known Answer #7 ([2.3669, 0.8871])
    coords = session.artifacts["reduced_coordinates"]
    expected_doc0_coords = [2.3669, 0.8871]
    assert np.allclose(coords[0], expected_doc0_coords, atol=1e-3), \
        f"Expected document 0 PCA coordinates close to {expected_doc0_coords}, got {coords[0]}"

    # Assert shape of full coordinates array
    assert list(coords.shape) == [8, 2]

    # Assert plotting requirement: session.pending_figure must be populated
    assert session.pending_figure is not None, "session.pending_figure must be populated with a matplotlib figure."
    assert isinstance(session.pending_figure, plt.Figure), "session.pending_figure must be an instance of matplotlib.figure.Figure."


def test_reduce_dimensions_tool_tsne_pending_figure():
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

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Clear pending_figure before testing
    session.pending_figure = None

    # Invoke t-SNE
    result = reduce_dimensions_tool.invoke({"method": "tsne", "perplexity": 2.0, "random_state": 42})

    # Assert session.pending_figure is set for t-SNE
    assert session.pending_figure is not None, "session.pending_figure must be populated for t-SNE method."
    assert isinstance(session.pending_figure, plt.Figure)
