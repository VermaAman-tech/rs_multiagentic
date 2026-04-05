from agents.base_agent import AgentResult
from framework.episode_runner import EpisodeRunner


def test_detect_error_result_keeps_handled_tool_failures_non_fatal():
    runner = EpisodeRunner()
    result = AgentResult(
        output={
            "ans": "Insufficient data: no accessible GeoTIFF raster inputs were provided.",
            "raw_output": "handled failure",
        },
        confidence=0.7,
        tool_calls=[
            {
                "tool": "AddIndexLayer",
                "args": {"geotiff_path": "missing.tif", "index_name": "NDBI"},
                "output": {"success": False, "error": "missing.tif: No such file"},
            }
        ],
        n_turns=1,
        trace=[],
    )

    is_error, reason = runner._detect_error_result(result)

    assert is_error is False
    assert reason is None


def test_detect_error_result_flags_unhandled_tool_failure():
    runner = EpisodeRunner()
    result = AgentResult(
        output={"raw_output": "tool failed and no answer"},
        confidence=0.2,
        tool_calls=[
            {
                "tool": "SegmentObjectPixels",
                "args": {"image_path": "x.png", "bboxes": [[0, 0, 1, 1]]},
                "output": {"success": False, "error": "No module named 'sam2'"},
            }
        ],
        n_turns=1,
        trace=[],
    )

    is_error, reason = runner._detect_error_result(result)

    assert is_error is True
    assert "sam2" in str(reason)
