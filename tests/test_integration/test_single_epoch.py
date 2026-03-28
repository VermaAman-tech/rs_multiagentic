from framework.episode_runner import EpisodeRunner


def test_single_epoch_runs():
    runner = EpisodeRunner(tool_server="http://localhost:9000")
    ep = runner.run_episode("test query")
    assert ep["final"]["status"] == "terminated"
