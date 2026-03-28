from tools.novel.evacuation_route_planner import evacuation_route_planner
from tools.novel.prithvi_embed import prithvi_embed
from tools.novel.road_damage_scorer import road_damage_scorer
from tools.novel.temporal_stack_loader import temporal_stack_loader


def test_temporal_stack_loader_returns_two_epochs():
    out = temporal_stack_loader([0, 0, 1, 1], ["2024-01-01", "2024-02-01"])
    assert out["n_epochs"] == 2
    assert len(out["stack_paths"]) == 2


def test_road_damage_scorer_schema():
    out = road_damage_scorer("roads.gpkg", "roads", "damage.tif")
    assert "segment_scores" in out
    assert all("segment_id" in x and "traversability" in x for x in out["segment_scores"])


def test_evacuation_route_planner_schema():
    out = evacuation_route_planner("graph.gpkg", ["A"], ["B"])
    assert "routes" in out
    assert out["routes"][0]["origin"] == "A"
    assert out["routes"][0]["destination"] == "B"


def test_prithvi_embed_schema():
    out = prithvi_embed("scene.tif")
    assert out["embedding_dim"] == 768
    assert {"mean", "std"}.issubset(out["summary_stats"])
