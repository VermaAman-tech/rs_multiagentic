from __future__ import annotations

from fastapi import FastAPI

from tools.schemas import *

from tools.novel import (
    temporal_stack_loader,
    road_damage_scorer,
    evacuation_route_planner,
    prithvi_embed,
)
from tools.vision import (
    object_detection,
    segmentation,
    description,
    text_to_bbox,
    region_attribute,
    counting,
    ocr,
    change_detection,
    draw_box,
    add_text,
)
from tools.gis import (
    area_boundary,
    pois_layer,
    compute_distance,
    display_map,
    bbox_from_geotiff,
    display_geotiff,
    display_on_geotiff,
)
from tools.spectral import add_index_layer, compute_index_change, show_index_layer
from tools.math_tools import calculator as calculator_tool
from tools.math_tools import solver as solver_tool
from tools.math_tools import plot as plot_tool
from tools.utility import google_search as google_search_tool
from tools.utility import terminate as terminate_tool

app = FastAPI(title="MAGRF Tool Server", version="1.0")


@app.get("/health")
def health() -> dict:
    # Plan-conformant tool count: 28 tools.
    return {"status": "ok", "n_tools": 28}


# --- Novel tools (4) ---
@app.post("/tools/TemporalStackLoader", response_model=TemporalStackLoaderOutput)
def run_temporal_stack(req: TemporalStackLoaderInput):
    res = temporal_stack_loader.temporal_stack_loader(
        aoi_bbox=req.aoi_bbox,
        date_range=req.date_range,
        sensor=req.sensor,
        max_cloud_pct=req.max_cloud_pct,
        local_stack_path=req.local_stack_path,
        image_paths=req.image_paths,
    )
    return TemporalStackLoaderOutput(**res)


@app.post("/tools/RoadDamageScorer", response_model=RoadDamageScorerOutput)
def run_road_damage(req: RoadDamageScorerInput):
    res = road_damage_scorer.road_damage_scorer(
        gpkg_path=req.gpkg_path,
        road_layer_name=req.road_layer_name,
        damage_raster_layer=req.damage_raster_layer,
        buffer_meters=req.buffer_meters,
    )
    return RoadDamageScorerOutput(**res)


@app.post("/tools/EvacuationRoutePlanner", response_model=EvacuationRoutePlannerOutput)
def run_evacuation(req: EvacuationRoutePlannerInput):
    res = evacuation_route_planner.evacuation_route_planner(
        graph_path=req.graph_path,
        origins=req.origins,
        destinations=req.destinations,
        blocked_segments=req.blocked_segments,
        mode=req.mode,
    )
    return EvacuationRoutePlannerOutput(**res)


@app.post("/tools/PrithviEmbed", response_model=PrithviEmbedOutput)
def run_prithvi(req: PrithviEmbedInput):
    res = prithvi_embed.prithvi_embed(raster_path=req.raster_path)
    return PrithviEmbedOutput(**res)


# --- Vision tools (10) ---
@app.post("/tools/ObjectDetection", response_model=ObjectDetectionOutput)
def run_object_detection(req: ObjectDetectionInput):
    return ObjectDetectionOutput(**object_detection.run(req))


@app.post("/tools/SegmentObjectPixels", response_model=SegmentObjectPixelsOutput)
def run_segment(req: SegmentObjectPixelsInput):
    return SegmentObjectPixelsOutput(**segmentation.run(req))


@app.post("/tools/ImageDescription", response_model=ImageDescriptionOutput)
def run_image_desc(req: ImageDescriptionInput):
    return ImageDescriptionOutput(**description.run(req))


@app.post("/tools/TextToBbox", response_model=TextToBboxOutput)
def run_text_to_bbox(req: TextToBboxInput):
    return TextToBboxOutput(**text_to_bbox.run(req))


@app.post("/tools/RegionAttributeDescription", response_model=RegionAttributeDescriptionOutput)
def run_region_attr(req: RegionAttributeDescriptionInput):
    return RegionAttributeDescriptionOutput(**region_attribute.run(req))


@app.post("/tools/CountGivenObject", response_model=CountGivenObjectOutput)
def run_count_obj(req: CountGivenObjectInput):
    return CountGivenObjectOutput(**counting.run(req))


@app.post("/tools/OCR", response_model=OCROutput)
def run_ocr(req: OCRInput):
    return OCROutput(**ocr.run(req))


@app.post("/tools/ChangeDetection", response_model=ChangeDetectionOutput)
def run_change_detection(req: ChangeDetectionInput):
    return ChangeDetectionOutput(**change_detection.run(req))


@app.post("/tools/DrawBox", response_model=DrawBoxOutput)
def run_draw_box(req: DrawBoxInput):
    return DrawBoxOutput(**draw_box.run(req))


@app.post("/tools/AddText", response_model=AddTextOutput)
def run_add_text(req: AddTextInput):
    return AddTextOutput(**add_text.run(req))


# --- GIS tools (6) ---
@app.post("/tools/GetAreaBoundary", response_model=GetAreaBoundaryOutput)
def run_get_area_boundary(req: GetAreaBoundaryInput):
    return GetAreaBoundaryOutput(**area_boundary.run(req))


@app.post("/tools/AddPoisLayer", response_model=AddPoisLayerOutput)
def run_add_pois(req: AddPoisLayerInput):
    return AddPoisLayerOutput(**pois_layer.run(req))


@app.post("/tools/ComputeDistance", response_model=ComputeDistanceOutput)
def run_compute_distance(req: ComputeDistanceInput):
    return ComputeDistanceOutput(**compute_distance.run(req))


@app.post("/tools/DisplayOnMap", response_model=DisplayOnMapOutput)
def run_display_map(req: DisplayOnMapInput):
    return DisplayOnMapOutput(**display_map.run(req))


@app.post("/tools/GetBboxFromGeotiff", response_model=GetBboxFromGeotiffOutput)
def run_get_bbox_geotiff(req: GetBboxFromGeotiffInput):
    return GetBboxFromGeotiffOutput(**bbox_from_geotiff.run(req))


@app.post("/tools/DisplayGeotiff", response_model=DisplayGeotiffOutput)
def run_display_geotiff(req: DisplayGeotiffInput):
    return DisplayGeotiffOutput(**display_geotiff.run(req))


@app.post("/tools/DisplayOnGeotiff", response_model=DisplayOnGeotiffOutput)
def run_display_on_geotiff_alias(req: DisplayOnGeotiffInput):
    return DisplayOnGeotiffOutput(**display_on_geotiff.run(req))


# --- Spectral tools (3) ---
@app.post("/tools/AddIndexLayer", response_model=AddIndexLayerOutput)
def run_add_index(req: AddIndexLayerInput):
    return AddIndexLayerOutput(**add_index_layer.run(req))


@app.post("/tools/ComputeIndexChange", response_model=ComputeIndexChangeOutput)
def run_compute_index_change(req: ComputeIndexChangeInput):
    return ComputeIndexChangeOutput(**compute_index_change.run(req))


@app.post("/tools/ShowIndexLayer", response_model=ShowIndexLayerOutput)
def run_show_index(req: ShowIndexLayerInput):
    return ShowIndexLayerOutput(**show_index_layer.run(req))


# --- Math tools (3) ---
@app.post("/tools/Calculator", response_model=CalculatorOutput)
def run_calculator(req: CalculatorInput):
    return CalculatorOutput(**calculator_tool(req.expression))


@app.post("/tools/Solver", response_model=SolverOutput)
def run_solver(req: SolverInput):
    return SolverOutput(**solver_tool(req.equation))


@app.post("/tools/Plot", response_model=PlotOutput)
def run_plot(req: PlotInput):
    return PlotOutput(**plot_tool(req.x_values, req.y_values, req.output_path))


# --- Utility tools (2) ---
@app.post("/tools/GoogleSearch", response_model=GoogleSearchOutput)
def run_google_search(req: GoogleSearchInput):
    return GoogleSearchOutput(**google_search_tool(query=req.query, k=req.k))


@app.post("/tools/Terminate", response_model=TerminateOutput)
def run_terminate(req: TerminateInput):
    return TerminateOutput(**terminate_tool(final_answer=req.final_answer, ans=req.ans))
