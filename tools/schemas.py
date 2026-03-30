from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- NOVEL TOOLS ---

class TemporalStackLoaderInput(BaseModel):
    aoi_bbox: list[float] = Field(..., min_length=4, max_length=4)
    date_range: list[str] = Field(..., min_length=2, max_length=2)
    sensor: str = "sentinel-2"
    max_cloud_pct: float = 20.0

class TemporalStackLoaderOutput(BaseModel):
    stack_paths: list[str]
    n_epochs: int

class RoadDamageScorerInput(BaseModel):
    gpkg_path: str
    road_layer_name: str = "roads"
    damage_raster_layer: str
    buffer_meters: int = 10

class RoadDamageScorerOutput(BaseModel):
    segment_scores: list[dict]

class EvacuationRoutePlannerInput(BaseModel):
    graph_path: str
    origins: list[str]
    destinations: list[str]
    blocked_segments: list[str] = []
    mode: str = "evacuation"

class EvacuationRoutePlannerOutput(BaseModel):
    routes: list[dict]

class PrithviEmbedInput(BaseModel):
    raster_path: str

class PrithviEmbedOutput(BaseModel):
    embedding_dim: int
    summary_stats: dict

# --- VISION TOOLS ---

class ObjectDetectionInput(BaseModel):
    image_path: str
    text_prompt: str

class ObjectDetectionOutput(BaseModel):
    bboxes: List[List[float]]
    labels: List[str]

class SegmentObjectPixelsInput(BaseModel):
    image_path: str
    bboxes: List[List[float]]

class SegmentObjectPixelsOutput(BaseModel):
    polygons: List[Any]

class ImageDescriptionInput(BaseModel):
    image_path: str

class ImageDescriptionOutput(BaseModel):
    description: str

class TextToBboxInput(BaseModel):
    image_path: str
    text_prompt: str

class TextToBboxOutput(BaseModel):
    bboxes: List[List[float]]

class RegionAttributeDescriptionInput(BaseModel):
    image_path: str
    bbox: List[float]

class RegionAttributeDescriptionOutput(BaseModel):
    description: str

class CountGivenObjectInput(BaseModel):
    image_path: str
    text_prompt: str

class CountGivenObjectOutput(BaseModel):
    count: int

class OCRInput(BaseModel):
    image_path: str

class OCROutput(BaseModel):
    text: str

class ChangeDetectionInput(BaseModel):
    image_path_1: str
    image_path_2: str

class ChangeDetectionOutput(BaseModel):
    change_map_path: str
    changed_pixels: int

class DrawBoxInput(BaseModel):
    image_path: str
    bboxes: List[List[float]]
    output_path: str

class DrawBoxOutput(BaseModel):
    success: bool
    drawn_image_path: str

class AddTextInput(BaseModel):
    image_path: str
    text: str
    position: List[int]
    output_path: str

class AddTextOutput(BaseModel):
    success: bool
    drawn_image_path: str

# --- GIS TOOLS ---

class GetAreaBoundaryInput(BaseModel):
    area_name: str

class GetAreaBoundaryOutput(BaseModel):
    boundary_wkt: str

class AddPoisLayerInput(BaseModel):
    poi_category: str
    bbox: List[float]

class AddPoisLayerOutput(BaseModel):
    pois: List[Dict[str, Any]]

class ComputeDistanceInput(BaseModel):
    point_a: List[float]
    point_b: List[float]

class ComputeDistanceOutput(BaseModel):
    distance_meters: float

class DisplayOnMapInput(BaseModel):
    features: List[Dict[str, Any]]
    output_html: str

class DisplayOnMapOutput(BaseModel):
    success: bool
    html_path: str

class GetBboxFromGeotiffInput(BaseModel):
    geotiff_path: str

class GetBboxFromGeotiffOutput(BaseModel):
    bbox: List[float]
    crs: str

class DisplayGeotiffInput(BaseModel):
    geotiff_path: str
    features: List[Dict[str, Any]]
    output_path: str

class DisplayGeotiffOutput(BaseModel):
    success: bool
    raster_path: str

# Backward-compat aliases
class DisplayOnGeotiffInput(DisplayGeotiffInput):
    pass

class DisplayOnGeotiffOutput(DisplayGeotiffOutput):
    pass

# --- SPECTRAL TOOLS ---

class AddIndexLayerInput(BaseModel):
    geotiff_path: str
    index_name: str

class AddIndexLayerOutput(BaseModel):
    index_array_path: str
    mean_val: float

class ComputeIndexChangeInput(BaseModel):
    index_path_pre: str
    index_path_post: str

class ComputeIndexChangeOutput(BaseModel):
    diff_path: str
    mean_diff: float

class ShowIndexLayerInput(BaseModel):
    index_array_path: str

class ShowIndexLayerOutput(BaseModel):
    success: bool

# --- MATH TOOLS ---

class CalculatorInput(BaseModel):
    expression: str

class CalculatorOutput(BaseModel):
    result: float

class SolverInput(BaseModel):
    equation: str

class SolverOutput(BaseModel):
    roots: List[float]

class PlotInput(BaseModel):
    x_values: List[float]
    y_values: List[float]
    output_path: str

class PlotOutput(BaseModel):
    success: bool
    output_path: Optional[str] = None
    error: Optional[str] = None

# --- UTILITY TOOLS ---

class GoogleSearchInput(BaseModel):
    query: str
    k: int = 10

class GoogleSearchOutput(BaseModel):
    results: List[str]
    n_results: int = 0
    success: bool = True

class TerminateInput(BaseModel):
    final_answer: Any = None
    ans: Any = None

class TerminateOutput(BaseModel):
    status: str
    final_answer: Any = None
