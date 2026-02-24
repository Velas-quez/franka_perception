"""Core perception pipeline for cube detection."""

from .geometry.cube_fitting import CubeEstimate
from .pipelines.pipeline import CubeDetectionPipeline, CubeDetectionResult
