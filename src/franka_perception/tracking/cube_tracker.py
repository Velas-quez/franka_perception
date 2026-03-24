#!/usr/bin/env python3
"""Temporal tracker for persistent cube IDs."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import List, Optional, Sequence

import numpy as np

from ..geometry.cube_fitting import CubeEstimate


@dataclass
class DetectionObservation:
    """Per-frame observation used to update the tracker."""

    cube: CubeEstimate
    mask_centroid: Optional[np.ndarray] = None


@dataclass
class TrackedCubeState:
    """Externally visible state of a tracked cube."""

    track_id: int
    cube: CubeEstimate
    is_occluded: bool


@dataclass
class _Track:
    track_id: int
    cube: CubeEstimate
    velocity: np.ndarray
    mask_centroid: Optional[np.ndarray]
    last_timestamp: float
    missed_frames: int = 0


def _copy_cube(cube: CubeEstimate) -> CubeEstimate:
    return CubeEstimate(
        transform=np.asarray(cube.transform, dtype=float).copy(),
        mesh=cube.mesh,
        initial_mesh=cube.initial_mesh,
        icp_fitness=float(cube.icp_fitness),
    )


class CubeTracker:
    """Track cubes over time using 3D position first and 2D masks as a tie-breaker."""

    def __init__(self,
                 max_match_distance: float = 0.06,
                 max_missed_frames: int = 5,
                 mask_max_distance: float = 120.0,
                 position_weight: float = 1.0,
                 mask_weight: float = 0.2,
                 velocity_alpha: float = 0.6) -> None:
        self.max_match_distance = max(1e-6, float(max_match_distance))
        self.max_missed_frames = max(0, int(max_missed_frames))
        self.mask_max_distance = max(1.0, float(mask_max_distance))
        self.position_weight = max(0.0, float(position_weight))
        self.mask_weight = max(0.0, float(mask_weight))
        self.velocity_alpha = float(np.clip(velocity_alpha, 0.0, 1.0))
        self._tracks: List[_Track] = []
        self._next_track_id = 0

    def reset(self) -> None:
        self._tracks = []

    def update(self,
               detections: Sequence[DetectionObservation],
               timestamp: Optional[float] = None) -> List[TrackedCubeState]:
        now = float(time.time() if timestamp is None else timestamp)
        detections = list(detections)
        predicted_centers = [self._predict_track_center(track, now) for track in self._tracks]
        matches, unmatched_track_indices, unmatched_detection_indices = self._match(
            detections,
            predicted_centers,
        )

        for track_idx, detection_idx in matches:
            self._update_track(self._tracks[track_idx], detections[detection_idx], now)

        for track_idx in unmatched_track_indices:
            self._mark_occluded(self._tracks[track_idx], predicted_centers[track_idx], now)

        self._tracks = [
            track for track in self._tracks
            if track.missed_frames <= self.max_missed_frames
        ]

        for detection_idx in unmatched_detection_indices:
            self._tracks.append(self._create_track(detections[detection_idx], now))

        self._tracks.sort(key=lambda track: track.track_id)
        return [
            TrackedCubeState(
                track_id=track.track_id,
                cube=_copy_cube(track.cube),
                is_occluded=track.missed_frames > 0,
            )
            for track in self._tracks
        ]

    def _predict_track_center(self, track: _Track, now: float) -> np.ndarray:
        dt = max(0.0, now - float(track.last_timestamp))
        center = np.asarray(track.cube.transform[:3, 3], dtype=float)
        return center + track.velocity * dt

    def _match(self,
               detections: Sequence[DetectionObservation],
               predicted_centers: Sequence[np.ndarray]):
        candidate_pairs = []
        for track_idx, track in enumerate(self._tracks):
            predicted_center = np.asarray(predicted_centers[track_idx], dtype=float)
            for detection_idx, detection in enumerate(detections):
                detection_center = np.asarray(detection.cube.transform[:3, 3], dtype=float)
                position_distance = float(np.linalg.norm(predicted_center - detection_center))
                if position_distance > self.max_match_distance:
                    continue
                cost = self.position_weight * (position_distance / self.max_match_distance)

                if (
                    self.mask_weight > 0.0
                    and track.mask_centroid is not None
                    and detection.mask_centroid is not None
                ):
                    mask_distance = float(np.linalg.norm(track.mask_centroid - detection.mask_centroid))
                    cost += self.mask_weight * min(
                        1.0,
                        mask_distance / self.mask_max_distance,
                    )

                candidate_pairs.append((cost, position_distance, track_idx, detection_idx))

        candidate_pairs.sort(key=lambda item: (item[0], item[1]))
        matches = []
        used_tracks = set()
        used_detections = set()
        for _, _, track_idx, detection_idx in candidate_pairs:
            if track_idx in used_tracks or detection_idx in used_detections:
                continue
            matches.append((track_idx, detection_idx))
            used_tracks.add(track_idx)
            used_detections.add(detection_idx)

        unmatched_track_indices = [
            idx for idx in range(len(self._tracks))
            if idx not in used_tracks
        ]
        unmatched_detection_indices = [
            idx for idx in range(len(detections))
            if idx not in used_detections
        ]
        return matches, unmatched_track_indices, unmatched_detection_indices

    def _update_track(self,
                      track: _Track,
                      detection: DetectionObservation,
                      now: float) -> None:
        previous_center = np.asarray(track.cube.transform[:3, 3], dtype=float)
        new_cube = _copy_cube(detection.cube)
        new_center = np.asarray(new_cube.transform[:3, 3], dtype=float)
        dt = max(1e-6, now - float(track.last_timestamp))
        measured_velocity = (new_center - previous_center) / dt
        track.velocity = (
            self.velocity_alpha * measured_velocity
            + (1.0 - self.velocity_alpha) * track.velocity
        )
        track.cube = new_cube
        track.mask_centroid = (
            None if detection.mask_centroid is None
            else np.asarray(detection.mask_centroid, dtype=float).copy()
        )
        track.last_timestamp = now
        track.missed_frames = 0

    def _mark_occluded(self,
                       track: _Track,
                       predicted_center: np.ndarray,
                       now: float) -> None:
        track.cube.transform[:3, 3] = np.asarray(predicted_center, dtype=float)
        track.last_timestamp = now
        track.missed_frames += 1

    def _create_track(self,
                      detection: DetectionObservation,
                      now: float) -> _Track:
        track = _Track(
            track_id=self._next_track_id,
            cube=_copy_cube(detection.cube),
            velocity=np.zeros(3, dtype=float),
            mask_centroid=(
                None if detection.mask_centroid is None
                else np.asarray(detection.mask_centroid, dtype=float).copy()
            ),
            last_timestamp=now,
            missed_frames=0,
        )
        self._next_track_id += 1
        return track
