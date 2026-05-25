"""Trajectory smoothing and plotting utilities for TrafficLab outputs."""

__all__ = ["SmoothStats", "TrajectoryPlotter", "smooth_file", "smooth_trajectories"]


def __getattr__(name):
    if name == "TrajectoryPlotter":
        from trafficlab.trajectory.plotting import TrajectoryPlotter

        return TrajectoryPlotter
    if name in {"SmoothStats", "smooth_file", "smooth_trajectories"}:
        from trafficlab.trajectory.smoothing import (
            SmoothStats,
            smooth_file,
            smooth_trajectories,
        )

        exports = {
            "SmoothStats": SmoothStats,
            "smooth_file": smooth_file,
            "smooth_trajectories": smooth_trajectories,
        }
        return exports[name]
    raise AttributeError(f"module 'trafficlab.trajectory' has no attribute {name!r}")
