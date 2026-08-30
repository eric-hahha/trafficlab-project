"""Trajectory plotting utilities for TrafficLab outputs."""

__all__ = ["TrajectoryPlotter"]


def __getattr__(name):
    if name == "TrajectoryPlotter":
        from trafficlab.trajectory.plotting import TrajectoryPlotter

        return TrajectoryPlotter
    raise AttributeError(f"module 'trafficlab.trajectory' has no attribute {name!r}")
