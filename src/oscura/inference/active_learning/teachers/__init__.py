"""Teachers for L* active learning.

This module provides different oracle implementations (teachers) for the
L* algorithm. Teachers can be based on:

- Simulator: Replay from captured protocol traces
- Interactive: Live device interaction
- Model: From formal protocol specification
"""

from oscura.inference.active_learning.teachers.simulator import SimulatorTeacher

__all__ = [
    "SimulatorTeacher",
]
