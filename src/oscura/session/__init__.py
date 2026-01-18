"""Session management for Oscura analysis sessions.

This module provides session save/restore, trace annotations, and
operation history tracking.


Example:
    >>> import oscura as osc
    >>> session = osc.Session()
    >>> session.load_trace('capture.wfm')
    >>> session.annotate(time=1.5e-6, text='Glitch here')
    >>> session.save('debug_session.tks')
    >>>
    >>> # Later...
    >>> session = osc.load_session('debug_session.tks')
    >>> print(session.annotations)
"""

from oscura.session.annotations import Annotation, AnnotationLayer, AnnotationType
from oscura.session.history import HistoryEntry, OperationHistory
from oscura.session.session import Session, load_session

__all__ = [
    # Annotations (SESS-002)
    "Annotation",
    "AnnotationLayer",
    "AnnotationType",
    # History (SESS-003)
    "HistoryEntry",
    "OperationHistory",
    # Session (SESS-001)
    "Session",
    "load_session",
]
