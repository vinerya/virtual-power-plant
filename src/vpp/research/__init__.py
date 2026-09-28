"""Research/AI layer — NEVER used in production paths.

All models in this module are for research, benchmarking, and offline
comparison against the rule-based production system.  No model here
should ever be called from a production code path.
"""

from vpp.research.base import ResearchExperiment, ResearchModel

__all__ = ["ResearchExperiment", "ResearchModel"]
