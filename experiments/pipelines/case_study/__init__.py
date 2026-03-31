"""Compatibility wrapper for the future case study pipeline package.

This package is the target location for gradually extracting the current
`experiments.case_study` workflow into a dedicated pipeline namespace.

At this stage it only re-exports the existing implementation to avoid
breaking imports while establishing the new structure.
"""

from experiments.case_study.run_case_study import main, run_case_study

__all__ = ["main", "run_case_study"]
