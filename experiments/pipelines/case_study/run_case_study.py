"""Compatibility entry point for the case study pipeline."""

from experiments.case_study.run_case_study import main, run_case_study

__all__ = ["main", "run_case_study"]


if __name__ == "__main__":
    main()
