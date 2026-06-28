"""Sudoku RLVR: train a small open model to solve Sudoku with GRPO on one GPU."""

from . import data, prompts, rewards, sudoku

__all__ = ["sudoku", "prompts", "rewards", "data"]
__version__ = "0.1.0"
