"""Run automated selection experiments from the repository root.

    uv run python experiments/run_selection.py random --steps 100 --seed 7
    uv run --extra imagenet python experiments/run_selection.py imagenet --epsilon 0.1
    uv run --extra imagenet python experiments/run_selection.py offspring-value-imagenet --gamma 1
    uv run python experiments/run_selection.py --help
    uv run python experiments/run_selection.py offspring-value-imagenet --help
"""

from automated_picbreeder.selection_cli import main


if __name__ == "__main__":
    main()
