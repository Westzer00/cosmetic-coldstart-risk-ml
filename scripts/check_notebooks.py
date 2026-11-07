"""Notebook 구조와 Python/Colab 코드 문법을 확인합니다. 학습은 실행하지 않습니다."""
from __future__ import annotations

from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell


def main() -> None:
    notebooks = sorted(Path("notebooks").glob("*.ipynb"))
    if len(notebooks) < 4:
        raise RuntimeError(f"Expected four research notebooks, found {len(notebooks)}")

    shell = InteractiveShell.instance()
    total = 0
    failures = []
    for path in notebooks:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        count = 0
        for i, cell in enumerate(nb.cells, 1):
            if cell.cell_type != "code" or not cell.source.strip():
                continue
            try:
                transformed = shell.input_transformer_manager.transform_cell(cell.source)
                compile(transformed, f"{path}::cell{i}", "exec")
                count += 1
            except Exception as exc:
                failures.append(f"{path} cell {i}: {type(exc).__name__}: {exc}")
        total += count
        print(f"{path}: {count} code cells checked")

    if failures:
        raise RuntimeError("\n".join(failures))
    print(f"Notebook structure and syntax: {len(notebooks)} files, {total} code cells OK")


if __name__ == "__main__":
    main()
