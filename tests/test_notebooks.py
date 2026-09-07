from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
STUDENT = ROOT / "notebooks" / "00_python_and_tensors.ipynb"
SOLUTION = ROOT / "notebooks" / "solutions" / "00_python_and_tensors_solution.ipynb"


def load_notebook(path):
    return nbformat.read(path, as_version=4)


def cell_ids(notebook):
    return [cell["id"] for cell in notebook.cells]


def source(notebook, cell_type=None):
    cells = notebook.cells
    if cell_type is not None:
        cells = [cell for cell in cells if cell.cell_type == cell_type]
    return "\n".join(cell.source for cell in cells)


def test_student_and_solution_have_matching_structure():
    student = load_notebook(STUDENT)
    solution = load_notebook(SOLUTION)

    assert cell_ids(student) == cell_ids(solution)
    exercise_ids = [
        cell["id"]
        for cell in student.cells
        if "exercise" in cell.get("metadata", {}).get("tags", [])
    ]
    assert exercise_ids == [f"exercise-{number}" for number in range(1, 7)]

    student_headings = [
        line
        for line in source(student, "markdown").splitlines()
        if line.startswith("## ")
    ]
    solution_headings = [
        line
        for line in source(solution, "markdown").splitlines()
        if line.startswith("## ")
    ]
    assert student_headings == solution_headings


def test_todos_are_only_in_student_notebook():
    assert "TODO" in source(load_notebook(STUDENT))
    assert "TODO" not in source(load_notebook(SOLUTION))


def test_notebooks_have_no_runtime_network_gpu_or_absolute_home_dependency():
    for path in (STUDENT, SOLUTION):
        code = source(load_notebook(path), "code")
        forbidden = ("!pip", "requests.", "urlopen(", ".cuda(", 'device="cuda"')
        assert not any(item in code for item in forbidden)
        assert "/home/" not in code


def test_solution_executes_from_top_to_bottom_on_cpu():
    notebook = load_notebook(SOLUTION)
    executed = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()

    assert all(
        output.get("output_type") != "error"
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
    )

