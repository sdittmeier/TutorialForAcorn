from pathlib import Path
import subprocess

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PAIRS = [
    (
        ROOT / "notebooks" / "00_python_and_tensors.ipynb",
        ROOT / "notebooks" / "solutions" / "00_python_and_tensors_solution.ipynb",
    ),
    (
        ROOT / "notebooks" / "01_one_acorn_event.ipynb",
        ROOT / "notebooks" / "solutions" / "01_one_acorn_event_solution.ipynb",
    ),
    (
        ROOT / "notebooks" / "02_run_the_toy_pipeline.ipynb",
        ROOT / "notebooks" / "solutions" / "02_run_the_toy_pipeline_solution.ipynb",
    ),
    (
        ROOT / "notebooks" / "03_create_an_edge_classifier.ipynb",
        ROOT / "notebooks" / "solutions" / "03_create_an_edge_classifier_solution.ipynb",
    ),
]


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
    for student_path, solution_path in NOTEBOOK_PAIRS:
        student = load_notebook(student_path)
        solution = load_notebook(solution_path)

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
    for student_path, solution_path in NOTEBOOK_PAIRS:
        assert "TODO" in source(load_notebook(student_path))
        assert "TODO" not in source(load_notebook(solution_path))


def test_notebooks_have_no_runtime_network_gpu_or_absolute_home_dependency():
    for pair in NOTEBOOK_PAIRS:
        for path in pair:
            code = source(load_notebook(path), "code")
            forbidden = ("!pip", "requests.", "urlopen(", ".cuda(", 'device="cuda"')
            assert not any(item in code for item in forbidden)
            assert "/home/" not in code


def test_solution_executes_from_top_to_bottom_on_cpu():
    for _, solution_path in NOTEBOOK_PAIRS:
        notebook = load_notebook(solution_path)
        executed = NotebookClient(
            notebook,
            timeout=300,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()

        assert all(
            output.get("output_type") != "error"
            for cell in executed.cells
            if cell.cell_type == "code"
            for output in cell.get("outputs", [])
        )


def test_one_event_solution_has_expected_data_contract():
    notebook = load_notebook(NOTEBOOK_PAIRS[1][1])
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, str(NOTEBOOK_PAIRS[1][1]), "exec"), namespace)

    event = namespace["event"]
    assert event.num_nodes == 8
    assert event.r.shape == event.phi.shape == event.z.shape == (8,)
    assert event.particle_id.shape == (8,)
    assert event.edge_index.shape == (2, 12)
    assert event.edge_y.shape == event.edge_scores.shape == (12,)
    assert int(event.edge_y.sum()) == 5
    assert namespace["graph_efficiency"] == 5 / 6
    assert namespace["graph_purity"] == 5 / 12


def test_bundled_pipeline_data_has_expected_contract():
    import torch

    data_root = ROOT / "tutorial_data" / "edge_classifier"
    expected_counts = {"trainset": 8, "valset": 2, "testset": 2}
    required = {
        "hit_x",
        "hit_r",
        "hit_phi",
        "hit_z",
        "edge_index",
        "edge_y",
        "track_edges",
        "track_to_edge_map",
        "event_id",
        "config",
    }
    for split, count in expected_counts.items():
        paths = sorted((data_root / split).glob("*.pyg"))
        assert len(paths) == count
        for path in paths:
            event = torch.load(path, map_location="cpu", weights_only=False)
            assert required <= set(event.keys())
            assert event.edge_index.shape[0] == 2
            assert event.edge_y.shape == (event.edge_index.shape[1],)
            assert event.track_to_edge_map.shape == (event.track_edges.shape[1],)
            assert event.edge_index.max() < event.num_nodes


def test_acorn_submodule_is_pinned_and_clean_after_notebook_execution():
    acorn_root = ROOT / "vendor" / "acorn"
    assert (acorn_root / ".git").exists(), "Run: git submodule update --init --recursive"
    assert (acorn_root / "acorn" / "core").is_dir()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=acorn_root, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=acorn_root, text=True,
        capture_output=True, check=True,
    ).stdout
    assert revision == "f8b8787e269e0ba504d1bf0a555806b7f69d04e2"
    assert status == ""
