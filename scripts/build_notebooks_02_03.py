"""Build the paired student/solution notebooks for tutorials 02 and 03."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
SOLUTIONS = NOTEBOOKS / "solutions"


def md(cell_id, source):
    cell = nbf.v4.new_markdown_cell(source)
    cell["id"] = cell_id
    return cell


def code(cell_id, source, exercise=False):
    cell = nbf.v4.new_code_cell(source)
    cell["id"] = cell_id
    if exercise:
        cell.metadata["tags"] = ["exercise"]
    return cell


def notebook(cells):
    result = nbf.v4.new_notebook(cells=cells)
    result.metadata.kernelspec = {
        "display_name": "Python 3 (acorn-tutorial)",
        "language": "python",
        "name": "python3",
    }
    result.metadata.language_info = {"name": "python", "version": "3.10"}
    return result


def pipeline_cells(solution):
    role = "Solution" if solution else "Student"
    event_answer = (
        'event_path = sorted((DATA / "trainset").glob("*.pyg"))[0]\n'
        'event = torch.load(event_path, map_location="cpu", weights_only=False)'
        if solution
        else 'event_path = None  # TODO: first .pyg path in DATA / "trainset"\n'
        'event = None       # TODO: torch.load it on CPU with weights_only=False'
    )
    mapping_answer = (
        'key_meanings = {\n'
        '    "stage": "which ACORN pipeline stage owns the model",\n'
        '    "model": "exported Python class selected at runtime",\n'
        '    "input_dir": "directory containing input split folders",\n'
        '    "stage_dir": "directory for checkpoints, logs, and scored events",\n'
        '}'
        if solution
        else 'key_meanings = {\n'
        '    "stage": None,      # TODO\n'
        '    "model": None,      # TODO\n'
        '    "input_dir": None,  # TODO\n'
        '    "stage_dir": None,  # TODO\n'
        '}'
    )
    train_answer = (
        'train_config["input_dir"] = str(DATA)\n'
        'train_config["stage_dir"] = str(STAGE_DIR)\n'
        'train_config["accelerator"] = "cpu"\n'
        'train_config["num_workers"] = [0, 0, 0]\n'
        'train_config["log_wandb"] = False'
        if solution
        else 'train_config["input_dir"] = None       # TODO: str(DATA)\n'
        'train_config["stage_dir"] = None       # TODO: str(STAGE_DIR)\n'
        'train_config["accelerator"] = None     # TODO: "cpu"\n'
        'train_config["num_workers"] = None     # TODO: [0, 0, 0]\n'
        'train_config["log_wandb"] = None       # TODO: False'
    )
    checkpoint_answer = (
        'checkpoints = sorted((STAGE_DIR / "artifacts").glob("*.ckpt"))\n'
        'checkpoint = checkpoints[0]'
        if solution
        else 'checkpoints = None  # TODO: sorted .ckpt files in STAGE_DIR / "artifacts"\n'
        'checkpoint = None   # TODO: select the first checkpoint'
    )
    inference_answer = (
        'scored_paths = sorted((STAGE_DIR / "testset").glob("*.pyg"))\n'
        'scored_event = torch.load(scored_paths[0], map_location="cpu", weights_only=False)'
        if solution
        else 'scored_paths = None  # TODO: sorted .pyg files in STAGE_DIR / "testset"\n'
        'scored_event = None  # TODO: load the first scored event on CPU'
    )
    metric_answer = (
        'predicted = scored_event.edge_scores >= score_cut\n'
        'true_positive = int((predicted & scored_event.edge_y.bool()).sum())\n'
        'selected = int(predicted.sum())\n'
        'truth = int(scored_event.edge_y.sum())\n'
        'edge_efficiency = true_positive / truth\n'
        'edge_purity = true_positive / selected if selected else 0.0'
        if solution
        else 'predicted = None       # TODO: edge_scores >= score_cut\n'
        'true_positive = None   # TODO: selected edges that are also true\n'
        'selected = None        # TODO: number of selected edges\n'
        'truth = None           # TODO: number of true edges\n'
        'edge_efficiency = None # TODO: true_positive / truth\n'
        'edge_purity = None     # TODO: true_positive / selected, or 0 if none'
    )
    return [
        md("title", f"# 02 — Run the toy ACORN pipeline\n\n**{role} notebook · 45–60 minutes · CPU only**\n\nFollow one tiny dataset through ACORN training, inference, and evaluation. The model is deliberately treated as a black box here; tutorial 03 opens that box."),
        md("how-to-use", "## How to use this notebook\n\nRun cells in order with **Shift+Enter**. The three ACORN commands train only a tiny one-step GNN and normally finish within a few minutes on CPU. Fill all six exercise cells and use their assertions as immediate feedback."),
        code("setup", '''from pathlib import Path
import os
import subprocess
import sys
import tempfile

import pandas as pd
import torch
import yaml

ROOT = Path.cwd()
if not (ROOT / "vendor" / "acorn").is_dir():
    ROOT = ROOT.parent.resolve()
assert (ROOT / "vendor" / "acorn" / "acorn").is_dir(), "Initialize the ACORN submodule first"

DATA = ROOT / "tutorial_data" / "edge_classifier"
runtime = tempfile.TemporaryDirectory(prefix="acorn_tutorial_02_")
RUNTIME = Path(runtime.name)
STAGE_DIR = RUNTIME / "interaction_gnn"

ACORN_ENV = os.environ.copy()
ACORN_ENV["PYTHONPATH"] = str(ROOT / "vendor" / "acorn") + os.pathsep + ACORN_ENV.get("PYTHONPATH", "")
ACORN_ENV["MPLCONFIGDIR"] = str(RUNTIME / "matplotlib")

def run_acorn(operation, config_path):
    command = [sys.executable, "-m", "acorn.core.entrypoint_stage", operation, str(config_path)]
    result = subprocess.run(command, cwd=ROOT, env=ACORN_ENV, text=True,
                            capture_output=True, timeout=240, check=False)
    lines = (result.stdout + result.stderr).splitlines()
    print("$ acorn", operation, config_path.name)
    print("\\n".join(lines[-20:]))
    result.check_returncode()
    return result

print("runtime workspace:", RUNTIME)'''),
        md("data-heading", "## 1. Inputs: split folders of PyG events\n\nACORN stages exchange `.pyg` event files. This repository bundles eight training, two validation, and two test events. They extend tutorial 01's graph with the metadata required by `EdgeClassifierStage`."),
        code("exercise-1", event_answer + '''

required = {"hit_x", "hit_r", "hit_phi", "hit_z", "edge_index", "edge_y",
            "track_edges", "track_to_edge_map", "event_id", "config"}
print(event)
assert required <= set(event.keys())
assert event.num_nodes == 8
assert event.edge_index.shape == (2, 12)''', True),
        md("yaml-heading", "## 2. Configuration is the pipeline interface\n\nA short YAML file becomes a Python dictionary. `stage` chooses an ACORN subpackage and `model` chooses a class exported by that stage. `input_dir` reads the preceding stage's artifacts; `stage_dir` receives this stage's logs, checkpoints, plots, and predictions."),
        code("exercise-2", mapping_answer + '''

assert key_meanings["stage"].startswith("which ACORN")
assert "class" in key_meanings["model"]
assert "input" in key_meanings["input_dir"]
assert "checkpoints" in key_meanings["stage_dir"]''', True),
        md("train-heading", "## 3. Train a one-step InteractionGNN\n\nThe supplied template uses one graph iteration, hidden width 16, and ten epochs. The important reliability settings are CPU acceleration, zero worker processes, and disabled W&B logging. The resolved config is written only to this notebook's temporary workspace."),
        code("exercise-3", '''train_template = ROOT / "configs" / "02_interaction_gnn_train.yaml"
train_config = yaml.safe_load(train_template.read_text())
''' + train_answer + '''

assert train_config["accelerator"] == "cpu"
assert train_config["num_workers"] == [0, 0, 0]
assert train_config["log_wandb"] is False
assert train_config["n_graph_iters"] == 1

train_path = RUNTIME / "train.yaml"
train_path.write_text(yaml.safe_dump(train_config, sort_keys=False))
run_acorn("train", train_path)''', True),
        md("checkpoint-heading", "## 4. A checkpoint connects training to inference\n\nLightning stores learned parameters and hyperparameters in `.ckpt` files. ACORN inference searches `stage_dir` for the newest checkpoint when none is supplied explicitly. CSV logs make the short loss history inspectable without an online service."),
        code("exercise-4", checkpoint_answer + '''

metric_paths = sorted(STAGE_DIR.rglob("metrics.csv"))
history = pd.read_csv(metric_paths[0])
print("checkpoint:", checkpoint.name)
print(history.dropna(subset=["val_loss"])[["epoch", "val_loss"]].tail())
assert checkpoint.suffix == ".ckpt"
assert "val_loss" in history.columns''', True),
        md("infer-heading", "## 5. Inference writes scored events\n\nTraining changes parameters; inference keeps them fixed, calculates one probability per candidate edge, stores it as `edge_scores`, appends the stage configuration, and writes new `.pyg` files under `stage_dir`."),
        code("exercise-5", '''infer_config = yaml.safe_load((ROOT / "configs" / "02_interaction_gnn_infer.yaml").read_text())
infer_config["input_dir"] = str(DATA)
infer_config["stage_dir"] = str(STAGE_DIR)
infer_path = RUNTIME / "infer.yaml"
infer_path.write_text(yaml.safe_dump(infer_config, sort_keys=False))
run_acorn("infer", infer_path)

''' + inference_answer + '''

print("scores:", scored_event.edge_scores)
assert scored_event.edge_scores.shape == scored_event.edge_y.shape
assert torch.isfinite(scored_event.edge_scores).all()
assert len(scored_event.config) >= 1''', True),
        md("eval-heading", "## 6. Evaluation applies an operating point\n\n`acorn eval` reads scored events, applies the requested target-track selection, and produces configured plots. A score cut converts probabilities into decisions. Edge efficiency is the fraction of truth edges retained; edge purity is the fraction of selected edges that are true."),
        code("eval-run", '''eval_config = yaml.safe_load((ROOT / "configs" / "02_interaction_gnn_eval.yaml").read_text())
eval_config["input_dir"] = str(DATA)
eval_config["stage_dir"] = str(STAGE_DIR)
eval_path = RUNTIME / "eval.yaml"
eval_path.write_text(yaml.safe_dump(eval_config, sort_keys=False))
run_acorn("eval", eval_path)'''),
        code("exercise-6", '''score_cut = eval_config["score_cut"]
''' + metric_answer + '''

print(f"edge efficiency at {score_cut}: {edge_efficiency:.1%}")
print(f"edge purity at {score_cut}: {edge_purity:.1%}")
assert 0.0 <= edge_efficiency <= 1.0
assert 0.0 <= edge_purity <= 1.0''', True),
        md("flow-heading", "## 7. The complete data flow\n\n```text\ntutorial_data/edge_classifier/*.pyg\n        │  acorn train + train.yaml\n        ▼\nstage_dir/artifacts/*.ckpt + CSV logs\n        │  acorn infer + infer.yaml\n        ▼\nstage_dir/{trainset,valset,testset}/*.pyg with edge_scores\n        │  acorn eval + eval.yaml\n        ▼\nevaluation plots and operating-point metrics\n```"),
        md("recap-heading", "## 8. Recap and next step\n\nYou can now explain how YAML selects a stage and model, why input and output directories differ, why training/inference/evaluation are separate operations, and how a checkpoint connects them. Tutorial 03 implements the model class that sits inside this workflow."),
    ]


def model_source(solution):
    if solution:
        init = '''n_features = len(hparams["node_features"])
        self.network = torch.nn.Sequential(
            torch.nn.Linear(2 * n_features, hparams["hidden"]),
            torch.nn.ReLU(),
            torch.nn.Linear(hparams["hidden"], 1),
        )'''
        forward = '''x = torch.stack(
            [batch[name] for name in self.hparams["node_features"]], dim=-1
        ).float()
        start, end = batch.edge_index
        edge_input = torch.cat([x[start], x[end]], dim=-1)
        return self.network(edge_input).squeeze(-1)'''
    else:
        init = '''n_features = None  # TODO: number of configured node features
        self.network = None  # TODO: Linear(2*n_features, hidden), ReLU, Linear(hidden, 1)'''
        forward = '''x = None  # TODO: stack configured batch features along the last dimension
        start, end = None, None  # TODO: unpack batch.edge_index
        edge_input = None  # TODO: concatenate source and destination rows of x
        return None  # TODO: network output with its final size-1 dimension removed'''
    return f'''import torch
from ..edge_classifier_stage import EdgeClassifierStage


class TinyEdgeMLP(EdgeClassifierStage):
    def __init__(self, hparams):
        super().__init__(hparams)
        {init}

    def forward(self, batch):
        {forward}
'''


def model_cells(solution):
    role = "Solution" if solution else "Student"
    width = "input_width = 2 * len(node_features)" if solution else "input_width = None  # TODO: two endpoints times the number of features"
    stack = 'x = torch.stack([event[name] for name in node_features], dim=-1).float()' if solution else 'x = None  # TODO: stack event[name] for each configured feature'
    gather = 'start, end = event.edge_index\nsource_features = x[start]\ndestination_features = x[end]' if solution else 'start, end = None, None       # TODO: unpack event.edge_index\nsource_features = None          # TODO: gather x at start\ndestination_features = None     # TODO: gather x at end'
    concat = 'edge_input = torch.cat([source_features, destination_features], dim=-1)\nnetwork = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))\nlogits = network(edge_input).squeeze(-1)' if solution else 'edge_input = None  # TODO: concatenate endpoint features along the last dimension\nnetwork = None     # TODO: Linear(6, 8), ReLU, Linear(8, 1)\nlogits = None      # TODO: network(edge_input), then squeeze the last dimension'
    registration = 'registration = \'\\nfrom .models.tiny_edge_mlp import TinyEdgeMLP\\n__all__.append("TinyEdgeMLP")\\n\'' if solution else 'registration = None  # TODO: import TinyEdgeMLP and append its name to __all__'
    source = model_source(solution)
    return [
        md("title", f"# 03 — Create an ACORN edge classifier\n\n**{role} notebook · 45–60 minutes · CPU only**\n\nBuild `TinyEdgeMLP`, test its tensor contract, register it in a disposable copy of the pinned ACORN source, and select it through real YAML."),
        md("how-to-use", "## How to use this notebook\n\nRun cells in order with **Shift+Enter**. The disposable source copy exists only for this kernel session, so the pinned submodule remains clean. Fill the six exercise cells; the final cells run the registered model through ACORN."),
        code("setup", '''from pathlib import Path
import importlib
import os
import shutil
import subprocess
import sys
import tempfile

import torch
import yaml

ROOT = Path.cwd()
if not (ROOT / "vendor" / "acorn").is_dir():
    ROOT = ROOT.parent.resolve()
DATA = ROOT / "tutorial_data" / "edge_classifier"
event = torch.load(sorted((DATA / "trainset").glob("*.pyg"))[0], map_location="cpu", weights_only=False)
node_features = ["hit_r", "hit_phi", "hit_z"]
print(event)'''),
        md("contract-heading", "## 1. One raw logit per candidate edge\n\nAn edge classifier receives `N` nodes with `F` selected features and `E` candidate edges. Each edge has two endpoints, so a plain endpoint MLP receives `2F` values and returns `E` raw logits. The base class applies the sigmoid where probabilities are needed; `forward` must not apply it."),
        code("exercise-1", width + '''
assert input_width == 6''', True),
        md("stack-heading", "## 2. Stack named node features\n\nACORN stores features as separate node-like tensors and lists their names in `hparams`. Stacking them produces the `[N, F]` matrix consumed by a model."),
        code("exercise-2", stack + '''
print("nodes:", x.shape)
assert x.shape == (8, 3)
assert torch.isfinite(x).all()''', True),
        md("gather-heading", "## 3. Gather both edge endpoints\n\nThe two rows of `edge_index` provide source and destination node indices. Indexing `x` with them produces two `[E, F]` tensors."),
        code("exercise-3", gather + '''
assert source_features.shape == destination_features.shape == (12, 3)
assert torch.equal(source_features[0], x[0])
assert torch.equal(destination_features[0], x[2])''', True),
        md("network-heading", "## 4. Convert endpoint features into logits\n\nConcatenate the endpoints to `[E, 2F]`, then use a small MLP ending in one output. Removing only the final singleton dimension gives `[E]`, the contract expected by `EdgeClassifierStage`."),
        code("exercise-4", concat + '''
assert edge_input.shape == (12, 6)
assert logits.shape == (12,)
assert torch.isfinite(logits).all()''', True),
        md("class-heading", "## 5. Put the computation in an ACORN subclass\n\nLayers belong in `__init__`; tensor computation belongs in `forward`. Calling `super().__init__(hparams)` activates the base stage's data, loss, optimizer, validation, checkpoint, and inference behavior. The module text below is also what we will place in ACORN's model package."),
        code("exercise-5", 'MODEL_SOURCE = r\'\'\'' + source.replace("'''", "\\'\\'\\'") + "'''\n\ncompile(MODEL_SOURCE, \"tiny_edge_mlp.py\", \"exec\")\nassert \"class TinyEdgeMLP(EdgeClassifierStage)\" in MODEL_SOURCE\nassert \"torch.sigmoid\" not in MODEL_SOURCE", True),
        md("register-heading", "## 6. Register the class for YAML lookup\n\nACORN resolves `stage: edge_classifier` and `model: TinyEdgeMLP` with `str_to_class`. Therefore the class file must live in the stage package and the stage's `__init__.py` must import and export it. We perform those edits in a temporary copy, never in the pinned submodule."),
        code("scratch-copy", '''scratch = tempfile.TemporaryDirectory(prefix="acorn_tutorial_03_")
SCRATCH = Path(scratch.name)
ACORN_COPY = SCRATCH / "acorn-source"
shutil.copytree(ROOT / "vendor" / "acorn", ACORN_COPY)
model_path = ACORN_COPY / "acorn" / "stages" / "edge_classifier" / "models" / "tiny_edge_mlp.py"
model_path.write_text(MODEL_SOURCE)
init_path = ACORN_COPY / "acorn" / "stages" / "edge_classifier" / "__init__.py"'''),
        code("exercise-6", registration + '''
assert "TinyEdgeMLP" in registration
init_path.write_text(init_path.read_text() + registration)

scratch_env = os.environ.copy()
scratch_env["PYTHONPATH"] = str(ACORN_COPY) + os.pathsep + scratch_env.get("PYTHONPATH", "")
check = subprocess.run(
    [sys.executable, "-c", "from acorn.core.core_utils import str_to_class; print(str_to_class('edge_classifier', 'TinyEdgeMLP').__name__)"],
    env=scratch_env, cwd=ROOT, text=True, capture_output=True, timeout=60, check=True,
)
print(check.stdout.strip())
assert "TinyEdgeMLP" in check.stdout''', True),
        md("pipeline-heading", "## 7. Select the new model from YAML\n\nNothing in ACORN's CLI needs a special case. Changing `model` to the exported class name selects it; the inherited stage still supplies training and checkpointing. This smoke run overfits two toy events and runs inference from the resulting checkpoint."),
        code("pipeline-run", '''stage_dir = SCRATCH / "tiny_edge_mlp_run"
train_config = yaml.safe_load((ROOT / "configs" / "02_interaction_gnn_train.yaml").read_text())
train_config.update({
    "model": "TinyEdgeMLP", "input_dir": str(DATA), "stage_dir": str(stage_dir),
    "data_split": [2, 1, 0], "hidden": 8, "max_epochs": 15,
})
train_path = SCRATCH / "tiny_train.yaml"
train_path.write_text(yaml.safe_dump(train_config, sort_keys=False))

def run_scratch(operation, path):
    result = subprocess.run(
        [sys.executable, "-m", "acorn.core.entrypoint_stage", operation, str(path)],
        env=scratch_env, cwd=ROOT, text=True, capture_output=True, timeout=240, check=False,
    )
    print("$ acorn", operation, path.name)
    print("\\n".join((result.stdout + result.stderr).splitlines()[-12:]))
    result.check_returncode()

run_scratch("train", train_path)
assert list((stage_dir / "artifacts").glob("*.ckpt"))

infer_config = {key: train_config[key] for key in [
    "stage", "model", "input_dir", "stage_dir", "project", "accelerator",
    "devices", "nodes", "data_split", "variable_with_prefix", "num_workers",
]}
infer_path = SCRATCH / "tiny_infer.yaml"
infer_path.write_text(yaml.safe_dump(infer_config, sort_keys=False))
run_scratch("infer", infer_path)
output_path = sorted((stage_dir / "trainset").glob("*.pyg"))[0]
output_event = torch.load(output_path, map_location="cpu", weights_only=False)
assert output_event.edge_scores.shape == output_event.edge_y.shape
assert torch.isfinite(output_event.edge_scores).all()'''),
        md("recap-heading", "## 8. Recap and contribution checklist\n\nA new ACORN edge classifier inherits `EdgeClassifierStage`, calls `super()`, creates layers in `__init__`, returns one raw finite logit per edge from `forward`, is imported by the stage package, is named under `model:` in YAML, and is smoke-tested on a tiny dataset before expensive training."),
    ]


def write_pair(stem, builder):
    nbf.write(notebook(builder(False)), NOTEBOOKS / f"{stem}.ipynb")
    nbf.write(notebook(builder(True)), SOLUTIONS / f"{stem}_solution.ipynb")


def main():
    write_pair("02_run_the_toy_pipeline", pipeline_cells)
    write_pair("03_create_an_edge_classifier", model_cells)
    print("Built tutorials 02 and 03 with matching solution notebooks")


if __name__ == "__main__":
    main()
