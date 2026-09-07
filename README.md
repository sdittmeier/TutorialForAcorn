# ACORN School: From Python to Your First Model

This repository is a beginner-friendly companion to
[ACORN](https://github.com/GNN4ITkTeam/CommonFramework), the GNN4ITk common
framework. The first tutorial teaches the small part of Python and PyTorch that
you need before reading an ACORN edge-classification model.

## Start here

You need Git and a Conda-compatible package manager (Conda, Mamba, or
Micromamba). No GPU, CERN account, or external dataset is required. Clone with
submodules so tutorials 02 and 03 can use the pinned ACORN source tree.

```bash
git clone --recurse-submodules <this-repository-url>
cd TutorialForAcorn
conda env create -f environment.yml
conda activate acorn-tutorial
jupyter lab
```

If you already cloned the repository, initialize ACORN with
`git submodule update --init --recursive` before creating the environment.

Work through the numbered notebooks in order. Fill the six cells marked
**Exercise** in each notebook; every exercise has a small assertion that confirms
your answer. If you get stuck, compare your work with the matching notebook in
`notebooks/solutions/`.

Expected completion time: **45–60 minutes per notebook**.

## What you will learn

By the end of the tutorial you should be able to:

- use variables, lists, dictionaries, loops, functions, imports, and classes;
- inspect tensor shapes and dtypes;
- index tensors, select values with boolean masks, and combine tensors;
- recognize `self`, inheritance, `super()`, and `forward()` in model code;
- read a Python traceback from the final line upward; and
- follow the data flow through the first lines of an ACORN edge classifier.

The second tutorial then gives those tensors physical meaning. You will build a
real PyTorch Geometric `Data` event, distinguish node and edge attributes,
visualize true and fake candidate edges, and calculate graph efficiency and
purity.

The examples deliberately reuse tracking names such as `r`, `phi`, `z`,
`node_features`, and `edge_index` across both tutorials, so the Python concepts
connect directly to the graph-data contract.

## Test the tutorial

Both solution notebooks are executable from beginning to end on CPU:

```bash
python -m pytest
```

The tests execute a temporary in-memory copy, check the exercise/solution
structure, and ensure the notebooks do not require a GPU, network call, or an
absolute filesystem path.

## ACORN compatibility

The tutorial was designed against ACORN `dev` commit
[`f8b8787e269e0ba504d1bf0a555806b7f69d04e2`](https://github.com/GNN4ITkTeam/CommonFramework/tree/f8b8787e269e0ba504d1bf0a555806b7f69d04e2)
(package version 2.0.1). In particular, its final example mirrors the current
`InteractionGNN` conventions: hyperparameters in a dictionary, feature lookup
by name, `torch.stack`, unpacking `edge_index`, and concatenating endpoint
features.

The first two tutorials use only PyTorch and PyG. Tutorials 02 and 03 use the
pinned ACORN submodule and its real command-line, stage, checkpoint, and model
registration behavior. Their examples remain CPU-only and operate exclusively
on the bundled toy events.

## Planned learning path

1. `00_python_and_tensors.ipynb` — Python survival kit
2. `01_one_acorn_event.ipynb` — hits, tracks, tensors, and graphs (available)
3. `02_run_the_toy_pipeline.ipynb` — stages, YAML, training, inference, and evaluation (available)
4. `03_create_an_edge_classifier.ipynb` — implement and register `TinyEdgeMLP` (available)
