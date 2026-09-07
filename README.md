<<<<<<< HEAD
# TutorialForAcorn
=======
# ACORN School: Python and Tensors

This repository is a beginner-friendly companion to
[ACORN](https://github.com/GNN4ITkTeam/CommonFramework), the GNN4ITk common
framework. The first tutorial teaches the small part of Python and PyTorch that
you need before reading an ACORN edge-classification model.

## Start here

You need Git and a Conda-compatible package manager (Conda, Mamba, or
Micromamba). No GPU, CERN account, external dataset, or full ACORN installation
is required.

```bash
conda env create -f environment.yml
conda activate acorn-tutorial
jupyter lab
```

Open `notebooks/00_python_and_tensors.ipynb`. Work from top to bottom and fill
the six cells marked **Exercise**. Each exercise has a small assertion that
confirms your answer. If you get stuck, compare your work with
`notebooks/solutions/00_python_and_tensors_solution.ipynb`.

Expected completion time: **45–60 minutes**.

## What you will learn

By the end of the tutorial you should be able to:

- use variables, lists, dictionaries, loops, functions, imports, and classes;
- inspect tensor shapes and dtypes;
- index tensors, select values with boolean masks, and combine tensors;
- recognize `self`, inheritance, `super()`, and `forward()` in model code;
- read a Python traceback from the final line upward; and
- follow the data flow through the first lines of an ACORN edge classifier.

The examples deliberately use tracking names such as `r`, `phi`, `z`,
`node_features`, and `edge_index`. The physics and graph-data contract are
introduced in the next tutorial rather than assumed here.

## Test the tutorial

The solution notebook is executable from beginning to end on CPU:

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

Installing the full ACORN dependency stack is intentionally deferred until the
pipeline tutorial, where the package is actually used. This keeps the first
experience small and reliable while retaining a precise upstream reference.

## Planned learning path

1. `00_python_and_tensors.ipynb` — Python survival kit (this pilot)
2. `01_one_acorn_event.ipynb` — hits, tracks, tensors, and graphs
3. `02_run_the_toy_pipeline.ipynb` — stages, YAML, training, inference, and evaluation
4. `03_create_an_edge_classifier.ipynb` — implement and register `TinyEdgeMLP`

>>>>>>> d8a2f47 (initial commit)
