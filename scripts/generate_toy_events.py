"""Generate the tiny, deterministic PyG dataset used by tutorials 02 and 03."""

from pathlib import Path

import torch
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tutorial_data" / "edge_classifier"
SPLITS = {"trainset": range(0, 8), "valset": range(8, 10), "testset": range(10, 12)}

TRACK_EDGES = torch.tensor([[0, 2, 4, 1, 3, 5], [2, 4, 6, 3, 5, 7]])
EDGE_INDEX = torch.tensor(
    [
        [0, 2, 4, 1, 3, 0, 1, 2, 3, 4, 5, 0],
        [2, 4, 6, 3, 5, 3, 2, 5, 4, 7, 6, 5],
    ]
)
EDGE_Y = torch.tensor(
    [True, True, True, True, True, False, False, False, False, False, False, False]
)


def make_event(event_id: int) -> Data:
    generator = torch.Generator().manual_seed(10_000 + event_id)
    r = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
    phi = torch.tensor([0.15, -0.20, 0.20, -0.14, 0.27, -0.08, 0.35, -0.02])
    z = torch.tensor([-3.0, 2.0, -1.5, 1.0, 0.0, 0.0, 1.5, -1.0])
    phi = phi + 0.012 * torch.randn(8, generator=generator)
    z = z + 0.08 * torch.randn(8, generator=generator)
    x = torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], dim=-1)

    return Data(
        num_nodes=8,
        hit_id=torch.arange(8),
        hit_x=x,
        hit_r=r,
        hit_phi=phi,
        hit_z=z,
        hit_region=torch.div(torch.arange(8), 2, rounding_mode="floor"),
        edge_index=EDGE_INDEX.clone(),
        edge_y=EDGE_Y.clone(),
        track_edges=TRACK_EDGES.clone(),
        track_to_edge_map=torch.tensor([0, 1, 2, 3, 4, -1]),
        track_particle_id=torch.tensor([0, 0, 0, 1, 1, 1]),
        track_particle_pt=torch.tensor([2000.0, 2000.0, 2000.0, 2500.0, 2500.0, 2500.0]),
        track_particle_nhits=torch.full((6,), 4),
        track_particle_primary=torch.ones(6, dtype=torch.bool),
        track_particle_pdgId=torch.tensor([13, 13, 13, -13, -13, -13]),
        event_id=event_id,
        config=[],
    )


def main() -> None:
    for split, event_ids in SPLITS.items():
        split_dir = OUTPUT / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for event_id in event_ids:
            torch.save(make_event(event_id), split_dir / f"event{event_id:03d}.pyg")
    print(f"Wrote 12 events to {OUTPUT}")


if __name__ == "__main__":
    main()
