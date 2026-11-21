"""Truth-rated diagnostics only for the no-oracle cell-report simulator.

NEVER import into MATD3 Actor/Critic or radio admission code. This utility
reads ground truth only after a transition to score/report diagnostic progress.
"""
import torch
import warp as wp


def true_cell_progress(world, grid_n, cell_width):
    """Vectorized per-world true discovery and GCS delivery, and false cells.

    All source maps remain inaccessible to decentralized policies. A REPORT is
    counted as true only when its actually observed cell is occupied.
    """
    targets = wp.to_torch(world.targets)
    cells = wp.to_torch(world.report_cell).long()
    known = (wp.to_torch(world.known) != 0).any(dim=1)
    delivered = wp.to_torch(world.delivered) != 0
    target_x = (targets[..., 0] / cell_width).long().clamp(0, grid_n - 1)
    target_y = (targets[..., 1] / cell_width).long().clamp(0, grid_n - 1)
    target_cells = target_y * grid_n + target_x
    valid = cells >= 0
    # Use a scatter accumulator to avoid E x T x S broadcast intermediates.
    idx = cells.clamp(0, grid_n * grid_n - 1)
    found_grid = torch.zeros((len(cells), grid_n * grid_n), device=cells.device, dtype=torch.int32)
    delivered_grid = torch.zeros_like(found_grid)
    found_grid.scatter_add_(1, idx, (known & valid).int())
    delivered_grid.scatter_add_(1, idx, (delivered & valid).int())
    found = found_grid.gather(1, target_cells) > 0
    received = delivered_grid.gather(1, target_cells) > 0
    true_locations = torch.zeros_like(found_grid)
    true_locations.scatter_add_(1, target_cells, torch.ones_like(target_cells, dtype=torch.int32))
    false_slot = valid & (true_locations.gather(1, idx) == 0)
    return {
        'found': found,
        'delivered': received,
        'false_confirmed_cells': (known & false_slot).sum(dim=1),
        'false_gcs_reports': (delivered & false_slot).sum(dim=1),
        'hypothesis_count': valid.sum(dim=1),
    }
