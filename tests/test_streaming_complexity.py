"""Static complexity guards for bounded Warp report streaming."""
import ast
from pathlib import Path


def _is_packet_capacity_range(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Attribute)
        and isinstance(node.args[0].value, ast.Name)
        and node.args[0].value.id == "p"
        and node.args[0].attr == "packet_capacity"
    )


def test_fill_session_does_not_rescan_full_packet_pool_per_fragment():
    source = Path("uav_network/communication/gpu/streaming.py").read_text()
    tree = ast.parse(source)
    fill = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "fill_session")
    fragment_loop = next(
        node for node in ast.walk(fill)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "offset"
    )
    rescans = [
        node for node in ast.walk(fragment_loop)
        if isinstance(node, ast.For) and _is_packet_capacity_range(node.iter)
    ]
    assert not rescans, "fill_session must not restart a full packet-capacity scan for every fragment"
