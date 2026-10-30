import json
import os
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def generate_visuals():
    # Automatically find the newest run
    base_dir = Path(__file__).resolve().parent / "runs"
    runs = list(base_dir.glob("*/reflection_pipeline_summary.json"))
    if not runs:
        print("❌ No run files found.")
        return
    json_path = max(runs, key=os.path.getmtime)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tree_snapshot = data.get("tree")
    summary = data.get("summary", [])

    # 1. Generate ASCII tree for terminal
    if tree_snapshot and tree_snapshot.get("nodes"):
        print_ascii_tree_from_tree(tree_snapshot)
        plot_graph_from_tree(tree_snapshot, json_path.parent.name)
    else:
        print_ascii_tree(summary)
        plot_graph(summary, json_path.parent.name)

def print_ascii_tree(summary):
    children = defaultdict(list)
    node_info = {0: {"reward": 0.0, "status": "ROOT", "step": 0}}
    for i, step_data in enumerate(summary):
        node_id = i + 1
        parent_id = step_data.get("parent_id", 0)
        children[parent_id].append(node_id)
        node_info[node_id] = {"reward": step_data.get("reward", 0.0), "status": "SUCCESS" if step_data.get("success") else "FAILED", "step": step_data.get("step")}

    def draw(node, prefix=""):
        kids = children[node]
        for i, k in enumerate(kids):
            is_last = (i == len(kids) - 1)
            marker = "└── " if is_last else "├── "
            info = node_info[k]
            print(f"{prefix}{marker}[Node {k}] Step {info['step']} | {'🟢' if info['status']=='SUCCESS' else '🔴'}")
            draw(k, prefix + ("    " if is_last else "│   "))
    
    print("\n🌲 Tree Structure:")
    draw(0)

def print_ascii_tree_from_tree(tree_snapshot):
    nodes = tree_snapshot.get("nodes", [])
    children = defaultdict(list)
    node_info = {}

    for node in nodes:
        node_id = node.get("node_id")
        parent_id = node.get("parent_id")
        if parent_id is not None:
            children[parent_id].append(node_id)
        node_info[node_id] = {
            "visits": node.get("visits", 0),
            "q_value": node.get("q_value", 0.0),
            "type": node.get("type"),
        }

    def draw(node, prefix=""):
        kids = children.get(node, [])
        for i, k in enumerate(kids):
            is_last = (i == len(kids) - 1)
            marker = "└── " if is_last else "├── "
            info = node_info.get(k, {})
            q_value = info.get("q_value", 0.0)
            visits = info.get("visits", 0)
            node_type = info.get("type", "unknown")
            print(
                f"{prefix}{marker}[Node {k}] type={node_type} | q={q_value:.4f} | visits={visits}"
            )
            draw(k, prefix + ("    " if is_last else "│   "))

    print("\n🌲 Tree Structure:")
    draw(0)

def plot_graph(summary, run_name):
    G = nx.DiGraph()
    G.add_node(0, status="ROOT")
    for i, step_data in enumerate(summary):
        G.add_node(i + 1, status="SUCCESS" if step_data.get("success") else "FAILED")
        G.add_edge(step_data.get("parent_id", 0), i + 1)

    # 1. Manually calculate depth (y-coordinate) for hierarchy
    depths = nx.single_source_shortest_path_length(G, 0)
    pos = {}
    
    # Organize nodes by their depth
    nodes_by_depth = {}
    for node, depth in depths.items():
        nodes_by_depth.setdefault(depth, []).append(node)
    
    # 2. Assign positions: 
    # y = -depth, x = spread out nodes at the same depth
    for depth, nodes in nodes_by_depth.items():
        width = len(nodes)
        for i, node in enumerate(nodes):
            x = (i - (width - 1) / 2.0) * 2.0 # Spread horizontally
            y = -depth                        # Force vertical hierarchy
            pos[node] = (x, y)

    plt.figure(figsize=(14, 8))
    colors = ['#2ecc71' if G.nodes[n]['status'] == 'SUCCESS' 
              else '#e74c3c' if G.nodes[n]['status'] == 'FAILED' 
              else '#3498db' for n in G.nodes()]
    
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=1200, font_size=9, edge_color='#bdc3c7')
    
    plt.title(f"MCTS Hierarchy Tree: {run_name}", pad=20)
    plt.margins(0.2) # Add space around the graph
    plt.savefig("tree_visualization.png")
    print(f"\n✅ Plot saved to tree_visualization.png with hierarchical layout.")

def plot_graph_from_tree(tree_snapshot, run_name):
    nodes = tree_snapshot.get("nodes", [])
    G = nx.DiGraph()

    for node in nodes:
        node_id = node.get("node_id")
        parent_id = node.get("parent_id")
        G.add_node(
            node_id,
            q_value=node.get("q_value", 0.0),
            visits=node.get("visits", 0),
            node_type=node.get("type", "unknown"),
        )
        if parent_id is not None:
            G.add_edge(parent_id, node_id)

    depths = nx.single_source_shortest_path_length(G, 0)
    pos = {}
    nodes_by_depth = {}
    for node, depth in depths.items():
        nodes_by_depth.setdefault(depth, []).append(node)

    for depth, depth_nodes in nodes_by_depth.items():
        width = len(depth_nodes)
        for i, node in enumerate(depth_nodes):
            x = (i - (width - 1) / 2.0) * 2.0
            y = -depth
            pos[node] = (x, y)

    labels = {}
    for node_id, attrs in G.nodes(data=True):
        labels[node_id] = (
            f"{node_id}\n"
            f"q={attrs.get('q_value', 0.0):.2f}\n"
            f"v={attrs.get('visits', 0)}"
        )

    colors = []
    for node_id, attrs in G.nodes(data=True):
        if node_id == 0:
            colors.append("#3498db")
        else:
            colors.append("#2ecc71" if attrs.get("q_value", 0.0) >= 0 else "#e74c3c")

    plt.figure(figsize=(14, 8))
    nx.draw(
        G,
        pos,
        labels=labels,
        node_color=colors,
        node_size=1400,
        font_size=8,
        edge_color="#bdc3c7",
    )

    plt.title(f"MCTS Hierarchy Tree: {run_name}", pad=20)
    plt.margins(0.2)
    plt.savefig("tree_visualization.png")
    print(f"\n✅ Plot saved to tree_visualization.png with hierarchical layout.")

if __name__ == "__main__":
    generate_visuals()