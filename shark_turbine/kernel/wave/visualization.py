graphviz_disabled = False
try:
    import pygraphviz as pgv
except:
    graphviz_disabled = True
pandas_disabled = False
try:
    import pandas as pd
except:
    pandas_disabled = True
from torch import fx
from .scheduling.graph_utils import Edge
from ..ops.wave_ops import Output, get_custom
import math


def number_nodes(graph: fx.Graph) -> dict[int, int]:
    return {id(node): i for i, node in enumerate(graph.nodes)}


def visualize_graph(graph: fx.Graph, file_name: str):
    if graphviz_disabled:
        raise ImportError("pygraphviz not installed, cannot visualize graph")
    node_numbering = number_nodes(graph)
    G = pgv.AGraph(directed=True)
    for node in graph.nodes:
        G.add_node(node_numbering[id(node)], label=node.name)
    for node in graph.nodes:
        for user in node.users.keys():
            G.add_edge(node_numbering[id(node)], node_numbering[id(user)])
    G.layout(prog="dot")
    G.draw(file_name)


def visualize_edges(edges: list[Edge], file_name: str):
    if graphviz_disabled:
        raise ImportError("pygraphviz not installed, cannot visualize graph")
    G = pgv.AGraph(directed=True)
    node_map = {}
    count = 0
    for edge in edges:
        if edge._from not in node_map:
            node_map[edge._from] = count
            count += 1
            G.add_node(node_map[edge._from], label=f"{edge._from}")
        if edge._to not in node_map:
            node_map[edge._to] = count
            count += 1
            G.add_node(node_map[edge._to], label=f"{edge._to}")
        G.add_edge(
            node_map[edge._from],
            node_map[edge._to],
            label=f"({edge.weight.iteration_difference}, {edge.weight.delay})",
        )
    G.layout(prog="dot")
    G.draw(file_name)


def visualize_schedule(
    schedule: dict[fx.Graph, int], initiation_interval: int, file_name: str
):
    if pandas_disabled:
        raise ImportError("pandas not installed, cannot visualize schedule")

    max_time = max(schedule.values())
    max_stage = math.ceil(max_time / initiation_interval)
    rows = max_time + 1 + max_stage * initiation_interval
    cols = max_stage

    table = [["" for _ in range(cols)] for _ in range(rows)]
    for stage in range(max_stage):
        for key, value in schedule.items():
            table[value + stage * initiation_interval][stage] += f"{key}<br>"

    df = pd.DataFrame(table, columns=[f"Stage {i}" for i in range(cols)])
    s = df.style.set_properties(**{"text-align": "center"})
    s = s.set_table_styles(
        [
            {"selector": "", "props": [("border", "1px solid grey")]},
            {"selector": "tbody td", "props": [("border", "1px solid grey")]},
            {"selector": "th", "props": [("border", "1px solid grey")]},
            {"selector": "th", "props": [("min-width", "300px")]},
        ]
    )
    output = s.apply(
        lambda x: [
            (
                "background: lightgreen"
                if int(x.name) >= (max_stage - 1) * initiation_interval
                and int(x.name) < max_stage * initiation_interval
                else ""
            )
            for _ in x
        ],
        axis=1,
    ).to_html()
    with open(f"{file_name}", "w") as f:
        f.write(output)


def visualize_mapped_graphs(
    second: fx.Graph,
    mappings: list[dict[fx.Node, fx.Node]],
    file_name: str,
):
    """
    Given the pipelined graph and a list of mappings of nodes from the original
    graph to the pipelined graph (per stage), visualize the pipelined graph (with their original labels)

    """

    if graphviz_disabled:
        raise ImportError("pygraphviz not installed, cannot visualize graph")
    second_numbering = number_nodes(second)
    inverse_mapping: list[dict[fx.Node, fx.Node]] = []
    for stage, mapping in enumerate(mappings):
        inverse_mapping.append({v: k for k, v in mapping.items()})

    # Draw nodes and edges in the pipelined graph.
    G = pgv.AGraph(directed=True)
    G0 = G.add_subgraph(name="pipelined")
    for node in second.nodes:
        if hasattr(node, "scheduling_parameters"):
            stage = node.scheduling_parameters["stage"]
            name = inverse_mapping[stage][node].name
        else:
            name = node.name
        G0.add_node(
            second_numbering[id(node)],
            label=name,
            color="lightblue",
            style="filled",
        )
        for user in node.users.keys():
            if user not in second.nodes:
                continue
            if isinstance(get_custom(user), Output):
                continue
            G0.add_edge(
                second_numbering[id(node)],
                second_numbering[id(user)],
                color="black",
            )

    # Draw nodes and edges in the original graph.
    colors = ["red", "green", "orange", "purple", "orange", "cyan", "magenta"]
    max_stage = len(mappings)
    for stage, mapping in enumerate(mappings):
        for node, mapped_node in mapping.items():
            for user in node.users.keys():
                if user not in mapping:
                    continue
                mapped_user = mapping[user]
                G.add_edge(
                    second_numbering[id(mapped_node)],
                    second_numbering[id(mapped_user)],
                    label=f"{stage}",
                    color=colors[stage % max_stage],
                )

    # Draw edges between rotating registers for the same variable.
    for stage, mapping in enumerate(mappings):
        for node, mapped_node in mapping.items():
            for second_stage, second_mapping in enumerate(mappings):
                if stage == second_stage:
                    continue
                for second_node, second_mapped_node in second_mapping.items():
                    if node == second_node and stage < second_stage:
                        G.add_edge(
                            second_numbering[id(mapped_node)],
                            second_numbering[id(second_mapped_node)],
                            label=f"{stage} -> {second_stage}",
                            color="magenta",
                        )

    G.layout(prog="dot")
    G.draw(file_name)
