from typing import List, Set
import pyformlang.cfg
from pyformlang.cfg import Epsilon
import networkx as nx
import scipy.sparse as scpy


def cfg_to_weak_normal_form(cfg: pyformlang.cfg.CFG) -> pyformlang.cfg.CFG:
    # Determine tmp_eps so that to_normal_form function does not miss it, cause we need weak cnf
    tmp_eps = pyformlang.cfg.Terminal("Epsilon#Terminal")
    while tmp_eps in cfg.terminals or tmp_eps in cfg.variables:
        tmp_eps += "_"

    # terminals and variables auto-detect from productions field
    cfg = pyformlang.cfg.CFG(
        start_symbol=cfg.start_symbol,
        productions={
            pyformlang.cfg.Production(
                head=prod.head,
                body=[
                    cfg_obj if not isinstance(cfg_obj, Epsilon) else tmp_eps
                    for cfg_obj in prod.body
                ]
                if len(prod.body) > 0
                else [tmp_eps],
            )
            for prod in cfg.productions
        },
    )
    cfg = cfg.to_normal_form()

    # replace the epsilons back
    cfg = pyformlang.cfg.CFG(
        start_symbol=cfg.start_symbol,
        productions={
            pyformlang.cfg.Production(
                head=prod.head,
                body=[
                    cfg_obj if cfg_obj != tmp_eps else Epsilon()
                    for cfg_obj in prod.body
                ],
            )
            for prod in cfg.productions
        },
    )
    return cfg


def hellings_based_cfpq(
    cfg: pyformlang.cfg.CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    wcnf = cfg_to_weak_normal_form(cfg)
    m = set()

    for u, v, label in graph.edges(data="label"):
        if not label:
            continue
        for var in label_to_variables(wcnf, label):
            m.add((var, u, v))

    nullable = wcnf.get_nullable_symbols()
    for var in nullable:
        for node in graph.nodes:
            m.add((var, node, node))

    r = m.copy()

    def update_paths(new_triple, buffer, stack, result):
        if new_triple not in result:
            buffer.add(new_triple)
            stack.add(new_triple)

    while m:
        (var1, u1, v1) = m.pop()
        new_m = set()
        for var2, u2, v2 in r:
            if v2 == u1:
                for head in body_to_head(wcnf, [var2, var1]):
                    new_path = (head, u2, v1)
                    update_paths(new_path, new_m, m, r)

            if v1 == u2:
                for head in body_to_head(wcnf, [var1, var2]):
                    new_path = (head, u1, v2)
                    update_paths(new_path, new_m, m, r)

        r.update(new_m)

    ans = set()
    for var, u, v in r:
        if var.value == "S":
            if (not start_nodes or u in start_nodes) and (
                not final_nodes or v in final_nodes
            ):
                ans.add((u, v))

    return ans


def label_to_variables(
    wcnf: pyformlang.cfg.CFG, label: str
) -> List[pyformlang.cfg.Variable]:
    terminal = pyformlang.cfg.Terminal(label)
    return [
        prod.head
        for prod in wcnf.productions
        if terminal in prod.body and len(prod.body) == 1
    ]


def body_to_head(wcnf: pyformlang.cfg.CFG, body):
    return {prod.head for prod in wcnf.productions if prod.body == body}


def matrix_based_cfpq(
    cfg: pyformlang.cfg.CFG,
    graph: nx.DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> set[tuple[int, int]]:
    wcnf = cfg_to_weak_normal_form(cfg)

    n = graph.number_of_nodes()
    bool_matrices: dict[any, scpy.csr_matrix] = {
        var: scpy.csr_matrix((n, n), dtype=bool) for var in wcnf.variables
    }

    index_to_node = {i: node for i, node in enumerate(graph.nodes)}
    node_to_index = {node: i for i, node in index_to_node.items()}

    for u, v, label in graph.edges(data="label"):
        index_u, index_v = node_to_index[u], node_to_index[v]
        for var in label_to_variables(wcnf, label):
            bool_matrices[var][index_u, index_v] = True

    nullable = wcnf.get_nullable_symbols()
    for var in nullable:
        for node in graph.nodes:
            index = node_to_index[node]
            bool_matrices[var][index, index] = True

    m = list(wcnf.variables)

    while m:
        updated_not_terminal = m.pop(0)
        for production in wcnf.productions:
            if updated_not_terminal in production.body:
                matrix_only_new_edges: scpy.csr_matrix = (
                    bool_matrices[production.body[0]]
                    @ bool_matrices[production.body[1]]
                )
                prev_count_value = bool_matrices[production.head].count_nonzero()
                bool_matrices[production.head] += matrix_only_new_edges
                curr_count_value = bool_matrices[production.head].count_nonzero()
                if prev_count_value < curr_count_value:
                    m.append(production.head)

    ans = set()

    start_symbol = wcnf.start_symbol

    # maybe check not needed, need to see if the start symbol is included in wcnf.variables
    if start_symbol not in bool_matrices.keys():
        return ans

    r = bool_matrices[start_symbol]

    for i in range(n):
        for j in range(n):
            u, v = index_to_node[i], index_to_node[j]
            if (
                r[i, j]
                and (not start_nodes or u in start_nodes)
                and (not final_nodes or v in final_nodes)
            ):
                ans.add((u, v))

    return ans
