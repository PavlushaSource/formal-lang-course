from typing import List
import pyformlang.cfg
from pyformlang.cfg import Epsilon
import networkx as nx
from itertools import product
import scipy.sparse as scpy

from project.graph_utils import graph_to_nfa
from project.finite_automata import AdjacencyMatrixFA


def cfg_to_weak_normal_form(cfg: pyformlang.cfg.CFG) -> pyformlang.cfg.CFG:
    # Determine tmp_eps so that to_normal_form function does not miss it, cause we need weak cnf
    tmp_eps = pyformlang.cfg.Terminal("Epsilon#Terminal")
    while tmp_eps in cfg.terminals or tmp_eps in cfg.variables:
        tmp_eps += "_"
    print(f"BEFORE REPALACE EPSILON - {[x for x in cfg.productions]}")
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
    print(f"REPLACE EPLISOL - {[x for x in cfg.productions]}")
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


def body_to_head(wcnf: pyformlang.cfg.CFG, body):
    res = {}
    for prod in wcnf.productions:
        # print(f"PB {prod.body} == OUT {body}")
        if prod.body == body:
            res[prod.head] = prod

    return {prod.head for prod in wcnf.productions if prod.body == body}


def hellings_based_cfpq(
    cfg: pyformlang.cfg.CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    nfa = graph_to_nfa(graph, start_nodes, final_nodes)

    wcnf = cfg_to_weak_normal_form(cfg)

    adjFA = AdjacencyMatrixFA(nfa)
    n = adjFA.count_states

    old_labels = [x for x in adjFA.adj_matrix.keys()]
    for label in old_labels:
        for var in label_to_variables(wcnf, label):
            adjFA.adj_matrix[var] = adjFA.adj_matrix[label]

        del adjFA.adj_matrix[label]

    m = {
        (var, adjFA.states[u], adjFA.states[v])
        for (u, label, v) in nfa._transition_function
        for var in label_to_variables(wcnf, label)
    }
    r = m.copy()

    labels = []
    n = adjFA.count_states
    while len(m) > 0:
        labels = list(adjFA.adj_matrix.keys())
        # print(f"\nLABELS {labels}\n")
        (var, u, v) = m.pop()

        for k in range(n):
            for label in labels:
                if adjFA.adj_matrix[label][k, u]:
                    body = [label, var]
                    for head in body_to_head(wcnf, body):
                        if head not in adjFA.adj_matrix.keys():
                            adjFA.adj_matrix[head] = scpy.csr_matrix((n, n), dtype=bool)

                        adjFA.adj_matrix[head][k, v] = True

                        new_pair = (head, k, v)
                        if new_pair not in r:
                            r.add((head, k, v))
                            m.add((head, k, v))

                if adjFA.adj_matrix[label][v, k]:
                    body = [var, label]
                    for head in body_to_head(wcnf, body):
                        if head not in adjFA.adj_matrix.keys():
                            adjFA.adj_matrix[head] = scpy.csr_matrix((n, n), dtype=bool)

                        adjFA.adj_matrix[head][u, k] = True

                        new_pair = (head, u, k)
                        if new_pair not in r:
                            r.add((head, u, k))
                            m.add((head, u, k))

        # print(f"\nCurrent STATE R - {r}\n")
        # print(f"\nCurrent STATE M - {m}\n")

    ans = set()

    if "S" not in adjFA.adj_matrix.keys():
        return ans
    for start_state, final_state in product(adjFA.start_states, adjFA.final_states):
        ind_start = adjFA.states[start_state]
        ind_end = adjFA.states[final_state]
        if adjFA.adj_matrix["S"][ind_start, ind_end]:
            ans.add((start_state, final_state))

    return ans


def label_to_variables(
    wcnf: pyformlang.cfg.CFG, label: str
) -> List[pyformlang.cfg.Variable]:
    terminal = pyformlang.cfg.Terminal(label)
    return [prod.head for prod in wcnf.productions if terminal in prod.body]
