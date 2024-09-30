from .finite_automata import AdjacencyMatrixFA
from .graph_utils import graph_to_nfa
from networkx import MultiDiGraph
from .regex_utils import regex_to_dfa
from .finite_automata import intersect_automata
from itertools import product
from functools import reduce
import scipy.sparse as scpy


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    nfa1 = graph_to_nfa(graph, start_nodes, final_nodes)
    dfa2 = regex_to_dfa(regex)

    graph_AMFA = AdjacencyMatrixFA(nfa1)
    regex_AMFA = AdjacencyMatrixFA(dfa2)

    intersect_fa = intersect_automata(graph_AMFA, regex_AMFA)
    tr_closure = intersect_fa.transitive_closure()

    result = set()

    for start_graph_state, final_graph_state in product(start_nodes, final_nodes):
        for start_regex_state, final_regex_state in product(
            dfa2.start_states, dfa2.final_states
        ):
            start_index = intersect_fa.states[(start_graph_state, start_regex_state)]
            final_index = intersect_fa.states[(final_graph_state, final_regex_state)]
            if tr_closure[start_index, final_index]:
                result.add((start_graph_state, final_graph_state))
    return result


def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    regex_dfa = regex_to_dfa(regex)
    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)

    nfa = AdjacencyMatrixFA(graph_nfa)
    dfa = AdjacencyMatrixFA(regex_dfa)
    symbols = dfa.adj_matrix.keys() & nfa.adj_matrix.keys()

    nfa_final_states_ind_to_state = {
        index: state for state, index in nfa.states.items() if state in final_nodes
    }

    start_nodes_enumerate = {i: st_node for i, st_node in enumerate(start_nodes)}
    front = init_front(start_nodes_enumerate, nfa, dfa)
    visited = front

    transpose_dfa = {
        sym: matrix.transpose()
        for sym, matrix in dfa.adj_matrix.items()
        if sym in symbols
    }
    k = dfa.count_states

    while front.sum() > 0:
        fronts = {sym: front @ nfa.adj_matrix[sym] for sym in symbols}

        fronts = {
            sym: mult_with_transpose(
                transpose_dfa[sym], fronts[sym], len(start_nodes), k
            )
            for sym in symbols
        }

        front = reduce(lambda x, y: x + y, fronts.values(), front)
        front = front > visited
        visited = visited + front

    result = set()
    for i in range(len(start_nodes)):
        slice_visited = visited[i * k : (i + 1) * k]

        for dfa_final_state in dfa.final_states:
            ind = dfa.states[dfa_final_state]

            for column in slice_visited.getrow(ind).indices:
                if column in nfa_final_states_ind_to_state.keys():
                    result.add(
                        (
                            start_nodes_enumerate[i],
                            nfa_final_states_ind_to_state[column],
                        )
                    )

    return result


def mult_with_transpose(
    transpose_dfa: scpy.csr_matrix,
    front: scpy.csr_matrix,
    count_start_nodes: int,
    k: int,
) -> scpy.csr_matrix:
    new_front = scpy.csr_matrix((front.shape[0], front.shape[1]), dtype=bool)

    for i in range(count_start_nodes):
        new_front[i * k : (i + 1) * k] = transpose_dfa @ front[i * k : (i + 1) * k]

    return new_front


def init_front(
    start_nodes: dict, nfa: AdjacencyMatrixFA, dfa: AdjacencyMatrixFA
) -> scpy.csr_matrix:
    front = scpy.csr_matrix(
        (dfa.count_states * len(start_nodes), nfa.count_states),
        dtype=bool,
    )

    for i, start_node in start_nodes.items():
        column = nfa.states[start_node]

        for start_dfa_state in dfa.start_states:
            front[dfa.states[start_dfa_state] + i * dfa.count_states, column] = True

    return front
