import numpy as np
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor
from pydantic_numpy.typing import Np1DArrayBool

# TODO: Fix arg > 2-"A and not B or C" is incorrectly evaluated

_heuristic_merge_grammar = Grammar(
    """
    expr = and_expr / or_expr / var
    or_expr = var ws ("or" / "OR" / "|") ws expr
    and_expr = var ws ("and" / "AND" / "&") ws expr
    var = not_var / var_atom
    not_var = ("not" / "~") ws var_atom
    var_atom = ~"\\w+"
    ws = ~"\\s*"
    """
)


class HeuristicMergeVisitor(NodeVisitor):
    def __init__(self, context):
        self.context = context

    def visit_expr(self, node, children):
        return children[0]

    def visit_or_expr(self, node, children):
        lhs: list[Np1DArrayBool]
        rhs: list[Np1DArrayBool]
        lhs, _, _, _, rhs = children
        return [np.logical_or(po_lhs, po_rhs) for po_lhs, po_rhs in zip(lhs, rhs)]

    def visit_and_expr(self, node, children):
        lhs: list[Np1DArrayBool]
        rhs: list[Np1DArrayBool]
        lhs, _, _, _, rhs = children
        return [np.logical_and(po_lhs, po_rhs) for po_lhs, po_rhs in zip(lhs, rhs)]

    def visit_var(self, node, children):
        return children[0]

    def visit_not_var(self, node, children):
        var_atom: list[Np1DArrayBool]
        _, _, var_atom = children
        return [np.logical_not(po_var_atom) for po_var_atom in var_atom]

    def visit_var_atom(self, node, children):
        return self.context[node.text]

    def visit_ws(self, node, children):
        return None

    def generic_visit(self, node, children):
        return children or node


def parse_heuristic_merge_equation(formula: str, alias_to_heuristic_result: dict[str, Np1DArrayBool]) -> Np1DArrayBool:
    visitor = HeuristicMergeVisitor(alias_to_heuristic_result)
    tree = _heuristic_merge_grammar.parse(formula)
    return visitor.visit(tree)
