import numpy as np
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor


_heuristic_merge_grammar = Grammar(
    """
    expr = or_expr / and_expr / var
    or_expr = var ws ("or" / "|") ws expr
    and_expr = var ws ("and" / "&") ws expr
    var = not_var / var_atom
    not_var = ("not" / "~") ws var_atom
    var_atom = ~"\w+"
    ws = ~"\s*"
    """
)


class HeuristicMergeVisitor(NodeVisitor):
    def __init__(self, context):
        self.context = context

    def visit_expr(self, node, children):
        return children[0]

    def visit_or_expr(self, node, children):
        var, _, _, _, expr = children
        return np.logical_or(var, expr)

    def visit_and_expr(self, node, children):
        var, _, _, _, expr = children
        return np.logical_and(var, expr)

    def visit_var(self, node, children):
        return children[0]

    def visit_not_var(self, node, children):
        _, _, var_atom = children
        return np.logical_not(var_atom)

    def visit_var_atom(self, node, children):
        return self.context[node.text]

    def visit_ws(self, node, children):
        return None

    def generic_visit(self, node, children):
        return children or node


def parse_heuristic_merge_equation(
    formula: str, name2boolean_index: dict[str, np.ndarray[bool, bool]]
) -> np.ndarray[bool, bool]:
    visitor = HeuristicMergeVisitor(name2boolean_index)
    tree = _heuristic_merge_grammar.parse(formula)
    return visitor.visit(tree)
