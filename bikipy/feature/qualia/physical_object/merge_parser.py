from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

from bikipy.feature.qualia.physical_object.analysis.i import QualiaAnalysis

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
        var: QualiaAnalysis
        expr: QualiaAnalysis
        var, _, _, _, expr = children
        return var | expr

    def visit_and_expr(self, node, children):
        var: QualiaAnalysis
        expr: QualiaAnalysis
        var, _, _, _, expr = children
        return var & expr

    def visit_var(self, node, children):
        return children[0]

    def visit_not_var(self, node, children):
        var_atom: QualiaAnalysis
        _, _, var_atom = children
        return ~var_atom

    def visit_var_atom(self, node, children):
        return self.context[node.text]

    def visit_ws(self, node, children):
        return None

    def generic_visit(self, node, children):
        return children or node


def parse_heuristic_merge_equation(formula: str, alias_to_qualia_analysis: dict[str, QualiaAnalysis]) -> QualiaAnalysis:
    visitor = HeuristicMergeVisitor(alias_to_qualia_analysis)
    tree = _heuristic_merge_grammar.parse(formula)
    return visitor.visit(tree)
