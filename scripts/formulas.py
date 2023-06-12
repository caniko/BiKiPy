import numpy as np
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

# define the grammar
grammar = Grammar(
    """
    expr = or_expr / and_expr / not_expr
    or_expr = var "or" var
    and_expr = var "and" var
    not_expr = ("not" / "~") var
    var = ~r"\w+"
    """
)


class BooleanExprVisitor(NodeVisitor):
    def __init__(self, variables):
        self.variables = variables

    def visit_expr(self, node, visited_children):
        return visited_children[0]

    def visit_or_expr(self, node, visited_children):
        left, _, right = visited_children
        return np.logical_or(left, right)

    def visit_and_expr(self, node, visited_children):
        left, _, right = visited_children
        return np.logical_and(left, right)

    def visit_not_expr(self, node, visited_children):
        _, value = visited_children
        return np.logical_not(value)

    def visit_parens(self, node, visited_children):
        _, expr, _ = visited_children
        return expr

    def visit_value(self, node, visited_children):
        return visited_children[0]

    def visit_var(self, node, visited_children):
        return self.variables[node.text]

    def generic_visit(self, node, visited_children):
        return visited_children or node


variables = {
    "a": np.array([True, False, True]),
    "b": np.array([False, False, True]),
}

visitor = BooleanExprVisitor(variables)
tree = grammar.parse("not a and b")
result = visitor.visit(tree)
print(result)
