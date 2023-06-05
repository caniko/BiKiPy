from formulaic.parser import DefaultFormulaParser

f = [
    f"{token.token} : {token.kind.value}"
    for token in DefaultFormulaParser(include_intercept=False).get_tokens("~ fov | proximity")
]
1
