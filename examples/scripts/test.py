class a:
    def __init__(self, d):
        self.d = d


l = [a(2), a(1)]

c = sorted(l, key=lambda item: item.d)
print(c)
