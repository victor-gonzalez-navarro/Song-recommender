class Node:

    def __init__(self, attribute=None, avail_attrs=[], depth=None, children=dict(), terminal_cases=[], is_leaf=False,
                 xvic=[]):
        self.is_leaf = is_leaf
        self.terminal_cases = terminal_cases
        self.attribute = attribute
        self.avail_attrs = avail_attrs
        self.children = children

        self.xvic = xvic
        self.depth = depth

    def add_child(self, value, child):
        self.children[value] = child

    def add_case(self, case):
        self.terminal_cases.append(case)

    def set_terminal(self, terminal_cases):
        self.is_leaf = True
        self.terminal_cases = terminal_cases

    def get_instances(self):
        out = set()
        for val in self.children.values():
            if isinstance(val, Node):
                out.update(val.get_instances())
            else:
                out.update(set(val))

        return list(out)
