import numpy as np


class Node:

    def __init__(self, attribute=None, avail_attrs=[], depth=None, children=dict(), case_ids=[], is_leaf=False):
        self.is_leaf = is_leaf
        self.case_ids = case_ids
        self.attribute = attribute
        self.avail_attrs = avail_attrs
        self.children = children
        # self.xvic = None

        self.depth = depth

    def add_child(self, value, child):
        self.children[value] = child

    def add_case(self, case):
        self.case_ids.append(case)

    def set_cases(self, case_ids):
        self.case_ids = case_ids

    def set_terminal(self, case_ids):
        self.is_leaf = True
        self.case_ids = case_ids

    def get_instances(self):
        out = set()
        for val in self.children.values():
            if isinstance(val, Node):
                out.update(val.get_instances())
            else:
                out.update(set(val))

        return list(out)
