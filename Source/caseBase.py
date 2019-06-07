import numpy as np
import pandas as pd
from Source.node import Node
from Source.preprocess import Preprocess


class CaseBase:

    def __init__(self, x):
        # Preprocess of the data to be stored in the Case Base
        n_clusters = 5
        self.num_class = 2
        self.prep = Preprocess()
        self.attr_names, self.attr_vals, self.attr_types = self.prep.extract_attr_info(x, self.num_class)
        x = x.values
        aux_x, self.attr_vals = self.prep.fit_predict(x[:,:-self.num_class], n_clusters=n_clusters)

        self.tree = None
        self.feat_selected = np.zeros((x.shape[1], 1))  # Depth at which each feature is selected
        self.max_depth = aux_x.shape[1]                     # Maximum depth corresponds to the number of attributes

        self.make_tree(x, aux_x)

    def make_tree(self, x, x_aux):
        self.tree = self.find_best_partition(x_aux, avail_attrs=list(range(x_aux.shape[1])), depth=0)
        self.tree.xvic = x  ####################################################################
        self.expand_tree(self.tree, x, x_aux, depth=1)

    def find_best_partition(self, x, avail_attrs, depth):
        best_score = -1
        best_aux_score = -1
        for feat_ix in avail_attrs:
            unique_vals, counts = np.unique(x[:, feat_ix], return_counts=True)
            score = len(unique_vals) / len(self.attr_vals[feat_ix])

            # Check the number of possible values for this attribute are in the remaining dataset
            if score > best_score:
                best_feat_ix = feat_ix
                best_score = score
                best_aux_score = np.std(counts)
            elif score == best_score:
                # In case of draw, select the attribute which values cover the most similar number of instances
                aux_score = np.std(counts)
                if aux_score < best_aux_score:
                    best_feat_ix = feat_ix
                    best_score = score
                    best_aux_score = aux_score

        # Annotate the depth at which this feature has been selected
        self.feat_selected[best_feat_ix] = depth

        # Remove the attribute from the list of available attributes
        avail_attrs = [attr for attr in avail_attrs if attr != best_feat_ix]

        # Create the Node and add a child per value of the selected attribute
        out_node = Node(attribute=best_feat_ix, avail_attrs=avail_attrs, depth=depth, children={})
        for val in self.attr_vals[best_feat_ix]:
            out_node.add_child(val, np.argwhere(x[:, best_feat_ix] == val)[:, 0])

        return out_node

    def expand_tree(self, tree, x, x_aux, depth):
        for key, val in tree.children.items():
            if len(val) == 0:
                # If the split left this branch empty, set the terminal boolean to True without adding any case
                tree.children[key] = Node(is_leaf=True, depth=depth)
            elif depth == self.max_depth:
                # If the maximum depth has been reached, add the terminal cases in the leaf node
                terminal_cases = x[val, :].tolist()
                tree.children[key] = Node(terminal_cases=terminal_cases, is_leaf=True, depth=depth, xvic=x[val, :])
            else:
                # Otherwise, find the best partition for this leaf and expand the subtree
                tree.children[key] = self.find_best_partition(x_aux[val, :], tree.avail_attrs, depth)
                tree.children[key].xvic = x[val, :]  #########################################################################
                self.expand_tree(tree.children[key], x[val, :], x_aux[val, :], depth + 1)

        return

    def check_node(self, x_tst, tree):
        if tree.is_leaf:
            return tree.terminal_cases
        else:
            return self.check_node(x_tst, tree.children[x_tst[tree.attribute]])

    def print_tree(self):
        print()
        print('--------------------')
        print('--------------------')
        print('The Case Base is:')
        print('--------------------')
        print('--------------------')
        print()
        self.print_tree_aux('Root', self.tree)

    def print_tree_aux(self, branch, tree):
        if not tree.is_leaf:
            print('\t\t\t\t|' * tree.depth + '\u2919\033[92m' + str(branch) + '\033[0m\u291a\u27f6attr_\033[1m' +
                  str(self.attr_names[tree.attribute]) + '\033[0m')
            for branch, child_tree in tree.children.items():
                self.print_tree_aux(branch, child_tree)
        else:
            if tree.terminal_cases:
                first = True
                for case in tree.terminal_cases:
                    if first:
                        print('\t\t\t\t|' * tree.depth + '\u2919\033[92m' + str(branch) +
                              '\033[0m\u291a\u27f6\tcase_\033[94m' + str(case) + '\033[0m')
                        first = False
                    else:
                        print('\t\t\t\t|' * tree.depth + '\u2919\033[92m' + ' ' * len(str(branch)) +
                              '\033[0m\u291a\u27f6\tcase_\033[94m' + str(case) + '\033[0m')
            else:
                print('\t\t\t\t|' * tree.depth + '\u2919\033[92m' + str(branch) + '\033[0m\u291a\u27f6case_\033[94m' +
                      'No cases yet' + '\033[0m')

    def retrieve(self, new_case):
        object = self.tree
        feat = object.attribute
        instances_ant = []
        while (object.is_leaf != True):
            distances = np.abs(new_case[feat] - self.prep.models[feat].cluster_centers_)
            featvals = np.argsort(distances[:, 0])
            instances_ant = object.xvic
            object = object.children[featvals[0]]
            feat = object.attribute
        if len(object.xvic) != 0:
            return object.xvic
        else:
            return instances_ant

    def update(self, retrieved_cases, sol_types):
        solution = []

        # We stop in a leaf that has depth smaller the number of attributes
        if retrieved_cases.shape[0] == 0:
            dict = {}
            dict['leaf'] = False
            dict['num_inst'] = retrieved_cases.shape[0]
            for i in range(self.num_class):
                if sol_types[i] == 'num_continuous':
                    dict['min'] = np.min(retrieved_cases[:, -self.num_class + i])
                    dict['mean'] = np.mean(retrieved_cases[:, -self.num_class + i])
                    dict['max'] = np.max(retrieved_cases[:, -self.num_class + i])
                    solution.append(dict)
                elif sol_types[i] == 'categorical':
                    unique_vals, counts = np.unique(retrieved_cases[:, -self.num_class + i], return_counts=True)
                    if len(counts) > 1:
                        dict['2Pop'] = unique_vals[1]
                    else:
                        dict['2Pop'] = 'None'
                    dict['1Pop'] = unique_vals[0]
                    solution.append(dict)
        # We stop in a leaf that has depth equal to the number of attributes
        else:
            for i in range(self.num_class):
                dict = {}
                dict['leaf'] = True
                dict['num_inst'] = retrieved_cases[:, -self.num_class + i].shape[0]
                if sol_types[i] == 'num_continuous':
                    dict['min'] = np.min(retrieved_cases[:, -self.num_class + i])
                    dict['mean'] = np.mean(retrieved_cases[:, -self.num_class + i])
                    dict['max'] = np.max(retrieved_cases[:, -self.num_class + i])
                    solution.append(dict)
                elif sol_types[i] == 'categorical':
                    unique_vals, counts = np.unique(retrieved_cases[:, -self.num_class + i], return_counts=True)
                    if len(counts) > 1:
                        dict['2Pop'] = unique_vals[1]
                    else:
                        dict['2Pop'] = 'None'
                    dict['1Pop'] = unique_vals[0]
                    solution.append(dict)
        return solution