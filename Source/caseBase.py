import numpy as np
from Source.node import Node
from Source.preprocess import Preprocess


class CaseBase:

    def __init__(self, x, num_class, attr_categ):
        # Preprocess of the data to be stored in the Case Base
        n_clusters = 5
        self.num_class = num_class
        self.prep = Preprocess(attr_categ)
        self.attr_names, self.attr_vals, self.attr_types = self.prep.extract_attr_info(x, self.num_class)
        self.x = x.values
        aux_x, self.attr_vals = self.prep.fit_predict(self.x[:, :-self.num_class], n_clusters=n_clusters)  # Auxiliary X with the
                                                                                                           # preprocessed data

        self.tree = None
        self.feat_selected = np.zeros((self.x.shape[1], 1))  # Depth at which each feature is selected
        self.max_depth = aux_x.shape[1]                      # Maximum depth corresponds to the number of attributes
                                                             # (+ leaf)

        self.make_tree(self.x, aux_x)

    def make_tree(self, x, x_aux):
        self.tree = self.find_best_partition(x_aux, avail_attrs=list(range(x_aux.shape[1])), depth=0)
        self.tree.set_cases(list(range(x_aux.shape[0])))
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
            prev_val = np.copy(val)
            if len(val) == 0:
                # If the split left this branch empty, set the terminal boolean to True without adding any case
                tree.children[key] = Node(is_leaf=True, depth=depth)
            elif depth == self.max_depth:
                # If the maximum depth has been reached, add the terminal cases in the leaf node
                terminal_cases = np.array(tree.case_ids)[prev_val].tolist()  # x[val, :].tolist()
                tree.children[key] = Node(case_ids=terminal_cases, is_leaf=True, depth=depth)
            else:
                # Otherwise, find the best partition for this leaf and expand the subtree
                tree.children[key] = self.find_best_partition(x_aux[val, :], tree.avail_attrs, depth)
                tree.children[key].set_cases(np.array(tree.case_ids)[prev_val].tolist())
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
            if tree.case_ids:
                first = True
                for case in tree.case_ids:
                    if first:
                        print('\t\t\t\t|' * tree.depth + '\u2919\033[92m' + str(branch) +
                              '\033[0m\u291a\u27f6\tcase_\033[94m' + str(self.x[case, :]) + '\033[0m')
                        first = False
                    else:
                        print('\t\t\t\t|' * tree.depth + '\u2919\033[92m' + ' ' * len(str(branch)) +
                              '\033[0m\u291a\u27f6\tcase_\033[94m' + str(self.x[case, :]) + '\033[0m')
            else:
                print('\t\t\t\t|' * tree.depth + '\u2919\033[92m' + str(branch) + '\033[0m\u291a\u27f6case_\033[94m' +
                      'No cases yet' + '\033[0m')

    # It follows the best path, and in case it arrives to a point with no more instance it returns the instances
    # included by the parent
    def retrieve(self, new_case):
        object = self.tree
        feat = object.attribute
        instances_ant = []
        while (object.is_leaf != True) and (len(object.case_ids) > 0):
            distances = self.compute_distances(new_case[feat], self.prep.models[feat].cluster_centers_, feat)
            featvals = np.argsort(distances[:, 0])
            instances_ant = object.case_ids
            object = object.children[featvals[0]]
            feat = object.attribute
        if len(object.case_ids) > 0:
            return self.x[object.case_ids,:]
        else:
            return self.x[instances_ant,:]

    # PARTIAL MATCHING
    def retrieve_v2(self, new_case):
        retrieved_cases = np.empty((0, len(new_case)+self.num_class))
        object = self.tree
        feat = object.attribute
        instances_ant = []
        while (object.is_leaf != True) and (len(object.case_ids) > 0):
            distances, seclosecat = self.compute_distances(new_case[feat], self.prep.models[feat], object.children, feat)
            # Retrieve instances second best and then following the best path
            if self.attr_types[feat] == 'num_continuous':
                featvals = np.argsort(distances[:, 0])
                retr = object.children[featvals[1]].retrieve_best(new_case, self.prep.models, self.x, self.attr_types)
            elif self.attr_types[feat] == 'categorical':
                retr = object.children[seclosecat].retrieve_best(new_case, self.prep.models, self.x, self.attr_types)
            retrieved_cases = np.append(retrieved_cases, retr, axis=0)
            instances_ant = object.case_ids
            if self.attr_types[feat] == 'num_continuous':
                object = object.children[featvals[0]]
            elif self.attr_types[feat] == 'categorical':
                object = object.children[new_case[feat]]
            feat = object.attribute
        if len(object.case_ids) > 0:
            return np.concatenate((self.x[object.case_ids,:], retrieved_cases), axis=0)
        else:
            return np.concatenate((self.x[instances_ant,:], retrieved_cases), axis=0)

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

    def compute_distances(self, inst1, inst2, categories, feat):
        diction = dict()
        diction[3] = ['not at all', 'some concentration', 'yes, my full concentration']
        diction[4] = ['really unhappy', 'not so happy', 'neutral', 'happy', 'very happy']
        diction[5] = ['very clam', 'calm', 'neutral', 'with some energy', 'with a lot of energy']
        diction[6] = ['I get less happy', 'I keep the same', 'I get happier']
        diction[7] = ['I get more relaxed', 'I keep the same', 'I feel with more energy']

        distances = []
        seclosecat = ''
        if self.attr_types[feat] == 'num_continuous':
            for i in range(inst2.cluster_centers_.shape[0]):
                distances.append(np.abs(inst1 - inst2.cluster_centers_[i,0]))
        elif self.attr_types[feat] == 'categorical':
            categ = list(categories.keys())
            if feat == 2 or feat == 1:
                print('Change this [Victor]')
                for i in range(len(categ)):
                    if inst1 != categ[i]:
                        seclosecat = categ[i]
            else:
                idx = diction[feat].index(inst1)
                if idx == len(diction[feat])-1:
                    seclosecat = diction[feat][idx-1]
                else:
                    seclosecat = diction[feat][idx+1]

        return np.array(distances).reshape((len(distances),1)), seclosecat
