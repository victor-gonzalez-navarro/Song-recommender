import numpy as np
from Source.node import Node
from Source.preprocess import Preprocess
from scipy.linalg import norm
from sklearn.preprocessing import MinMaxScaler
from Source.SpotifyAPI import BINS, NORM_BINS, BINS_BEGIN, BINS_STEPS


class CaseBase:

    def __init__(self, x, num_class, attr_categ, songs_info, n_clusters=5):
        # Preprocess of the data to be stored in the Case Base
        self.num_class = num_class
        self.prep = Preprocess(attr_categ)
        self.attr_names, self.attr_vals, self.attr_types, self.sol_cols = self.prep.extract_attr_info(x, self.num_class)
        self.songs_info = songs_info

        self.scaler = MinMaxScaler()
        self.scaler.fit(self.songs_info[self.sol_cols])
        self.x = x.values
        aux_x, self.attr_vals = self.prep.fit_predict(self.x[:, :-self.num_class], n_clusters=n_clusters)  # Auxiliary X with the
                                                                                                           # preprocessed data

        self.tree = None
        self.feat_selected = np.zeros((self.x.shape[1], 1))  # Depth at which each feature is selected
        self.max_depth = aux_x.shape[1]                      # Maximum depth corresponds to the number of attributes
                                                             # (+ leaf)

        self.make_tree(self.x, aux_x)

    def get_tree_depth(self, tree=None, depth=0):
        if tree is None:
            tree = self.tree

        if tree.is_leaf:
            return depth + 1

        depths = []
        for _, child_tree in tree.children.items():
            depths.append(self.get_tree_depth(child_tree, depth) + 1)

        return max(depths)

    def get_n_cases(self, tree=None):
        if tree is None:
            tree = self.tree

        if tree.is_leaf:
            return len(tree.case_ids)

        cases = 0
        for _, child_tree in tree.children.items():
            cases += self.get_n_cases(child_tree)

        return cases

    def make_tree(self, x, x_aux):
        self.tree = self.find_best_partition(x_aux, avail_attrs=list(range(x_aux.shape[1])), depth=0)
        self.tree.set_cases(list(range(x_aux.shape[0])))
        self.expand_tree(self.tree, x, x_aux, depth=1)

    def find_best_partition(self, x, avail_attrs, depth):
        best_score = -1
        best_aux_score = -1
        for feat_ix in avail_attrs:
            unique_vals, counts = np.unique(x[:, feat_ix], return_counts=True)
            score = len(unique_vals) / len(self.attr_vals[feat_ix]) - np.exp(len(self.attr_vals[feat_ix])-len(self.attr_vals[2]))

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
            return tree.case_ids
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
        if tree.is_leaf and tree.case_ids:
            first = True
            for case in tree.case_ids:
                if first:
                    print('\t\t\t\t|' * tree.depth + '\u2919\033[92m' + str(branch) +
                          '\033[0m\u291a\u27f6\tcase_\033[94m' + str(self.x[case, :]) + '\033[0m')
                    first = False
                else:
                    print('\t\t\t\t|' * tree.depth + '\u2919\033[92m' + ' ' * len(str(branch)) +
                          '\033[0m\u291a\u27f6\tcase_\033[94m' + str(self.x[case, :]) + '\033[0m')

        elif tree.is_leaf and not tree.case_ids:
            print('\t\t\t\t|' * tree.depth + '\u2919\033[92m' + str(branch) + '\033[0m\u291a\u27f6case_\033[94m' +
                  'No cases yet' + '\033[0m')

        else:
            print('\t\t\t\t|' * tree.depth + '\u2919\033[92m' + str(branch) + '\033[0m\u291a\u27f6attr_\033[1m' +
                  str(self.attr_names[tree.attribute]) + '\033[0m')
            for branch, child_tree in tree.children.items():
                self.print_tree_aux(branch, child_tree)

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
            distances, closecat, seclosecat = self.compute_distances(new_case[feat], self.prep.models[feat],
                                object.children, feat)
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
                object = object.children[closecat]
            feat = object.attribute
        if len(object.case_ids) > 0:
            return np.concatenate((self.x[object.case_ids,:], retrieved_cases), axis=0)
        else:
            return np.concatenate((self.x[instances_ant,:], retrieved_cases), axis=0)

    def update(self, retrieved_cases, sol_types, new_case):
        num_conc = 3; num_happ = 4; num_energ = 5; num_futHappy = 6; num_futEnerg = 7;
        diction = {}
        diction[num_conc] = ['not at all', 'some concentration', 'yes, my full concentration']
        diction[num_happ] = ['really unhappy', 'not so happy', 'neutral', 'happy', 'very happy']
        diction[num_energ] = ['very calm', 'calm', 'neutral', 'with some energy', 'with a lot of energy']
        diction[num_futHappy] = ['I get less happy', 'I keep the same', 'I get happier']
        diction[num_futEnerg] = ['I get more relaxed', 'I keep the same', 'I feel with more energy']
        solution = []
        dict = {}
        dict['num_inst'] = retrieved_cases.shape[0]
        for i in range(self.num_class):
            if sol_types[i] == 'num_continuous':
                dict['min'] = np.min(retrieved_cases[:, -self.num_class + i])
                dict['max'] = np.max(retrieved_cases[:, -self.num_class + i])
                if i == 6:  # Attribute number 6 is Tempo
                    # + Concentration --> - Tempo
                    unique_vals, counts = np.unique(retrieved_cases[:, num_conc], return_counts=True)
                    distance = diction[num_conc].index(new_case[num_conc]) - diction[num_conc].index(unique_vals[0])
                    mean_Tempo = np.mean(retrieved_cases[:, -self.num_class + i])
                    dict['mean'] = mean_Tempo - distance*0.1*(np.max(self.x[:,-1])-np.min(self.x[:,-1]))
                    # + Energy --> + Tempo
                    unique_vals, counts = np.unique(retrieved_cases[:, num_energ], return_counts=True)
                    distance = diction[num_energ].index(new_case[num_energ]) - diction[num_energ].index(unique_vals[0])
                    dict['mean'] = dict['mean'] + distance*0.1*(np.max(self.x[:,-1])-np.min(self.x[:,-1]))
                    # + Future Energy --> + Tempo
                    unique_vals, counts = np.unique(retrieved_cases[:, num_futEnerg], return_counts=True)
                    distance = diction[num_futEnerg].index(new_case[num_futEnerg]) - diction[num_futEnerg].index(unique_vals[0])
                    dict['mean'] = dict['mean'] + distance*0.1*(np.max(self.x[:,-1])-np.min(self.x[:,-1]))
                    # Update min and max the same as the mean
                    dict['min'] = dict['min'] + (mean_Tempo-dict['mean'])
                    dict['max'] = dict['max'] + (mean_Tempo-dict['mean'])
                elif i == 5:  # Attribute number 5 is Valence
                    # + Happininess --> + Valence
                    unique_vals, counts = np.unique(retrieved_cases[:, num_happ], return_counts=True)
                    distance = diction[num_happ].index(new_case[num_happ]) - diction[num_happ].index(unique_vals[0])
                    mean_Valence = np.mean(retrieved_cases[:, -self.num_class + i])
                    dict['mean'] = mean_Valence + distance*0.1*(np.max(self.x[:,-2])-np.min(self.x[:,-2]))
                    # + Future Happiness --> + Valence
                    unique_vals, counts = np.unique(retrieved_cases[:, num_futHappy], return_counts=True)
                    distance = diction[num_futHappy].index(new_case[num_futHappy]) - diction[num_futHappy].index(unique_vals[0])
                    dict['mean'] = dict['mean'] + distance*0.1*(np.max(self.x[:,-2])-np.min(self.x[:,-2]))
                    # Update min and max the same as the mean
                    dict['min'] = dict['min'] + (mean_Valence-dict['mean'])
                    dict['max'] = dict['max'] + (mean_Valence-dict['mean'])
            solution.append(dict)
        return solution

    def compute_distances(self, inst1, inst2, categories, feat):
        diction = dict()
        diction[2] = ['Instrumental-Classical', 'Vocal', 'Blues', 'Jazz', 'Rock', 'Hard Rock', 'Dance', 'Pop', 'Reggaeton', 'Reggae', 'Latin']
        diction[3] = ['not at all', 'some concentration', 'yes, my full concentration']
        diction[4] = ['really unhappy', 'not so happy', 'neutral', 'happy', 'very happy']
        diction[5] = ['very calm', 'calm', 'neutral', 'with some energy', 'with a lot of energy']
        diction[6] = ['I get less happy', 'I keep the same', 'I get happier']
        diction[7] = ['I get more relaxed', 'I keep the same', 'I feel with more energy']

        distances = []
        seclosecat = ''
        closecat = ''
        if self.attr_types[feat] == 'num_continuous':
            for i in range(inst2.cluster_centers_.shape[0]):
                distances.append(np.abs(inst1 - inst2.cluster_centers_[i,0]))
        elif self.attr_types[feat] == 'categorical':
            closecat = inst1
            categ = list(categories.keys())
            if feat == 1:
                for i in range(len(categ)):
                    if inst1 != categ[i]:
                        seclosecat = categ[i]
            elif feat == 2:
                intersectt = []
                genres_newcase = inst1.split(',')
                for i in range(len(categ)):
                    genres_possible = categ[i].split(',')
                    intersectt.append(len(set(genres_newcase).intersection(set(genres_possible))))
                sort_index = np.argsort(np.array(intersectt))
                closecat = categ[sort_index[-1]]
                seclosecat = categ[sort_index[-2]]
            else:
                idx = diction[feat].index(inst1)
                if idx == len(diction[feat])-1:
                    seclosecat = diction[feat][idx-1]
                else:
                    seclosecat = diction[feat][idx+1]

        return np.array(distances).reshape((len(distances), 1)), closecat, seclosecat

    def create_playlist(self, new_case, solution, max_length, norm_feats, debug):
        diction = dict()
        diction[2] = ['Instrumental-Classical', 'Vocal', 'Blues', 'Jazz', 'Rock', 'Hard Rock', 'Dance', 'Pop', 'Reggaeton', 'Reggae', 'Latin']
        diction[3] = ['not at all', 'some concentration', 'yes, my full concentration']
        diction[4] = ['really unhappy', 'not so happy', 'neutral', 'happy', 'very happy']
        diction[5] = ['very calm', 'calm', 'neutral', 'with some energy', 'with a lot of energy']
        diction[6] = ['I get less happy', 'I keep the same', 'I get happier']
        diction[7] = ['I get more relaxed', 'I keep the same', 'I feel with more energy']

        # Weighting the Previous attributes from 3 to 7
        feat_weights = np.array([[0.2, 0.1, 0.1, 0.3, 0.3]])

        # Each attribute vote on the value to start from for each song feature
        start = ['min', 'mean', 'max']
        feat_start = [[] for _ in range(self.num_class)]
        for feat in range(3, 8):
            ix = diction[feat].index(new_case[feat])
            ix = ix / len(diction[feat]) * 3
            inverse_ix = -ix % 2
            if feat in [3, 6, 7]:
                feat_start[0].append(inverse_ix)
                feat_start[1].append(inverse_ix)
                feat_start[2].append(inverse_ix)
                feat_start[3].append(ix)
                feat_start[4].append(ix)
                feat_start[5].append(inverse_ix)
                feat_start[6].append(inverse_ix)
            elif feat in [4]:
                feat_start[0].append(ix)
                feat_start[1].append(ix)
                feat_start[2].append(ix)
                feat_start[3].append(inverse_ix)
                feat_start[4].append(inverse_ix)
                feat_start[5].append(ix)
                feat_start[6].append(ix)
            elif feat in [5]:
                feat_start[0].append(ix)
                feat_start[1].append(ix)
                feat_start[2].append(ix)
                feat_start[3].append(inverse_ix)
                feat_start[4].append(inverse_ix)
                feat_start[5].append(inverse_ix)
                feat_start[6].append(ix)

        # Normalize song features with MinMaxScaling
        if norm_feats:
            song_data = self.scaler.transform(self.songs_info[self.sol_cols].values)
        else:
            song_data = self.songs_info[self.sol_cols].values

        # Weighted voting takes place
        feat_start = [np.array(votes).dot(feat_weights.T)[0] for votes in feat_start]
        init_point = []
        search_directions = []

        # The initial point will be selecting from the proportion of the range it lies in (0.86 min-mean)
        for point in range(self.num_class):
            low_ix = int(feat_start[point])
            min_val = solution[point][start[low_ix]]
            max_val = solution[point][start[low_ix+1]]
            init_point.append((max_val - min_val) * (feat_start[point] % 1) + min_val)

            # Searching Directions are selected based on the region it lies
            # (min-mean goes upward, mean-max goes downward)
            search_directions.append(-np.sign(feat_start[point]-1))

        if debug:
            print('The initial point is: ' + str(init_point))
            print('The search direction is: ' + str(search_directions))

        if norm_feats:
            init_point = self.scaler.transform([init_point])[0, :]
        best_score = np.inf

        # Extract the 3 styles this user prefers and the respective songs in the dataset
        styles = new_case[2].split(',')
        avail_songs = [i for i in range(self.songs_info.shape[0]) if self.songs_info['Genre'].values[i] in styles]

        # Check on the length of the available songs at the dataset
        if max_length > len(avail_songs):
            max_length = len(avail_songs)
            print('Setting the Maximum Length of the Playlist to ' + str(len(avail_songs)) + ' due to lack of more '
                  'songs of the user genre tastes')

        # Select the closest song to the initial point
        for song_id in avail_songs:
            score = norm(song_data[song_id] - init_point)
            if score < best_score:
                best_score = score
                selected_song = song_id

        if debug:
            print('The selected song is:\n' + str(self.songs_info.ix[selected_song]) + '\n')

        n_selected = 1
        avail_songs.remove(selected_song)
        playlist = [selected_song]

        # Continue the playlist until it is full
        while n_selected < max_length:
            # Try to find a song following the current search directions
            prev_song = selected_song
            selected_song = None
            best_score = np.inf
            for song_id in avail_songs:
                diff = song_data[song_id] - prev_song
                if np.array_equal(np.sign(diff), search_directions):
                    score = norm(diff)
                    if score < best_score:
                        best_score = score
                        selected_song = song_id

            # In case there is no such song, modify the less possible amount of search directions
            # and find a song following it
            new_search_directions = np.copy(search_directions)
            if selected_song is None:
                best_score_directions = np.inf
                for song_id in avail_songs:
                    diff = song_data[song_id] - prev_song
                    score_directions = sum(np.abs(np.sign(diff) - search_directions))
                    if score_directions < best_score_directions:
                        score = norm(diff)
                        if score < best_score:
                            best_score = score
                            selected_song = song_id
                            new_search_directions = np.sign(diff)

                if debug:
                    print()
                    print('The search direction has changed to : ' + str(new_search_directions))
                    print()

            if debug:
                print('The selected song is:\n' + str(self.songs_info.ix[selected_song]) + '\n')

            n_selected += 1
            avail_songs.remove(selected_song)
            playlist.append(selected_song)
            search_directions = np.copy(new_search_directions)

        # Replace the list of indexes with the list of info about the songs
        playlist = self.songs_info[['Song', 'Artist', 'Genre', 'Link_Spotify']].ix[playlist]
        playlist.reset_index(inplace=True, drop=True)

        return playlist

    def revise(self, solution, new_case, automatic=True):
        DANCEABILITY = 1
        ENERGY = 2
        LOUDNESS = 3
        ACOUSTICNESS = 4
        INSTRUMENTALNESS = 5
        VALENCE = 6
        TEMPO = 7

        if automatic:
            for i, parameter in enumerate(solution):
                print(parameter['mean'], i)

            print(len(solution))
            print(solution)
            print(self.get_n_cases())

        return