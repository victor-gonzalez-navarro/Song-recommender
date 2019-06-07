import numpy as np
from sklearn.cluster import KMeans


class Preprocess:

    def __init__(self):
        self.attr_names = None
        self.attr_values = {}
        self.attr_types = {}
        self.models = None

    def extract_attr_info(self, data, num_class):
        # Extract the column names of the attributes
        self.attr_names = data.columns.values.tolist()[:-num_class]
        #self.attr_names.remove('class')

        # For each column extract the possible values
        for attr_ix in range(len(self.attr_names)):
            attr_values = np.array(data[self.attr_names[attr_ix]].unique())

            if len(attr_values) > 20:
                self.attr_types[attr_ix] = "num_continuous"
                self.attr_values[attr_ix] = []
            else:
                self.attr_types[attr_ix] = "categorical"
                self.attr_values[attr_ix] = attr_values

        return self.attr_names, self.attr_values, self.attr_types

    def fit_predict(self, data, n_clusters):
        print_section = True
        self.models = [None] * data.shape[1]
        aux_data = np.copy(data)

        for i in range(len(self.attr_names)):
            if self.attr_types[i] == 'num_continuous':
                if print_section:
                    print('\n\033[1mDISCRETIZATION:\033[0m')
                    print_section = False

                km = KMeans(n_clusters=n_clusters)
                km.fit(aux_data[:, i].reshape(-1, 1))
                self.models[i] = km
                aux_data[:, i] = km.predict(aux_data[:, i].reshape(-1, 1))
                self.attr_values[i] = np.array(range(n_clusters))

                print('The cluster centers obtained for the variable \033[1m' + self.attr_names[i] + '\033[0m are:')
                for c in range(n_clusters):
                    print('Cluster ' + str(c) + ': ' + str(km.cluster_centers_[c]))

        return aux_data, self.attr_values

    def predict(self, instance):
        pred_instance = np.copy(instance)

        for i in range(len(self.attr_names)):
            if self.attr_types[i] == 'num_continuous':
                pred_instance[i] = self.models[i].predict(instance[i])

        return pred_instance
