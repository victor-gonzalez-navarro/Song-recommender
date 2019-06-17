import numpy as np
import pandas as pd
from Source.caseBase import CaseBase
from Source.auxiliaryFunct import menu
from Source.SpotifyAPI import normalize_vars
from sklearn.metrics.pairwise import euclidean_distances


DATA_PATH = '../Data/'
TRAIN_PATH = DATA_PATH + 'songs_train.csv'
TEST_PATH = DATA_PATH + 'songs_test.csv'
SONGS_PATH = DATA_PATH + 'all_songs.csv'
PLAYLIST_LENGTH = 2
NUM_ATTRIB_SOLUTION = 7
TYPE_ATTRIB_SOLUTION = ['num_continuous'] * NUM_ATTRIB_SOLUTION
ATTR_CATEG = 2


def euclidean(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def run(cb, new_case, retain_when='good', retain_threshold=0.2, evaluation_auto=True, print_playlist=False, plot_retain=False):
    # [R1]: RETRIEVE ****
    retrieved_cases = cb.retrieve_v2(new_case)

    # [R2]: REUSE ****
    solution = cb.update(retrieved_cases, TYPE_ATTRIB_SOLUTION, new_case)

    # [EXTRA]: PLAYLIST CREATION FROM solution ****
    playlist = cb.create_playlist(new_case, solution, max_length=PLAYLIST_LENGTH, norm_feats=True, debug=False)

    # [R3]: REVISE ****
    if evaluation_auto:
        goodness = cb.revise(solution, new_case, plot=plot_retain)
    else:
        print('\nThe generated playlist is:\n{:s}\n'.format(playlist.to_string()))
        goodness = 1.0 if input('Do you like this playlist? [yes|no]: ') == 'yes' else 0.0

    retain = goodness > retain_threshold
    # 1) If retain only for bad cases, change the boolean value
    # 2) Default retain good cases
    # 3) If retain is None, do not retain anything
    # 4) Note that retaining the bad cases is normally not a good option because you will be storing
    #    erroneous predictions as part of the database !
    if retain_when is not None and retain_when == 'bad':
        retain = not retain
    if retain_when == 'all':
        retain = True
    if print_playlist and evaluation_auto:
        print('\nThe generated playlist is:\n{:s}\n'.format(playlist.to_string()))

    # [R4]: RETAIN ****
    retained = False
    if retain_when is not None and retain:
        retained = True
        cb.retain(solution, new_case)

    return playlist, retained


def find_song_properties(data_songs, song, artist, genre, variables):
    for _, row in data_songs.iterrows():
        if row['Song'] == song and row['Artist'] == artist and row['Genre'] == genre:
            data = []
            for variable in variables:
                data.append(row[variable])
            return data
    return None


def run_tests(data_test, data_train, data_songs, retain_when='good'):
    variables = list(data_test)[8:]
    euclidean_max = euclidean([0] * 7, [1] * 7)

    # Create CB
    cb = CaseBase(data_train, NUM_ATTRIB_SOLUTION, ATTR_CATEG, data_songs)

    # Distances between all songs to get threshold
    # distances = []
    # for row1 in range(len(data_test.index)):
    #    for row2 in range(row1 + 1, len(data_test.index)):
    #        song1 = normalize_vars(variables, data_test.loc[row1, :].values[8:])
    #        song2 = normalize_vars(variables, data_test.loc[row2, :].values[8:])
    #        distances.append(euclidean(song1, song2))

    # similarities = 1 - np.array(distances) / euclidean_max
    # threshold = np.mean(similarities)

    # Distances between returned songs and target
    results = []  # True or False for a prediction with similarity > threshold
    count_retained = 0
    for idx, row in data_test.iterrows():
        data = row.values[:8]
        result = row.values[8:]
        result = normalize_vars(variables, result)
        playlist, retained = run(cb, data, retain_when=retain_when)

        if retained:
            count_retained += 1

        # DEBUG 1
        # if idx == 70:
        #    print(playlist.to_string())

        distances = []
        for _, row in playlist.iterrows():
            output = find_song_properties(data_songs, row['Song'], row['Artist'], row['Genre'], variables)
            output = normalize_vars(variables, output)
            distances.append(euclidean(output, result))

        similarities = 1 - np.array(distances) / euclidean_max
        results.append(np.mean(similarities))
    return np.mean(results), count_retained


def main(mode='ask', debug=False, evaluation_auto=True):
    pd.options.display.max_colwidth = 100
    example_case = [200, 'Male', "Instrumental-Classical,Dance,Blues", 'some concentration', 'really unhappy', 'calm', 'I keep the same', 'I keep the same']

    np.random.seed(3)

    # Read files
    data_test = pd.read_csv(TEST_PATH)
    data_train = pd.read_csv(TRAIN_PATH)
    data_songs = pd.read_csv(SONGS_PATH)
    data_train = data_train.drop(data_train.columns[[8, 9, 10, 13, 15, 16, 19, 22, 23, 24, 25, 26]], axis=1)
    data_test = data_test.drop(data_test.columns[[8, 9, 10, 13, 15, 16, 19, 22, 23, 24, 25, 26]], axis=1)

    if mode == 'example':
        cb = CaseBase(data_train, NUM_ATTRIB_SOLUTION, ATTR_CATEG, data_songs)
        playlist, _ = run(cb, example_case, print_playlist=True, evaluation_auto=evaluation_auto)

    elif mode == 'ask':
        cb = CaseBase(data_train, NUM_ATTRIB_SOLUTION, ATTR_CATEG, data_songs)
        playlist, _ = run(cb, menu(), print_playlist=True, evaluation_auto=evaluation_auto)
        print('Process finished successfully')

    elif mode == 'tests':
        for retain in [None, 'good', 'bad', 'all']:
            results, count_retained = run_tests(data_test, data_train, data_songs, retain_when=retain)
            print('Tests performance: {:.4f} Avg Sim (retain: {:s}, retained: {:d})\n'.format(results, str(retain), count_retained))


if __name__ == '__main__':
    main(mode='tests', evaluation_auto=True)
