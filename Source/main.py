import numpy as np
import pandas as pd
from Source.case_base import CaseBase
from Source.utils.aux_functions import menu, euclidean, find_song_properties
from Source.spotify_api import normalize_vars


# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS
# ----------------------------------------------------------------------------------------------------------------------

# Paths
DATA_PATH  = '../Data/'
TRAIN_PATH = DATA_PATH + 'songs_train.csv'
TEST_PATH  = DATA_PATH + 'songs_test.csv'
SONGS_PATH = DATA_PATH + 'all_songs.csv'

# CBR
PLAYLIST_LENGTH  = 5    # x >= 1
N_SIM            = 2    # x >= 1
RETAIN_THRESHOLD = 0.6  # 0 < x < 1
EVALUATION_AUTO  = True
MODE             = 'example'   # ['ask', 'tests', 'example']
RETAIN_WHEN      = 'good'  # [None, 'good', 'bad' (Do not use), 'all' (Do not use)]

# Example instance
EXAMPLE_CASE = [37, 'Male', "Instrumental-Classical,Dance,Blues", 'some concentration', 'really unhappy',
                'calm', 'I keep the same', 'I keep the same']

# Do not modify
NUM_ATTRIB_SOLUTION = 7
TYPE_ATTRIB_SOLUTION = ['num_continuous'] * NUM_ATTRIB_SOLUTION
ATTR_CATEG = 2


# ----------------------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------------------
def main():
    # For debugging purposes
    np.random.seed(3)

    # Read files
    data_test = pd.read_csv(TEST_PATH)
    data_train = pd.read_csv(TRAIN_PATH)
    data_songs = pd.read_csv(SONGS_PATH)
    data_train = data_train.drop(data_train.columns[[8, 9, 10, 13, 15, 16, 19, 22, 23, 24, 25, 26]], axis=1)
    data_test = data_test.drop(data_test.columns[[8, 9, 10, 13, 15, 16, 19, 22, 23, 24, 25, 26]], axis=1)

    # Run the example
    if MODE == 'example':
        cb = CaseBase(data_train, NUM_ATTRIB_SOLUTION, ATTR_CATEG, data_songs)
        playlist, _, goodness = run(cb, EXAMPLE_CASE, print_playlist=True)
        print('Solution quality [0,1]:', goodness)

    # Run the CBR in a normal case
    elif MODE == 'ask':
        cb = CaseBase(data_train, NUM_ATTRIB_SOLUTION, ATTR_CATEG, data_songs)
        playlist, _, goodness = run(cb, menu(), print_playlist=True)
        print('Solution quality [0,1]:', goodness)
        print('Process finished successfully')

    # Run tests for the CBR
    elif MODE == 'tests':
        for retain in [None, 'good', 'bad', 'all']:
            results, count_retained = run_tests(data_test, data_train, data_songs, retain_when=retain)
            print('Tests performance: {:.4f} Avg Sim (retain: {:s}, retained: {:d})'.format(
                results, str(retain), count_retained))


# ----------------------------------------------------------------------------------------------------------------------
# RUN AN INSTANCE OF THE CBR IN A CASE
# ----------------------------------------------------------------------------------------------------------------------
def run(cb, new_case, retain_when=RETAIN_WHEN, evaluation_auto=EVALUATION_AUTO, print_playlist=False, plot_retain=False):
    # [R1]: RETRIEVE ****
    retrieved_cases = cb.retrieve_v2(new_case)
    # [R2]: REUSE ****
    solution = cb.update(retrieved_cases, TYPE_ATTRIB_SOLUTION, new_case)
    # [EXTRA]: PLAYLIST CREATION FROM solution ****
    playlist = cb.create_playlist(new_case, solution, max_length=PLAYLIST_LENGTH, norm_feats=True, debug=False)
    if print_playlist:
        print('\nThe generated playlist is:\n{:s}\n'.format(playlist.to_string()))

    # [R3]: REVISE ****
    goodness = cb.revise(solution, new_case, evaluation_auto, plot=plot_retain)

    # Check if the solution should be retained based in the goodness
    if retain_when is None:
        retain = False
    elif retain_when == 'good':
        retain = goodness > RETAIN_THRESHOLD
    elif retain_when == 'bad':
        retain = (1 - goodness) > RETAIN_THRESHOLD
    elif retain_when == 'all':
        retain = True
    else:
        retain = False

    # [R4]: RETAIN ****
    if retain:
        cb.retain(solution, new_case)

    return playlist, retain, goodness


# ----------------------------------------------------------------------------------------------------------------------
# TESTS
# ----------------------------------------------------------------------------------------------------------------------
def run_tests(data_test, data_train, data_songs, retain_when='good'):
    variables = list(data_test)[8:]
    euclidean_max = euclidean([0] * 7, [1] * 7)
    # print(distances_between_all_songs(data_test, euclidean_max))

    # Create CB
    cb = CaseBase(data_train, NUM_ATTRIB_SOLUTION, ATTR_CATEG, data_songs)

    # Distances between returned songs and target
    mean_sims = 0
    count_retained = 0
    for idx, row in data_test.iterrows():
        # Run the CBR
        data = row.values[:8]
        result = row.values[8:]
        result = normalize_vars(variables, result)
        playlist, retained, _ = run(cb, data, evaluation_auto=True, retain_when=retain_when)

        # Count if the solution was retained
        if retained:
            count_retained += 1

        # Measure distance between ground truth song and playlist songs
        distances = []
        for _, song in playlist.iterrows():
            output = find_song_properties(data_songs, song['Song'], song['Artist'], song['Genre'], variables)
            output = normalize_vars(variables, output)
            distances.append(euclidean(output, result))

        # Get mean similarity of n bests
        similarities = 1 - np.array(distances) / euclidean_max
        similarities = np.sort(similarities)[-N_SIM:]
        mean_sims += np.mean(similarities) / len(data_test)

    return mean_sims, count_retained


# ----------------------------------------------------------------------------------------------------------------------
# RUN MAIN
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
