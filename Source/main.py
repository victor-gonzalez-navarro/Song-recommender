import numpy as np
import pandas as pd
from Source.caseBase import CaseBase
from Source.auxiliaryFunct import menu


def main():
    # HYPERPARAMETERS---------------------------------------------------------------------------------------------------
    new_case = [200, 'Male', "Instrumental-Classical,Dance,Blues", 'some concentration', 'really unhappy', 'calm', 'I keep the same', 'I keep the same']
    num_attrib_solution = 7
    type_attrib_solution = ['num_continuous']*num_attrib_solution
    attr_categ = 2  # force attribute 2 to be categorical
    playlist_length = 5
    # ------------------------------------------------------------------------------------------------------------------

    # User introduces new input
    # new_case = menu(new_case)

    path = '../Data/'
    dataset = 'songs.csv'
    songs_dataset = 'songs_DB.csv'

    # For testing purposes, set seed of numpy (used by sklearn as well)
    np.random.seed(3)

    data = pd.read_csv(path + dataset)
    songs_info = pd.read_csv(path + songs_dataset)  # , delimiter='|')
    dat = data.drop(data.columns[[8,9,10,13,15,16,19,22,23,24,25,26]], axis=1)
    cb = CaseBase(dat, num_attrib_solution, attr_categ, songs_info)

    # [R1]: RETRIEVE ****
    retrieved_cases = cb.retrieve_v2(new_case)
    print(retrieved_cases)

    # [R2]: REUSE ****
    solution = cb.update(retrieved_cases, type_attrib_solution, new_case)

    # [R3]: REVISE ****
    positive = cb.revise(solution, new_case)

    # [R4]: RETAIN ****
    if positive:
        cb.retain(solution, new_case)

    # [EXTRA]: PLAYLIST CREATION FROM solution ****
    playlist = cb.create_playlist(new_case, solution, max_length=playlist_length, norm_feats=True, debug=False)

    pd.options.display.max_colwidth = 100
    print()
    print('The generated playlist is:\n')
    print(playlist.to_string())
    print()



    # cb.print_tree()

    print('Process finished successfully')


if __name__ == '__main__':
    main()
