import numpy as np
import pandas as pd
from Source.caseBase import CaseBase
from Source.auxiliaryFunct import menu


def main():
    # HYPERPARAMETERS---------------------------------------------------------------------------------------------------
    new_case = [37, 'Male', "Blues,Instrumental-Classical,Rock", 'some concentration', 'happy', 'calm','I keep the same', 'I keep the same']
    num_attrib_solution = 7
    type_attrib_solution = ['num_continuous']*num_attrib_solution
    attr_categ = 2  # force attribute 2 to be categorical
    # ------------------------------------------------------------------------------------------------------------------

    # User introduces new input
    new_case = menu(new_case)

    path = '../Data/'
    dataset = 'songs.csv'

    # For testing purposes, set seed of numpy (used by sklearn as well)
    np.random.seed(3)

    data = pd.read_csv(path + dataset)
    songs_ID = data[['Song','Artist']]
    dat = data.drop(data.columns[[8,9,10,13,15,16,19,22,23,24,25,26]], axis = 1)
    cb = CaseBase(dat, num_attrib_solution, attr_categ)

    # [R1]: RETRIEVE ****
    retrieved_cases = cb.retrieve_v2(new_case)

    # [R2]: REUSE ****
    solution = cb.update(retrieved_cases, type_attrib_solution, new_case)

    # [R3]: REVISE ****

    # [R4]: RETAIN ****

    # [EXTRA]: PLAYLIST CREATION FROM solution ****

    # cb.print_tree()

    print('Process finished successfully')


main()
