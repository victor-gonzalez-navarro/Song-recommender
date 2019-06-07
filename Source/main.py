import numpy as np
import pandas as pd
from Source.caseBase import CaseBase


def main():
    path = '../Data/'
    dataset = 'iris.csv'

    # For testing purposes, set seed of numpy (used by sklearn as well)
    np.random.seed(3)

    data = pd.read_csv(path + dataset)
    cb = CaseBase(data)

    # INPUT SOFTWARE--------------------------------------------
    # new_case = [6.7,3.8,1.3]    # stops before
    new_case = [4.7,3.2,1.3]      # stops end but same class
    # new_case = [6.9, 3.1, 5.1]  # stops enf and different class
    # ----------------------------------------------------------

    # R1: RETRIEVE ****
    retrieved_cases = cb.retrieve(new_case)

    # R2: REUSE ****
    solution = cb.update(retrieved_cases, ['num_continuous', 'categorical'])

    cb.print_tree()

    print('Process finished successfully')


main()
