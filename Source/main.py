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

    cb.print_tree()


main()
