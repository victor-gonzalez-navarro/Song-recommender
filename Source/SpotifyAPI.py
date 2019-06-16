import numpy as np


def normalize(bins):
    normalized = {}
    for k, v in bins.items():
        normalized[k] = np.array(v) / np.sum(v)
    return normalized


BINS = {
    'acousticnes':      [3070, 825, 615, 510, 410, 365, 330, 320, 285, 275, 275, 270, 275, 275, 220, 250, 310, 315, 325, 450],
    'danceability':     [15, 30, 75, 160, 185, 260, 370, 450, 600, 760, 950, 1035, 1040, 1060, 1020, 835, 640, 340, 150, 70],
    'energy':           [185, 155, 200, 225, 250, 305, 340, 395, 490, 550, 590, 695, 635, 750, 765, 800, 730, 720, 710, 540],
    'instrumentalness': [7650, 200, 95, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 120, 160, 235, 355, 430, 160],
    'liveness':         [255, 2460, 2910, 960, 645, 540, 615, 450, 180, 120, 90, 90, 90, 120, 90, 70, 75, 75, 90, 135],
    'loudness':         [15, 15, 15, 40, 40, 70, 70, 130, 180, 320, 510, 920, 1610, 2750, 2850, 585, 15],
    'speechiness':      [4700, 2625, 745, 430, 310, 265, 215, 170, 120, 60, 30, 30, 15, 15, 15, 15, 15, 30, 60, 190],
    'valence':          [325, 370, 440, 555, 510, 590, 600, 635, 555, 570, 590, 630, 530, 550, 490, 490, 475, 390, 360, 320],
    'tempo':            [10, 0, 0, 10, 30, 165, 800, 1130, 1390, 1170, 1730, 1330, 795, 535, 535, 265, 90, 50, 10],
}

NORM_BINS = normalize(BINS)

BINS_STEPS = {
    'acousticnes':      0.05,
    'danceability':     0.485,
    'energy':           0.05,
    'instrumentalness': 0.05,
    'liveness':         0.05,
    'loudness':         2.55882353,
    'speechiness':      0.0485,
    'valence':          0.05,
    'tempo':            11.74515235,
}

BINS_BEGIN = {
    'acousticnes':      0,
    'danceability':     0,
    'energy':           0,
    'instrumentalness': 0,
    'liveness':         0,
    'loudness':         -42.05714,
    'speechiness':      0,
    'valence':          0,
    'tempo':            0,
}