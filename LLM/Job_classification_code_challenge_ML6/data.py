#!/usr/bin/env python3

"""Load text files and labels

This file contains the method that creates data and labels from a directory.
"""
import csv

import numpy as np

label_map = {
    0: 'Sales Jobs',
    1: 'Customer Services Jobs',
    2: 'IT Jobs',
    3: 'HR & Recruitment Jobs',
    4: 'Accounting & Finance Jobs',
}

label_map_inv = dict(map(reversed, label_map.items()))


def load_dataset(dataset_file):
    """Gets lists from data and label array from the csv file.

    Parameters:gsutil
        dataset_dir: A string specifying the directory of a dataset.

    Returns:
        data: A numpy array containing the texts
        labels: A numpy array containing labels corresponding to the text
    """

    data = []
    labels = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        data_reader = csv.reader(f, delimiter=",", quotechar='"')
        next(data_reader)
        for lbl, desc in data_reader:
            data.append(desc)
            labels.append(label_map_inv[lbl])

    return np.array(data), np.array(labels)
