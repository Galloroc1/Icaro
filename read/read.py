import os
import pandas as pd
import numpy as np
from config import Config


def read_csv(path):
    reader = {
        'single': read_csv_by_pandas,
        'multiprocess': read_csv_by_pandas,
        'cluster': read_csv_by_line,
    }
    data = reader[Config.compute_type](path)
    return data


def read_csv_by_pandas(path):
    return pd.read_csv(path)


def read_csv_by_line(path):
    file = open(path, 'r', encoding='utf-8')
    for line in file:
        yield line.strip()


def make_test_csv():
    data = np.random.random_sample((3, 3))
    data = pd.DataFrame(data)
    data.to_csv("test.csv", index=False)

