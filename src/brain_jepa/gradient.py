from importlib.resources import files, as_file

import torch
import pandas as pd

import brain_jepa


def load_gradient():
    source = files(brain_jepa) / "gradient_mapping_400.csv"
    with as_file(source) as p:
        df = pd.read_csv(p, header=None)
    gradient = torch.tensor(df.values, dtype=torch.float32)
    return gradient.unsqueeze(0)
