import torch
from torch.utils.data import default_collate

class SimCLRTransform:

    def __init__(self, base_transform):

        self.base_transform = base_transform

    def __call__(self, raw_sample):

        view_1 = self.base_transform(raw_sample)
        view_2 = self.base_transform(raw_sample)
        return view_1, view_2

def simclr_collate(batch):

    views_1 = [item[0] for item in batch]
    views_2 = [item[1] for item in batch]

    collated_view_1 = default_collate(views_1)
    collated_view_2 = default_collate(views_2)

    return collated_view_1, collated_view_2
