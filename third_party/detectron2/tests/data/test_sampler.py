# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import math
import operator
import unittest
import torch
from torch.utils import data
from torch.utils.data.sampler import SequentialSampler

from detectron2.data.build import worker_init_reset_seed
from detectron2.data.common import DatasetFromList, ToIterableDataset
from detectron2.data.samplers import (
    GroupedBatchSampler,
    RepeatFactorTrainingSampler,
    TrainingSampler,
)
from detectron2.utils.env import seed_all_rng


class TestGroupedBatchSampler(unittest.TestCase):
    def test_missing_group_id(self):
        sampler = SequentialSampler(list(range(100)))
        group_ids = [1] * 100
        samples = GroupedBatchSampler(sampler, group_ids, 2)

        for mini_batch in samples:
            self.assertEqual(len(mini_batch), 2)

    def test_groups(self):
        sampler = SequentialSampler(list(range(100)))
        group_ids = [1, 0] * 50
        samples = GroupedBatchSampler(sampler, group_ids, 2)

        for mini_batch in samples:
            self.assertEqual((mini_batch[0] + mini_batch[1]) % 2, 0)


class TestSamplerDeterministic(unittest.TestCase):
    def test_to_iterable(self):
        sampler = TrainingSampler(100, seed=10)
        dataset = DatasetFromList(list(range(100)))
        dataset = ToIterableDataset(dataset, sampler)
        data_loader = data.DataLoader(dataset, num_workers=0, collate_fn=operator.itemgetter(0))

        output = list(itertools.islice(data_loader, 100))
        self.assertEqual(set(output), set(range(100)))

        data_loader = data.DataLoader(
            dataset,
            num_workers=2,
            collate_fn=operator.itemgetter(0),
            worker_init_fn=worker_init_reset_seed,
            # reset seed should not affect behavior of TrainingSampler
        )
        output = list(itertools.islice(data_loader, 100))
        # multiple workers should not lead to duplicate or different data
        self.assertEqual(set(output), set(range(100)))

    def test_training_sampler_seed(self):
        seed_all_rng(42)
        sampler = TrainingSampler(30)
        data = list(itertools.islice(sampler, 65))

        seed_all_rng(42)
        sampler = TrainingSampler(30)
        seed_all_rng(999)  # should be ineffective
        data2 = list(itertools.islice(sampler, 65))
        self.assertEqual(data, data2)


class TestRepeatFactorTrainingSampler(unittest.TestCase):
    def test_repeat_factors_from_category_frequency(self):
        repeat_thresh = 0.5

        dataset_dicts = [
            {"annotations": [{"category_id": 0}, {"category_id": 1}]},
            {"annotations": [{"category_id": 0}]},
            {"annotations": []},
        ]

        rep_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts, repeat_thresh
        )

        expected_rep_factors = torch.tensor([math.sqrt(3 / 2), 1.0, 1.0])
        self.assertTrue(torch.allclose(rep_factors, expected_rep_factors))
