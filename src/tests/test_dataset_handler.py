from torch.utils import data
from src import GOD_CLIENT_NAME
import unittest
import torch

from src import federated_dataset

class DatasetsTestCase(unittest.TestCase):
    def test_deterministic_load_data_synthetic(self) -> None:
        trainset_1 = federated_dataset.load_data(
            dataset_name="synthetic_0_0_123",
            client_names=["f_00000"],
            train_test_split=0.8,
            type="train",
            min_no_samples=1,
            is_embedded=False)
        trainset_2 = federated_dataset.load_data(
            dataset_name="synthetic_0_0_123",
            client_names=["f_00000"],
            train_test_split=0.8,
            type="train",
            min_no_samples=1,
            is_embedded=False)
        assert len(trainset_1) == len(trainset_2)
        dataloader_1 = torch.utils.data.DataLoader(trainset_1, batch_size=1, shuffle=False)
        dataloader_2 = torch.utils.data.DataLoader(trainset_2, batch_size=1, shuffle=False)
        for sample_1, sample_2 in zip(dataloader_1, dataloader_2):
            assert torch.all(torch.eq(sample_1[0], sample_2[0]))
            assert torch.eq(sample_1[1], sample_2[1])

    def test_combined_client_load_data_matches_single_client_load_data_synthetic(self) -> None:
        trainset_all = federated_dataset.load_data(
            dataset_name="synthetic_0_0_123",
            client_names=[GOD_CLIENT_NAME],
            train_test_split=0.8,
            type="train",
            min_no_samples=1,
            is_embedded=False)
        client_names = sorted(federated_dataset.FederatedDataset.clients.keys())
        trainsets = []
        for client_name in client_names:
            trainset = federated_dataset.load_data(
                dataset_name="synthetic_0_0_123",
                client_names=[client_name],
                train_test_split=0.8,
                type="train",
                min_no_samples=1,
                is_embedded=False)
            trainsets.append(trainset)
        trainset_individual = torch.utils.data.ConcatDataset(trainsets)
        samples_1 = list(trainset_all)
        samples_2 = list(trainset_individual)
        assert len(samples_1) == len(samples_2)
        for sample_1 in samples_1:
            in_samples_2 = False
            for sample_2 in samples_2:
                if torch.all(torch.eq(sample_1[0], sample_2[0])) and \
                    torch.eq(sample_1[1], sample_2[1]):
                    in_samples_2 = True
            assert in_samples_2

    def test_train_and_test_set_do_not_overlap(self) -> None:
        trainset_all = federated_dataset.load_data(
            dataset_name="synthetic_0_0_123",
            client_names=[GOD_CLIENT_NAME],
            train_test_split=0.8,
            type="train",
            min_no_samples=1,
            is_embedded=False)
        testset_all = federated_dataset.load_data(
            dataset_name="synthetic_0_0_123",
            client_names=[GOD_CLIENT_NAME],
            train_test_split=0.8,
            type="test",
            min_no_samples=1,
            is_embedded=False)
        assert len(trainset_all) + len(testset_all) == 119
        samples_train = list(trainset_all)
        samples_test = list(testset_all)
        for (train_sample, label) in samples_train:
            for (test_sample, label) in samples_test:
                assert not torch.all(train_sample == test_sample)

    def test_train_and_test_set_ratio_correct(self) -> None:
        trainset_all = federated_dataset.load_data(
            dataset_name="synthetic_0_0_123",
            client_names=[GOD_CLIENT_NAME],
            train_test_split=0.8,
            type="train",
            min_no_samples=1,
            is_embedded=False)
        testset_all = federated_dataset.load_data(
            dataset_name="synthetic_0_0_123",
            client_names=[GOD_CLIENT_NAME],
            train_test_split=0.8,
            type="test",
            min_no_samples=1,
            is_embedded=False)
        assert abs(len(trainset_all)/len(testset_all) - 4) < 0.05


if __name__ == "__main__":
    unittest.main(verbosity=2)
