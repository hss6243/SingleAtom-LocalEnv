import torch
import math

from torch.utils.data.sampler import Sampler


class ImbalancedDatasetSampler(Sampler):
    """
    Source: https://github.com/ufoym/imbalanced-dataset-sampler/
            blob/master/torchsampler/imbalanced.py
    Sampling class to make sure positive and negative labels
    are represented equally during training.
    Attributes:
        data_length (int): length of dataset
        weights (torch.Tensor): weights of each index in the
            dataset depending.
    """

    def __init__(self, target_name, props):
        """
        Args:
            target_name (str): name of the property being classified
            props (dict): property dictionary
        """

        data_length = len(props[target_name])

        negative_idx = [
            i
            for i, target in enumerate(props[target_name])
            if round(target.item()) == 0
        ]
        positive_idx = [i for i in range(data_length) if i not in negative_idx]

        num_neg = len(negative_idx)
        num_pos = len(positive_idx)

        if num_neg == 0:
            num_neg = 1
        if num_pos == 0:
            num_pos = 1

        negative_weight = num_neg
        positive_weight = num_pos

        self.data_length = data_length
        self.weights = torch.zeros(data_length)
        self.weights[negative_idx] = 1 / negative_weight
        self.weights[positive_idx] = 1 / positive_weight

    def __iter__(self):

        return (
            i
            for i in torch.multinomial(self.weights, self.data_length, replacement=True)
        )

    def __len__(self):
        return self.data_length


class MinDFTBatchSampler(Sampler):
    """Batch sampler that enforces a minimum number of DFT-labeled samples.

    A sample is considered DFT-labeled when its `target` tensor contains at
    least one finite value. This is useful for multifidelity training where
    UMA labels can dominate and DFT supervision may vanish in random batches.
    """

    def __init__(
        self,
        targets,
        batch_size,
        min_dft_per_batch,
        drop_last=False,
        shuffle=True,
        dft_replacement=True,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if min_dft_per_batch < 0:
            raise ValueError("min_dft_per_batch must be >= 0")
        if min_dft_per_batch > batch_size:
            raise ValueError("min_dft_per_batch cannot exceed batch_size")

        self.batch_size = int(batch_size)
        self.min_dft_per_batch = int(min_dft_per_batch)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.dft_replacement = bool(dft_replacement)

        self.dft_indices = []
        self.other_indices = []
        for idx, targ in enumerate(targets):
            if self._has_any_finite(targ):
                self.dft_indices.append(idx)
            else:
                self.other_indices.append(idx)

        if self.min_dft_per_batch > 0 and len(self.dft_indices) == 0:
            raise ValueError(
                "No DFT-labeled samples were found, but min_dft_per_batch > 0"
            )

        self.num_samples = len(targets)

    @staticmethod
    def _has_any_finite(targ):
        if not isinstance(targ, torch.Tensor):
            targ = torch.as_tensor(targ)
        return torch.isfinite(targ).any().item()

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        return math.ceil(self.num_samples / self.batch_size)

    def _prepare_pool(self, indices):
        if self.shuffle:
            perm = torch.randperm(len(indices)).tolist()
            return [indices[i] for i in perm]
        return list(indices)

    def _draw_from_pool(self, pool, n):
        take_n = min(n, len(pool))
        taken = pool[:take_n]
        del pool[:take_n]
        return taken, n - take_n

    def _draw_with_replacement(self, base_indices, n):
        if n <= 0:
            return []
        rand_idx = torch.randint(low=0, high=len(base_indices), size=(n,)).tolist()
        return [base_indices[i] for i in rand_idx]

    def __iter__(self):
        dft_pool = self._prepare_pool(self.dft_indices)
        other_pool = self._prepare_pool(self.other_indices)
        all_indices = self.dft_indices + self.other_indices

        num_batches = len(self)
        for batch_id in range(num_batches):
            if self.drop_last:
                current_bs = self.batch_size
            else:
                remaining = self.num_samples - batch_id * self.batch_size
                current_bs = min(self.batch_size, remaining)

            need_dft = min(self.min_dft_per_batch, current_bs)
            batch = []

            taken_dft, missing_dft = self._draw_from_pool(dft_pool, need_dft)
            batch.extend(taken_dft)
            if missing_dft > 0:
                if not self.dft_replacement:
                    raise RuntimeError(
                        "Insufficient DFT samples to satisfy min_dft_per_batch "
                        "without replacement"
                    )
                batch.extend(self._draw_with_replacement(self.dft_indices, missing_dft))

            remaining_slots = current_bs - len(batch)
            if remaining_slots > 0:
                taken_other, missing_other = self._draw_from_pool(other_pool, remaining_slots)
                batch.extend(taken_other)

                if missing_other > 0:
                    taken_dft_extra, missing_other = self._draw_from_pool(
                        dft_pool, missing_other
                    )
                    batch.extend(taken_dft_extra)

                if missing_other > 0:
                    batch.extend(self._draw_with_replacement(all_indices, missing_other))

            if self.shuffle:
                perm = torch.randperm(len(batch)).tolist()
                batch = [batch[i] for i in perm]

            yield batch


class ImbalancedDFTdLabelBatchSampler(Sampler):
    """Batch sampler for multifidelity per-site targets with label-aware balancing.

    This sampler uses the number of finite target labels per sample to define
    sampling probabilities, while optionally enforcing a minimum number of
    DFT-labeled samples in each batch.

    A sample is considered DFT-labeled when its target contains at least one
    finite entry.
    """

    def __init__(
        self,
        targets,
        batch_size,
        drop_last=False,
        dft_group_weight=0.5,
        label_count_power=1.0,
        replacement=True,
        min_dft_per_batch=0,
        dft_replacement=True,
        shuffle_within_batch=True,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not (0.0 <= dft_group_weight <= 1.0):
            raise ValueError("dft_group_weight must be in [0, 1]")
        if label_count_power <= 0:
            raise ValueError("label_count_power must be > 0")
        if min_dft_per_batch < 0:
            raise ValueError("min_dft_per_batch must be >= 0")
        if min_dft_per_batch > batch_size:
            raise ValueError("min_dft_per_batch cannot exceed batch_size")

        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.dft_group_weight = float(dft_group_weight)
        self.label_count_power = float(label_count_power)
        self.replacement = bool(replacement)
        self.min_dft_per_batch = int(min_dft_per_batch)
        self.dft_replacement = bool(dft_replacement)
        self.shuffle_within_batch = bool(shuffle_within_batch)

        self.num_samples = len(targets)

        self.label_counts = []
        self.dft_indices = []
        self.other_indices = []
        for idx, targ in enumerate(targets):
            cnt = self._count_finite_labels(targ)
            self.label_counts.append(cnt)
            if cnt > 0:
                self.dft_indices.append(idx)
            else:
                self.other_indices.append(idx)

        if self.min_dft_per_batch > 0 and len(self.dft_indices) == 0:
            raise ValueError(
                "No DFT-labeled samples were found, but min_dft_per_batch > 0"
            )

        self.sample_weights = self._build_sample_weights()

    @staticmethod
    def _count_finite_labels(targ):
        if not isinstance(targ, torch.Tensor):
            targ = torch.as_tensor(targ)
        return int(torch.isfinite(targ).sum().item())

    def _build_sample_weights(self):
        weights = torch.zeros(self.num_samples, dtype=torch.double)

        has_dft = len(self.dft_indices) > 0
        has_other = len(self.other_indices) > 0

        if has_dft and has_other:
            dft_raw = torch.tensor(
                [
                    float(max(self.label_counts[i], 1)) ** self.label_count_power
                    for i in self.dft_indices
                ],
                dtype=torch.double,
            )
            dft_raw_sum = dft_raw.sum().item()
            if dft_raw_sum <= 0:
                dft_raw = torch.ones(len(self.dft_indices), dtype=torch.double)
                dft_raw_sum = float(len(self.dft_indices))

            other_weight_each = (1.0 - self.dft_group_weight) / len(self.other_indices)
            for i in self.other_indices:
                weights[i] = other_weight_each

            for j, i in enumerate(self.dft_indices):
                weights[i] = self.dft_group_weight * (dft_raw[j].item() / dft_raw_sum)

        elif has_dft:
            dft_raw = torch.tensor(
                [
                    float(max(self.label_counts[i], 1)) ** self.label_count_power
                    for i in self.dft_indices
                ],
                dtype=torch.double,
            )
            dft_raw_sum = dft_raw.sum().item()
            if dft_raw_sum <= 0:
                dft_raw = torch.ones(len(self.dft_indices), dtype=torch.double)
                dft_raw_sum = float(len(self.dft_indices))

            for j, i in enumerate(self.dft_indices):
                weights[i] = dft_raw[j].item() / dft_raw_sum

        else:
            uniform_weight = 1.0 / len(self.other_indices)
            for i in self.other_indices:
                weights[i] = uniform_weight

        if weights.sum().item() <= 0:
            raise RuntimeError("Failed to construct non-zero sampling weights")

        return weights

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        return math.ceil(self.num_samples / self.batch_size)

    def _sample_epoch_indices(self):
        return torch.multinomial(
            self.sample_weights,
            self.num_samples,
            replacement=self.replacement,
        ).tolist()

    def _draw_dft_indices(self, n):
        if n <= 0:
            return []
        if len(self.dft_indices) == 0:
            raise RuntimeError("Cannot draw DFT indices because dft_indices is empty")

        dft_weights = self.sample_weights[self.dft_indices]
        dft_weights = dft_weights / dft_weights.sum()

        if not self.dft_replacement and n > len(self.dft_indices):
            raise RuntimeError(
                "Insufficient DFT samples to satisfy min_dft_per_batch without replacement"
            )

        draw = torch.multinomial(
            dft_weights,
            n,
            replacement=self.dft_replacement,
        ).tolist()
        return [self.dft_indices[k] for k in draw]

    def _is_dft(self, idx):
        return self.label_counts[idx] > 0

    def _enforce_min_dft(self, batch):
        if self.min_dft_per_batch <= 0:
            return batch

        need_dft = min(self.min_dft_per_batch, len(batch))
        num_dft = sum(1 for idx in batch if self._is_dft(idx))
        missing = need_dft - num_dft
        if missing <= 0:
            return batch

        non_dft_positions = [k for k, idx in enumerate(batch) if not self._is_dft(idx)]
        if len(non_dft_positions) < missing:
            missing = len(non_dft_positions)
        if missing <= 0:
            return batch

        replacements = self._draw_dft_indices(missing)
        for pos, rep in zip(non_dft_positions[:missing], replacements):
            batch[pos] = rep

        return batch

    def __iter__(self):
        sampled = self._sample_epoch_indices()

        for start in range(0, self.num_samples, self.batch_size):
            end = min(start + self.batch_size, self.num_samples)
            batch = sampled[start:end]

            if self.drop_last and len(batch) < self.batch_size:
                continue

            batch = self._enforce_min_dft(batch)

            if self.shuffle_within_batch and len(batch) > 1:
                perm = torch.randperm(len(batch)).tolist()
                batch = [batch[i] for i in perm]

            yield batch
