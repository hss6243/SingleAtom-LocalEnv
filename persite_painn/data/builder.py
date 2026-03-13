from tqdm import tqdm
import random
import pickle as pkl
from sklearn.model_selection import train_test_split

from pymatgen.io.ase import AseAtomsAdaptor as AA
import numpy as np
import torch
from .dataset import Dataset, Dataset_lite


def _to_python_id(val):
    if isinstance(val, torch.Tensor):
        if val.numel() == 1:
            return val.item()
        return val.detach().cpu().numpy().tolist()
    return val


def _to_key(val):
    return str(_to_python_id(val))


def _extract_embedding_value(raw_val):
    if isinstance(raw_val, dict):
        if "embedding" in raw_val:
            raw_val = raw_val["embedding"]
        elif "embeddings" in raw_val:
            raw_val = raw_val["embeddings"]

    if hasattr(raw_val, "detach") and hasattr(raw_val, "cpu"):
        return raw_val.detach().cpu().numpy()

    return np.asarray(raw_val)


def load_external_features(path):
    """Load external per-node embeddings from a .pkl or .pt file.

    Supported formats:
    - dict: {sample_id: embedding_or_record}
    - list: [{"sample_id": ..., "embedding": ...}, ...]
    """
    if path.endswith(".pt") or path.endswith(".pth"):
        payload = torch.load(path, map_location="cpu")
    else:
        payload = pkl.load(open(path, "rb"))

    feature_map = {}
    if isinstance(payload, dict):
        for sample_id, emb in payload.items():
            feature_map[_to_key(sample_id)] = _extract_embedding_value(emb)
        return feature_map

    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                raise TypeError("Each list item must be a dict with sample_id and embedding")
            sample_id = row.get("sample_id", row.get("id", row.get("name")))
            if sample_id is None:
                raise KeyError("Missing sample_id/id/name in external feature row")
            emb = row.get("embedding", row.get("embeddings"))
            if emb is None:
                raise KeyError("Missing embedding/embeddings in external feature row")
            feature_map[_to_key(sample_id)] = _extract_embedding_value(emb)
        return feature_map

    raise TypeError("Unsupported external feature payload type")


def attach_external_features_to_dataset(dataset, external_features, strict=True):
    """Attach per-node external features to an already constructed Dataset."""
    new_features_list = []
    for name, nxyz in zip(dataset.props["name"], dataset.props["nxyz"]):
        key = _to_key(name)
        if key not in external_features:
            if strict:
                raise KeyError(f"Missing external feature for sample id: {key}")
            raise KeyError(
                f"Non-strict mode is not supported yet for missing sample id: {key}"
            )

        feat = np.asarray(_extract_embedding_value(external_features[key]))
        if feat.ndim == 1:
            feat = feat.reshape(-1, 1)

        num_atoms = int(nxyz.shape[0])
        if feat.shape[0] != num_atoms:
            raise ValueError(
                f"External feature atom count mismatch for {key}: "
                f"{feat.shape[0]} vs {num_atoms}"
            )

        new_features_list.append(torch.tensor(feat, dtype=torch.float32))

    dataset.props["new_features"] = new_features_list
    return dataset


def build_dataset(
    raw_data,
    cutoff=5.0,
    multifidelity=False,
    seed=1234,
    external_features=None,
) -> Dataset:
    """Summary: Builds a dataset from raw data. (Modified by SS to include d_uma)
    """
    samples = [[id_, struct] for id_, struct in raw_data.items()]
    props = gen_props_from_file(
        samples=samples,
        multifidelity=multifidelity,
        seed=seed,
        external_features=external_features,
    )
    dataset = Dataset(props=props)
    # dataset = Dataset_lite(props=props)
    dataset.generate_neighbor_list(cutoff=cutoff, undirected=False)

    return dataset


def compute_prop(id_, crystal, multifidelity):
    if multifidelity:
        target = crystal.site_properties["target"]
        fidelity = crystal.site_properties["fidelity"]
        index = compute_balanced_batch_index(target)
        if len(target) == 1:
            target = np.array(target).reshape(-1, 1)
    else:
        target = crystal.site_properties["target"]
        if len(target) == 1:
            target = np.array(target).reshape(-1, 1)
        fidelity = None
        index = None

    d_uma = crystal.site_properties["d_uma"]    
    if d_uma is not None and len(d_uma) == 1:
        d_uma = np.array(d_uma).reshape(-1, 1)
  

    structure = AA.get_atoms(crystal)
    n = np.asarray(structure.numbers).reshape(-1, 1)
    xyz = np.asarray(structure.positions)
    nxyz = np.concatenate((n, xyz), axis=1)
    lattice = structure.cell[:]

    return id_, nxyz, lattice, target, fidelity, index, d_uma


def gen_props_from_file(
    samples,
    multifidelity=True,
    seed=1234,
    external_features=None,
):
    """Summary

    Args:
        path (TYPE): Description

    Returns:
        TYPE: Description

    Raises:
        TypeError: Description
    """
    print("Creating props...")
    random.seed(seed)
    random.shuffle(samples)

    props = {}
    name_list = []
    nxyz_list = []
    lattice_list = []
    fidelity_list = []
    target_list = []
    index_list = []
    d_uma_list = []
    new_features_list = []
    for idx in tqdm(range(len(samples)), position=0, leave=True):
        id_, nxyz, lattice, target, fidelity, index, d_uma = compute_prop(
            samples[idx][0],
            samples[idx][1],
            multifidelity,
        )

        name_list.append(id_)
        nxyz_list.append(nxyz)
        lattice_list.append(lattice)
        target_list.append(target)
        fidelity_list.append(fidelity)
        index_list.append(index)
        d_uma_list.append(d_uma)
        if external_features is not None:
            key = _to_key(id_)
            if key not in external_features:
                raise KeyError(f"Missing external feature for sample id: {key}")

            feat = np.asarray(_extract_embedding_value(external_features[key]))
            if feat.ndim == 1:
                feat = feat.reshape(-1, 1)

            if feat.shape[0] != nxyz.shape[0]:
                raise ValueError(
                    f"External feature atom count mismatch for {key}: "
                    f"{feat.shape[0]} vs {nxyz.shape[0]}"
                )
            new_features_list.append(feat)
        
    props["nxyz"] = nxyz_list
    props["lattice"] = lattice_list
    props["name"] = name_list
    props["target"] = target_list
    props["fidelity"] = fidelity_list
    props["classification"] = index_list
    props["d_uma"] = d_uma_list
    if external_features is not None:
        props["new_features"] = new_features_list

    return props


def binary_split(dataset, targ_name, test_size, seed):
    """
    Split the dataset with proportional amounts of a binary label in each.
    Args:
        dataset (nff.data.dataset): NFF dataset
        targ_name (str, optional): name of the binary label to use
            in splitting.
        test_size (float, optional): fraction of dataset for test
    Returns:
        idx_train (list[int]): indices of species in the training set
        idx_test (list[int]): indices of species in the test set
    """

    # get indices of positive and negative values
    pos_idx = [i for i, targ in enumerate(dataset.props[targ_name]) if targ]
    neg_idx = [i for i in range(len(dataset)) if i not in pos_idx]

    # split the positive and negative indices separately
    pos_idx_train, pos_idx_test = train_test_split(
        pos_idx, test_size=test_size, random_state=seed
    )
    neg_idx_train, neg_idx_test = train_test_split(
        neg_idx, test_size=test_size, random_state=seed
    )

    # combine the negative and positive test idx to get the test idx
    # do the same for train

    idx_train = pos_idx_train + neg_idx_train
    idx_test = pos_idx_test + neg_idx_test

    return idx_train, idx_test


def split_train_test(dataset, test_size=0.2, binary=False, targ_name=None, seed=None, test_ids=None):
    """Splits the current dataset in two, one for training and
    another for testing.

    Args:
        dataset (nff.data.dataset): NFF dataset
        test_size (float, optional): fraction of dataset for test
        binary (bool, optional): whether to split the dataset with
            proportional amounts of a binary label in each.
        targ_name (str, optional): name of the binary label to use
            in splitting.
    Returns:
        TYPE: Description
    """

    if binary:
        idx_train, idx_test = binary_split(
            dataset=dataset, targ_name=targ_name, test_size=test_size, seed=seed
        )
    elif test_ids is not None:
        idx_train = []
        idx_test = []

        for i, data in enumerate(dataset):
            if data['name'].item() in test_ids:
                idx_test.append(i)
            else:
                idx_train.append(i)
    else:
        idx = list(range(len(dataset)))
        idx_train, idx_test = train_test_split(
            idx, test_size=test_size, random_state=seed
        )

    train = Dataset(
        props={key: [val[i] for i in idx_train] for key, val in dataset.props.items()},
    )
    test = Dataset(
        props={key: [val[i] for i in idx_test] for key, val in dataset.props.items()},
    )

    return train, test


def split_train_validation_test(
    dataset, val_size=0.2, test_size=0.2, seed=None, test_ids=None, val_ids=None
):
    """Summary
    Args:
        dataset (TYPE): Description
        val_size (float, optional): Description
        test_size (float, optional): Description
    Returns:
        TYPE: Description
    """
    train, test = split_train_test(dataset, test_size=test_size, seed=seed, test_ids=test_ids)
    train, validation = split_train_test(
        train, test_size=val_size / (1 - test_size), seed=seed, test_ids=val_ids
    )

    return train, validation, test


def compute_balanced_batch_index(target):
    target = np.array(target).reshape(-1)
    nan_mask = np.isnan(target)
    # print(target[~nan_mask])
    check = len(target[~nan_mask])
    if check > 0:
        return 0.0
    else:
        return 1.0
