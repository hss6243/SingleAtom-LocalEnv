import argparse
import os
import pickle as pkl
import sys

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, RandomSampler

from persite_painn.data import collate_dicts
from persite_painn.data.builder import (
    attach_external_features_to_dataset,
    build_dataset,
    load_external_features,
    split_train_validation_test,
)
from persite_painn.data.preprocess import convert_site_prop
from persite_painn.data.sampler import (
    ImbalancedDFTdLabelBatchSampler,
    MinDFTBatchSampler,
)
from persite_painn.nn.builder import get_model, load_params_from_path
from persite_painn.train import AverageMeter
from persite_painn.train.builder import (
    get_loss_metric_fn,
    get_optimizer,
    get_scheduler,
)
from persite_painn.train.evaluate import test_model as _package_test_model
from persite_painn.train.trainer import Trainer
from persite_painn.utils import batch_to, inference
from persite_painn.utils.train_utils import Normalizer
from persite_painn.utils.wandb_utils import save_artifacts


def test_model(
    model,
    test_loader,
    metric_fn,
    device,
    normalizer=None,
    multifidelity=False,
):
    """Local override of persite_painn.train.evaluate.test_model.

    This version fixes multifidelity evaluation for datasets where the UMA
    target is stored under "d_uma" (per-site), while "fidelity" is a
    per-structure scalar. We use `d_uma` as the target for the UMA head and
    keep the original logic otherwise.
    """

    model.to(device)
    model.eval()
    test_targets = []
    test_preds = []
    test_ids = []
    test_targets_fidelity = []
    test_preds_fidelity = []
    metrics = AverageMeter()

    with torch.no_grad():
        for batch in test_loader:
            batch = batch_to(batch, device)
            target = batch["target"]

            # DFT output for metric
            output = inference(
                model=model,
                data=batch,
                normalizer=normalizer,
                output_key="target",
                device=device,
            )
            if device == "cpu":
                metric_output = model(batch, inference=True)
            else:
                metric_output = model(batch)

            metric = metric_fn(metric_output, batch)
            metrics.update(metric.cpu().item(), target.size(0))

            # Per-site indexing
            test_pred = output.data.cpu()
            test_target = target.detach().cpu()

            batch_ids = []
            count = 0
            num_bin = []
            for i, val in enumerate(batch["num_atoms"].detach().cpu().numpy()):
                count += val
                num_bin.append(count)
                if i == 0:
                    change = list(np.arange(val))
                else:
                    adding_val = num_bin[i - 1]
                    change = list(np.arange(val) + adding_val)
                batch_ids.append(change)

            test_preds += [test_pred[i].tolist() for i in batch_ids]
            test_targets += [test_target[i].tolist() for i in batch_ids]

            if multifidelity:
                # Use per-site UMA labels stored under "d_uma" as targets
                target_fidelity = batch["d_uma"].detach().cpu()
                output_fidelity = inference(
                    model=model,
                    data=batch,
                    normalizer=normalizer,
                    output_key="fidelity",
                    device=device,
                ).data.cpu()

                test_preds_fidelity += [
                    output_fidelity[i].tolist() for i in batch_ids
                ]
                test_targets_fidelity += [
                    target_fidelity[i].tolist() for i in batch_ids
                ]

            metric_out = metrics.avg
            if isinstance(batch["name"], list):
                test_ids += batch["name"]
            else:
                test_ids += batch["name"].detach().tolist()

    return (
        test_preds,
        test_targets,
        test_ids,
        metric_out,
        test_preds_fidelity,
        test_targets_fidelity,
    )

parser = argparse.ArgumentParser(description="Per-site PaiNN multifidelity (DFT + UMA)")
parser.add_argument("--data_raw", default="", type=str, help="path to raw data")
parser.add_argument(
    "--data_cache",
    default="dataset_cache",
    type=str,
    help="cache where data is / will be stored",
)
parser.add_argument(
    "--details", default="details.json", type=str, help="json file of model parameters"
)
parser.add_argument("--savedir", default="./results", type=str, help="saving directory")
parser.add_argument(
    "--workers", default=0, type=int, help="number of data loading workers"
)
parser.add_argument(
    "--epochs", default=500, type=int, help="number of total epochs to run"
)
parser.add_argument(
    "--start_epoch",
    default=0,
    type=int,
    help="manual epoch number (useful on restarts)",
)
parser.add_argument("-b", "--batch_size", default=64, type=int, help="mini-batch size")
parser.add_argument("--print_freq", default=10, type=int, help="print frequency")
parser.add_argument("--resume", default="", type=str, help="path to latest checkpoint")
parser.add_argument("--cuda", default=0, type=int, help="GPU setting")
parser.add_argument("--device", default="cuda", type=str, help="cpu or cuda")
parser.add_argument(
    "--early_stop_val",
    default=20,
    type=int,
    help="early stopping condition for validation loss update count",
)
parser.add_argument(
    "--early_stop_train",
    default=0.05,
    type=float,
    help="early stopping condition for train loss tolerance",
)
parser.add_argument(
    "--seed",
    default=None,
    type=int,
    help="Seed of random initialization to control the experiment",
)
parser.add_argument(
    "--wandb",
    action="store_true",
    default=False,
    help="Whether to run with W & B",
)
parser.add_argument(
    "--test_ids",
    default=None,
    type=str,
    help="pickle filename where test ids are stored",
)
parser.add_argument(
    "--val_ids",
    default=None,
    type=str,
    help="pickle filename where val ids are stored",
)
parser.add_argument(
    "--external_features",
    default="",
    type=str,
    help="path to external per-node feature map (.pkl or .pt)",
)
parser.add_argument(
    "--external_feature_dim",
    default=None,
    type=int,
    help="external feature dimension (auto-infer from file when omitted)",
)
parser.add_argument(
    "--external_alpha",
    default=1.0,
    type=float,
    help="residual fusion scale for external features",
)
parser.add_argument(
    "--external_fusion_mode",
    default="concat",
    choices=["none", "add", "concat", "concat_unprojected"],
    type=str,
    help=(
        "fusion mode for external features: "
        "none (ignore external), add (Wu+b residual), "
        "concat (concat + 2d->d projection), "
        "concat_unprojected (keep concatenated 2d feature)"
    ),
)
parser.add_argument(
    "--train_min_dft_per_batch",
    default=0,
    type=int,
    help=(
        "minimum number of DFT-labeled samples per training batch; "
        "0 disables enforced DFT batch composition"
    ),
)
parser.add_argument(
    "--train_dft_replacement",
    action="store_true",
    default=False,
    help="allow replacement when enforcing minimum DFT samples per batch",
)
parser.add_argument(
    "--disable_imbalanced_dftd_sampler",
    action="store_true",
    default=False,
    help=(
        "disable ImbalancedDFTdLabelBatchSampler in multifidelity training and "
        "fall back to previous sampler logic"
    ),
)
parser.add_argument(
    "--dft_group_weight",
    default=0.5,
    type=float,
    help="sampling probability mass allocated to DFT-labeled samples in MF mode",
)
parser.add_argument(
    "--label_count_power",
    default=1.0,
    type=float,
    help="power exponent for per-sample finite-label count weighting in MF mode",
)
parser.add_argument(
    "--temporary_val_target_only",
    action="store_true",
    default=False,
    help=(
        "TEMPORARY EXPERIMENT ONLY: use DFT target-only validation loss for "
        "model selection/early stopping. Remove this option after this experiment."
    ),
)


def main(args):
    # Load details
    wandb_config, details, modelparams, model_type = load_params_from_path(args.details)

    external_features = None
    if args.external_features:
        external_features = load_external_features(args.external_features)
        print(f"Loaded external feature map: {len(external_features)} samples")

    # wandb config
    if args.wandb:
        wandb_config.update(details)
        wandb_config.update(modelparams)
        wandb.init(
            project=wandb_config["project"],
            name=wandb_config["name"],
            config=wandb_config,
        )

    # Load data (prefer cached Dataset as in existing scripts)
    if os.path.exists(args.data_cache):
        print("Cached dataset exists...")
        dataset = torch.load(args.data_cache)
        if external_features is not None:
            dataset = attach_external_features_to_dataset(dataset, external_features)
        print(f"Number of Data: {len(dataset)}")
    else:
        try:
            data = pkl.load(open(args.data_raw, "rb"))
        except ValueError:
            print("Path to data should be given --data_raw")
        else:
            print("Start making dataset...")
            if details["multifidelity"]:
                new_data = convert_site_prop(
                    data,
                    details["output_keys"],
                    details["fidelity_keys"],
                )
                dataset = build_dataset(
                    raw_data=new_data,
                    cutoff=modelparams["cutoff"],
                    multifidelity=details["multifidelity"],
                    seed=args.seed,
                    external_features=external_features,
                )
            else:
                new_data = convert_site_prop(data, details["output_keys"])
                dataset = build_dataset(
                    raw_data=new_data,
                    cutoff=modelparams["cutoff"],
                    multifidelity=details["multifidelity"],
                    seed=args.seed,
                    external_features=external_features,
                )

            print(f"Number of Data: {len(dataset)}")
            print("Done creating dataset, caching...")
            dataset.save(args.data_cache)
            print("Done caching dataset")

    # Optional fixed train/val split
    if args.test_ids is not None and args.val_ids is not None:
        test_ids_bin = pkl.load(open(args.test_ids, "rb"))
        val_ids_bin = pkl.load(open(args.val_ids, "rb"))
        val_size = details["val_size"]
        test_size = 0
    else:
        test_ids_bin = None
        val_ids_bin = None
        val_size = details["val_size"]
        test_size = details["test_size"]

    train_set, val_set, test_set = split_train_validation_test(
        dataset,
        val_size=val_size,
        test_size=test_size,
        seed=args.seed,
        test_ids=test_ids_bin,
        val_ids=val_ids_bin,
    )

    # Normalizers
    normalizer = {}

    # DFT target normalizer (same as DFT-only)
    targs_target = []
    for batch in train_set:
        targs_target.append(batch["target"])
    targs_target = torch.concat(targs_target)
    normalizer_target = Normalizer(targs_target, "target")
    normalizer["target"] = normalizer_target

    # UMA (d_uma) normalizer: used as "fidelity" output target
    targs_duma = []
    for batch in train_set:
        targs_duma.append(batch["d_uma"])
    targs_duma = torch.concat(targs_duma)
    normalizer_duma = Normalizer(targs_duma, "d_uma")
    normalizer["d_uma"] = normalizer_duma
    # Alias for model's fidelity head so evaluation can denormalize it
    normalizer["fidelity"] = normalizer_duma

    # Means / stddevs passed into PainnMultifidelity
    modelparams["means"] = {
        "target": normalizer_target.mean,
        "fidelity": normalizer_duma.mean,
    }
    modelparams["stddevs"] = {
        "target": normalizer_target.std,
        "fidelity": normalizer_duma.std,
    }

    modelparams["use_external_features"] = external_features is not None
    modelparams["external_alpha"] = args.external_alpha
    modelparams["external_fusion_mode"] = args.external_fusion_mode
    if external_features is not None:
        if args.external_feature_dim is not None:
            ext_dim = args.external_feature_dim
        else:
            first_key = next(iter(external_features.keys()))
            first_val = external_features[first_key]
            ext_dim = int(first_val.shape[-1]) if first_val.ndim > 1 else 1
        modelparams["external_feature_dim"] = ext_dim
        print(
            f"External feature fusion enabled: dim={modelparams['external_feature_dim']}, "
            f"alpha={modelparams['external_alpha']}, "
            f"mode={modelparams['external_fusion_mode']}"
        )

    # Get model (PainnMultifidelity when details["multifidelity"] is True)
    model = get_model(
        modelparams,
        model_type=model_type,
        multifidelity=details["multifidelity"],
    )
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    # Optimizer
    optimizer = get_optimizer(
        optim=details["optim"],
        trainable_params=trainable_params,
        lr=details["lr"],
        weight_decay=details["weight_decay"],
    )

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            if args.start_epoch != 0:
                print(f"=> loading checkpoint '{args.resume}'")
                checkpoint = torch.load(args.resume)
                best_metric = checkpoint["best_metric"]
                best_loss = checkpoint["best_loss"]
                model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                args.start_epoch = checkpoint["epoch"]
                normalizer.load_state_dict(checkpoint["normalizer"])
            elif args.start_epoch == 0:
                checkpoint = torch.load(args.resume)
                best_metric = checkpoint["best_metric"]
                best_loss = checkpoint["best_loss"]
                model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                normalizer.load_state_dict(checkpoint["normalizer"])

            print(
                f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})"
            )
        else:
            print(f"=> no checkpoint found at '{args.resume}'")
    else:
        best_metric = 1e10
        best_loss = 1e10

    # Loss function: multi-task on DFT target + UMA (d_uma)
    if details["multifidelity"]:
        # modelparams["loss_coeff"]["target"] in [0,1]; remainder goes to UMA
        lam_target = modelparams["loss_coeff"]["target"]
        # Use "d_uma" as the loss key (we will map model's fidelity output to this)
        loss_coeff = {"d_uma": 1.0 - lam_target, "target": lam_target}
        correspondence_keys_loss = {"d_uma": "d_uma", "target": "target"}

        base_loss_fn = get_loss_metric_fn(
            loss_coeff=loss_coeff,
            correspondence_keys=correspondence_keys_loss,
            operation_name=details["loss_fn"],
            normalizer=normalizer,
        )

        def loss_fn(results, ground_truth):
            # Map model's "fidelity" output to the logical "d_uma" key
            remapped_results = dict(results)
            remapped_results["d_uma"] = results["fidelity"]
            return base_loss_fn(remapped_results, ground_truth)

    else:
        loss_coeff = {"target": 1.0}
        correspondence_keys_loss = {"target": "target"}

        loss_fn = get_loss_metric_fn(
            loss_coeff=loss_coeff,
            correspondence_keys=correspondence_keys_loss,
            operation_name=details["loss_fn"],
            normalizer=normalizer,
        )

    # Metric: report MAE on DFT target only (for clarity)
    metric_coeff = {"target": 1.0}
    correspondence_keys_metric = {"target": "target"}
    metric_fn = get_loss_metric_fn(
        loss_coeff=metric_coeff,
        correspondence_keys=correspondence_keys_metric,
        operation_name=details["metric_fn"],
        normalizer=normalizer,
    )

    # Validation loss selection
    # TEMPORARY EXPERIMENT ONLY: this branch exists for short-term comparison and
    # should be removed after finishing this experiment.
    if args.temporary_val_target_only:
        val_loss_coeff = {"target": 1.0}
        val_correspondence = {"target": "target"}
        validation_loss_fn = get_loss_metric_fn(
            loss_coeff=val_loss_coeff,
            correspondence_keys=val_correspondence,
            operation_name=details["loss_fn"],
            normalizer=normalizer,
        )
        print(
            "[TEMP] Validation loss is set to DFT target-only. "
            "Remove --temporary_val_target_only after this experiment."
        )
    else:
        # Default behavior: validation loss follows the same loss function as train.
        validation_loss_fn = None

    # Scheduler
    scheduler = get_scheduler(
        sched=details["sched"], optimizer=optimizer, epochs=args.epochs
    )

    # DataLoaders
    if details["multifidelity"] and not args.disable_imbalanced_dftd_sampler:
        train_batch_sampler = ImbalancedDFTdLabelBatchSampler(
            targets=train_set.props["target"],
            batch_size=args.batch_size,
            drop_last=False,
            dft_group_weight=args.dft_group_weight,
            label_count_power=args.label_count_power,
            replacement=True,
            min_dft_per_batch=args.train_min_dft_per_batch,
            dft_replacement=args.train_dft_replacement,
            shuffle_within_batch=True,
        )
        train_loader = DataLoader(
            train_set,
            num_workers=args.workers,
            collate_fn=collate_dicts,
            batch_sampler=train_batch_sampler,
        )
        print(
            "Using ImbalancedDFTdLabelBatchSampler for MF: "
            f"dft_group_weight={args.dft_group_weight}, "
            f"label_count_power={args.label_count_power}, "
            f"min_dft_per_batch={args.train_min_dft_per_batch}, "
            f"dft_replacement={args.train_dft_replacement}"
        )
    elif details["multifidelity"] and args.train_min_dft_per_batch > 0:
        train_batch_sampler = MinDFTBatchSampler(
            targets=train_set.props["target"],
            batch_size=args.batch_size,
            min_dft_per_batch=args.train_min_dft_per_batch,
            drop_last=False,
            shuffle=True,
            dft_replacement=args.train_dft_replacement,
        )
        train_loader = DataLoader(
            train_set,
            num_workers=args.workers,
            collate_fn=collate_dicts,
            batch_sampler=train_batch_sampler,
        )
        print(
            "Using MinDFTBatchSampler fallback for MF: "
            f"min_dft_per_batch={args.train_min_dft_per_batch}, "
            f"dft_replacement={args.train_dft_replacement}"
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=args.workers,
            collate_fn=collate_dicts,
            sampler=RandomSampler(train_set),
        )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_dicts,
    )

    # Save ids
    train_ids = []
    for item in train_set:
        if isinstance(item["name"], str):
            train_ids.append(item["name"])
        else:
            train_ids.append(item["name"].item())
    val_ids = []
    for item in val_set:
        if isinstance(item["name"], str):
            val_ids.append(item["name"])
        else:
            val_ids.append(item["name"].item())

    os.makedirs(args.savedir, exist_ok=True)
    pkl.dump(train_ids, open(f"{args.savedir}/train_ids.pkl", "wb"))
    pkl.dump(val_ids, open(f"{args.savedir}/val_ids.pkl", "wb"))

    early_stop = [args.early_stop_val, args.early_stop_train]

    # (Optional) turn off gradient of conv_to_fc as in original multifidelity script
    if details["multifidelity"] and hasattr(model, "fn_target") and hasattr(
        model.fn_target, "conv_to_fc"
    ):
        for param in model.fn_target.conv_to_fc.parameters():
            param.requires_grad = False

    # Trainer
    trainer = Trainer(
        model_path=args.savedir,
        model=model,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        validation_loader=val_loader,
        run_wandb=args.wandb,
        validation_loss_fn=validation_loss_fn,
        normalizer=normalizer,
    )

    # Train
    _ = trainer.train(
        device=args.device,
        start_epoch=args.start_epoch,
        n_epochs=args.epochs,
        best_loss=best_loss,
        best_metric=best_metric,
        early_stop=early_stop,
    )

    # Test (only if we actually have a test set)
    if len(test_set) > 0:
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            num_workers=args.workers,
            collate_fn=collate_dicts,
        )

        best_checkpoint = torch.load(f"{args.savedir}/best_model.pth.tar")
        model.load_state_dict(best_checkpoint["state_dict"])

        (
            test_preds,
            test_targets,
            test_ids,
            metric_out,
            test_preds_fidelity,
            test_targets_fidelity,
        ) = test_model(
            model=model,
            test_loader=test_loader,
            metric_fn=metric_fn,
            device="cpu",
            normalizer=normalizer,
            multifidelity=details["multifidelity"],
        )
        print(f"TEST Accuracy (DFT target metric): {metric_out}")

        # Save test results
        pkl.dump(test_ids, open(f"{args.savedir}/test_ids.pkl", "wb"))
        pkl.dump(test_preds, open(f"{args.savedir}/test_preds.pkl", "wb"))
        pkl.dump(test_targets, open(f"{args.savedir}/test_targs.pkl", "wb"))
        if details["multifidelity"]:
            pkl.dump(
                test_preds_fidelity,
                open(f"{args.savedir}/test_preds_fidelity.pkl", "wb"),
            )
            pkl.dump(
                test_targets_fidelity,
                open(f"{args.savedir}/test_targs_fidelity.pkl", "wb"),
            )
    else:
        print("No test set defined (test_size=0). Skipping test evaluation.")

    # save wandb artifacts
    if args.wandb:
        save_artifacts(args.savedir, details["multifidelity"])


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
        assert torch.cuda.is_available(), "cuda is not available"

    main(args)
