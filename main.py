import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import csv
import torch
import numpy as np

from dataset.dataset_survival import Generic_MIL_Survival_Dataset
from utils.options import parse_args
from utils.util import get_split_loader, set_seed

from utils.loss import define_loss
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler


def main(args):
    # set random seed for reproduction
    set_seed(args.seed)
    
    if args.mode != 'debug':
    # create results directory
        results_dir = "./results/{dataset}/{model}-{fusion}-{alpha}-{optimizer}".format(
            dataset=args.dataset,
            model=args.model,
            fusion=args.fusion,
            alpha=args.alpha,
            optimizer = args.optimizer
        )
    else:
        results_dir = f'./results/debug/{args.dataset}'
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    print('*' * 100)
    print(f'current gpu:{torch.cuda.current_device()}')
    print(f'current dataset:{args.dataset}')
    print(f'results_dir:{results_dir}')
    print('*' * 100)

    # 5-fold cross validation
    header = ["folds", "fold 0", "fold 1", "fold 2", "fold 3", "fold 4", "mean", "std"]
    best_epoch = ["best epoch"]
    best_score = ["best cindex"]

    # start 5-fold CV evaluation.
    for fold in range(5):
        # build dataset
        dataset = Generic_MIL_Survival_Dataset(
            csv_path=f"./csv/tcga_{args.dataset}_all_clean.csv",
            modal=args.modal,
            OOM=args.OOM,
            apply_sig=True,
            data_dir=args.data_root_dir,
            shuffle=False,
            seed=args.seed,
            patient_strat=False,
            n_bins=4,
            label_col="survival_months",
        )
        split_dir = os.path.join("./splits", args.which_splits, args.dataset.upper())
        train_dataset, val_dataset = dataset.return_splits(
            from_id=False, csv_path="{}/splits_{}.csv".format(split_dir, fold)
        )
        train_loader = get_split_loader(
            train_dataset,
            training=True,
            weighted=args.weighted_sample,
            modal=args.modal,
            batch_size=args.batch_size,
        )
        val_loader = get_split_loader(
            val_dataset, modal=args.modal, batch_size=args.batch_size
        )
        print(
            "training: {}, validation: {}".format(len(train_dataset), len(val_dataset))
        )

        # build model, criterion, optimizer, schedular
        if args.model == "ptcma":
            from models.ptcma.network import PTCMA
            from models.ptcma.engine import Engine

            model_dict = {
                # "text_sizes": train_dataset.text_sizes,
                "n_classes": 4,
                "fusion": args.fusion,
                "model_size": args.model_size,
            }
            model = PTCMA(**model_dict)
            criterion = define_loss(args)
            optimizer = define_optimizer(args, model)
            scheduler = define_scheduler(args, optimizer)
            engine = Engine(args, results_dir, fold)

        elif args.model == 'abmil':
            from models.abacmil.engine import Engine
            from models.abacmil.network import ABMIL

            model_dict = {
                # "text_sizes": train_dataset.text_sizes,
                "n_classes": 4,
            }
            model = ABMIL(**model_dict)
            criterion = define_loss(args)
            optimizer = define_optimizer(args, model)
            scheduler = define_scheduler(args, optimizer)
            engine = Engine(args, results_dir, fold)

        elif args.model == 'acmil_ga':
            from models.abacmil.engine import Engine
            from models.abacmil.network import ACMIL_GA

            model_dict = {
                # "text_sizes": train_dataset.text_sizes,
                "n_classes": 4,
            }
            model = ACMIL_GA(**model_dict)
            criterion = define_loss(args)
            optimizer = define_optimizer(args, model)
            scheduler = define_scheduler(args, optimizer)
            engine = Engine(args, results_dir, fold)
        
        elif args.model == 'acmil_mha':
            from models.abacmil.engine import Engine
            from models.abacmil.network import ACMIL_MHA

            model_dict = {
                # "text_sizes": train_dataset.text_sizes,
                "n_classes": 4,
            }
            model = ACMIL_MHA(**model_dict)
            criterion = define_loss(args)
            optimizer = define_optimizer(args, model)
            scheduler = define_scheduler(args, optimizer)
            engine = Engine(args, results_dir, fold)

        elif args.model == 'transmil':
            from models.transmil.engine import Engine
            from models.transmil.network import TransMIL

            model_dict = {
                # "text_sizes": train_dataset.text_sizes,
                "n_classes": 4,
            }
            model = TransMIL(**model_dict)
            criterion = define_loss(args)
            optimizer = define_optimizer(args, model)
            scheduler = define_scheduler(args, optimizer)
            engine = Engine(args, results_dir, fold)

        else:
            raise NotImplementedError(
                "Model [{}] is not implemented".format(args.model)
            )
        # start training
        score, epoch = engine.learning(
            model, train_loader, val_loader, criterion, optimizer, scheduler
        )
        # save best score and epoch for each fold
        best_epoch.append(epoch)
        best_score.append(score)

    # finish training
    # mean and std
    best_epoch.append("~")
    best_epoch.append("~")
    best_score.append(np.mean(best_score[1:6]))
    best_score.append(np.std(best_score[1:6]))

    csv_path = os.path.join(results_dir, "results.csv")
    print("############", csv_path)
    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        writer.writerow(best_epoch)
        writer.writerow(best_score)


if __name__ == "__main__":
    args = parse_args()
    results = main(args)
    print("finished!")
