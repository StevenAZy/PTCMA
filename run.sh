CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv \
  --mode train \
  --dataset blca \
  --data_root_dir /data/lichangyong/TCGA_FEATURE/BLCA \
  --modal coattn \
  --model cmta \
  --num_epoch 30 \
  --batch_size 1 \
  --loss nll_surv_l1 \
  --lr 0.001 \
  --optimizer Adam \
  --scheduler None \
  --alpha 1.0

CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv \
  --mode train \
  --dataset brca \
  --data_root_dir /data/lichangyong/TCGA_FEATURE/BRCA \
  --modal coattn \
  --model cmta \
  --num_epoch 30 \
  --batch_size 1 \
  --loss nll_surv_l1 \
  --lr 0.001 \
  --optimizer Adam \
  --scheduler None \
  --alpha 1.0

CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv \
  --dataset luad \
  --data_root_dir /data/lichangyong/TCGA_FEATURE/LUAD \
  --modal coattn \
  --model cmta \
  --num_epoch 30 \
  --batch_size 1 \
  --loss nll_surv_l1 \
  --lr 0.001 \
  --optimizer Adam \
  --scheduler None \
  --alpha 1.0

CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv \
  --mode train \
  --dataset ucec \
  --data_root_dir /data/lichangyong/TCGA_FEATURE/UCEC \
  --modal coattn \
  --model cmta \
  --num_epoch 30 \
  --batch_size 1 \
  --loss nll_surv_l1 \
  --lr 0.001 \
  --optimizer Adam \
  --scheduler None \
  --alpha 1.0