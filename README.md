# PTCMA
**Summary:** Here is the official implementation of PTCMA.

### Pre-requisites:
```bash
torch 1.12.0+cu116
scikit-survival 0.19.0
```
### Prepare your data
#### WSIs
1. Download diagnostic WSIs from [TCGA](https://portal.gdc.cancer.gov/)
2. Use the WSI processing tool provided by [CLAM](https://github.com/mahmoodlab/CLAM) to extract resnet-50 pretrained 1024-dim feature for each 256 $\times$ 256 patch (20x), which we then save as `.pt` files for each WSI. So, we get one `pt_files` folder storing `.pt` files for all WSIs of one study.

The final structure of datasets should be as following:
```bash
DATA_ROOT_DIR/
    └──pt_files/
        ├── slide_1.pt
        ├── slide_2.pt
        └── ...
```

DATA_ROOT_DIR is the base directory of cancer type (e.g. the directory to TCGA_BLCA), which should be passed to the model with the argument `--data_root_dir` as shown in [run.sh](run.sh).

Splits for each cancer type are found in the `splits/5foldcv ` folder, which are randomly partitioned each dataset using 5-fold cross-validation. Each one contains splits_{k}.csv for k = 1 to 5. To compare with MCAT, we follow the same splits as that of MCAT.

## Running Experiments
To train PTCMA, you can specify the argument in the bash `run.sh` and run the command:
```bash
bash run.sh
```
or use the following generic command-line and specify the arguments:
```bash
CUDA_VISIBLE_DEVICES=<DEVICE_ID> python main.py \
                                      --which_splits 5foldcv \
                                      --dataset <CANCER_TYPE> \
                                      --data_root_dir <DATA_ROOT_DIR>\
                                      --modal coattn \
                                      --model ptcma \
                                      --num_epoch 30 \
                                      --batch_size 1 \
                                      --loss nll_surv_l1 \
                                      --lr 0.001 \
                                      --optimizer SGD \
                                      --scheduler None \
                                      --alpha 1.0
```
Commands for all experiments of PTCMA can be found in the [run.sh](run.sh) file.

__Tips: some patients may have multiple WSIs, especially in TCGA-GBMLGG, resulting in OOM issue.__ In such case, we can randomly sample certain number of patches for these special patients to reduce the computational requirements. That will not significantly impact the overall performance.
```bash
CUDA_VISIBLE_DEVICES=<DEVICE_ID> python main.py \
                                      --which_splits 5foldcv \
                                      --dataset <CANCER_TYPE> \
                                      --data_root_dir <DATA_ROOT_DIR>\
                                      --modal coattn \
                                      --model ptcma \
                                      --num_epoch 30 \
                                      --batch_size 1 \
                                      --loss nll_surv_l1 \
                                      --lr 0.001 \
                                      --optimizer SGD \
                                      --scheduler None \
                                      --alpha 1.0 \
                                      --OOM 4096
```
If the number of patches is larger than 4096, randomly sampling 4096 patches. __If there is still OOM issue, you can further reduce the number of sampled patches.__

## Acknowledgements
Huge thanks to the authors of following open-source projects:
- [CLAM](https://github.com/mahmoodlab/CLAM)

## License & Citation 
If you find our work useful in your research, please consider citing our paper at:
