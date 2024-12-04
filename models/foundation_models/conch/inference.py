import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from sksurv.metrics import concordance_index_censored
from open_clip_custom import create_model_from_pretrained


def read_data_csv(csv_file):
    data_df = pd.read_csv(csv_file)
    return data_df["val"].dropna()

if __name__ == "__main__":
    dataset = "BLCA"
    all_data_csv = f"/home/lichangyong/code/PTCMA/csv/tcga_{dataset.lower()}_all_clean.csv"
    dataset_csv = f"/home/lichangyong/code/PTCMA/splits/5foldcv/{dataset}/splits_0.csv"
    dataset_path = f"/data/lichangyong/TCGA_FEATURE/{dataset}/pt_files_by_case"

    model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path="checkpoints/pytorch_model.bin")
    model.eval() 
    params_before_inference = {name: param.clone() for name, param in model.named_parameters()}
    print(type(params_before_inference))

    val_data = read_data_csv(dataset_csv)

    all_data_df = pd.read_csv(all_data_csv)

    layer = nn.Linear(1024, 512)
    output_layer = nn.Linear(512, 4)

    all_risk_scores = np.zeros((len(val_data)))
    all_censorships = np.zeros((len(val_data)))
    all_event_times = np.zeros((len(val_data)))

    # print(len(val_data))
    for idx, data in enumerate(val_data):
        # print(idx)
        key_data = all_data_df.loc[all_data_df['case_id'] == data, ['slide_id', 'survival_months', 'censorship']].values
        slide_id = key_data[0][0]
        event_time = key_data[0][1]
        censorship = key_data[0][2]

        WSI_emb = torch.load(os.path.join(dataset_path, slide_id.replace('.svs', '.pt')), weights_only=False)
        WSI_emb_512 = layer(WSI_emb)

        with torch.no_grad():
            feature = model.visual.forward_project(WSI_emb_512)
            output = output_layer(feature)
            
        logits = torch.mean(output, dim=0)
        logits = logits.unsqueeze(0)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        
        # print(risk)
        all_risk_scores[idx] = risk[0]
        all_censorships[idx] = censorship
        all_event_times[idx] = event_time


    c_index = concordance_index_censored((1-all_censorships).astype(bool),
                                        all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print(c_index)
    # break