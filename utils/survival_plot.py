import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt

from models.cmta.network import CMTA


def plot_km(model, dataset):
    model = model.cuda()
    data = {'true_time':[], 'true_event':[], 'pred_time':[]}
    csv_data = pd.read_csv(f'csv/tcga_{dataset.lower()}_all_clean.csv')

    splits_path = f'splits/5foldcv/{dataset}'
    for fold in os.listdir(splits_path):
        fold_val = [val for val in pd.read_csv(os.path.join(splits_path, fold))['val']]

        checkpoint = torch.load('/home/lichangyong/code/PTCMA/results/blca/[cmta]-[concat]-[0.5]-[2024-11-12]-[15-00-03]/fold_0/model_best_0.5670_2.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])

        for index, row in csv_data.iterrows():
            if row['case_id'] not in fold_val:
                continue
            data['true_time'].append(row['survival_months'])
            data['true_event'].append(row['censorship'])
            
            slide = row['slide_id'].replace('.svs', '.pt')
            case_id = row['case_id']
            path_emb = torch.load(f'/data/lichangyong/TCGA_FEATURE/{dataset}/pt_files_by_case/{slide}')
            text_emb = torch.load(f'/data/lichangyong/TCGA_FEATURE/{dataset}/text_emb_by_case/{case_id}.pt')
            pred_time = model(x_path=path_emb.cuda(), x_text=text_emb.cuda())[0]
            data['pred_time'].append(pred_time)

        df = pd.DataFrame(data.cpu())

        # 将病例按预测生存时间分为两组
        df['risk_group'] = pd.qcut(df['pred_time'], q=2, labels=['Low Risk', 'High Risk'])

        # 创建 Kaplan-Meier 分析器
        kmf = KaplanMeierFitter()

        # 绘制分组 KM 曲线
        plt.figure(figsize=(8, 6))
        for group in df['risk_group'].unique():
            group_data = df[df['risk_group'] == group]
            kmf.fit(group_data['true_time'], event_observed=group_data['true_event'], label=group)
            kmf.plot_survival_function()

        # 设置图例和标题
        plt.title('Kaplan-Meier Survival Curve by Risk Group', fontsize=16)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Survival Probability', fontsize=14)
        plt.legend(title='Risk Groups')
        plt.grid(alpha=0.4)
        plt.savefig('KM.png')

        # 可选：Log-Rank 检验，比较风险组间差异
        group_low = df[df['risk_group'] == 'Low Risk']
        group_high = df[df['risk_group'] == 'High Risk']

        results = logrank_test(
            group_low['true_time'], group_high['true_time'],
            event_observed_A=group_low['true_event'],
            event_observed_B=group_high['true_event']
        )
        print(f"Log-Rank Test P-Value: {results.p_value}")



if __name__ == '__main__':
    model_dict = {"n_classes":4, "fusion":"concat", "model_size":"small"}
    model = CMTA(**model_dict)
    plot_km(model, 'BLCA')







# # 示例数据
# data = {
#     'true_time': [5, 6, 6, 2, 4, 4, 7, 8, 10, 10],  # 真实生存时间
#     'true_event': [1, 0, 1, 1, 1, 0, 1, 0, 1, 1],  # 真实事件指示符 (1:事件, 0:截尾)
#     'pred_time': [4.2, 6.5, 5.8, 3.1, 4.3, 4.0, 6.9, 7.5, 9.1, 9.8],  # 模型预测生存时间
# }


