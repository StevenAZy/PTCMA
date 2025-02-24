import numpy as np

import torch
import torch.nn as nn

from .util import initialize_weights
from .util import NystromAttention
from .util import BilinearFusion
from .util import MultiheadAttention


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,  # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,  # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class Modality_Encoder(nn.Module):
    def __init__(self, feature_dim=1024):
        super(Modality_Encoder, self).__init__()
        
        self.pos_layer = PPEG(dim=feature_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features):
        # ---->pad
        H = features.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([features, features[:, :add_length, :]], dim=1)  # [B, N, 512]
        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)
        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]
        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]
        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]
        # ---->cls_token
        h = self.norm(h)
        return h[:, 0], h[:, 1:]
 

class PTCMA(nn.Module):
    def __init__(self, n_classes=4, fusion="concat", model_size="small"):
        super(PTCMA, self).__init__()

        self.n_classes = n_classes
        self.fusion = fusion

        ###
        self.size_dict = {
            "pathomics": {"small": [1024, 256, 256], "large": [1024, 512, 256]},
            "text": {"small": [768, 256], "large": [768, 1024, 1024, 256]},
        }
        # Pathomics Embedding Network
        hidden = self.size_dict["pathomics"][model_size]
        fc = []
        for idx in range(len(hidden) - 1):
            fc.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(0.25))
        self.pathomics_fc = nn.Sequential(*fc)

        # Text Embedding Network
        hidden = self.size_dict["text"][model_size]
        fc = []
        for idx in range(len(hidden) - 1 ):
            fc.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(0.25))
        self.text_fc = nn.Sequential(*fc)


        # Pathomics Transformer
        # Encoder
        self.pathomics_encoder = Modality_Encoder(feature_dim=hidden[-1])
        # Decoder
        self.pathomics_decoder = Modality_Encoder(feature_dim=hidden[-1])

        # P->G Attention
        self.P_in_T_Att = MultiheadAttention(embed_dim=256, num_heads=1)
        # G->P Attention
        self.T_in_P_Att = MultiheadAttention(embed_dim=256, num_heads=1)

        # Pathomics Transformer Decoder
        # Encoder
        self.text_encoder = Modality_Encoder(feature_dim=hidden[-1])
        # Decoder
        self.text_decoder = Modality_Encoder(feature_dim=hidden[-1])

        # Classification Layer
        if self.fusion == "concat":
            self.mm = nn.Sequential(
                *[nn.Linear(hidden[-1] * 2, hidden[-1]), nn.ReLU(), nn.Linear(hidden[-1], hidden[-1]), nn.ReLU()]
            )
        elif self.fusion == "bilinear":
            self.mm = BilinearFusion(dim1=hidden[-1], dim2=hidden[-1], scale_dim1=8, scale_dim2=8, mmhid=hidden[-1])
        else:
            raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))

        self.classifier = nn.Linear(hidden[-1], self.n_classes)

        self.apply(initialize_weights)

    def forward(self, **kwargs):
        # meta text and pathomics features
        x_path = kwargs["x_path"]
        x_text = kwargs["x_text"]

        # Enbedding
        # pathomics embedding
        pathomics_features = self.pathomics_fc(x_path).unsqueeze(0)
        # text embedding
        text_features = self.text_fc(x_text).unsqueeze(0)

        # encoder
        # pathomics encoder
        cls_token_pathomics_encoder, patch_token_pathomics_encoder = self.pathomics_encoder(
            pathomics_features)  # cls token + patch tokens
        # text encoder
        cls_token_text_encoder, patch_token_text_encoder = self.text_encoder(
            text_features)  # cls token + patch tokens

        # cross-text attention
        pathomics_in_text, Att = self.P_in_T_Att(
            patch_token_pathomics_encoder.transpose(1, 0),
            patch_token_text_encoder.transpose(1, 0),
            patch_token_text_encoder.transpose(1, 0),
        )  # ([14642, 1, 256])
        text_in_pathomics, Att = self.T_in_P_Att(
            patch_token_text_encoder.transpose(1, 0),
            patch_token_pathomics_encoder.transpose(1, 0),
            patch_token_pathomics_encoder.transpose(1, 0),
        )  # ([7, 1, 256])
        # decoder
        # pathomics decoder
        cls_token_pathomics_decoder, _ = self.pathomics_decoder(
            pathomics_in_text.transpose(1, 0))  # cls token + patch tokens
        # text decoder
        cls_token_text_decoder, _ = self.text_decoder(
            text_in_pathomics.transpose(1, 0))  # cls token + patch tokens

        # fusion
        if self.fusion == "concat":
            fusion = self.mm(
                torch.concat(
                    (
                        (cls_token_pathomics_encoder + cls_token_pathomics_decoder) / 2,
                        (cls_token_text_encoder + cls_token_text_decoder) / 2,
                    ),
                    dim=1,
                )
            )  # take cls token to make prediction
        elif self.fusion == "bilinear":
            fusion = self.mm(
                (cls_token_pathomics_encoder + cls_token_pathomics_decoder) / 2,
                (cls_token_text_encoder + cls_token_text_decoder) / 2,
            )  # take cls token to make prediction
        else:
            raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))

        # predict
        logits = self.classifier(fusion)  # [1, n_classes]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S, cls_token_pathomics_encoder, cls_token_pathomics_decoder, cls_token_text_encoder, cls_token_text_decoder
