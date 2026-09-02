import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqChannelSoftCluster(nn.Module):
    def __init__(self, C, K=20, use_phase=True):
        super().__init__()
        self.C = C
        self.K = K
        self.S = nn.Parameter(torch.randn(C, K) * 0.1)
        self.W_K = nn.Parameter(torch.eye(K))
        self.ln = nn.LayerNorm(C)
        self.use_phase = use_phase


    def forward(self, X):
        X = X.float()
        B, T, C = X.shape

        # ===== 1) Frequency domain transformation =====
        X_f = torch.fft.rfft(X, dim=1)
        X_phase = torch.angle(X_f)
        X_feat = torch.abs(X_f)


        X_feat_max = X_feat.max(dim=1, keepdim=True)[0] + 1e-8
        X_feat_norm = X_feat / X_feat_max  # 归一化到 [0, 1]

        # ===== 2) soft clustering =====
        S_soft = F.softmax(self.S, dim=-1)


        if torch.isinf(X_feat_norm).any() or torch.isnan(X_feat_norm).any():
            X_feat_norm = torch.nan_to_num(X_feat_norm, nan=0.0, posinf=1e6, neginf=-1e6)
            X_feat_norm = torch.clamp(X_feat_norm, 0, 1e6)

        Z = torch.einsum("bfc,ck->bfk", X_feat_norm, S_soft)


        if torch.isinf(Z).any() or torch.isnan(Z).any():
            Z = torch.nan_to_num(Z, nan=0.0, posinf=1e6, neginf=-1e6)
            Z = torch.clamp(Z, 0, 1e6)

        Z_mix = torch.einsum("bfk,km->bfm", Z, self.W_K)

        X_hat_f_mag = torch.einsum("bfm,ck->bfc", Z_mix, S_soft)
        X_hat_f_mag = X_hat_f_mag * X_feat_max
        # ===== 3) Reconstruct complex spectrum =====
        X_hat_f = X_hat_f_mag * torch.exp(1j * X_phase)



        # ===== 4) Inverse Transform (using double precision)=====
        X_hat = torch.fft.irfft(X_hat_f.double(), n=T, dim=1).float()

        X_hat = torch.nan_to_num(X_hat, nan=0.0, posinf=1e6, neginf=-1e6)
        X_hat = torch.clamp(X_hat, -1e6, 1e6)

        # ===== 5) Residual+LN=====
        return self.ln(X + X_hat)

class Muti_Period_Learner(nn.Module):
    def __init__(self, configs):
        super(Muti_Period_Learner, self).__init__()
        # Basic Parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.model_type = configs.model_type
        self.dropout_rate = configs.dropout

        assert self.model_type in ["linear", "mlp"]

        self.periods = configs.period_lens

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=2 * (p // 2) + 1,
                stride=1,
                padding=p // 2,
                bias=False,
            ) for p in self.periods
        ])


        # Backbone
        self.backbones = nn.ModuleList()
        self.seg_nums_x, self.seg_nums_y = [], []
        valid_periods = []
        for p in self.periods:
            if self.seq_len % p != 0 or self.pred_len % p != 0:
                continue

            seg_x = self.seq_len // p
            seg_y = self.pred_len // p
            self.seg_nums_x.append(seg_x)
            self.seg_nums_y.append(seg_y)
            valid_periods.append(p)

            if self.model_type == "linear":
                 self.backbones.append(nn.Linear(seg_x, seg_y, bias=False))
            else:
                self.backbones.append(nn.Sequential(
                     nn.Linear(seg_x, self.d_model),
                     nn.ReLU(),
                     nn.Dropout(self.dropout_rate),
                     nn.Linear(self.d_model, seg_y),
                  ))




    def forward(self, x):
        batch_size = x.shape[0]

        preds_multi = []
        intermediates = {}

        for p, conv, seg_x, seg_y, backbone in zip(
                self.periods, self.convs, self.seg_nums_x, self.seg_nums_y, self.backbones
        ):

            x_conv = conv(x.reshape(-1, 1, self.seq_len)).reshape(batch_size, self.enc_in, self.seq_len) + x

            x_ds = x_conv.reshape(batch_size * self.enc_in, seg_x, p).permute(0, 2, 1)

            y_full = backbone(x_ds)  # (b*c, p, seg_y)
            y_full = y_full.permute(0, 2, 1).reshape(batch_size, self.enc_in, seg_y * p)
            y_full = y_full.permute(0, 2, 1)

            # === Dynamic segmentation processing ===
            total_len = seg_y * p
            if total_len < self.pred_len:
                residual = self.pred_len - total_len

                # Extract the last residual part of the input
                x_res = x_conv[:, :, -p:]
                x_res = x_res.reshape(batch_size * self.enc_in, 1, p).squeeze(1)  # (b*c, p)
                proj = nn.Linear(p, residual).to(x.device)
                y_res = proj(x_res)  # (b*c, residual)

                # reshape -> (b, residual, c)
                y_res = y_res.reshape(batch_size, self.enc_in, residual).permute(0, 2, 1)

                y = torch.cat([y_full, y_res], dim=1)
            else:
                y = y_full[:, :self.pred_len, :]

            preds_multi.append(y)

        # === Multi periods fusion===
        y_stack = torch.stack(preds_multi, dim=0)  # (num_periods, B, T, C)
        y1 = torch.mean(y_stack, dim=0)


        return y1




class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.channel_cluster = FreqChannelSoftCluster(
            C=configs.enc_in,
            K=configs.cluster_k)


        self.multi_periods = Muti_Period_Learner(configs)

        self.linear_head = nn.Sequential(nn.Linear(configs.seq_len, configs.pred_len), nn.Dropout(configs.dropout))

        self.fuse_alpha = nn.Parameter(torch.zeros(configs.enc_in))



    def forward(self, x):

        # Normalize
        seq_mean = torch.mean(x, dim=1, keepdim=True)
        x_norm = (x - seq_mean).permute(0, 2, 1)  # (B, C, T)

        # ===== 1) Frequency Soft Clustering =====
        x_cluster = self.channel_cluster(x_norm.permute(0, 2, 1))  # (B, T, C)
        y1 = self.linear_head(x_cluster.permute(0, 2, 1)).permute(0, 2, 1)

        # ===== 2) Multi-Period Basis Module =====
        y2 = self.multi_periods(x_norm)

        # ===== 3) fusion =====
        alpha = torch.sigmoid(self.fuse_alpha).view(1, 1, -1)
        y = y1 * (1 - alpha) + y2 * alpha


        return y + seq_mean



