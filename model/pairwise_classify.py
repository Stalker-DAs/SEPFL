import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange, repeat

def get_concat_feature(x_ori, x_k, x_weight):
    x_k = torch.einsum('b k, b k d -> b d', x_weight, x_k)
    return torch.cat((x_ori, x_k), dim=1)

class pairwise_classify(nn.Module):
    def __init__(self, feature_dim, nclass, k):
        super(pairwise_classify, self).__init__()
        self.dim = feature_dim
        self.nhid_2 = int(feature_dim / 2)
        self.nhid_4 = int(feature_dim / 4)

        self.channel_bn = nn.BatchNorm1d(self.nhid_2)
        self.token_bn = nn.BatchNorm1d(self.nhid_2)
        self.attention_bn = nn.BatchNorm1d(self.nhid_2)


        self.channel_mix = nn.Sequential(nn.Linear(self.nhid_2, 2 * self.nhid_2),
                                         nn.GELU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(2 * self.nhid_2, self.nhid_2),
                                         nn.Dropout(0.2))

        self.token_mix = nn.Sequential(nn.Linear(k, 2 * k),
                                       nn.GELU(),
                                       nn.Dropout(0.2),
                                       nn.Linear(2 * k, k),
                                       nn.Dropout(0.2))

        self.attention_layer = nn.Sequential(nn.Linear(self.nhid_2, self.nhid_4),
                                             nn.GELU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(self.nhid_4, 1))

        self.classifier = nn.Sequential(nn.BatchNorm1d(feature_dim),
                                        nn.Linear(feature_dim, self.nhid_2),
                                        nn.BatchNorm1d(self.nhid_2),
                                        nn.GELU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(self.nhid_2, self.nhid_4),
                                        nn.BatchNorm1d(self.nhid_4),
                                        nn.GELU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(self.nhid_4, nclass))

        self.channel_mix.apply(self._initialize_weights)
        self.token_mix.apply(self._initialize_weights)
        self.attention_layer.apply(self._initialize_weights)
        self.classifier.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, bf, cf, bkf, ckf):
        B, N = bf.shape[0], bf.shape[1]
        k = bkf.shape[1]

        bf_copy = repeat(bf, 'b d -> b k d', k=k)
        b_fin = torch.cat((bf_copy, bkf), dim=2)

        b_fin_temp = self.channel_mix(b_fin)
        b_fin_temp = rearrange(b_fin_temp, 'B N D -> B D N')
        b_fin_temp = self.channel_bn(b_fin_temp)
        b_fin_temp = rearrange(b_fin_temp, 'B D N -> B N D')

        b_fin = b_fin + b_fin_temp

        b_fin = rearrange(b_fin, 'B N D -> B D N')

        b_fin_temp = self.token_mix(b_fin)
        b_fin_temp = self.token_bn(b_fin_temp)

        b_fin = b_fin + b_fin_temp

        b_fin = self.attention_bn(b_fin)
        b_fin = rearrange(b_fin, 'B D N -> B N D')
        b_fin = self.attention_layer(b_fin)

        b_fin = b_fin.reshape(B, k)
        bf = get_concat_feature(bf, bkf, b_fin)

        cf_copy = repeat(cf, 'b d -> b k d', k=k)
        c_fin = torch.cat((cf_copy, ckf), dim=2)

        c_fin_temp = self.channel_mix(c_fin)
        c_fin_temp = rearrange(c_fin_temp, 'B N D -> B D N')
        c_fin_temp = self.channel_bn(c_fin_temp)
        c_fin_temp = rearrange(c_fin_temp, 'B D N -> B N D')

        c_fin = c_fin + c_fin_temp

        c_fin = rearrange(c_fin, 'B N D -> B D N')

        c_fin_temp = self.token_mix(c_fin)
        c_fin_temp = self.token_bn(c_fin_temp)

        c_fin = c_fin + c_fin_temp

        c_fin = self.attention_bn(c_fin)
        c_fin = rearrange(c_fin, 'B D N -> B N D')
        c_fin = self.attention_layer(c_fin)

        c_fin = c_fin.reshape(B, k)
        cf = get_concat_feature(cf, ckf, c_fin)

        x = torch.cat((bf, cf), dim=-1).cuda()

        x = self.classifier(x)

        return x


