import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
# from torch.nn.modules.activation import MultiheadAttention

import random
from einops import rearrange
import os
import torch
from models.base.clip_fsar import load, tokenize
from utils import getcombinations
from .tcn import TemporalConvNet

'''
把self-loss换成所有子序列
'''


class PreNormattention_qkv(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        return self.fn(self.norm(q), self.norm(k), self.norm(v), **kwargs) + q


class Transformer_v1(nn.Module):
    def __init__(self, heads=8, dim=2048, dim_head_k=256, dim_head_v=256, dropout_atte=0.05, mlp_dim=2048,
                 dropout_ffn=0.05, depth=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([  # PreNormattention(2048, Attention(2048, heads = 8, dim_head = 256, dropout = 0.2))
                    # PreNormattention(heads, dim, dim_head_k, dim_head_v, dropout=dropout_atte),
                    PreNormattention_qkv(dim,
                                         Attention_qkv(dim, heads=heads, dim_head=dim_head_k, dropout=dropout_atte)),
                    FeedForward(dim, mlp_dim, dropout=dropout_ffn),
                ]))

    def forward(self, q, k, v):
        # if self.depth
        for attn, ff in self.layers[:1]:
            x = attn(q, k, v)
            x = ff(x) + x
        if self.depth > 1:
            for attn, ff in self.layers[1:]:
                x = attn(x, x, x)
                x = ff(x) + x
        return x


class Transformer_v2(nn.Module):
    def __init__(self, heads=8, dim=2048, dim_head_k=256, dim_head_v=256, dropout_atte=0.05, mlp_dim=2048,
                 dropout_ffn=0.05, depth=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([  # PreNormattention(2048, Attention(2048, heads = 8, dim_head = 256, dropout = 0.2))
                    # PreNormattention(heads, dim, dim_head_k, dim_head_v, dropout=dropout_atte),
                    PreNormattention(dim, Attention(dim, heads=heads, dim_head=dim_head_k, dropout=dropout_atte)),
                    FeedForward(dim, mlp_dim, dropout=dropout_ffn),
                ]))

    def forward(self, x):
        # if self.depth
        for attn, ff in self.layers[:1]:
            x = attn(x)
            x = ff(x) + x
        if self.depth > 1:
            for attn, ff in self.layers[1:]:
                x = attn(x)
                x = ff(x) + x
        return x


class Transformer_v3(nn.Module):
    def __init__(self, heads=8, dim=2048, dim_head_k=256, dim_head_v=256, dropout_atte=0.05, mlp_dim=2048,
                 dropout_ffn=0.05, depth=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([  # PreNormattention(2048, Attention(2048, heads = 8, dim_head = 256, dropout = 0.2))
                    # PreNormattention(heads, dim, dim_head_k, dim_head_v, dropout=dropout_atte),
                    PreNormattention_qkv(dim,
                                         Attention_v3(dim, heads=heads, dim_head=dim_head_k, dropout=dropout_atte)),
                    FeedForward(dim, mlp_dim, dropout=dropout_ffn),
                ]))

    def forward(self, q, k, v):
        # if self.depth
        for attn, ff in self.layers[:1]:
            x = attn(q, k, v)
            x = ff(x) + x
        if self.depth > 1:
            for attn, ff in self.layers[1:]:
                x = attn(x, x, x)
                x = ff(x) + x
        return x


class PreNormattention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x


class Attention_qkv(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads
        bk = k.shape[0]
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', b=bk, h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', b=bk, h=h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)  # [30, 8, 8, 5]

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Attention_v3(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads
        bk = k.shape[0]
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b  i d, b  j d -> b  i j', q, k) * self.scale

        attn = self.attend(dots)  # [30, 8, 8, 5]

        out = einsum('b  i j, b  j d -> b  i d', attn, v)

        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# PostNormattention似乎没有用
class PostNormattention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs) + x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


####################################################################

def cos_sim(x, y, epsilon=0.01):
    """
    Calculates the cosine similarity between the last dimension of two tensors.
    """
    numerator = torch.matmul(x, y.transpose(-1, -2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1, -2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists


def extract_class_indices(labels, which_class):
    """
    Helper method to extract the indices of elements which have the specified label.
    :param labels: (torch.tensor) Labels of the context set.
    :param which_class: Label for which indices are extracted.
    :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
    """
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask, as_tuple=False)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


def OTAM_cum_dist(dists, lbda=0.1):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len]
    TODO: clearn up if possible - currently messy to work with pt1.8. Possibly due to stack operation?
    """
    dists = F.pad(dists, (1, 1), 'constant', 0)  # [25, 25, 8, 10]

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        # cum_dists[:,:,0,m] = dists[:,:,0,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,0,m-1]))
        # paper does continuous relaxation of the cum_dists entry, but it trains faster without, so using the simpler version for now:
        cum_dists[:, :, 0, m] = dists[:, :, 0, m] + cum_dists[:, :, 0, m - 1]

        # remaining rows
    for l in range(1, dists.shape[2]):
        # first non-zero column
        cum_dists[:, :, l, 1] = dists[:, :, l, 1] - lbda * torch.log(
            torch.exp(- cum_dists[:, :, l - 1, 0] / lbda) + torch.exp(- cum_dists[:, :, l - 1, 1] / lbda) + torch.exp(
                - cum_dists[:, :, l, 0] / lbda))

        # middle columns
        for m in range(2, dists.shape[3] - 1):
            cum_dists[:, :, l, m] = dists[:, :, l, m] - lbda * torch.log(
                torch.exp(- cum_dists[:, :, l - 1, m - 1] / lbda) + torch.exp(- cum_dists[:, :, l, m - 1] / lbda))

        # last column
        # cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
        cum_dists[:, :, l, -1] = dists[:, :, l, -1] - lbda * torch.log(
            torch.exp(- cum_dists[:, :, l - 1, -2] / lbda) + torch.exp(- cum_dists[:, :, l - 1, -1] / lbda) + torch.exp(
                - cum_dists[:, :, l, -2] / lbda))

    return cum_dists[:, :, -1, -1]


def OTAM_cum_dist_v2(dists, lbda=0.5):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len]
    TODO: clearn up if possible - currently messy to work with pt1.8. Possibly due to stack operation?
    """
    dists = F.pad(dists, (1, 1), 'constant', 0)  # [25, 25, 8, 10]

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        # cum_dists[:,:,0,m] = dists[:,:,0,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,0,m-1]))
        # paper does continuous relaxation of the cum_dists entry, but it trains faster without, so using the simpler version for now:
        cum_dists[:, :, 0, m] = dists[:, :, 0, m] + cum_dists[:, :, 0, m - 1]

        # remaining rows
    for l in range(1, dists.shape[2]):
        # first non-zero column
        cum_dists[:, :, l, 1] = dists[:, :, l, 1] - lbda * torch.log(
            torch.exp(- cum_dists[:, :, l - 1, 0] / lbda) + torch.exp(- cum_dists[:, :, l - 1, 1] / lbda) + torch.exp(
                - cum_dists[:, :, l, 0] / lbda))

        # middle columns
        for m in range(2, dists.shape[3] - 1):
            cum_dists[:, :, l, m] = dists[:, :, l, m] - lbda * torch.log(
                torch.exp(- cum_dists[:, :, l - 1, m - 1] / lbda) + torch.exp(- cum_dists[:, :, l, m - 1] / lbda))

        # last column
        # cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
        cum_dists[:, :, l, -1] = dists[:, :, l, -1] - lbda * torch.log(
            torch.exp(- cum_dists[:, :, l - 1, -2] / lbda) + torch.exp(- cum_dists[:, :, l - 1, -1] / lbda) + torch.exp(
                - cum_dists[:, :, l, -2] / lbda))

    return cum_dists[:, :, -1, -1]


class MDMF_CLIPFSAR(nn.Module):
    """
    OTAM with a CNN backbone.
    """

    def __init__(self, args, cfg):
        super(MDMF_CLIPFSAR, self).__init__()
        self.argss = args
        self.args = cfg
        if cfg.VIDEO.HEAD.BACKBONE_NAME == "RN50":
            clip_backbone, self.preprocess = load(cfg.VIDEO.HEAD.BACKBONE_NAME, device="cuda", cfg=cfg,
                                                  jit=False)  # ViT-B/16
            self.backbone = clip_backbone.visual  # model.load_state_dict(state_dict)
            self.class_real_train = cfg.TRAIN.CLASS_NAME  # 所有得训练样本 标签名称
            self.class_real_test = cfg.TEST.CLASS_NAME  # 所有得测试样本 标签名称
            self.mid_dim = 1024
        elif cfg.VIDEO.HEAD.BACKBONE_NAME == "ViT-B/16":
            # 将clip模型加载了进来
            clip_backbone, self.preprocess = load(cfg.VIDEO.HEAD.BACKBONE_NAME, device="cuda", cfg=cfg, jit=False)  # ViT-B/16
            self.backbone = clip_backbone.visual  # model.load_state_dict(state_dict)
            self.class_real_train = cfg.TRAIN.CLASS_NAME
            self.class_real_test = cfg.TEST.CLASS_NAME
            # backbone, self.preprocess = load("RN50", device="cuda", cfg=cfg, jit=False)
            # self.backbone = backbone.visual model.load_state_dict(state_dict)
            # self.backbone = CLIP
            self.mid_dim = 512
        with torch.no_grad():
            if hasattr(self.args.TEST, "PROMPT") and self.args.TEST.PROMPT:
                text_templete = [self.args.TEST.PROMPT.format(self.class_real_train[int(ii)]) for ii in
                                 range(len(self.class_real_train))]
            else:
                text_templete = ["a photo of {}".format(self.class_real_train[int(ii)]) for ii in
                                 range(len(self.class_real_train))]

            text_templete = tokenize(text_templete).cuda()
            self.text_features_train = clip_backbone.encode_text(text_templete)  # 对文本进行编码，数量就是support 中类别数量 例如 kinetics就是分别对64个单词进行编码

            if hasattr(self.args.TEST, "PROMPT") and self.args.TEST.PROMPT:
                text_templete = [self.args.TEST.PROMPT.format(self.class_real_test[int(ii)]) for ii in
                                 range(len(self.class_real_test))]
            else:
                text_templete = ["a photo of {}".format(self.class_real_test[int(ii)]) for ii in
                                 range(len(self.class_real_test))]

            text_templete = tokenize(text_templete).cuda()

            self.text_features_test = clip_backbone.encode_text(text_templete)

        self.mid_layer = nn.Sequential()
        self.classification_layer = nn.Sequential()
        self.scale = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.scale.data.fill_(1.0)

        if hasattr(self.args.TRAIN, "TRANSFORMER_DEPTH") and self.args.TRAIN.TRANSFORMER_DEPTH:
            self.context2 = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2,
                                           depth=int(self.args.TRAIN.TRANSFORMER_DEPTH))

            self.context_local = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2, depth=int(self.args.TRAIN.TRANSFORMER_DEPTH))
        else:
            self.context2 = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2)

            self.context_local = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2)

        self.context_conv1 = nn.Conv1d(self.mid_dim // 1, self.mid_dim // 1, kernel_size=3, padding=1, groups=1)
        self.context_conv2 = nn.Conv1d(self.mid_dim // 1, self.mid_dim // 1, kernel_size=3, padding=1, groups=1)

        self.context_conv11 = nn.Conv1d(self.mid_dim // 1, self.mid_dim // 1, kernel_size=3, padding=1, groups=1)
        self.context_conv22 = nn.Conv1d(self.mid_dim // 1, self.mid_dim // 1, kernel_size=3, padding=1, groups=1)
        # set_trace()
        self.distillation = 0
        self.tcn = TemporalConvNet(self.mid_dim, [self.mid_dim, self.mid_dim, self.mid_dim], 3, dropout=0.1)

    '''
    @class_real_train: train的类别标签
    @support_real_class： 一次episode中train N way K shot个 样本 属于 标签序列的 索引， 例如 25个样本，每个样本是第几个标号
    '''

    def get_context_feats(self, support_features, target_features):
        support_features_conv = self.context_conv1(support_features.permute(0, 2, 1))
        target_features_conv = self.context_conv1(target_features.permute(0, 2, 1))
        support_features_conv = self.context_conv2(support_features_conv)
        target_features_conv = self.context_conv2(target_features_conv)
        return support_features_conv.permute(0, 2, 1), target_features_conv.permute(0, 2, 1)

    def get_context_feats_g(self, support_features, target_features):
        support_context_g = self.tcn(support_features.transpose(1, 2)).transpose(1, 2)
        target_context_g = self.tcn(target_features.transpose(1, 2)).transpose(1, 2)

        support_context_g = support_context_g[:, -1, :].unsqueeze(1).repeat(1, 8, 1) + support_features
        target_context_g = target_context_g[:, -1, :].unsqueeze(1).repeat(1, 8, 1) + target_features
        return support_context_g, target_context_g

    def get_context_feats2(self, support_features, target_features):
        support_features_conv = self.context_conv11(support_features.permute(0, 2, 1))
        target_features_conv = self.context_conv11(target_features.permute(0, 2, 1))
        support_features_conv = self.context_conv22(support_features_conv)
        target_features_conv = self.context_conv22(target_features_conv)
        return support_features_conv.permute(0, 2, 1), target_features_conv.permute(0, 2, 1)

    def get_feats(self, support_images, target_images, support_real_class=False, support_labels=False):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        if self.training:
            support_features = self.backbone(support_images).squeeze()  # self.backbone 为 clip的视觉分支visual ModifeidResnet 输出维度 [(way*shot*frames_number),1024]
            target_features = self.backbone(target_images).squeeze()

            dim = int(support_features.shape[1])

            support_features = support_features.reshape(-1, self.argss.seq_len, dim)  # 输出维度 [(way*shot), frame_numbers,1024]
            target_features = target_features.reshape(-1, self.argss.seq_len, dim)  # 输出维度 [(way*shot), frame_numbers,1024]
            support_features_text = None

        else:
            support_features = self.backbone(support_images).squeeze()
            target_features = self.backbone(target_images).squeeze()
            dim = int(target_features.shape[1])
            support_features = support_features.reshape(-1, self.argss.seq_len, dim)  # 输出维度 [(way*shot), frame_numbers,1024]
            target_features = target_features.reshape(-1, self.argss.seq_len, dim)  # 输出维度 [(way*shot), frame_numbers,1024]
            support_features_text = self.text_features_test[support_real_class.long()]  # self.text_features_test 为clip的 文本分支 backbone.encode_text

        return support_features, target_features, support_features_text

    def fliter_support_feature(self, context_support, support_features, topk=4):
        _, fliter_index = cos_sim(context_support, support_features).topk(k=topk, dim=-1)
        fliter_index = fliter_index.squeeze()
        fliter_index, _ = torch.sort(fliter_index, dim=-1)
        fliter_frame = [torch.index_select(each, 0, fliter_index[index]) for index, each in enumerate(support_features)]
        return torch.stack(fliter_frame)

    def fliter_target_feature(self, context_support, target_features, support_labels, topk=4):
        bs, _, _ = target_features.shape
        unique_labels = torch.unique(support_labels)
        context_support = [
            torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)), dim=0)
            for c in unique_labels]
        context_support = torch.stack(context_support).squeeze().repeat(bs, 1, 1)  # 25 5  1024

        fliter_value, fliter_index = cos_sim(context_support, target_features).topk(k=topk, dim=-1)  # 25 5 8 --->topk   25 5 4
        # 求出最小值  所有值减去最小值
        value = torch.mean(fliter_value, dim=-1)  # 25 5
        # 根据因子获取context
        factors = F.softmax(value * 200, dim=-1)

        list_indices = [random.choices([i for i in range(self.argss.way)], factor)[0] for factor in factors]

        list_indices = torch.Tensor(list_indices).cuda()

        _, indices = torch.mean(fliter_value, dim=-1).max(-1)

        fliter_index, _ = torch.sort(fliter_index, dim=-1)
        fliter_frame = [torch.index_select(each, 0, fliter_index[index][indices[index]]) for index, each in enumerate(target_features)]
        return torch.stack(fliter_frame), indices, list_indices

    def forward(self, inputs):  # 获得support support labels, query, support real class
        support_images, support_labels, target_images, support_real_class = inputs['support_set'], inputs['support_labels'], inputs['target_set'], inputs['real_support_labels']  # [200, 3, 224, 224] inputs["real_support_labels"]

        # set_trace()
        # 一个episode中所有的样本对应的文本语句特征
        if self.training:  # 取5个support样本类型对应的样本标签编码，输出维度为5 1 1024
            context_support = self.text_features_train[support_real_class.long()].unsqueeze(1)  # .repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1)
        else:
            context_support = self.text_features_test[support_real_class.long()].unsqueeze(1)  # .repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1) # .repeat(support_bs+target_bs, 1, 1)
        support_features, target_features, text_features = self.get_feats(support_images, target_images, support_real_class)

        # 通过语义信息过滤support视频帧，取出最大topk=4帧
        support_features_fliter = self.fliter_support_feature(context_support, support_features)

        # 通过语义信息过滤target视频帧，取出最大topk=4帧，同时得到target大概率指向的文本indices
        target_features_fliter, target_indices, factors = self.fliter_target_feature(context_support, target_features, support_labels)

        unique_labels = torch.unique(support_labels)

        # 常规图文匹配，将过滤的视频文本匹配
        class_text_logits = self.video2imagetext_adapter_mean(support_features_fliter, target_features_fliter)

        # transform通过对target、context混合得到新特征，后两个为context头特征
        soft_target_context = self.create_soft_context(support_labels, context_support, factors)
        is_pps = self.argss.is_pps
        if not is_pps:
            soft_target_context = torch.zeros_like(soft_target_context).cuda()

        support_features_context, target_features_context = self.get_context_feats(support_features, target_features)
        support_features_up, target_features_up = self.text_eh_temporal_transformer(context_support,
                                                                                    soft_target_context,
                                                                                    support_features,
                                                                                    target_features,
                                                                                    support_features_context,
                                                                                    target_features_context,
                                                                                    support_labels,
                                                                                    0
                                                                                    )

        support_features_up, target_features_up = support_features_up[:, 1:, :], target_features_up[:, 1:, :]
        support_features_up_g, target_features_up_g = support_features_up[:, 0, :], target_features_up[:, 0, :]

        support_features_context2, target_features_context2 = self.get_context_feats_g(support_features, target_features)
        support_features_down, target_features_down = self.text_eh_temporal_transformer(context_support,
                                                                                        soft_target_context,
                                                                                        support_features,
                                                                                        target_features,
                                                                                        support_features_context2,
                                                                                        target_features_context2,
                                                                                        support_labels,
                                                                                        1
                                                                                        )

        support_features_down, target_features_down = support_features_down[:, 1:, :], target_features_down[:, 1:, :]
        support_features_down_g, target_features_down_g = support_features_down[:, 0, :], target_features_down[:, 0, :]

        if not self.training:
            # 测试过程中 target和文本context归一化后，softmax得到比对值
            unique_labels = torch.unique(support_labels)
            support_features_s = [
                torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)), dim=0)
                for c in unique_labels]
            support_features_s = torch.stack(support_features_s)
            support_features_s = support_features_up.mean(1)
            # unique_labels = torch.unique(support_labels)
            image_features = self.classification_layer(target_features_down.mean(1))
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = support_features_up_g / support_features_s.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logit_scale = self.scale  # 1. # self.backbone.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_image = F.softmax(logits_per_image, dim=1)
            # logits_per_image = cos_sim(image_features,text_features)

        sim_up = cos_sim(target_features_up_g, support_features_up_g)
        sim_down = cos_sim(target_features_down_g, support_features_down_g)
        sim_up_s = F.softmax(sim_up, dim=-1)
        sim_down_s = F.softmax(sim_down, dim=-1)
        tu, _ = torch.max(sim_up_s, dim=-1)
        td, _ = torch.max(sim_down_s, dim=-1)
        t_judgements = tu >= td

        # 图像特征比对
        cum_dist_up = self.otam_distance(support_features_up, target_features_up)
        cum_dist_down = self.otam_distance(support_features_down, target_features_down)
        # 全部otam得到loss
        # cum_dists = 1*(-cum_dist_up) + 1*(-cum_dist_down)
        # 混合获取
        # if not self.training:
        #     cum_dists = -self.fusion_textimage_distance(-cum_dists, logits_per_image)

        cum_up_s = F.softmax(-cum_dist_up, dim=-1)
        cum_down_s = F.softmax(-cum_dist_down, dim=-1)

        vu, _ = torch.max(cum_up_s, dim=-1)
        vd, _ = torch.max(cum_down_s, dim=-1)

        judgements = vu >= vd
        '''
        for i, v in enumerate(judgements):
            if v:
                pass
            else:
                cum_dist_up[i], cum_dist_down[i] = cum_dist_down[i], cum_dist_up[i]
       '''
        cv_up_reli = []
        cv_down_reli = []
        ct_up_reli = []
        ct_down_reli = []
        cum_dist_reli_up = []
        cum_dist_unreli_down = []
        cum_dist_unreli_up = []
        cum_dist_reli_down = []
        up_reli_num = 0
        down_reli_num = 0

        # 代码有些冗余，做消融用，只需要修改t_compare和v_compare 防止每次试验代码改动太多
        t_compare = self.argss.dist_condition[0]
        v_compare = self.argss.dist_condition[1]

        #####------内部函数-----#########
        def dist_condition_up(v, t):
            if t_compare and v_compare:
                return v and t
            elif t_compare:
                return t
            elif v_compare:
                return v

        def dist_condition_down(v, t):
            if t_compare and v_compare:
                return not v and not t
            elif t_compare:
                return not t
            elif v_compare:
                return not v

        for i, (v, t) in enumerate(zip(judgements, t_judgements)):
            # up是reliable的
            if dist_condition_up(v, t):
                cum_dist_reli_up.append(cum_dist_up[i])
                cum_dist_unreli_down.append(cum_dist_down[i])
                # 上层分支可靠性系数
                cv_up_reli.append(vu[i])
                ct_up_reli.append(tu[i])
                up_reli_num = up_reli_num + 1
            # donw 是reliable的
            elif dist_condition_down(v, t):
                cum_dist_reli_down.append(cum_dist_down[i])
                cum_dist_unreli_up.append(cum_dist_up[i])
                # 下层分支可靠性系数
                cv_down_reli.append(vd[i])
                ct_down_reli.append(td[i])
                down_reli_num = down_reli_num + 1
            else:
                pass

        down_reliable_loss = 0
        up_reliable_loss = 0
        if up_reli_num > 0:
            cum_dist_reli_up = torch.stack(cum_dist_reli_up, dim=0)
            cum_dist_unreli_down = torch.stack(cum_dist_unreli_down, dim=0)
            cv_up_reli = torch.stack(cv_up_reli, dim=0)
            ct_up_reli = torch.stack(ct_up_reli, dim=0)

            up_reliable_loss = F.kl_div((-cum_dist_unreli_down).softmax(-1).log(), (-cum_dist_reli_up).softmax(-1), reduction='none')
            up_reliable_loss = torch.sum(up_reliable_loss, dim=1)
            c_up_reli = cv_up_reli + ct_up_reli
            up_reliable_loss = torch.matmul(up_reliable_loss, c_up_reli) / torch.sum(c_up_reli)

        if down_reli_num > 0:
            cum_dist_reli_down = torch.stack(cum_dist_reli_down, dim=0)
            cum_dist_unreli_up = torch.stack(cum_dist_unreli_up, dim=0)
            cv_down_reli = torch.stack(cv_down_reli, dim=0)
            ct_down_reli = torch.stack(ct_down_reli, dim=0)

            down_reliable_loss = F.kl_div((-cum_dist_unreli_up).softmax(-1).log(), (-cum_dist_reli_down).softmax(-1), reduction='none')
            down_reliable_loss = torch.sum(down_reliable_loss, dim=1)
            c_down_reli = cv_down_reli + ct_down_reli
            down_reliable_loss = torch.matmul(down_reliable_loss, c_down_reli) / torch.sum(c_down_reli)

        #不使用任何蒸馏条件，无条件蒸馏的KL散度, loss
        if t_compare==0 and v_compare ==0:
            up_reliable_loss = F.kl_div((-cum_dist_down).softmax(-1).log(), (-cum_dist_up).softmax(-1), reduction='none')
            down_reliable_loss = F.kl_div((-cum_dist_up).softmax(-1).log(), (-cum_dist_down).softmax(-1), reduction='none')
            down_reliable_loss = torch.sum(torch.sum(down_reliable_loss, dim=1), dim=0)
            up_reliable_loss = torch.sum(torch.sum(up_reliable_loss, dim=1), dim=0)

        dist_direction = self.argss.dist_direction
        diss_loss = 0
        if dist_direction == "bid":
            diss_loss = up_reliable_loss + down_reliable_loss
        elif dist_direction == "up2down":
            diss_loss = down_reliable_loss
        elif dist_direction == "down2up":
            diss_loss = up_reliable_loss

        cum_dists = 1 * (-cum_dist_up) + 1 * (-cum_dist_down)

        # if not self.training:
        #     cum_dists = -cum_dist_up
        # if not self.training:
        # cummax=F.softmax(cum_dists,dim=-1)
        # logitsmax = F.softmax(logits_per_image, dim=-1)
        # vcum, _ = torch.max(cummax,dim=-1)
        # vlog, _ = torch.max(logitsmax,dim=-1)

        # judgements = vcum >= vlog
        # for i, v in enumerate(judgements):
        #     if v:
        #         pass
        #     else:
        #         cum_dists[i], logits_per_image[i] = logits_per_image[i], cum_dists[i]
        # cum_dists = -self.fusion_textimage_distance(-cum_dists, logits_per_image)

        return_dict = {
            'dis_logits': cum_dists,
            "class_logits": class_text_logits,
            "self_logits": diss_loss,
        }  # [5， 5] , [10 64]
        return return_dict

    def get_target_features_combie(self, target_features, target_features_context_up, target_features_context_down, soft_target_context):
        combinations = torch.as_tensor(getcombinations(8, 4)).cuda()
        combinations, _ = torch.sort(combinations, dim=-1)
        random_index = torch.as_tensor(random.sample(range(combinations.__len__()), 10)).cuda()
        # 得到 c82数据取随机4个，还是排序
        combinations = torch.index_select(combinations, 0, random_index)
        # 取出特征 reshape操作   进入context local
        target_features_comb = torch.stack([torch.index_select(target_features, 1, each) for each in combinations], dim=1)
        target_features_context_comb_up = torch.stack([torch.index_select(target_features_context_up, 1, each) for each in combinations], dim=1)  # 25 4 2 1024
        target_features_context_comb_down = torch.stack([torch.index_select(target_features_context_down, 1, each) for each in combinations], dim=1)  # 25 4 2 1024
        bs, local_query_len, select_len, _ = target_features_comb.shape
        target_features_comb = target_features_comb.reshape(bs * local_query_len, select_len, -1)  # 100 2 1024
        target_features_context_comb_up = target_features_context_comb_up.reshape(bs * local_query_len, select_len, -1)
        target_features_context_comb_down = target_features_context_comb_down.reshape(bs * local_query_len, select_len, -1)

        target_features_comb = torch.cat([target_features_comb, soft_target_context.repeat_interleave(local_query_len, dim=0)], dim=1)
        target_features_context_comb_up = torch.cat([target_features_context_comb_up, soft_target_context.repeat_interleave(local_query_len, dim=0)], dim=1)
        target_features_context_comb_down = torch.cat([target_features_context_comb_down, soft_target_context.repeat_interleave(local_query_len, dim=0)], dim=1)

        target_features_down_comb = self.context_local(target_features_context_comb_down, target_features_comb, target_features_comb)[:, :select_len, :]  # 100 2
        target_features_up_comb = self.context2(target_features_context_comb_up, target_features_comb, target_features_comb)[:, :select_len, :]  # 100 2
        return target_features_up_comb, target_features_down_comb, local_query_len

    def fusion_textimage_distance(self, cum_dists_visual, logits_per_image):
        cum_dists_visual_soft = F.softmax((8 - cum_dists_visual) / 8., dim=1)
        if hasattr(self.args.TRAIN, "TEXT_COFF") and self.args.TRAIN.TEXT_COFF:
            cum_dists = -(logits_per_image.pow(self.args.TRAIN.TEXT_COFF) * cum_dists_visual_soft.pow(1.0 - self.args.TRAIN.TEXT_COFF))
        else:
            cum_dists = -(logits_per_image.pow(0) * cum_dists_visual_soft.pow(1))
        return cum_dists

    def create_soft_context(self, support_labels, context_support, target_indice):
        # target model混合生成
        unique_labels = torch.unique(support_labels)
        context_support_pro = [
            torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)), dim=0)
            for c in unique_labels]
        context_support_pro = torch.stack(context_support_pro)

        context_target = torch.index_select(context_support_pro, 0, target_indice.long())
        return context_target

    def text_eh_temporal_transformer(self, context_support, context_target, support_features, target_features,
                                     support_feature_other, target_feature_other, support_labels, branch):

        unique_labels = torch.unique(support_labels)

        target_features_pro = torch.cat([context_target, target_features], dim=1)
        support_features_pro = torch.cat([context_support, support_features], dim=1)

        target_feature_other_pro = torch.cat([context_target, target_feature_other], dim=1)
        support_feature_other_pro = torch.cat([context_support, support_feature_other], dim=1)
        if branch == 0:
            target_features = self.context2(target_feature_other_pro, target_features_pro, target_features_pro)
            support_features = self.context2(support_feature_other_pro, support_features_pro, support_features_pro)
        elif branch == 1:
            target_features = self.context_local(target_feature_other_pro, target_features_pro, target_features_pro)
            support_features = self.context_local(support_feature_other_pro, support_features_pro, support_features_pro)

        # target_features_pro = self.context3(target_features_pro, target_feature_other_pro, target_feature_other_pro)[:,:frame_num,:]
        # support_features_pro = self.context3(support_features_pro, support_feature_other_pro, support_feature_other_pro)[:,:frame_num,:]

        # unique_labels = torch.unique(support_labels)
        support_features = [
            torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0)
            for c in unique_labels]
        support_features = torch.stack(support_features)

        # support_features_pro = [
        #     torch.mean(torch.index_select(support_features_pro, 0, extract_class_indices(support_labels, c)), dim=0)
        #     for c in unique_labels]
        # support_features_pro = torch.stack(support_features_pro)

        return support_features, target_features

    def otam_distance(self, support_features, target_features):
        n_queries = target_features.shape[0]
        n_support = support_features.shape[0]
        support_features = rearrange(support_features, 'b s d -> (b s) d')  # 5 8 1024-->40  1024
        target_features = rearrange(target_features, 'b s d -> (b s) d')
        frame_sim = cos_sim(target_features, support_features)  # 类别数量*每个类的样本数量， 类别数量
        frame_dists = 1 - frame_sim
        # dists维度为 query样本数量， support类别数量，帧数，帧数
        dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb=n_queries, sb=n_support)  # [25, 25, 8, 8]
        # calculate query -> support and support -> query  双向匹配还是单向匹配
        if hasattr(self.args.TRAIN, "SINGLE_DIRECT") and self.args.TRAIN.SINGLE_DIRECT:
            cum_dists = OTAM_cum_dist_v2(dists)
        else:
            cum_dists = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))
        return cum_dists

    def video2imagetext_adapter_mean(self, support_features, target_features):
        if self.training:
            text_features = self.text_features_train
        else:
            text_features = self.text_features_test
        # 是否使用分类标签 也就是文本分支
        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION") and self.args.TRAIN.USE_CLASSIFICATION:
            feature_classification_in = torch.cat([support_features, target_features], dim=0)  # 第一维度是support query一共的样本数量
            feature_classification = self.classification_layer(feature_classification_in).mean(1)  # 在第二位 帧的维度进行平均  维度为 [样本数量，1024] 公式2中的GAP操作
            class_text_logits = cos_sim(feature_classification, text_features) * self.scale  # 10 64， 10是10个视频样本，64是64个标签，对应于公式2
        else:
            class_text_logits = None
        return class_text_logits

    def video2imagetext_adapter_mean_episode(self, support_features, target_features, context_support):
        text_features = context_support
        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION") and self.args.TRAIN.USE_CLASSIFICATION:
            feature_classification_in = torch.cat([support_features, target_features], dim=0)  # 第一维度是support query一共的样本数量
            feature_classification = self.classification_layer(feature_classification_in).mean(1)  # 在第二位 帧的维度进行平均  维度为 [样本数量，1024] 公式2中的GAP操作
            class_text_logits = cos_sim(feature_classification, text_features) * self.scale  # 10 64， 10是10个视频样本，64是64个标签，对应于公式2
        else:
            class_text_logits = None
        return class_text_logits

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())

    def distribute_model(self):

        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        gpus_use_number = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        if self.argss.num_gpus > 1:
            self.backbone.cuda()
            self.backbone = torch.nn.DataParallel(self.backbone, device_ids=[i for i in range(gpus_use_number)])


