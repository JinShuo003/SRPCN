from functools import partial
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from pointnet2_ops import pointnet2_utils
import einops
from utils.loss import cd_loss_L1


def fps(data, number):
    """
        data B N 3
        number int
    """
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data


def jitter_points(pc, std=0.01, clip=0.05):
    bsize = pc.size()[0]
    for i in range(bsize):
        jittered_data = pc.new(pc.size(1), 3).normal_(
            mean=0.0, std=std
        ).clamp_(-clip, clip)
        pc[i, :, 0:3] += jittered_data
    return pc


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # 1 for mask, 0 for not mask
            # mask shape N, N
            mask_value = -torch.finfo(attn.dtype).max
            mask = (mask > 0)  # convert to boolen, shape torch.BoolTensor[N, N]
            attn = attn.masked_fill(mask, mask_value) # B h N N

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, v):
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DeformableLocalAttention(nn.Module):
    r''' DeformabelLocalAttention for only self attn
        Query a local region for each token (k x C)
        Conduct the Self-Attn among them and use the region feat after maxpooling to update the token feat
    '''
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., k=10, n_group=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v_off = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Deformable related
        self.k = k  # To be controlled
        self.n_group = n_group
        self.group_dims = dim // self.n_group
        assert num_heads % self.n_group == 0
        self.linear_offset = nn.Sequential(
            nn.Linear(2 * self.group_dims, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 3, bias=False)
        )
    def forward(self, x, pos, idx=None):
        B, N, C = x.shape
        # given N token and pos
        assert len(pos.shape) == 3 and pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for pos, expect it to be B N 3, but got {pos.shape}'
        # first query a neighborhood for one query token
        if idx is None:
            idx = knn_point(self.k, pos, pos) # B N k
        assert idx.size(-1) == self.k
        # project the qeury feat into shared space
        q = self.proj_q(x)
        v_off = self.proj_v_off(x)
        # Then we extract the region feat for a neighborhood
        local_v = index_points(v_off, idx) # B N k C
        # And we split it into several group on channels
        off_local_v = einops.rearrange(local_v, 'b n k (g c) -> (b g) n k c', g=self.n_group, c=self.group_dims) # Bg N k c
        group_q = einops.rearrange(q, 'b n (g c) -> (b g) n c', g=self.n_group, c=self.group_dims) # Bg N c
        # calculate offset
        shift_feat = torch.cat([
            off_local_v,
            group_q.unsqueeze(-2).expand(-1, -1, self.k, -1)
        ], dim=-1)  # Bg N k 2c
        offset  = self.linear_offset(shift_feat) # Bg N k 3
        offset = offset.tanh() # Bg N k 3
        # add offset for each point
        # The position in R3 for these points
        local_v_pos = index_points(pos, idx) # B N k 3
        local_v_pos = local_v_pos.unsqueeze(1).expand(-1, self.n_group, -1, -1, -1) # B g N k 3
        local_v_pos = einops.rearrange(local_v_pos, 'b g n k c -> (b g) n k c') # Bg N k 3
        shift_pos = local_v_pos + offset # Bg N 2*k 3
        # interpolate
        shift_pos = einops.rearrange(shift_pos, 'bg n k c -> bg (n k) c') # Bg k*N 3
        pos = pos.unsqueeze(1).expand(-1, self.n_group, -1, -1) # B g N 3
        pos = einops.rearrange(pos, 'b g n c -> (b g) n c') # Bg N 3
        v = einops.rearrange(x, 'b n (g c) -> (b g) n c', g=self.n_group, c=self.group_dims) # Bg N c
        # three_nn and three_interpolate
        dist, _idx = pointnet2_utils.three_nn(shift_pos.contiguous(), pos.contiguous())  #  Bg k*N 3, Bg k*N 3
        dist_reciprocal = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
        weight = dist_reciprocal / norm
        interpolated_feats = pointnet2_utils.three_interpolate(v.transpose(-1, -2).contiguous(), _idx, weight).transpose(-1, -2).contiguous()
        interpolated_feats = einops.rearrange(interpolated_feats, '(b g) (n k) c  -> b n k (g c)', b=B, g=self.n_group, n=N, k=self.k) # B N k gc

        # some assert to ensure the right feature shape
        assert interpolated_feats.size(1) == local_v.size(1)
        assert interpolated_feats.size(2) == local_v.size(2)
        assert interpolated_feats.size(3) == local_v.size(3)
        # SE module to select 1/2k out of k
        pass

        # calculate local attn
        # local_q : B N k C
        # interpolated_feats : B N k C
        # extrate the feat for a local region
        local_q = index_points(q, idx) # B N k C
        q = einops.rearrange(local_q, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim) # BHN k c
        k = self.proj_k(interpolated_feats)
        k = einops.rearrange(k, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim) # BHN k c
        v = self.proj_v(interpolated_feats)
        v = einops.rearrange(v, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim) # BHN k c

        attn = torch.einsum('b m c, b n c -> b m n', q, k) # BHN, k, k
        attn = attn.mul(self.scale)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b n c -> b m c', attn, v) # BHN k c
        out = einops.rearrange(out, '(b h n) k c -> b n k (h c)', b=B, n=N, h=self.num_heads) # B N k C
        out = out.max(dim=2, keepdim=False)[0]  # B N C
        out = self.proj(out)
        out = self.proj_drop(out)

        assert out.size(0) == B
        assert out.size(1) == N
        assert out.size(2) == C

        return out


class DeformableLocalCrossAttention(nn.Module):
    r''' DeformabelLocalAttention for self attn or cross attn
        Query a local region for each token (k x C) and then perform a cross attn among query token(1 x C) and local region (k x C)
        These can convert local self-attn as a local cross-attn
    '''

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., k=10, n_group=2):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v_off = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Deformable related
        self.k = k  # To be controlled
        self.n_group = n_group
        self.group_dims = dim // self.n_group
        assert num_heads % self.n_group == 0
        self.linear_offset = nn.Sequential(
            nn.Linear(2 * self.group_dims, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 3, bias=False)
        )

    def forward(self, q, q_pos, v=None, v_pos=None, idx=None, denoise_length=None):
        r'''
            If perform a self-attn, just use
                q = x, v = x, q_pos = pos, v_pos = pos
        '''
        if denoise_length is None:
            if v is None:
                v = q
            if v_pos is None:
                v_pos = q_pos

            B, N, C = q.shape
            k = v
            NK = k.size(1)
            # given N token and pos
            assert len(v_pos.shape) == 3 and v_pos.size(
                -1) == 3, f'[ERROR] Got an unexpected shape for v_pos, expect it to be B N 3, but got {v_pos.shape}'
            assert len(q_pos.shape) == 3 and q_pos.size(
                -1) == 3, f'[ERROR] Got an unexpected shape for q_pos, expect it to be B N 3, but got {q_pos.shape}'

            # first query a neighborhood for one query token
            if idx is None:
                idx = knn_point(self.k, v_pos, q_pos)  # B N k
            assert idx.size(-1) == self.k
            # project the qeury feat into shared space
            q = self.proj_q(q)
            v_off = self.proj_v_off(v)
            # Then we extract the region feat for a neighborhood
            local_v = index_points(v_off, idx)  # B N k C
            # And we split it into several group on channels
            off_local_v = einops.rearrange(local_v, 'b n k (g c) -> (b g) n k c', g=self.n_group,
                                           c=self.group_dims)  # Bg N k c
            group_q = einops.rearrange(q, 'b n (g c) -> (b g) n c', g=self.n_group, c=self.group_dims)  # Bg N c
            # calculate offset
            shift_feat = torch.cat([
                off_local_v,
                group_q.unsqueeze(-2).expand(-1, -1, self.k, -1)
            ], dim=-1)  # Bg N k 2c
            offset = self.linear_offset(shift_feat)  # Bg N k 3
            offset = offset.tanh()  # Bg N k 3
            # add offset for each point
            # The position in R3 for these points
            local_v_pos = index_points(v_pos, idx)  # B N k 3
            local_v_pos = local_v_pos.unsqueeze(1).expand(-1, self.n_group, -1, -1, -1)  # B g N k 3
            local_v_pos = einops.rearrange(local_v_pos, 'b g n k c -> (b g) n k c')  # Bg N k 3
            shift_pos = local_v_pos + offset  # Bg N k 3
            # interpolate
            shift_pos = einops.rearrange(shift_pos, 'bg n k c -> bg (n k) c')  # Bg k*N 3
            v_pos = v_pos.unsqueeze(1).expand(-1, self.n_group, -1, -1)  # B g Nk 3
            v_pos = einops.rearrange(v_pos, 'b g n c -> (b g) n c')  # Bg Nk 3
            v = einops.rearrange(v, 'b n (g c) -> (b g) n c', g=self.n_group, c=self.group_dims)  # Bg Nk c
            # three_nn and three_interpolate
            dist, idx = pointnet2_utils.three_nn(shift_pos.contiguous(), v_pos.contiguous())  # Bg k*N 3, Bg k*N 3
            dist_reciprocal = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
            weight = dist_reciprocal / norm
            interpolated_feats = pointnet2_utils.three_interpolate(v.transpose(-1, -2).contiguous(), idx,
                                                                   weight).transpose(-1, -2).contiguous()
            interpolated_feats = einops.rearrange(interpolated_feats, '(b g) (n k) c  -> b n k (g c)', b=B,
                                                  g=self.n_group, n=N, k=self.k)  # B N k gc

            # some assert to ensure the right feature shape
            assert interpolated_feats.size(1) == local_v.size(1)
            assert interpolated_feats.size(2) == local_v.size(2)
            assert interpolated_feats.size(3) == local_v.size(3)
            # SE module to select 1/2k out of k
            pass

            # calculate local attn
            # local_q : B N k C
            # interpolated_feats : B N k C
            q = einops.rearrange(q, 'b n (h c) -> (b h n) c', h=self.num_heads, c=self.head_dim).unsqueeze(
                -2)  # BHN 1 c
            k = self.proj_k(interpolated_feats)
            k = einops.rearrange(k, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim)  # BHN k c
            v = self.proj_v(interpolated_feats)
            v = einops.rearrange(v, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim)  # BHN k c

            attn = torch.einsum('b m c, b n c -> b m n', q, k)  # BHN, 1, k
            attn = attn.mul(self.scale)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            out = torch.einsum('b m n, b n c -> b m c', attn, v)  # BHN 1 c
            out = einops.rearrange(out, '(b h n) k c -> b n k (h c)', b=B, n=N, h=self.num_heads)  # B N 1 C
            assert out.size(2) == 1
            out = out.squeeze(2)
            out = self.proj(out)
            out = self.proj_drop(out)

            assert out.size(0) == B
            assert out.size(1) == N
            assert out.size(2) == C

        else:
            assert idx is None, f'we need online index calculation when denoise_length is set, denoise_length {denoise_length}'
            # when v_pos and v are given, that to say, it's a cross attn.
            # we only consider self-attn
            assert v is None, f'mask for denoise_length is only consider in self-attention, but v is given'
            assert v_pos is None, f'mask for denoise_length is only consider in self-attention, but v_pos is given'

            v = q
            v_pos = q_pos
            # given N token and pos
            assert len(v_pos.shape) == 3 and v_pos.size(
                -1) == 3, f'[ERROR] Got an unexpected shape for v_pos, expect it to be B N 3, but got {v_pos.shape}'
            assert len(q_pos.shape) == 3 and q_pos.size(
                -1) == 3, f'[ERROR] Got an unexpected shape for q_pos, expect it to be B N 3, but got {q_pos.shape}'
            assert q.size(-1) == v.size(-1) == self.dim
            B, N, C = q.shape

            q = self.proj_q(q)
            v_off = self.proj_v_off(v)

            ######################################### produce local_v by two knn #########################################
            # normal reconstruction task:
            # first query a neighborhood for one query token for normal part
            idx = knn_point(self.k, v_pos[:, :-denoise_length], q_pos[:, :-denoise_length])  # B N_r k
            assert idx.size(-1) == self.k
            # gather the neighbor point feat
            local_v_r = index_points(v_off[:, :-denoise_length], idx)  # B N_r k C
            local_v_r_pos = index_points(v_pos[:, :-denoise_length], idx)  # B N_r k 3

            # Then query a nerighborhood for denoise token within all token
            idx = knn_point(self.k, v_pos, q_pos[:, -denoise_length:])  # B N_n k
            assert idx.size(-1) == self.k
            assert idx.size(1) == denoise_length
            # gather the neighbor point feat
            local_v_n = index_points(v_off, idx)  # B N_n k C
            local_v_n_pos = index_points(v_pos, idx)  # B N_n k 3
            ######################################### produce local_v by two knn #########################################

            # Concat two part
            local_v = torch.cat([local_v_r, local_v_n], dim=1)  # B N k C

            # And we split it into several group on channels
            off_local_v = einops.rearrange(local_v, 'b n k (g c) -> (b g) n k c', g=self.n_group,
                                           c=self.group_dims)  # Bg N k c
            group_q = einops.rearrange(q, 'b n (g c) -> (b g) n c', g=self.n_group, c=self.group_dims)  # Bg N c

            # calculate offset
            shift_feat = torch.cat([
                off_local_v,
                group_q.unsqueeze(-2).expand(-1, -1, self.k, -1)
            ], dim=-1)  # Bg N k 2c
            offset = self.linear_offset(shift_feat)  # Bg N k 3
            offset = offset.tanh()  # Bg N k 3
            # add offset for each point
            # The position in R3 for these points
            local_v_pos = torch.cat([local_v_r_pos, local_v_n_pos], dim=1)  # B N k 3
            local_v_pos = local_v_pos.unsqueeze(1).expand(-1, self.n_group, -1, -1, -1)  # B g N k 3
            local_v_pos = einops.rearrange(local_v_pos, 'b g n k c -> (b g) n k c')  # Bg N k 3
            shift_pos = local_v_pos + offset  # Bg N k 3
            # interpolate
            shift_pos = einops.rearrange(shift_pos, 'bg n k c -> bg (n k) c')  # Bg k*N 3
            v_pos = v_pos.unsqueeze(1).expand(-1, self.n_group, -1, -1)  # B g Nk 3
            v_pos = einops.rearrange(v_pos, 'b g n c -> (b g) n c')  # Bg Nk 3
            v = einops.rearrange(v, 'b n (g c) -> (b g) n c', g=self.n_group, c=self.group_dims)  # Bg Nk c
            # three_nn and three_interpolate
            dist, idx = pointnet2_utils.three_nn(shift_pos.contiguous(), v_pos.contiguous())  # Bg k*N 3, Bg k*N 3
            dist_reciprocal = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
            weight = dist_reciprocal / norm
            interpolated_feats = pointnet2_utils.three_interpolate(v.transpose(-1, -2).contiguous(), idx,
                                                                   weight).transpose(-1, -2).contiguous()
            interpolated_feats = einops.rearrange(interpolated_feats, '(b g) (n k) c  -> b n k (g c)', b=B,
                                                  g=self.n_group, n=N, k=self.k)  # B N k gc

            # some assert to ensure the right feature shape
            assert interpolated_feats.size(1) == local_v.size(1)
            assert interpolated_feats.size(2) == local_v.size(2)
            assert interpolated_feats.size(3) == local_v.size(3)
            # SE module to select 1/2k out of k
            pass

            # calculate local attn
            # local_q : B N k C
            # interpolated_feats : B N k C
            q = einops.rearrange(q, 'b n (h c) -> (b h n) c', h=self.num_heads, c=self.head_dim).unsqueeze(
                -2)  # BHN 1 c
            k = self.proj_k(interpolated_feats)
            k = einops.rearrange(k, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim)  # BHN k c
            v = self.proj_v(interpolated_feats)
            v = einops.rearrange(v, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim)  # BHN k c

            attn = torch.einsum('b m c, b n c -> b m n', q, k)  # BHN, 1, k
            attn = attn.mul(self.scale)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            out = torch.einsum('b m n, b n c -> b m c', attn, v)  # BHN 1 c
            out = einops.rearrange(out, '(b h n) k c -> b n k (h c)', b=B, n=N, h=self.num_heads)  # B N 1 C
            assert out.size(2) == 1
            out = out.squeeze(2)
            out = self.proj(out)
            out = self.proj_drop(out)

            assert out.size(0) == B
            assert out.size(1) == N
            assert out.size(2) == C

        return out


class DynamicGraphAttention(nn.Module):
    r''' DynamicGraphAttention for self attn or cross attn
        Query a local region for each token (k x C) and then perform Conv2d with maxpooling to build the graph feature for each token
        These can convert local self-attn as a local cross-attn
    '''

    def __init__(self, dim, k=10):
        super().__init__()
        self.dim = dim
        # Deformable related
        self.k = k  # To be controlled
        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, q, q_pos, v=None, v_pos=None, idx=None, denoise_length=None):
        r'''
            If perform a self-attn, just use
                q = x, v = x, q_pos = pos, v_pos = pos
        '''
        if denoise_length is None:
            if v is None:
                v = q
            if v_pos is None:
                v_pos = q_pos
            # given N token and pos
            assert len(v_pos.shape) == 3 and v_pos.size(
                -1) == 3, f'[ERROR] Got an unexpected shape for v_pos, expect it to be B N 3, but got {v_pos.shape}'
            assert len(q_pos.shape) == 3 and q_pos.size(
                -1) == 3, f'[ERROR] Got an unexpected shape for q_pos, expect it to be B N 3, but got {q_pos.shape}'
            assert q.size(-1) == v.size(-1) == self.dim
            B, N, C = q.shape
            # first query a neighborhood for one query token
            if idx is None:
                idx = knn_point(self.k, v_pos, q_pos)  # B N k
            assert idx.size(-1) == self.k
            # gather the neighbor point feat
            local_v = index_points(v, idx)  # B N k C
            q = q.unsqueeze(-2).expand(-1, -1, self.k, -1)  # B N k C
            feature = torch.cat((local_v - q, q), dim=-1)  # B N k C
            out = self.knn_map(feature).max(-2)[0]  # B N C

            assert out.size(0) == B
            assert out.size(1) == N
            assert out.size(2) == C
        else:
            assert idx is None, f'we need online index calculation when denoise_length is set, denoise_length {denoise_length}'
            # when v_pos and v are given, that to say, it's a cross attn.
            # we only consider self-attn
            assert v is None, f'mask for denoise_length is only consider in self-attention, but v is given'
            assert v_pos is None, f'mask for denoise_length is only consider in self-attention, but v_pos is given'

            v = q
            v_pos = q_pos
            # given N token and pos
            assert len(v_pos.shape) == 3 and v_pos.size(
                -1) == 3, f'[ERROR] Got an unexpected shape for v_pos, expect it to be B N 3, but got {v_pos.shape}'
            assert len(q_pos.shape) == 3 and q_pos.size(
                -1) == 3, f'[ERROR] Got an unexpected shape for q_pos, expect it to be B N 3, but got {q_pos.shape}'
            assert q.size(-1) == v.size(-1) == self.dim
            B, N, C = q.shape

            # normal reconstruction task:
            # first query a neighborhood for one query token for normal part
            idx = knn_point(self.k, v_pos[:, :-denoise_length], q_pos[:, :-denoise_length])  # B N_r k
            assert idx.size(-1) == self.k
            # gather the neighbor point feat
            local_v_r = index_points(v[:, :-denoise_length], idx)  # B N_r k C

            # Then query a nerighborhood for denoise token within all token
            idx = knn_point(self.k, v_pos, q_pos[:, -denoise_length:])  # B N_n k
            assert idx.size(-1) == self.k
            assert idx.size(1) == denoise_length
            # gather the neighbor point feat
            local_v_n = index_points(v, idx)  # B N_n k C

            # Concat two part
            local_v = torch.cat([local_v_r, local_v_n], dim=1)
            q = q.unsqueeze(-2).expand(-1, -1, self.k, -1)  # B N k C
            feature = torch.cat((local_v - q, q), dim=-1)  # B N k C
            out = self.knn_map(feature).max(-2)[0]  # B N C

            assert out.size(0) == B
            assert out.size(1) == N
            assert out.size(2) == C
        return out


class improvedDeformableLocalGraphAttention(nn.Module):
    r''' DeformabelLocalAttention for self attn or cross attn
        Query a local region for each token (k x C) and then perform a graph conv among query token(1 x C) and local region (k x C)
        These can convert local self-attn as a local cross-attn
        $ improved:
            Deformable within a local ball
    '''

    def __init__(self, dim, k=10):
        super().__init__()
        self.dim = dim

        self.proj_v_off = nn.Linear(dim, dim)

        # Deformable related
        self.k = k  # To be controlled
        self.linear_offset = nn.Sequential(
            nn.Linear(2 * self.dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 3, bias=False)
        )
        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, q, q_pos, v=None, v_pos=None, idx=None, denoise_length=None):
        r'''
            If perform a self-attn, just use
                q = x, v = x, q_pos = pos, v_pos = pos
        '''
        if denoise_length is None:
            if v is None:
                v = q
            if v_pos is None:
                v_pos = q_pos

            B, N, C = q.shape
            # given N token and pos
            assert len(v_pos.shape) == 3 and v_pos.size(
                -1) == 3, f'[ERROR] Got an unexpected shape for v_pos, expect it to be B N 3, but got {v_pos.shape}'
            assert len(q_pos.shape) == 3 and q_pos.size(
                -1) == 3, f'[ERROR] Got an unexpected shape for q_pos, expect it to be B N 3, but got {q_pos.shape}'
            assert q.size(-1) == v.size(-1) == self.dim
            # first query a neighborhood for one query token
            if idx is None:
                idx = knn_point(self.k, v_pos, q_pos)  # B N k
            assert idx.size(-1) == self.k
            # project the local feat into shared space
            v_off = self.proj_v_off(v)
            # Then we extract the region feat for a neighborhood
            off_local_v = index_points(v_off, idx)  # B N k C
            # calculate offset
            shift_feat = torch.cat([
                off_local_v,
                q.unsqueeze(-2).expand(-1, -1, self.k, -1)
            ], dim=-1)  # B N k 2c
            offset = self.linear_offset(shift_feat)  # B N k 3
            offset = offset.tanh()  # B N k 3

            # add offset for each point
            # The position in R3 for these points
            local_v_pos = index_points(v_pos, idx)  # B N k 3

            # calculate scale
            scale = local_v_pos.max(-2)[0] - local_v_pos.min(-2)[0]  # B N 3
            scale = scale.unsqueeze(-2) * 0.5  # B N 1 3
            shift_pos = local_v_pos + offset * scale  # B N k 3

            # interpolate
            shift_pos = einops.rearrange(shift_pos, 'b n k c -> b (n k) c')  # B k*N 3
            # three_nn and three_interpolate
            dist, idx = pointnet2_utils.three_nn(shift_pos.contiguous(), v_pos.contiguous())  # B k*N 3, B k*N 3
            dist_reciprocal = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
            weight = dist_reciprocal / norm
            interpolated_feats = pointnet2_utils.three_interpolate(v.transpose(-1, -2).contiguous(), idx,
                                                                   weight).transpose(-1, -2).contiguous()
            interpolated_feats = einops.rearrange(interpolated_feats, 'b (n k) c  -> b n k c', n=N, k=self.k)  # B N k c

            q = q.unsqueeze(-2).expand(-1, -1, self.k, -1)  # B N k C
            feature = torch.cat((interpolated_feats - q, q), dim=-1)  # B N k C
            out = self.knn_map(feature).max(-2)[0]  # B N C

            assert out.size(0) == B
            assert out.size(1) == N
            assert out.size(2) == C

        else:
            assert idx is None, f'we need online index calculation when denoise_length is set, denoise_length {denoise_length}'
            # when v_pos and v are given, that to say, it's a cross attn.
            # we only consider self-attn
            assert v is None, f'mask for denoise_length is only consider in self-attention, but v is given'
            assert v_pos is None, f'mask for denoise_length is only consider in self-attention, but v_pos is given'

            v = q
            v_pos = q_pos
            # given N token and pos
            assert len(v_pos.shape) == 3 and v_pos.size(
                -1) == 3, f'[ERROR] Got an unexpected shape for v_pos, expect it to be B N 3, but got {v_pos.shape}'
            assert len(q_pos.shape) == 3 and q_pos.size(
                -1) == 3, f'[ERROR] Got an unexpected shape for q_pos, expect it to be B N 3, but got {q_pos.shape}'
            assert q.size(-1) == v.size(-1) == self.dim
            B, N, C = q.shape

            v_off = self.proj_v_off(v)

            # normal reconstruction task:
            # first query a neighborhood for one query token for normal part
            idx = knn_point(self.k, v_pos[:, :-denoise_length], q_pos[:, :-denoise_length])  # B N_r k
            assert idx.size(-1) == self.k
            # gather the neighbor point feat
            local_v_r_off = index_points(v_off[:, :-denoise_length], idx)  # B N_r k C
            local_v_r_pos = index_points(v_pos[:, :-denoise_length], idx)  # B N_r k 3
            # Then query a nerighborhood for denoise token within all token
            idx = knn_point(self.k, v_pos, q_pos[:, -denoise_length:])  # B N_n k
            assert idx.size(-1) == self.k
            assert idx.size(1) == denoise_length
            # gather the neighbor point feat
            local_v_n_off = index_points(v_off, idx)  # B N_n k C
            local_v_n_pos = index_points(v_pos, idx)  # B N_n k 3
            # Concat two part
            off_local_v = torch.cat([local_v_r_off, local_v_n_off], dim=1)  # B N k C
            # calculate offset
            shift_feat = torch.cat([
                off_local_v,
                q.unsqueeze(-2).expand(-1, -1, self.k, -1)
            ], dim=-1)  # B N k 2c
            offset = self.linear_offset(shift_feat)  # B N k 3
            offset = offset.tanh()  # B N k 3

            # add offset for each point
            # The position in R3 for these points
            local_v_pos = torch.cat([local_v_r_pos, local_v_n_pos], dim=1)  # B N k 3

            # calculate scale
            scale = local_v_pos.max(-2)[0] - local_v_pos.min(-2)[0]  # B N 3
            scale = scale.unsqueeze(-2) * 0.5  # B N 1 3
            shift_pos = local_v_pos + offset * scale  # B N k 3

            # interpolate
            shift_pos = einops.rearrange(shift_pos, 'b n k c -> b (n k) c')  # B k*N 3
            # three_nn and three_interpolate
            dist, idx = pointnet2_utils.three_nn(shift_pos.contiguous(), v_pos.contiguous())  # B k*N 3, B k*N 3
            dist_reciprocal = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
            weight = dist_reciprocal / norm
            interpolated_feats = pointnet2_utils.three_interpolate(v.transpose(-1, -2).contiguous(), idx,
                                                                   weight).transpose(-1, -2).contiguous()
            interpolated_feats = einops.rearrange(interpolated_feats, 'b (n k) c  -> b n k c', n=N, k=self.k)  # B N k c

            q = q.unsqueeze(-2).expand(-1, -1, self.k, -1)  # B N k C
            feature = torch.cat((interpolated_feats - q, q), dim=-1)  # B N k C
            out = self.knn_map(feature).max(-2)[0]  # B N C

            assert out.size(0) == B
            assert out.size(1) == N
            assert out.size(2) == C
        return out


class SelfAttnBlockApi(nn.Module):
    r"""
        1. Norm Encoder Block
            block_style = 'attn'
        2. Concatenation Fused Encoder Block
            block_style = 'attn-deform'
            combine_style = 'concat'
        3. Three-layer Fused Encoder Block
            block_style = 'attn-deform'
            combine_style = 'onebyone'
    """

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, block_style='attn-deform', combine_style='concat',
            k=10, n_group=2
    ):

        super().__init__()
        self.combine_style = combine_style
        assert combine_style in ['concat',
                                 'onebyone'], f'got unexpect combine_style {combine_style} for local and global attn'
        self.norm1 = norm_layer(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Api desigin
        block_tokens = block_style.split('-')
        assert len(block_tokens) > 0 and len(block_tokens) <= 2, f'invalid block_style {block_style}'
        self.block_length = len(block_tokens)
        self.attn = None
        self.local_attn = None
        for block_token in block_tokens:
            assert block_token in ['attn', 'rw_deform', 'deform', 'graph',
                                   'deform_graph'], f'got unexpect block_token {block_token} for Block component'
            if block_token == 'attn':
                self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            elif block_token == 'rw_deform':
                self.local_attn = DeformableLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                                           attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif block_token == 'deform':
                self.local_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                                                attn_drop=attn_drop, proj_drop=drop, k=k,
                                                                n_group=n_group)
            elif block_token == 'graph':
                self.local_attn = DynamicGraphAttention(dim, k=k)
            elif block_token == 'deform_graph':
                self.local_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.attn is not None and self.local_attn is not None:
            if combine_style == 'concat':
                self.merge_map = nn.Linear(dim * 2, dim)
            else:
                self.norm3 = norm_layer(dim)
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, pos, idx=None):
        feature_list = []
        if self.block_length == 2:
            if self.combine_style == 'concat':
                norm_x = self.norm1(x)
                if self.attn is not None:
                    global_attn_feat = self.attn(norm_x)
                    feature_list.append(global_attn_feat)
                if self.local_attn is not None:
                    local_attn_feat = self.local_attn(norm_x, pos, idx=idx)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.merge_map(f)
                    x = x + self.drop_path1(self.ls1(f))
                else:
                    raise RuntimeError()
            else:  # onebyone
                x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                x = x + self.drop_path3(self.ls3(self.local_attn(self.norm3(x), pos, idx=idx)))

        elif self.block_length == 1:
            norm_x = self.norm1(x)
            if self.attn is not None:
                global_attn_feat = self.attn(norm_x)
                feature_list.append(global_attn_feat)
            if self.local_attn is not None:
                local_attn_feat = self.local_attn(norm_x, pos, idx=idx)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                x = x + self.drop_path1(self.ls1(f))
            else:
                raise RuntimeError()

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class CrossAttnBlockApi(nn.Module):
    r"""
        1. Norm Decoder Block
            self_attn_block_style = 'attn'
            cross_attn_block_style = 'attn'
        2. Concatenation Fused Decoder Block
            self_attn_block_style = 'attn-deform'
            self_attn_combine_style = 'concat'
            cross_attn_block_style = 'attn-deform'
            cross_attn_combine_style = 'concat'
        3. Three-layer Fused Decoder Block
            self_attn_block_style = 'attn-deform'
            self_attn_combine_style = 'onebyone'
            cross_attn_block_style = 'attn-deform'
            cross_attn_combine_style = 'onebyone'
        4. Design by yourself
            #  only deform the cross attn
            self_attn_block_style = 'attn'
            cross_attn_block_style = 'attn-deform'
            cross_attn_combine_style = 'concat'
            #  perform graph conv on self attn
            self_attn_block_style = 'attn-graph'
            self_attn_combine_style = 'concat'
            cross_attn_block_style = 'attn-deform'
            cross_attn_combine_style = 'concat'
    """

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            self_attn_block_style='attn-deform', self_attn_combine_style='concat',
            cross_attn_block_style='attn-deform', cross_attn_combine_style='concat',
            k=10, n_group=2
    ):
        super().__init__()
        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Api desigin
        # first we deal with self-attn
        self.norm1 = norm_layer(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.self_attn_combine_style = self_attn_combine_style
        assert self_attn_combine_style in ['concat',
                                           'onebyone'], f'got unexpect self_attn_combine_style {self_attn_combine_style} for local and global attn'

        self_attn_block_tokens = self_attn_block_style.split('-')
        assert len(self_attn_block_tokens) > 0 and len(
            self_attn_block_tokens) <= 2, f'invalid self_attn_block_style {self_attn_block_style}'
        self.self_attn_block_length = len(self_attn_block_tokens)
        self.self_attn = None
        self.local_self_attn = None
        for self_attn_block_token in self_attn_block_tokens:
            assert self_attn_block_token in ['attn', 'rw_deform', 'deform', 'graph',
                                             'deform_graph'], f'got unexpect self_attn_block_token {self_attn_block_token} for Block component'
            if self_attn_block_token == 'attn':
                self.self_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                           proj_drop=drop)
            elif self_attn_block_token == 'rw_deform':
                self.local_self_attn = DeformableLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                                                attn_drop=attn_drop, proj_drop=drop, k=k,
                                                                n_group=n_group)
            elif self_attn_block_token == 'deform':
                self.local_self_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                                                     attn_drop=attn_drop, proj_drop=drop, k=k,
                                                                     n_group=n_group)
            elif self_attn_block_token == 'graph':
                self.local_self_attn = DynamicGraphAttention(dim, k=k)
            elif self_attn_block_token == 'deform_graph':
                self.local_self_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.self_attn is not None and self.local_self_attn is not None:
            if self_attn_combine_style == 'concat':
                self.self_attn_merge_map = nn.Linear(dim * 2, dim)
            else:
                self.norm3 = norm_layer(dim)
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Then we deal with cross-attn
        self.norm_q = norm_layer(dim)
        self.norm_v = norm_layer(dim)
        self.ls4 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path4 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cross_attn_combine_style = cross_attn_combine_style
        assert cross_attn_combine_style in ['concat',
                                            'onebyone'], f'got unexpect cross_attn_combine_style {cross_attn_combine_style} for local and global attn'

        # Api desigin
        cross_attn_block_tokens = cross_attn_block_style.split('-')
        assert len(cross_attn_block_tokens) > 0 and len(
            cross_attn_block_tokens) <= 2, f'invalid cross_attn_block_style {cross_attn_block_style}'
        self.cross_attn_block_length = len(cross_attn_block_tokens)
        self.cross_attn = None
        self.local_cross_attn = None
        for cross_attn_block_token in cross_attn_block_tokens:
            assert cross_attn_block_token in ['attn', 'deform', 'graph',
                                              'deform_graph'], f'got unexpect cross_attn_block_token {cross_attn_block_token} for Block component'
            if cross_attn_block_token == 'attn':
                self.cross_attn = CrossAttention(dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                                 proj_drop=drop)
            elif cross_attn_block_token == 'deform':
                self.local_cross_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                                                      attn_drop=attn_drop, proj_drop=drop, k=k,
                                                                      n_group=n_group)
            elif cross_attn_block_token == 'graph':
                self.local_cross_attn = DynamicGraphAttention(dim, k=k)
            elif cross_attn_block_token == 'deform_graph':
                self.local_cross_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.cross_attn is not None and self.local_cross_attn is not None:
            if cross_attn_combine_style == 'concat':
                self.cross_attn_merge_map = nn.Linear(dim * 2, dim)
            else:
                self.norm_q_2 = norm_layer(dim)
                self.norm_v_2 = norm_layer(dim)
                self.ls5 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path5 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, q, v, q_pos, v_pos, self_attn_idx=None, cross_attn_idx=None, denoise_length=None):
        # q = q + self.drop_path(self.self_attn(self.norm1(q)))

        # calculate mask, shape N,N
        # 1 for mask, 0 for not mask
        # mask shape N, N
        # q: [ true_query; denoise_token ]
        if denoise_length is None:
            mask = None
        else:
            query_len = q.size(1)
            mask = torch.zeros(query_len, query_len).to(q.device)
            mask[:-denoise_length, -denoise_length:] = 1.

        # Self attn
        feature_list = []
        if self.self_attn_block_length == 2:
            if self.self_attn_combine_style == 'concat':
                norm_q = self.norm1(q)
                if self.self_attn is not None:
                    global_attn_feat = self.self_attn(norm_q, mask=mask)
                    feature_list.append(global_attn_feat)
                if self.local_self_attn is not None:
                    local_attn_feat = self.local_self_attn(norm_q, q_pos, idx=self_attn_idx,
                                                           denoise_length=denoise_length)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.self_attn_merge_map(f)
                    q = q + self.drop_path1(self.ls1(f))
                else:
                    raise RuntimeError()
            else:  # onebyone
                q = q + self.drop_path1(self.ls1(self.self_attn(self.norm1(q), mask=mask)))
                q = q + self.drop_path3(self.ls3(
                    self.local_self_attn(self.norm3(q), q_pos, idx=self_attn_idx, denoise_length=denoise_length)))

        elif self.self_attn_block_length == 1:
            norm_q = self.norm1(q)
            if self.self_attn is not None:
                global_attn_feat = self.self_attn(norm_q, mask=mask)
                feature_list.append(global_attn_feat)
            if self.local_self_attn is not None:
                local_attn_feat = self.local_self_attn(norm_q, q_pos, idx=self_attn_idx, denoise_length=denoise_length)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                q = q + self.drop_path1(self.ls1(f))
            else:
                raise RuntimeError()

        # q = q + self.drop_path(self.attn(self.norm_q(q), self.norm_v(v)))
        # Cross attn
        feature_list = []
        if self.cross_attn_block_length == 2:
            if self.cross_attn_combine_style == 'concat':
                norm_q = self.norm_q(q)
                norm_v = self.norm_v(v)
                if self.cross_attn is not None:
                    global_attn_feat = self.cross_attn(norm_q, norm_v)
                    feature_list.append(global_attn_feat)
                if self.local_cross_attn is not None:
                    local_attn_feat = self.local_cross_attn(q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos,
                                                            idx=cross_attn_idx)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.cross_attn_merge_map(f)
                    q = q + self.drop_path4(self.ls4(f))
                else:
                    raise RuntimeError()
            else:  # onebyone
                q = q + self.drop_path4(self.ls4(self.cross_attn(self.norm_q(q), self.norm_v(v))))
                q = q + self.drop_path5(self.ls5(
                    self.local_cross_attn(q=self.norm_q_2(q), v=self.norm_v_2(v), q_pos=q_pos, v_pos=v_pos,
                                          idx=cross_attn_idx)))

        elif self.cross_attn_block_length == 1:
            norm_q = self.norm_q(q)
            norm_v = self.norm_v(v)
            if self.cross_attn is not None:
                global_attn_feat = self.cross_attn(norm_q, norm_v)
                feature_list.append(global_attn_feat)
            if self.local_cross_attn is not None:
                local_attn_feat = self.local_cross_attn(q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos,
                                                        idx=cross_attn_idx)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                q = q + self.drop_path4(self.ls4(f))
            else:
                raise RuntimeError()

        q = q + self.drop_path2(self.ls2(self.mlp(self.norm2(q))))
        return q


######################################## Entry ########################################

class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 block_style_list=['attn-deform'], combine_style='concat', k=10, n_group=2):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(SelfAttnBlockApi(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                act_layer=act_layer, norm_layer=norm_layer,
                block_style=block_style_list[i], combine_style=combine_style, k=k, n_group=n_group
            ))

    def forward(self, x, pos):
        idx = idx = knn_point(self.k, pos, pos)
        for _, block in enumerate(self.blocks):
            x = block(x, pos, idx=idx)
        return x


class TransformerDecoder(nn.Module):
    """ Transformer Decoder without hierarchical structure
    """

    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 self_attn_block_style_list=['attn-deform'], self_attn_combine_style='concat',
                 cross_attn_block_style_list=['attn-deform'], cross_attn_combine_style='concat',
                 k=10, n_group=2):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(CrossAttnBlockApi(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                act_layer=act_layer, norm_layer=norm_layer,
                self_attn_block_style=self_attn_block_style_list[i], self_attn_combine_style=self_attn_combine_style,
                cross_attn_block_style=cross_attn_block_style_list[i],
                cross_attn_combine_style=cross_attn_combine_style,
                k=k, n_group=n_group
            ))

    def forward(self, q, v, q_pos, v_pos, denoise_length=None):
        if denoise_length is None:
            self_attn_idx = knn_point(self.k, q_pos, q_pos)
        else:
            self_attn_idx = None
        cross_attn_idx = knn_point(self.k, v_pos, q_pos)
        for _, block in enumerate(self.blocks):
            q = block(q, v, q_pos, v_pos, self_attn_idx=self_attn_idx, cross_attn_idx=cross_attn_idx,
                      denoise_length=denoise_length)
        return q


class PointTransformerEncoder(nn.Module):
    """ Vision Transformer for point cloud encoder/decoder
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Args:
        embed_dim (int): embedding dimension
        depth (int): depth of transformer
        num_heads (int): number of attention heads
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): enable bias for qkv if True
        init_values: (float): layer-scale init values
        drop_rate (float): dropout rate
        attn_drop_rate (float): attention dropout rate
        drop_path_rate (float): stochastic depth rate
        norm_layer: (nn.Module): normalization layer
        act_layer: (nn.Module): MLP activation layer
    """

    def __init__(
            self, embed_dim=384, depth=6, num_heads=6, mlp_ratio=2., qkv_bias=True, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=None, act_layer=None,
            block_style_list=['attn-deform'], combine_style='concat',
            k=8, n_group=2
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        assert len(block_style_list) == depth
        self.blocks = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dpr,
            norm_layer=norm_layer,
            act_layer=act_layer,
            block_style_list=block_style_list,
            combine_style=combine_style,
            k=k,
            n_group=n_group)
        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos):
        x = self.blocks(x, pos)
        return x


class PointTransformerDecoder(nn.Module):
    """ Vision Transformer for point cloud encoder/decoder
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self, embed_dim=384, depth=8, num_heads=6, mlp_ratio=2., qkv_bias=True, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=None, act_layer=None,
            self_attn_block_style_list=['attn-deform'], self_attn_combine_style='concat',
            cross_attn_block_style_list=['attn-deform'], cross_attn_combine_style='concat',
            k=8, n_group=2
    ):
        """
        Args:
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        assert len(self_attn_block_style_list) == len(cross_attn_block_style_list) == depth
        self.blocks = TransformerDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dpr,
            norm_layer=norm_layer,
            act_layer=act_layer,
            self_attn_block_style_list=self_attn_block_style_list,
            self_attn_combine_style=self_attn_combine_style,
            cross_attn_block_style_list=cross_attn_block_style_list,
            cross_attn_combine_style=cross_attn_combine_style,
            k=k,
            n_group=n_group
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, q, v, q_pos, v_pos, denoise_length=None):
        q = self.blocks(q, v, q_pos, v_pos, denoise_length=denoise_length)
        return q


class PointTransformerEncoderEntry(PointTransformerEncoder):
    def __init__(self):
        super().__init__()


class PointTransformerDecoderEntry(PointTransformerDecoder):
    def __init__(self):
        super().__init__()


######################################## Grouper ########################################
class DGCNN_Grouper(nn.Module):
    def __init__(self, k=16):
        super().__init__()
        '''
        K has to be 16
        '''
        print('using group version 2')
        self.k = k
        # self.knn = KNN(k=k, transpose_mode=False)
        self.input_trans = nn.Conv1d(3, 8, 1)

        self.layer1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 32),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 64),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 64),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 128),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )
        self.num_features = 128

    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous()  # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):
        # coor: bs, 3, np, x: bs, c, np

        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            # _, idx = self.knn(coor_k, coor_q)  # bs k np
            idx = knn_point(k, coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous())  # B G M
            idx = idx.transpose(-1, -2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, x, num):
        '''
            INPUT:
                x : bs N 3
                num : list e.g.[1024, 512]
            ----------------------
            OUTPUT:

                coor bs N 3
                f    bs N C(128) 
        '''
        x = x.transpose(-1, -2).contiguous()

        coor = x
        f = self.input_trans(x)

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer1(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, num[0])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer2(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer3(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, num[1])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer4(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        coor = coor.transpose(-1, -2).contiguous()
        f = f.transpose(-1, -2).contiguous()

        return coor, f


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class SimpleEncoder(nn.Module):
    def __init__(self, k=32, embed_dims=128):
        super().__init__()
        self.embedding = Encoder(embed_dims)
        self.group_size = k

        self.num_features = embed_dims

    def forward(self, xyz, n_group):
        # 2048 divide into 128 * 32, overlap is needed
        if isinstance(n_group, list):
            n_group = n_group[-1]

        center = fps(xyz, n_group)  # B G 3

        assert center.size(1) == n_group, f'expect center to be B {n_group} 3, but got shape {center.shape}'

        batch_size, num_points, _ = xyz.shape
        # knn to get the neighborhood
        idx = knn_point(self.group_size, xyz, center)
        assert idx.size(1) == n_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, n_group, self.group_size, 3).contiguous()

        assert neighborhood.size(1) == n_group
        assert neighborhood.size(2) == self.group_size

        features = self.embedding(neighborhood)  # B G C

        return center, features


######################################## Fold ########################################
class Fold(nn.Module):
    def __init__(self, in_channel, step, hidden_dim=512):
        super().__init__()

        self.in_channel = in_channel
        self.step = step

        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, 3, 1),
        )

    def forward(self, x):
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        return fd2


class SimpleRebuildFCLayer(nn.Module):
    def __init__(self, input_dims, step, hidden_dim=512):
        super().__init__()
        self.input_dims = input_dims
        self.step = step
        self.layer = Mlp(self.input_dims, hidden_dim, step * 3)

    def forward(self, rec_feature):
        '''
        Input BNC
        '''
        batch_size = rec_feature.size(0)
        g_feature = rec_feature.max(1)[0]
        token_feature = rec_feature

        patch_feature = torch.cat([
            g_feature.unsqueeze(1).expand(-1, token_feature.size(1), -1),
            token_feature
        ], dim=-1)
        rebuild_pc = self.layer(patch_feature).reshape(batch_size, -1, self.step, 3)
        assert rebuild_pc.size(1) == rec_feature.size(1)
        return rebuild_pc


######################################## PCTransformer ########################################
class PCTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.center_num = [512, 256]

        in_chans = 3
        self.num_query = query_num = 512
        global_feature_dim = 1024

        # base encoder
        self.grouper = DGCNN_Grouper(k=16)
        self.pos_embed = nn.Sequential(
            nn.Linear(in_chans, 128),
            nn.GELU(),
            nn.Linear(128, 384)
        )
        self.input_proj = nn.Sequential(
            nn.Linear(self.grouper.num_features, 512),
            nn.GELU(),
            nn.Linear(512, 384)
        )
        # Coarse Level 1 : Encoder
        self.encoder = PointTransformerEncoderEntry()

        self.increase_dim = nn.Sequential(
            nn.Linear(384, 1024),
            nn.GELU(),
            nn.Linear(1024, global_feature_dim))
        # query generator
        self.coarse_pred = nn.Sequential(
            nn.Linear(global_feature_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 3 * query_num)
        )
        self.mlp_query = nn.Sequential(
            nn.Linear(global_feature_dim + 3, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 384)
        )
        # assert decoder_config.embed_dim == encoder_config.embed_dim
        self.mem_link = nn.Identity()
        # Coarse Level 2 : Decoder
        self.decoder = PointTransformerDecoderEntry()

        self.query_ranking = nn.Sequential(
            nn.Linear(3, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, xyz):
        bs = xyz.size(0)
        coor, f = self.grouper(xyz, self.center_num)  # b n c
        pe = self.pos_embed(coor)
        x = self.input_proj(f)

        x = self.encoder(x + pe, coor)  # b n c
        global_feature = self.increase_dim(x)  # B 1024 N
        global_feature = torch.max(global_feature, dim=1)[0]  # B 1024

        coarse = self.coarse_pred(global_feature).reshape(bs, -1, 3)

        coarse_inp = fps(xyz, self.num_query // 2)  # B 128 3
        coarse = torch.cat([coarse, coarse_inp], dim=1)  # B 224+128 3?

        mem = self.mem_link(x)

        # query selection
        query_ranking = self.query_ranking(coarse)  # b n 1
        idx = torch.argsort(query_ranking, dim=1, descending=True)  # b n 1
        coarse = torch.gather(coarse, 1, idx[:, :self.num_query].expand(-1, -1, coarse.size(-1)))

        if self.training:
            # add denoise task
            # first pick some point : 64?
            picked_points = fps(xyz, 64)
            picked_points = jitter_points(picked_points)
            coarse = torch.cat([coarse, picked_points], dim=1)  # B 256+64 3?
            denoise_length = 64

            # produce query
            q = self.mlp_query(
                torch.cat([
                    global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                    coarse], dim=-1))  # b n c

            # forward decoder
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor, denoise_length=denoise_length)

            return q, coarse, denoise_length

        else:
            # produce query
            q = self.mlp_query(
                torch.cat([
                    global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                    coarse], dim=-1))  # b n c

            # forward decoder
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor)

            return q, coarse, 0


######################################## PoinTr ########################################

class AdaPoinTr(nn.Module):
    def __init__(self):
        super().__init__()
        self.trans_dim = 384
        self.num_query = 512
        self.num_points = 2048

        self.decoder_type = 'fc'
        assert self.decoder_type in ['fold', 'fc'], f'unexpected decoder_type {self.decoder_type}'

        self.fold_step = 8
        self.base_model = PCTransformer()

        if self.decoder_type == 'fold':
            self.factor = self.fold_step ** 2
            self.decode_head = Fold(self.trans_dim, step=self.fold_step, hidden_dim=256)  # rebuild a cluster point
        else:
            if self.num_points is not None:
                self.factor = self.num_points // self.num_query
                assert self.num_points % self.num_query == 0
                self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2,
                                                        step=self.num_points // self.num_query)  # rebuild a cluster point
            else:
                self.factor = self.fold_step ** 2
                self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.fold_step ** 2)
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.reduce_map = nn.Linear(self.trans_dim + 1027, self.trans_dim)
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = cd_loss_L1

    def get_loss(self, ret, gt, epoch=1):
        pred_coarse, denoised_coarse, denoised_fine, pred_fine = ret

        assert pred_fine.size(1) == gt.size(1)

        # denoise loss
        idx = knn_point(self.factor, gt, denoised_coarse)  # B n k
        denoised_target = index_points(gt, idx)  # B n k 3
        denoised_target = denoised_target.reshape(gt.size(0), -1, 3)
        assert denoised_target.size(1) == denoised_fine.size(1)
        loss_denoised = self.loss_func(denoised_fine, denoised_target)
        loss_denoised = loss_denoised * 0.5

        # recon loss
        loss_coarse = self.loss_func(pred_coarse, gt)
        loss_fine = self.loss_func(pred_fine, gt)
        loss_recon = loss_coarse + loss_fine

        return loss_denoised, loss_recon

    def forward(self, xyz):
        q, coarse_point_cloud, denoise_length = self.base_model(xyz)  # B M C and B M 3

        B, M, C = q.shape

        global_feature = self.increase_dim(q.transpose(1, 2)).transpose(1, 2)  # B M 1024
        global_feature = torch.max(global_feature, dim=1)[0]  # B 1024

        rebuild_feature = torch.cat([
            global_feature.unsqueeze(-2).expand(-1, M, -1),
            q,
            coarse_point_cloud], dim=-1)  # B M 1027 + C

        # NOTE: foldingNet
        if self.decoder_type == 'fold':
            rebuild_feature = self.reduce_map(rebuild_feature.reshape(B * M, -1))  # BM C
            relative_xyz = self.decode_head(rebuild_feature).reshape(B, M, 3, -1)  # B M 3 S
            rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2, 3)  # B M S 3

        else:
            rebuild_feature = self.reduce_map(rebuild_feature)  # B M C
            relative_xyz = self.decode_head(rebuild_feature)  # B M S 3
            rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-2))  # B M S 3

        if self.training:
            # split the reconstruction and denoise task
            pred_fine = rebuild_points[:, :-denoise_length].reshape(B, -1, 3).contiguous()
            pred_coarse = coarse_point_cloud[:, :-denoise_length].contiguous()

            denoised_fine = rebuild_points[:, -denoise_length:].reshape(B, -1, 3).contiguous()
            denoised_coarse = coarse_point_cloud[:, -denoise_length:].contiguous()

            assert pred_fine.size(1) == self.num_query * self.factor
            assert pred_coarse.size(1) == self.num_query

            ret = (pred_coarse, denoised_coarse, denoised_fine, pred_fine)
            return ret

        else:
            assert denoise_length == 0
            rebuild_points = rebuild_points.reshape(B, -1, 3).contiguous()  # B N 3

            assert rebuild_points.size(1) == self.num_query * self.factor
            assert coarse_point_cloud.size(1) == self.num_query

            ret = (coarse_point_cloud, rebuild_points)
            return ret
