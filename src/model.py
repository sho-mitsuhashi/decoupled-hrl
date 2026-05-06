import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


################################################################################
#
# Policy Network
#
################################################################################


class Actor(nn.Module):
    """
    The policy network
    """
    def __init__(self, args):
        super(Actor, self).__init__()
        self.max_action = args.max_action
        dim_state  = args.dim_state
        dim_hidden = args.dim_hidden
        dim_action = args.dim_action
        dim_goal   = args.dim_goal

        if args.actor_capable:
            self.net = nn.Sequential(
                nn.Linear(dim_state+dim_goal, dim_hidden),
                nn.LayerNorm(dim_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(dim_hidden, dim_hidden),
                nn.LayerNorm(dim_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(dim_hidden, dim_hidden),
                nn.LayerNorm(dim_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(dim_hidden, dim_action),
                nn.Tanh()
            )
        else:#Degraded actor
            self.net = nn.Sequential(
                nn.Linear(dim_state+dim_goal, 8),
                nn.LayerNorm(8),
                nn.ReLU(inplace=True),
                nn.Linear(8, dim_action),
                nn.Tanh()
            )



    def forward(self, s, g):
        x = torch.cat([s, g], -1)
        actions = self.max_action * self.net(x)
        return actions

class PrimitiveActor(nn.Module):
    """
    The primitive policy network (s,g)->primitive action
    """
    def __init__(self, args):
        super().__init__()
        self.max_action = args.max_action
        dim_state  = args.dim_state
        dim_hidden = args.dim_prm_hidden
        dim_action = args.dim_action
        dim_goal   = args.dim_goal
        dim_latent=args.dim_hierarchical_latent

        if args.actor_capable:
            self.net = nn.Sequential(
                nn.Linear(dim_state+dim_goal, dim_hidden),
                nn.LayerNorm(dim_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(dim_hidden, dim_hidden),
                nn.LayerNorm(dim_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(dim_hidden, dim_hidden),
                nn.LayerNorm(dim_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(dim_hidden, dim_action),
                nn.Tanh()
            )
        else:#Degraded actor
            self.net = nn.Sequential(
                nn.Linear(dim_state+dim_goal, 8),
                nn.LayerNorm(8),
                nn.ReLU(inplace=True),
                nn.Linear(8, dim_action),
                nn.Tanh()
            )

    def forward(self, s, zg):
        x = torch.cat([s, zg], -1)
        actions = self.max_action * self.net(x)
        return actions

class HighLevelActor(nn.Module):

    def __init__(self, args):
        super().__init__()
        dim_state  = args.dim_state
        dim_hidden = args.dim_hidden
        dim_action = args.dim_action
        dim_goal   = args.dim_goal
        dim_latent=args.dim_hierarchical_latent

        # Actor も同じ構造のエンコーダを使って latent を出力
        self.net = nn.Sequential(
            nn.Linear(dim_latent + dim_latent, dim_hidden),
            nn.LayerNorm(dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_latent),
            nn.LayerNorm(dim_latent),
            nn.ReLU(inplace=True),
            
        )


    def forward(self, fh, phih):
        x = torch.cat([fh, phih], dim=-1)
        z = self.net(x)


        return z


###Hierarchical 

class CriticHierarchical(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_action = args.max_action
        dim_state  = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_action = args.dim_action
        dim_goal   = args.dim_goal
        dim_embed  = args.dim_embed
        dim_latent=args.dim_hierarchical_latent
        self.dim_embed = dim_embed


        def make_encoder(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, dim_hidden),
                nn.LayerNorm(dim_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(dim_hidden, dim_latent),
                nn.LayerNorm(dim_latent),
                nn.ReLU(inplace=True),
                
            )
        
        self.f_emb   = make_encoder(dim_state + dim_action)
        self.phi_emb = make_encoder(dim_state +dim_goal)
        self.encoder_head  = nn.Identity()

        # sym/asym はそのまま
        self.sym  = nn.Sequential(
            nn.Linear(dim_latent, dim_hidden),nn.LayerNorm(dim_hidden),nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed)
        )
        self.asym = nn.Sequential(
            nn.Linear(dim_latent, dim_hidden),nn.LayerNorm(dim_hidden),nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed)
        )



    def forward_encoder_head(self, z):
        # 潜在埋め込み（L2正規化で単位長に）

        latent   =self.encoder_head(z) 
        return latent

    def forward_encoder_sa(self, s, a):
        # 潜在埋め込み
        x1 = torch.cat([s, a / self.max_action], dim=-1)
        fh   =self.encoder_head(self.f_emb(x1))
        return fh

    def forward_encoder_sg(self, s, g):
        # 潜在埋め込み
        x2 = torch.cat([s,g],             dim=-1)
        phih = self.encoder_head(self.phi_emb(x2))

        return phih

    def forward_latent(self, fh , phih):


        # sym/asym 部分
        sym1  = self.sym(fh)
        sym2  = self.sym(phih)


        asym1 = self.asym(fh)
        asym2 = self.asym(phih)

        # 距離計算
        dist_s = (sym1 - sym2).pow(2).sum(-1, keepdim=True).sqrt()
        res    = F.relu(asym1 - asym2)
        dist_a = res.max(-1, keepdim=True)[0].view(-1, 1)
        dist   = dist_s + dist_a

        return -dist

    def forward(self, s, a, g):
        """
        標準的な forward 関数。
        s: state tensor  (batch, dim_state)
        a: action tensor (batch, dim_action)
        g: goal tensor   (batch, dim_goal)
        """
        fh   = self.forward_encoder_sa(s, a)
        phih = self.forward_encoder_sg(s, g)
        return self.forward_latent(fh, phih)
