# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.



from calendar import c
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

from semilearn.algorithms.utils import concat_all_gather
from semilearn.algorithms.hooks import MaskingHook
from semilearn.core.hooks import Hook

class SoftMatchWeightingHook(MaskingHook):
    """
    SoftMatch learnable truncated Gaussian weighting
    """
    def __init__(self, num_classes, n_sigma=2, momentum=0.999, per_class=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.n_sigma = n_sigma
        self.per_class = per_class
        self.m = momentum

        # initialize Gaussian mean and variance
        if not self.per_class:
            self.prob_max_mu_t = torch.tensor(1.0 / self.num_classes)
            self.prob_max_var_t = torch.tensor(1.0)
        else:
            self.prob_max_mu_t = torch.ones((self.num_classes)) / self.args.num_classes
            self.prob_max_var_t =  torch.ones((self.num_classes))

    @torch.no_grad()
    def update(self, algorithm, probs_x_ulb):
        if algorithm.distributed and algorithm.world_size > 1:
            probs_x_ulb = self.concat_all_gather(probs_x_ulb)
        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        if not self.per_class:
            prob_max_mu_t = torch.mean(max_probs) # torch.quantile(max_probs, 0.5)
            prob_max_var_t = torch.var(max_probs, unbiased=True)
            self.prob_max_mu_t = self.m * self.prob_max_mu_t + (1 - self.m) * prob_max_mu_t.item()
            self.prob_max_var_t = self.m * self.prob_max_var_t + (1 - self.m) * prob_max_var_t.item()
        else:
            prob_max_mu_t = torch.zeros_like(self.prob_max_mu_t)
            prob_max_var_t = torch.ones_like(self.prob_max_var_t)
            for i in range(self.num_classes):
                prob = max_probs[max_idx == i]
                if len(prob) > 1:
                    prob_max_mu_t[i] = torch.mean(prob)
                    prob_max_var_t[i] = torch.var(prob, unbiased=True)
            self.prob_max_mu_t = self.m * self.prob_max_mu_t + (1 - self.m) * prob_max_mu_t
            self.prob_max_var_t = self.m * self.prob_max_var_t + (1 - self.m) * prob_max_var_t
        return max_probs, max_idx
    
    @torch.no_grad()
    def masking(self, algorithm, logits_x_ulb, softmax_x_ulb=True, *args, **kwargs):
        if not self.prob_max_mu_t.is_cuda:
            self.prob_max_mu_t = self.prob_max_mu_t.to(logits_x_ulb.device)
        if not self.prob_max_var_t.is_cuda:
            self.prob_max_var_t = self.prob_max_var_t.to(logits_x_ulb.device)

        if softmax_x_ulb:
            probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()

        self.update(algorithm, probs_x_ulb)

        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        # compute weight
        if not self.per_class:
            mu = self.prob_max_mu_t
            var = self.prob_max_var_t
        else:
            mu = self.prob_max_mu_t[max_idx]
            var = self.prob_max_var_t[max_idx]
        mask = torch.exp(-((torch.clamp(max_probs - mu, max=0.0) ** 2) / (2 * var / (self.n_sigma ** 2))))
        return mask

  

class WSCFilterHook(Hook):
    """
    Mainly used for extend the feats to be connected.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def get_feats(self, algorithm, feat_dict, y_lb, pseudo_label, mask):
        conf_idx = (mask > algorithm.p_cutoff).nonzero(as_tuple=True)[0]

        conf_ulb_s_0 = feat_dict['x_ulb_s_0'][conf_idx]
        conf_ulb_s_1 = feat_dict['x_ulb_s_1'][conf_idx]

        feat_lb_s_0 = torch.cat([feat_dict['x_lb_s_0'], conf_ulb_s_0], dim=0)
        feat_lb_s_1 = torch.cat([feat_dict['x_lb_s_1'], conf_ulb_s_1], dim=0)

        feat_dict['x_lb_s_0'] = feat_lb_s_0
        feat_dict['x_lb_s_1'] = feat_lb_s_1

        y_lb = F.one_hot(y_lb, num_classes=pseudo_label.shape[1]).float()

        new_y_lb = torch.cat([y_lb, pseudo_label[conf_idx]], dim=0)

        return feat_dict, new_y_lb

class WeakSpectralLoss(nn.Module):
    def __init__(self, alpha, beta, cons):
        super(WeakSpectralLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.cons = cons

    def forward(self, zq1, zq2, q, z1=None, z2=None, pi=None):

        if pi is None:
            pi = (torch.ones(q.shape[1], 1) / q.shape[1]).cuda()

        Q = torch.matmul(q, q.t())
        W = torch.matmul(q, pi)
        W_double = torch.cat([W, W], dim=0)
        WxW = torch.matmul(W, W.t())
        if z1 is not None and z1.shape[0] >= 2:
            anti_identity_matrix_zq = (1 - torch.eye(zq1.shape[0])).cuda()
            anti_identity_matrix_z = (1 - torch.eye(z1.shape[0])).cuda()

            zq1xzq2 = torch.matmul(zq1, zq2.t())
            zq1xzq1 = torch.matmul(zq1, zq1.t())
            zq2xzq2 = torch.matmul(zq2, zq2.t())
            z1xz1 = torch.matmul(z1, z1.t())
            z2xz2 = torch.matmul(z2, z2.t())
            z1xz2 = torch.matmul(z1, z2.t())
            zqxz = torch.matmul(torch.cat([zq1, zq2], dim=0), torch.cat([z1, z2], dim=0).t())
            pow_zqxz = zqxz ** 2
            pow_zq1xzq2 = zq1xzq2 ** 2
            pow_z1xz2 = z1xz2 ** 2
            pow_z1xz1 = z1xz1 ** 2
            pow_z2xz2 = z2xz2 ** 2
            pow_zq1xzq1 = zq1xzq1 ** 2
            pow_zq2xzq2 = zq2xzq2 ** 2


            # compute l1
            l1 = -2 * self.alpha * (torch.trace(zq1xzq2) + torch.trace(z1xz2)) / (zq1.shape[0] + z1.shape[0])

            # compute l2
            # print(f"zq1 shape: {zq1.shape}, zq2 shape: {zq2.shape}, Q shape: {Q.shape} z1 shape: {z1.shape}, z2 shape: {z2.shape}")
            l2_1 = -2 * self.beta * (torch.sum(zq1xzq2 * Q * anti_identity_matrix_zq)) / (
                    zq1.shape[0] * (zq1.shape[0] - 1))
            l2_2 = -2 * self.beta * (torch.sum(zq1xzq1 * Q * anti_identity_matrix_zq)) / (
                    zq1.shape[0] * (zq1.shape[0] - 1))
            l2_3 = -2 * self.beta * (torch.sum(zq2xzq2 * Q * anti_identity_matrix_zq)) / (
                    zq1.shape[0] * (zq1.shape[0] - 1))
            l2 = (l2_1 + l2_2 + l2_3) / 3

            # compute l3
            l3_1 = (self.alpha * self.alpha * (torch.sum(pow_zq1xzq2 * anti_identity_matrix_zq) +
                                               torch.sum(pow_z1xz2 * anti_identity_matrix_z))
                    / (zq1.shape[0] * (zq1.shape[0] - 1) + z1.shape[0] * (z1.shape[0] - 1)))
            l3_2 = (self.alpha * self.alpha * (torch.sum(pow_zq1xzq1 * anti_identity_matrix_zq) +
                                               torch.sum(pow_z1xz1 * anti_identity_matrix_z))
                    / (zq1.shape[0] * (zq1.shape[0] - 1) + z1.shape[0] * (z1.shape[0] - 1)))
            l3_3 = (self.alpha * self.alpha * (torch.sum(pow_zq2xzq2 * anti_identity_matrix_zq) +
                                               torch.sum(pow_z2xz2 * anti_identity_matrix_z))
                    / (zq1.shape[0] * (zq1.shape[0] - 1) + z1.shape[0] * (z1.shape[0] - 1)))
            l3 = (l3_1 + l3_2 + l3_3) / 3

            # compute l4

            l4_1 = (self.beta * self.beta * torch.sum(pow_zq1xzq2 * anti_identity_matrix_zq * WxW)
                    / (zq1.shape[0] * (zq1.shape[0] - 1)))
            l4_2 = (self.beta * self.beta * torch.sum(pow_zq1xzq1 * anti_identity_matrix_zq * WxW)
                    / (zq1.shape[0] * (zq1.shape[0] - 1)))
            l4_3 = (self.beta * self.beta * torch.sum(pow_zq2xzq2 * anti_identity_matrix_zq * WxW)
                    / (zq1.shape[0] * (zq1.shape[0] - 1)))
            l4 = (l4_1 + l4_2 + l4_3) / 3

            # compute l5
            l5 = (self.beta * self.alpha * 2 *
                  torch.sum(
                      torch.diag_embed(W_double[:, 0]) @ pow_zqxz
                #        + (torch.diag_embed(W[:, 0]) @ pow_zq1xzq2) * anti_identity_matrix_zq
                       )
                  / (4 * zq1.shape[0] * z1.shape[0]
                #    + zq1.shape[0] * (zq1.shape[0] - 1)
                   ))

            loss = l1 + l2 + self.cons * (l3 + l4 + l5)


        else:
            anti_identity_matrix_zq = (1 - torch.eye(zq1.shape[0])).cuda()
            anti_identity_matrix_za = (1 - torch.eye(2 * zq1.shape[0])).cuda()
            zq1xzq2 = torch.matmul(zq1, zq2.t())
            zq1xzq1 = torch.matmul(zq1, zq1.t())
            zq2xzq2 = torch.matmul(zq2, zq2.t())
            zaxza = torch.matmul(torch.cat([zq1, zq2], dim=0), torch.cat([zq1, zq2], dim=0).t())
            pow_zaxza = zaxza ** 2
            # pow_zq1xzq1 = zq1xzq1 ** 2
            # pow_zq2xzq2 = zq2xzq2 ** 2
            # pow_zq1xzq2 = zq1xzq2 ** 2
            #
            l1 = -2 * self.alpha * torch.trace(zq1xzq2) / zq1.shape[0]

            l2_1 = -2 * self.beta * (torch.sum(zq1xzq2 * Q * anti_identity_matrix_zq)) / (
                    zq1.shape[0] * (zq1.shape[0] - 1))
            l2_2 = -2 * self.beta * (torch.sum(zq1xzq1 * Q * anti_identity_matrix_zq)) / (
                    zq1.shape[0] * (zq1.shape[0] - 1))
            l2_3 = -2 * self.beta * (torch.sum(zq2xzq2 * Q * anti_identity_matrix_zq)) / (
                    zq1.shape[0] * (zq1.shape[0] - 1))
            l2 = (l2_1 + l2_2 + l2_3) / 3

            l3_new = self.cons *  (self.alpha + self.beta / q.shape[1]) * (self.alpha + self.beta / q.shape[1])  * torch.sum(pow_zaxza * anti_identity_matrix_za) / (2 * zq1.shape[0] * (2 * zq1.shape[0] - 1))
            # loss = l1 + l2 + l3 + l4 + l5
            loss = l1 + l2 + l3_new

        return loss, l1, l2


