# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import SoftMatchWeightingHook, WeakSpectralLoss, WSCFilterHook
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, DistAlignEMAHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool

class WSC_Net(nn.Module):
    """
    WSC Net
    """
    def __init__(self, base, proj_size=128):
        super(WSC_Net, self).__init__()
        self.backbone = base
        self.proj_size = proj_size
        self.num_features = base.num_features

        self.mlp_proj = nn.Sequential(nn.Linear(self.num_features, self.num_features),
                            nn.BatchNorm1d(self.num_features),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.num_features, self.proj_size),
                            nn.BatchNorm1d(self.proj_size),
                            )
        
    def forward(self, x):
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)

        feat_proj = F.normalize(self.mlp_proj(feat), dim=-1)
        return {
            'logits': logits,
            'feat': feat_proj
        }
    
    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher


@ALGORITHMS.register('wsc')
class WSC(AlgorithmBase):
    """
        WSC base on SoftMatch algorithm (https://openreview.net/forum?id=ymt1zQXBDiF&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2023%2FConference%2FAuthors%23your-submissions)).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
            - ema_p (`float`):
                exponential moving average of probability update
        """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        self.init(T=args.T, hard_label=args.hard_label, dist_align=args.dist_align, dist_uniform=args.dist_uniform, ema_p=args.ema_p, n_sigma=args.n_sigma, per_class=args.per_class, spec_alpha=args.spec_alpha, spec_beta=args.spec_beta, consist_lam=args.consist_lam, lambda_wsc=args.lambda_wsc, p_cutoff=args.p_cutoff)
    
    def init(self, T, hard_label=True, dist_align=True, dist_uniform=True, ema_p=0.999, n_sigma=2, per_class=False, spec_alpha=2.0, spec_beta=15.0, consist_lam=3.0, lambda_wsc=1.0, p_cutoff=0.5):
        self.T = T
        self.use_hard_label = hard_label
        self.dist_align = dist_align
        self.dist_uniform = dist_uniform
        self.ema_p = ema_p
        self.n_sigma = n_sigma
        self.per_class = per_class
        self.spec_alpha = spec_alpha
        self.spec_beta = spec_beta
        self.lambda_wsc = lambda_wsc
        self.p_cutoff = p_cutoff
        self.weak_spectral_loss = WeakSpectralLoss(alpha=self.spec_alpha, beta=self.spec_beta, cons=consist_lam)

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(
            DistAlignEMAHook(num_classes=self.num_classes, momentum=self.args.ema_p, p_target_type='uniform' if self.args.dist_uniform else 'model'), 
            "DistAlignHook")
        self.register_hook(SoftMatchWeightingHook(num_classes=self.num_classes, n_sigma=self.args.n_sigma, momentum=self.args.ema_p, per_class=self.args.per_class), "MaskingHook")
        # self.register_hook(FixedThresholdingHook(), "FixedThresholdingHook")
        self.register_hook(WSCFilterHook(), "WSCFilterHook")
        super().set_hooks()

    def set_model(self): 
        model = super().set_model()
        model = WSC_Net(model, proj_size=self.args.proj_size)
        return model
    
    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model = WSC_Net(ema_model, proj_size=self.args.proj_size)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def train_step(self, x_lb_w, x_lb_s_0, x_lb_s_1, y_lb, x_ulb_w, x_ulb_s_0, x_ulb_s_1):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb_w, x_lb_s_0, x_lb_s_1, x_ulb_w, x_ulb_s_0, x_ulb_s_1))
                outputs = self.model(inputs)
                logits_x_lb_w, logits_x_lb_s_0, logits_x_lb_s_1 = outputs['logits'][:num_lb*3].chunk(3)
                logits_x_ulb_w, logits_x_ulb_s_0, logits_x_ulb_s_1 = outputs['logits'][num_lb*3:].chunk(3)
                feats_x_lb_w, feat_x_lb_s_0, feat_x_lb_s_1 = outputs['feat'][:num_lb*3].chunk(3)
                feats_x_ulb_w, feats_x_ulb_s_0, feats_x_ulb_s_1 = outputs['feat'][num_lb*3:].chunk(3)
            else:
                raise NotImplementedError("WSC does not support use_cat=False")
            feat_dict = {'x_lb_w':feats_x_lb_w, 'x_lb_s_0':feat_x_lb_s_0, 'x_lb_s_1':feat_x_lb_s_1, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s_0':feats_x_ulb_s_0, 'x_ulb_s_1':feats_x_ulb_s_1}


            sup_loss = self.ce_loss(logits_x_lb_w, y_lb, reduction='mean')

            probs_x_lb = torch.softmax(logits_x_lb_w.detach(), dim=-1)
            probs_x_ulb_w = torch.softmax(logits_x_ulb_w.detach(), dim=-1)

            # uniform distribution alignment 
            probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w, probs_x_lb=probs_x_lb)

            # calculate weight
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            # mask_wsc = self.call_hook("masking", "FixedThresholdingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          # make sure this is logits, not dist aligned probs
                                          # uniform alignment in softmatch do not use aligned probs for generating pesudo labels
                                          logits=logits_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T)
            
            soft_pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          # make sure this is logits, not dist aligned probs
                                          # uniform alignment in softmatch do not use aligned probs for generating pesudo labels
                                          logits=logits_x_ulb_w,
                                          use_hard_label=False,
                                          T=self.T)

            # calculate loss
            unsup_loss = self.consistency_loss(logits_x_ulb_s_0,
                                          pseudo_label,
                                          'ce',
                                          mask=mask)
            unsup_loss += self.consistency_loss(logits_x_ulb_s_1,
                                          pseudo_label,
                                          'ce',
                                          mask=mask)
            
            wsc_feat_dict, wsc_q = self.call_hook("get_feats", "WSCFilterHook", feat_dict=feat_dict, y_lb=y_lb, pseudo_label=soft_pseudo_label, mask=mask)

            wsc_loss, _, _ = self.weak_spectral_loss(
                wsc_feat_dict['x_lb_s_0'], wsc_feat_dict['x_lb_s_1'],
                wsc_q, 
                wsc_feat_dict['x_ulb_s_0'], wsc_feat_dict['x_ulb_s_1'],
            )

            lambda_w = float(self.epoch) / float(self.epochs)


            total_loss = sup_loss + self.lambda_u / 2 * unsup_loss + lambda_w * wsc_loss


        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         wsc_loss=wsc_loss.item(),
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    # TODO: change these
    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu()
        save_dict['p_target'] = self.hooks_dict['DistAlignHook'].p_target.cpu()
        save_dict['prob_max_mu_t'] = self.hooks_dict['MaskingHook'].prob_max_mu_t.cpu()
        save_dict['prob_max_var_t'] = self.hooks_dict['MaskingHook'].prob_max_var_t.cpu()
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['DistAlignHook'].p_target = checkpoint['p_target'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].prob_max_mu_t = checkpoint['prob_max_mu_t'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].prob_max_var_t = checkpoint['prob_max_var_t'].cuda(self.args.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--dist_align', str2bool, True),
            SSL_Argument('--dist_uniform', str2bool, True),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--n_sigma', int, 2),
            SSL_Argument('--per_class', str2bool, False),
            SSL_Argument('--spec_alpha', float, 2.0),
            SSL_Argument('--spec_beta', float, 15.0),
            SSL_Argument('--consist_lam', float, 3.0),
            SSL_Argument('--lambda_wsc', float, 1.0),
            SSL_Argument('--p_cutoff', float, 0.5),
            SSL_Argument('--proj_size', int, 128),
        ]
