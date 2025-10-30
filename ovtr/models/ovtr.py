# Copyright (c) Jinyang Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MOTR (https://github.com/megvii-research/MOTR)
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
"""
DETR model and criterion classes.
"""

import torch
import torch.nn.functional as F
from torch import nn
from typing import List
import copy
from util import box_ops, checkpoint
from util.misc import (NestedTensor, nested_tensor_from_tensor_list, get_world_size,
                       is_dist_avail_and_initialized, inverse_sigmoid,)

from detectron2.structures import Instances, Boxes, matched_boxlist_iou
from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer
from .updater import build as build_updater
from .deformable_detr import SetCriterion
from .segmentation import sigmoid_focal_loss

from util.clip_utils import load_embeddings
from .utils import MLP, protect_det_preds, protect_track_preds, preprocess_for_masks
from util.list_LVIS import Frequency_list_total_1, Frequency_list_70, novel_class

class TrackerPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, processor_dct=None):
        super().__init__()
        self.processor_dct = processor_dct

    @torch.no_grad()
    def forward(self, track_instances: Instances, target_size) -> Instances:
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits = track_instances.pred_logits
        out_bbox = track_instances.pred_boxes

        prob = out_logits.sigmoid()
        scores, labels = prob.max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = boxes.clamp(0, 1)  
        
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_size
        scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes)
        boxes = boxes * scale_fct[None, :]

        track_instances.boxes = boxes
        track_instances.scores = scores
        track_instances.labels = labels

        track_instances.remove('pred_logits')
        track_instances.remove('pred_boxes')
        return track_instances

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.6, filter_score_thresh=0.6, miss_tolerance=5, maximum_quantity=50):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0
        self.maximum_quantity = maximum_quantity

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances, _track_discard, is_repeat=False):
        cancel_disappear = track_instances.scores >= self.score_thresh
        cancel_disappear[_track_discard] = False
        track_instances.disappear_time[cancel_disappear] = 0
        # Found valid index
        score_indx = track_instances.scores >= self.score_thresh
        obj_indx = track_instances.obj_idxes != -1 
        valid_indx = score_indx | obj_indx
        track_instances = track_instances[valid_indx]

        if len(track_instances) > self.maximum_quantity:
            top_indices = self.quantity_filter(track_instances, self.maximum_quantity) 
            track_instances = track_instances[top_indices]

        for i in range(len(track_instances)):
            if track_instances.obj_idxes[i] == -2:
                continue
            elif track_instances.obj_idxes[i] == -1 and track_instances.scores[i] >= self.score_thresh:
                # print("track {} has score {:.2f}, assign obj_id {}, cls is {}".format(i, track_instances.scores[i], self.max_obj_id, track_instances.cls_idxes[i]))
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1
            elif track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] < self.filter_score_thresh and is_repeat is False:
                track_instances.disappear_time[i] += 1
                # print(track_instances.obj_idxes[i])
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    # Set the obj_id to -1.
                    # Then this track will be removed by TrackEmbeddingLayer.
                    track_instances.obj_idxes[i] = -1
                    # print("track {} has score {:.2f}, disappear".format(i, track_instances.scores[i], self.max_obj_id))
            # elif (track_instances.obj_idxes[i] >= 0) and (track_instances.scores[i] >= self.filter_score_thresh) and (track_instances.keep_cls[i] == False):
            #     print("track {} keeps origin obj_id {}, cls changes to {}".format(i, track_instances.obj_idxes[i], track_instances.cls_idxes[i]))
                # track_instances.obj_idxes[i] = self.max_obj_id
                # self.max_obj_id += 1
        return track_instances
    
    @staticmethod
    def quantity_filter(track_instances, maximum_quantity):
        scores = track_instances.scores
        _, top_indices = torch.topk(scores, k=maximum_quantity, sorted=False)
        top_indices = torch.sort(top_indices).values
        return top_indices


class OVFrameMatcher(SetCriterion):
    def __init__(self, num_classes,
                        matcher,
                        weight_dict,
                        losses,
                        random_drop=0,
                        calculate_negative_samples=True,
                        num_queries=900,
                        train_with_artificial_img_seqs=False,
                        train_with_pseudo=False,
                        ):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__(num_classes, matcher, weight_dict, losses,)
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_loss = True
        self.losses_dict = {}
        self._current_frame_idx = 0
        self.random_drop = random_drop

        self.num_queries = num_queries
        self.calculate_negative_samples = calculate_negative_samples
        self.train_with_artificial_img_seqs = train_with_artificial_img_seqs
        self.train_with_pseudo = train_with_pseudo

    def initialize(self, gt_instances: List[Instances]):
        self.gt_instances = gt_instances
        self.num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = {}

    def _step(self):
        self._current_frame_idx += 1

    def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float, device=self.sample_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

    def get_loss(self, loss, outputs, gt_instances, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            "align": self.loss_align,
            "align_pre": self.loss_align_pre,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, gt_instances, indices, num_boxes, **kwargs)

    def loss_labels(self, outputs, gt_instances: List[Instances], indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs['pred_logits']
        weights_bq = None

        def get_src_permutation_idx(indices):
            batch_idx = torch.cat([torch.full_like(src[filt != -1], i) for i, (src, filt) in enumerate(indices)])
            src_idx = torch.cat([src[filt != -1] for (src, filt) in indices])
            return batch_idx, src_idx
        idx = get_src_permutation_idx(indices)
        
        if self.calculate_negative_samples:
            num_class = len(outputs["select_id"])
        else:
            num_class = len(torch.unique(gt_instances[0].labels))
        
        target_classes = torch.full(src_logits.shape[:2], num_class,
                                    dtype=torch.int64, device=src_logits.device)

        select_id = outputs["select_id"]
        labels_ori = torch.cat([t.labels[J[J != -1]] for t, (_, J) in zip(gt_instances, indices)])
        tgt_ids_all = torch.cat([(select_id == lid).nonzero(as_tuple=False)[0] for lid in labels_ori])
        target_classes[idx] = tgt_ids_all
        if self.train_with_pseudo:
            # gather scores for matched targets
            labels_scores = torch.cat([t.scores[J[J != -1]] for t, (_, J) in zip(gt_instances, indices)])
            weights_bq = torch.ones(src_logits.shape[:2], dtype=src_logits.dtype, device=src_logits.device)
            # idx selects positives (batch_idx, src_idx)
            weights_bq[idx] = labels_scores.to(weights_bq)
      
        if self.focal_loss:                                                                    
            gt_labels_target = F.one_hot(target_classes, num_classes=num_class + 1)[:, :,
                               :-1]  # no loss for the last (background) class
            gt_labels_target = gt_labels_target.to(src_logits)
            loss_ce = sigmoid_focal_loss(src_logits[:, :, :num_class],
                                             gt_labels_target,
                                             alpha=0.25,
                                             gamma=2,
                                             num_boxes=num_boxes, mean_in_dim1=True, weights=weights_bq) * src_logits.shape[1]
        else:
            loss_ce = F.cross_entropy(src_logits[:, :, num_class].transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        return losses
    
    def loss_boxes(self, outputs, gt_instances: List[Instances], indices: List[tuple], num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([gt_per_img.boxes[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)

        # for pad target, don't calculate regression loss, judged by whether obj_id=-1
        target_obj_ids = torch.cat([gt_per_img.obj_ids[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)
        mask = (target_obj_ids != -1)

        loss_bbox = F.l1_loss(src_boxes[mask], target_boxes[mask], reduction='none')
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes[mask]),
            box_ops.box_cxcywh_to_xyxy(target_boxes[mask])))

        # Optional weighting by gt scores
        use_weight = False
        weights = None
        if self.train_with_pseudo:
            target_scores = torch.cat([gt_per_img.scores[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)
            weights = target_scores[mask].to(loss_bbox)
            use_weight = weights.numel() > 0

        losses = {}
        if use_weight:
            losses['loss_bbox'] = (loss_bbox.sum(-1) * weights).sum() / num_boxes
            losses['loss_giou'] = (loss_giou * weights).sum() / num_boxes
        else:
            losses['loss_bbox'] = loss_bbox.sum() / num_boxes
            losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses
    
    def loss_align(self, outputs, targets, indices, num_boxes, l1_distillation=False):
        """Alignment mechanism guides generalization capabilities and aligned queries.
        """
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx

        idx = self._get_src_permutation_idx(indices)
        src_feature = outputs["pred_embed"][idx]

        select_id = outputs["select_id"]
        image_feat = outputs["image_feat"]
        target_feature = []
        for t, (_, i) in zip(targets, indices):
            for c in t.labels[i]:
                index = (select_id == c).nonzero(as_tuple=False)[0]
                target_feature.append(image_feat[index])
        target_feature = torch.cat(target_feature, dim=0)
        # l1 normalize the feature
        src_feature = nn.functional.normalize(src_feature, dim=1)
        if l1_distillation:
            loss_feature = F.l1_loss(src_feature, target_feature, reduction="none")
        else:
            loss_feature = F.mse_loss(src_feature, target_feature, reduction="none")
        losses = {"loss_align": loss_feature.sum() / num_boxes}
        return losses
    
    def loss_align_pre(self, outputs, targets, indices, num_boxes):
        """Preserve text features without sudden variations.
        """
        input_feat = outputs["input_feat"]
        loss_feature_all = []

        select_id = outputs["select_id"]
        uniq_labels = [torch.unique(t.labels) for t in targets]
        tgt_ids_all = []
        embed_bs_index = []
        for i, uniq_label in enumerate(uniq_labels):
            tgt_ids = []
            if len(uniq_label)==0:
                continue
            else:
                embed_bs_index.append(i)
                for lid in uniq_label:
                    index = (select_id == lid).nonzero(as_tuple=False)[0]
                    tgt_ids.append(index)
                tgt_ids = torch.cat(tgt_ids)
            tgt_ids_all.append(tgt_ids)
        
        input_feats = torch.cat([input_feat[i] for i in tgt_ids_all])
        encoder_embeds = outputs["text_embed"][:, embed_bs_index]

        for encoder_embed in encoder_embeds:
            src_feature = torch.cat([enc_embed[tgt_id] for enc_embed, tgt_id in zip(encoder_embed,tgt_ids_all)])
            # l2 normalize the feature
            src_feature = nn.functional.normalize(src_feature, dim=1)
            loss_feature = F.mse_loss(src_feature, input_feats, reduction="none")
            loss_feature_all.append(loss_feature.sum() / num_boxes)
        loss_feature_all = torch.stack(loss_feature_all)
        loss_encoder_align = loss_feature_all.sum()
        losses = {"loss_align_pre": loss_encoder_align}
        return losses
    
    def match_for_single_frame(self, outputs: dict, is_first=None):
        outputs_without_aux = {k: v for k, v in outputs.items() if 
                               k != 'aux_outputs' and k != 'enc_outputs'}

        def select_unmatched_indexes(matched_indexes: torch.Tensor, num_total_indexes: int) -> torch.Tensor:
            matched_indexes_set = set(matched_indexes.detach().cpu().numpy().tolist())
            all_indexes_set = set(list(range(num_total_indexes)))
            unmatched_indexes_set = all_indexes_set - matched_indexes_set
            unmatched_indexes = torch.as_tensor(list(unmatched_indexes_set), dtype=torch.long).to(matched_indexes)
            return unmatched_indexes

        gt_instances_i = self.gt_instances[self._current_frame_idx]  # gt instances of i-th image.
        track_instances_last: Instances = outputs_without_aux['track_instances']

        if self.train_with_artificial_img_seqs:
            shielded_ids = protect_det_preds(outputs_without_aux, num_queries=self.num_queries)
            keep_indices = torch.ones(len(track_instances_last), dtype=torch.bool, device=shielded_ids.device)
            keep_indices[shielded_ids] = False
            track_instances = track_instances_last[keep_indices]
        else:
            track_instances = track_instances_last
            keep_indices = torch.ones(len(track_instances_last), dtype=torch.bool, device=track_instances_last.obj_idxes.device)

        outputs_i = {
            'pred_logits': track_instances.pred_logits.unsqueeze(0),
            'pred_boxes': track_instances.pred_boxes.unsqueeze(0),
            'pred_embed': outputs_without_aux['pred_embed'][0, keep_indices].unsqueeze(0),
            'select_id':outputs_without_aux['select_id'],
            'image_feat':outputs_without_aux['image_feat'],
        }

        obj_idxes = gt_instances_i.obj_ids
        device = obj_idxes.device
        obj_idxes_list = obj_idxes.detach().cpu().numpy().tolist()
        obj_idx_to_gt_idx = {obj_idx: gt_idx for gt_idx, obj_idx in enumerate(obj_idxes_list)}

        # step1. inherit and update the previous tracks.
        num_disappear_track = 0
        track_instances.matched_gt_idxes[:] = -1
        valid_track_mask = track_instances.obj_idxes >= 0
        valid_track_idxes = torch.arange(len(track_instances), device=device)[valid_track_mask]
        valid_obj_idxes = track_instances.obj_idxes[valid_track_idxes]
        for j in range(len(valid_obj_idxes)):
            obj_id = valid_obj_idxes[j].item()
            if obj_id in obj_idx_to_gt_idx:
                track_instances.matched_gt_idxes[valid_track_idxes[j]] = obj_idx_to_gt_idx[obj_id]
            else:
                num_disappear_track += 1

        full_track_idxes = torch.arange(len(track_instances), dtype=torch.long, device=device)
        matched_track_idxes = (track_instances.obj_idxes >= 0) # occu 
        prev_matched_indices = torch.stack(
            [full_track_idxes[matched_track_idxes], track_instances.matched_gt_idxes[matched_track_idxes]], dim=1).to(device)

        # step2. select the unmatched slots.
        # note that the fp tracks (obj_idxes == -2) will not be selected here.
        unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes == -1]

        # step3. select the unmatched gt instances (new tracks).
        tgt_indexes = track_instances.matched_gt_idxes
        tgt_indexes = tgt_indexes[tgt_indexes != -1]

        unmatched_tgt_indexes = select_unmatched_indexes(tgt_indexes, len(gt_instances_i))
        unmatched_gt_instances = gt_instances_i[unmatched_tgt_indexes]

        def match_for_single_decoder_layer(unmatched_outputs, matcher, unmatched_track_idxes):
            new_track_indices = matcher(unmatched_outputs,
                                             [unmatched_gt_instances])   # list[tuple(src_idx, tgt_idx)]

            # map the matched pair indexes to original index-space.
            src_idx = new_track_indices[0][0]
            tgt_idx = new_track_indices[0][1]
            # concat src and tgt for loss calculation.
            new_matched_indices = torch.stack([unmatched_track_idxes[src_idx], unmatched_tgt_indexes[tgt_idx]],
                                              dim=1).to(device)
            return new_matched_indices

        # step4. do matching between the unmatched slots and GTs.
        unmatched_outputs = {
            'pred_logits': track_instances.pred_logits[unmatched_track_idxes].unsqueeze(0),
            'pred_boxes': track_instances.pred_boxes[unmatched_track_idxes].unsqueeze(0),
            'select_id':outputs_without_aux['select_id'],
        }

        new_matched_indices = match_for_single_decoder_layer(unmatched_outputs, self.matcher, unmatched_track_idxes)

        # step5. update obj_idxes according to the new matching result.
        track_instances.obj_idxes[new_matched_indices[:, 0]] = gt_instances_i.obj_ids[new_matched_indices[:, 1]].long()
        track_instances.matched_gt_idxes[new_matched_indices[:, 0]] = new_matched_indices[:, 1]

        # step6. calculate iou.
        active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.matched_gt_idxes >= 0)
        active_track_boxes = track_instances.pred_boxes[active_idxes]
        if len(active_track_boxes) > 0:
            gt_boxes = gt_instances_i.boxes[track_instances.matched_gt_idxes[active_idxes]]
            active_track_boxes = box_ops.box_cxcywh_to_xyxy(active_track_boxes)
            gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
            track_instances.iou[active_idxes] = matched_boxlist_iou(Boxes(active_track_boxes), Boxes(gt_boxes))

        # step7. merge the unmatched pairs and the matched pairs.
        matched_indices = torch.cat([new_matched_indices, prev_matched_indices], dim=0) 

        # step8. calculate losses.
        self.num_samples += len(gt_instances_i) + num_disappear_track
        self.sample_device = device

        for loss in self.losses:
            new_track_loss = self.get_loss(loss,
                                           outputs=outputs_i,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices[:, 0], matched_indices[:, 1])],
                                           num_boxes=1)
            self.losses_dict.update(
                {'frame_{}_{}'.format(self._current_frame_idx, key): value for key, value in new_track_loss.items()})

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):

                # Shield and match individually for each layer.
                if self.train_with_artificial_img_seqs:
                    _shielded_ids_layer = protect_det_preds(aux_outputs, num_queries=self.num_queries)
                    _keep_indices_layer = torch.ones(len(track_instances_last), dtype=torch.bool, device=_shielded_ids_layer.device)
                    _keep_indices_layer[_shielded_ids_layer] = False
                    track_instances_layer = track_instances_last[_keep_indices_layer]
                else:
                    _keep_indices_layer = torch.ones(len(track_instances_last), dtype=torch.bool, device=track_instances_last.obj_idxes.device)
                    track_instances_layer = track_instances_last

                # step1*. inherit and update the previous tracks.
                track_instances_layer.matched_gt_idxes[:] = -1
                valid_track_mask = track_instances_layer.obj_idxes >= 0
                valid_track_idxes = torch.arange(len(track_instances_layer), device=device)[valid_track_mask]
                valid_obj_idxes = track_instances_layer.obj_idxes[valid_track_idxes]
                for j in range(len(valid_obj_idxes)):
                    obj_id = valid_obj_idxes[j].item()
                    if obj_id in obj_idx_to_gt_idx:
                        track_instances_layer.matched_gt_idxes[valid_track_idxes[j]] = obj_idx_to_gt_idx[obj_id]

                full_track_idxes = torch.arange(len(track_instances_layer), dtype=torch.long, device=device)
                matched_track_idxes_layer = (track_instances_layer.obj_idxes >= 0)
                prev_matched_indices_layer = torch.stack(
                    [full_track_idxes[matched_track_idxes_layer], track_instances_layer.matched_gt_idxes[matched_track_idxes_layer]], dim=1).to(device)

                # step2*. select the unmatched slots.
                unmatched_track_idxes_layer = full_track_idxes[track_instances_layer.obj_idxes == -1]

                # step3*. do matching between the unmatched slots and GTs.
                unmatched_outputs_layer = {
                    'pred_logits': aux_outputs['pred_logits'][0, _keep_indices_layer][unmatched_track_idxes_layer].unsqueeze(0),
                    'pred_boxes': aux_outputs['pred_boxes'][0, _keep_indices_layer][unmatched_track_idxes_layer].unsqueeze(0),
                    'select_id': aux_outputs['select_id'],
                }
                new_matched_indices_layer = match_for_single_decoder_layer(unmatched_outputs_layer, self.matcher, unmatched_track_idxes_layer)

                # step4*. merge the unmatched pairs and the matched pairs.
                matched_indices_layer = torch.cat([new_matched_indices_layer, prev_matched_indices_layer], dim=0)

                # step5*. calculate losses.
                _keep_aux_outputs = {
                    'pred_logits': aux_outputs['pred_logits'][0, _keep_indices_layer].unsqueeze(0),
                    'pred_boxes': aux_outputs['pred_boxes'][0, _keep_indices_layer].unsqueeze(0),
                    'pred_embed': aux_outputs['pred_embed'][0, _keep_indices_layer].unsqueeze(0),
                    'select_id': aux_outputs['select_id'],
                    'image_feat': aux_outputs['image_feat'],
                }
                for loss in self.losses:
                    l_dict = self.get_loss(loss,
                                           _keep_aux_outputs,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices_layer[:, 0], matched_indices_layer[:, 1])],
                                           num_boxes=1, )
                    self.losses_dict.update(
                        {'frame_{}_aux{}_{}'.format(self._current_frame_idx, i, key): value for key, value in
                         l_dict.items()})
            
        self._step()
        return track_instances

    def forward(self, outputs):
        losses = outputs.pop("losses_dict")
        num_samples = self.get_num_boxes(self.num_samples)
        loss_avg = {}
        for loss_name, _ in losses.items():
            loss_avg[loss_name] = losses[loss_name] / num_samples
        return loss_avg
    

class OVTR(nn.Module):
    def __init__(self, backbone, transformer, num_feature_levels, criterion, track_embed,
                    aux_loss=True, with_box_refine=False, two_stage=False,
                    two_stage_bbox_embed_share=False,
                    dec_pred_bbox_embed_share=True,
                    use_checkpoint=None,
                    distribution_based_sampling=None,
                    text_embeddings=None,
                    image_embeddings=None,
                    max_len=None,
                    novel_cls_cpu=None,
                    computed_aux=None,
                    score_thresh=None,
                    filter_score_thresh=None,
                    miss_tolerance=None,
                    train_with_artificial_img_seqs=False,
                 ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()

        self.num_queries = transformer.num_queries
        self.track_embed = track_embed
        self.transformer = transformer
        hidden_dim = transformer.d_model
     
        self.max_pad_len = max_len
        self.text_embeddings=text_embeddings.t()
        self.image_embeddings=image_embeddings.t()
        self.patch2query = nn.Linear(512, 256)
        self.all_ids = torch.tensor(range(self.text_embeddings.shape[-1]))
        self.all_ids = [i + 1 for i in self.all_ids]
        self.select_id = list(range(0, len(Frequency_list_total_1)))

        self.frequency = torch.tensor(Frequency_list_70, dtype=torch.float32, device='cpu')
        print("0.7 power sampling | Training excludes rare categories.")
        self.frequency_eval = torch.tensor(Frequency_list_total_1, dtype=torch.float32, device='cpu')
        self.novel_cls_cpu = novel_cls_cpu
        self.computed_aux = computed_aux
   
        for layer in [self.patch2query]:
            nn.init.xavier_uniform_(self.patch2query.weight)
            nn.init.constant_(self.patch2query.bias, 0)
        
        # feature alignment
        self.feature_align = nn.Linear(256, 512) # alignment head
        nn.init.xavier_uniform_(self.feature_align.weight)
        nn.init.constant_(self.feature_align.bias, 0)
        num_pred = len(self.computed_aux)
        if with_box_refine:
            self.feature_align = _get_clones(self.feature_align, num_pred)
        else:
            self.feature_align = nn.ModuleList([self.feature_align for _ in range(num_pred)])

        # bbox
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)
            ]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        
        if two_stage:
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)
            self.refpoint_embed = None

        self.num_feature_levels = num_feature_levels
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.post_process = TrackerPostProcess()
        self.track_base = RuntimeTrackerBase(score_thresh=score_thresh, 
                                             filter_score_thresh=filter_score_thresh, 
                                             miss_tolerance=miss_tolerance)
        
        self.use_checkpoint = use_checkpoint
        self.distribution_based_sampling = distribution_based_sampling
        self.criterion = criterion
        self.train_with_artificial_img_seqs = train_with_artificial_img_seqs

    def _generate_empty_tracks(self, cls_pad_len=1203):
        track_instances = Instances((1, 1))
        num_queries = self.num_queries
        dim_h = self.transformer.d_model
        device = self.transformer.level_embed.device

        track_instances.ref_pts = torch.zeros((num_queries, 4), device=device)
        track_instances.query_tgt = self.transformer.tgt_embed.weight
        track_instances.query_pos = torch.zeros((num_queries, dim_h), device=device)

        track_instances.obj_idxes = torch.full((num_queries,), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full((num_queries,), -1, dtype=torch.long, device=device)
        track_instances.iou = torch.zeros((num_queries,), dtype=torch.float, device=device)
        track_instances.scores = torch.zeros((num_queries,), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros((num_queries, 4), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros((num_queries, cls_pad_len), dtype=torch.float, device=device)

        if not self.training:
            track_instances.cls_idxes = torch.full((num_queries,), -1, dtype=torch.long, device=device)
            track_instances.disappear_time = torch.zeros((num_queries, ), dtype=torch.long, device=device)
        return track_instances.to(device)

    def clear(self):
        self.track_base.clear()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_embed):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_embed':c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_embed[:-1])]
        
    def _distribution_based_sampling(self, pad_len, uniq_labels=None):
        frequency = self.frequency.clone()
        frequency[uniq_labels] = 0
        extra_labels = torch.multinomial(frequency, pad_len)
        extra_labels = extra_labels[torch.isin(extra_labels,self.novel_cls_cpu, invert=True)]
        extra_labels = extra_labels[torch.randperm(len(extra_labels))]
        return extra_labels
    
    def get_select_id(self, cls_num, labels_list, extra_labels, is_first):
        max_pad_len = max(cls_num, self.max_pad_len)
        # get input categories
        uniq_labels = torch.unique(labels_list).to("cpu")    
        if is_first: # first frame detection
            if len(uniq_labels) < max_pad_len:
                pad_len = max_pad_len - len(uniq_labels)
                if self.distribution_based_sampling: # Sample negative categories based on the distribution.
                    extra_labels = self._distribution_based_sampling(pad_len, uniq_labels=uniq_labels)
                else:
                    extra_list = torch.tensor([i for i in self.all_ids if i not in uniq_labels])
                    extra_labels = extra_list[torch.randperm(len(extra_list))][:pad_len]
                select_id = uniq_labels.tolist() + extra_labels.tolist()
            else:
                select_id = uniq_labels.tolist()
                extra_labels = torch.LongTensor([])
        else: # subsequent frame tracking
            extra_label_notin = torch.isin(extra_labels, uniq_labels, invert=True)
            extra_labels_cur = extra_labels[extra_label_notin]
            select_id = uniq_labels.tolist() + extra_labels_cur.tolist()
            if len(select_id) < max_pad_len:
                sampled_labels = self._distribution_based_sampling(max_pad_len-len(select_id), uniq_labels=torch.tensor(select_id))
                if extra_labels is not None:
                    extra_labels = torch.cat([extra_labels_cur, sampled_labels])
                else:
                    extra_labels = sampled_labels
                select_id = uniq_labels.tolist() + extra_labels.tolist()
            elif len(select_id) > max_pad_len:
                select_id = select_id[:max_pad_len]
        return select_id, extra_labels
    
    def _forward_single_image(self, samples, track_instances: Instances, targets=None, extra_labels=None ,is_first=True, cls_num=0):
        features, pos = self.backbone(samples)      
        src, mask = features[-1].decompose()
        assert mask is not None
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        # Get the selected category id
        if self.training:
            labels_list = torch.cat([targets.labels])
            select_id, extra_labels = self.get_select_id(cls_num, labels_list, extra_labels, is_first)
        else:
            select_id, extra_labels = self.select_id, None

        # Prepare queries and embeddings for alignment
        text_query = self.text_embeddings[:, select_id].to(masks[0].device).t()
        image_align = self.image_embeddings[:, select_id].to(masks[0].device).t()
        
        image_feat_ori = (image_align.float()).detach()

        dtype = self.patch2query.weight.dtype
        text_query = self.patch2query(text_query.type(dtype))
        select_id = torch.tensor(select_id).to(text_query.device)
        text_dict = preprocess_for_masks(srcs[0].shape[0], select_id, text_query)

        (hs_cti, hs_ofa, init_reference, inter_references, pre_outputs_classes, query_pos_track) = self.transformer(
            srcs, masks, pos, track_instances.query_pos, track_instances.query_tgt, ref_pts=track_instances.ref_pts, text_dict=text_dict)

        outputs_coords = []
        outputs_embeds = []

        for lvl in range(hs_cti.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            tmp = self.bbox_embed[lvl](hs_ofa[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
            outputs_embeds.append(self.feature_align[lvl](hs_ofa[lvl]))
        outputs_class = pre_outputs_classes
        outputs_coord = torch.stack(outputs_coords)
        outputs_embed = torch.stack(outputs_embeds)

        if init_reference.shape[-1]==4:
            ref_pts_all = torch.cat([init_reference[None], inter_references[:, :, :, :4]], dim=0)
        else:
            ref_pts_all = torch.cat([init_reference[None], inter_references[:, :, :, :2]], dim=0)
        out = {
            'pred_logits': outputs_class[-1], 
            'pred_boxes': outputs_coord[-1], 
            'ref_pts': ref_pts_all[-2],
            "pred_embed": outputs_embed[-1],
            "select_id": select_id,
            "image_feat": image_feat_ori,
            "extra_labels": extra_labels,
            }
            
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_embed)
            for temp in out["aux_outputs"]:
                temp["select_id"] = select_id
                temp["image_feat"] = image_feat_ori
            
        out['query_pos_track'] = query_pos_track.transpose(0, 1)
        out['hs_ofa'] = hs_ofa[-1]
        out['hs_cti'] = hs_cti[-1]
        return out
     
    def _post_process_single_image(self, frame_res, track_instances, is_last, is_repeat=None, is_first=False, target_size=None):
        with torch.no_grad():
            track_scores = frame_res['pred_logits'][0, :].sigmoid().max(dim=-1).values

        track_instances.scores = track_scores
        track_instances.pred_logits = frame_res['pred_logits'][0]
        track_instances.pred_boxes = frame_res['pred_boxes'][0]
        track_instances.output_embedding_txt = frame_res['hs_cti'][0]
        track_instances.output_embedding_img = frame_res['hs_ofa'][0]
        track_instances.query_pos = frame_res["query_pos_track"][0]

        if self.training:
            # the track id will be assigned by the mather.
            frame_res['track_instances'] = track_instances
            track_instances = self.criterion.match_for_single_frame(frame_res, is_first)
        else:
            if self.train_with_artificial_img_seqs:
                track_instances, _track_discard = protect_track_preds(track_instances, num_queries=self.num_queries, miss_tolerance=self.track_base.miss_tolerance, ious_thresh=self.ious_thresh) 
            track_instances = self.post_process_pre(track_instances, frame_res['select_id'], is_first)
            # each track will be assigned an unique global id by the track base.
            if is_first:
                self.track_base.clear()
            track_instances = self.track_base.update(track_instances, _track_discard, is_repeat=is_repeat)

        tmp = {}
        tmp['init_track_instances'] = self._generate_empty_tracks(cls_pad_len=track_instances.pred_logits.shape[1])
        tmp['track_instances'] = track_instances

        if not is_last:
            out_track_instances = self.track_embed(tmp)
            frame_res['track_instances'] = out_track_instances
        else:
            frame_res['track_instances'] = None
        frame_res['track_instances_pre'] = track_instances
        return frame_res
    
    def post_process_pre(self, track_instances, select_id, is_first):
        out_logits = track_instances.pred_logits

        prob = out_logits.sigmoid()
        scores, labels = prob.max(-1)
 
        track_instances.scores = scores
        cur_cls_idxes = select_id[labels]
        # track_instances.keep_cls = torch.eq(cur_cls_idxes, track_instances.cls_idxes)

        if is_first:
            track_instances.cls_idxes = cur_cls_idxes
        else:
            track_instances.cls_idxes[scores >= self.track_base.filter_score_thresh] = cur_cls_idxes[scores >= self.track_base.filter_score_thresh]
        return track_instances

    @torch.no_grad()
    def inference_single_image(self, data, track_instances=None, is_repeat=False, frame_id=None, ori_img_size=None, extra_labels=None):
        img = nested_tensor_from_tensor_list([data['imgs'][0]])
        if (track_instances is None) or (frame_id == 0):
            track_instances = self._generate_empty_tracks()
        if frame_id == 0:
            is_first = True
        else:
            is_first = False

        res = self._forward_single_image(img, track_instances, None, extra_labels, is_first, cls_num=None)
        res = self._post_process_single_image(res, track_instances, False, is_repeat=is_repeat, is_first=is_first, target_size=ori_img_size[:-1])

        track_instances = res['track_instances']
        track_instances = self.post_process(track_instances, ori_img_size[:-1])
        ret = {'track_instances': track_instances}
        if 'ref_pts' in res:
            ref_pts = res['ref_pts'] 
            img_h, img_w = ori_img_size[:-1]
            # scale_fct = torch.Tensor([img_w, img_h]).to(ref_pts)
            scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(ref_pts)
            ref_pts = ref_pts * scale_fct[None]
            ret['ref_pts'] = ref_pts
        return ret

    def forward(self, data):
        if self.training:
            self.criterion.initialize(data['gt_instances'])
        frames = data['imgs']
        cls_num = max([len(torch.unique(gt_instance.labels)) for gt_instance in data['gt_instances']])
        outputs = {
            'pred_logits': [],
            'pred_boxes': [],
            'track_instances': []
        }
        track_instances = self._generate_empty_tracks()

        keys = list(track_instances._fields.keys())
        for frame_index, (frame, targets) in enumerate(zip(frames, data['gt_instances'])):
            frame.requires_grad = False
            is_last = frame_index == len(frames) - 1
            is_first = frame_index == 0
            if is_first:
                extra_labels = None
            else:
                extra_labels = frame_res["extra_labels"]
            if self.use_checkpoint and frame_index < len(frames) - 3:
                def fn(frame, *args):
                    frame = nested_tensor_from_tensor_list([frame])
                    tmp = Instances((1, 1), **dict(zip(keys, args)))
                    frame_res = self._forward_single_image(frame, tmp, targets, extra_labels, is_first, cls_num)
                    return (
                        frame_res['pred_logits'],
                        frame_res['pred_boxes'],
                        frame_res['ref_pts'],
                        frame_res['pred_embed'],
                        frame_res['select_id'],
                        frame_res['image_feat'],
                        frame_res['extra_labels'],
                        frame_res['query_pos_track'],
                        frame_res['hs_cti'],
                        frame_res['hs_ofa'],
                        *[aux['pred_logits'] for aux in frame_res['aux_outputs']],
                        *[aux['pred_boxes'] for aux in frame_res['aux_outputs']],
                        *[aux['pred_embed'] for aux in frame_res['aux_outputs']],
                        *[aux['select_id'] for aux in frame_res['aux_outputs']],
                        *[aux['image_feat'] for aux in frame_res['aux_outputs']],
                    )
                args = [frame] + [track_instances.get(k) for k in keys] 
                params = tuple((p for p in self.parameters() if p.requires_grad))
                tmp = checkpoint.CheckpointFunction.apply(fn, len(args), *args, *params)
                frame_res = {
                    'pred_logits': tmp[0],
                    'pred_boxes': tmp[1],
                    'ref_pts': tmp[2],
                    'pred_embed': tmp[3],
                    'select_id': tmp[4],
                    'image_feat': tmp[5],
                    'extra_labels': tmp[6],
                    'query_pos_track': tmp[7],
                    'hs_cti': tmp[8],
                    'hs_ofa': tmp[9],
                    'aux_outputs': [{
                        'pred_logits': tmp[10+i],
                        'pred_boxes': tmp[10+5+i],
                        'pred_embed': tmp[10+10+i],
                        'select_id': tmp[10+15+i],
                        'image_feat': tmp[10+20+i],
                    } for i in range(len(self.computed_aux)-1)],
                }
            else:
                frame = nested_tensor_from_tensor_list([frame])
                frame_res = self._forward_single_image(frame, track_instances, targets, extra_labels, is_first, cls_num)
            frame_res = self._post_process_single_image(frame_res, track_instances, is_last, is_first=is_first)
            
            track_instances = frame_res['track_instances']
            outputs['pred_logits'].append(frame_res['pred_logits'])
            outputs['pred_boxes'].append(frame_res['pred_boxes'])
            outputs['track_instances'].append(frame_res['track_instances_pre'])

        outputs['losses_dict'] = self.criterion.losses_dict
        return outputs


def build(args, cfg):
    
    assert cfg.Clip_text_embeddings and cfg.Clip_image_embeddings, "Clip_text_embeddings or Clip_image_embeddings should not be None"
    text_embeddings, image_embeddings = load_embeddings(cfg.Clip_text_embeddings, cfg.Clip_image_embeddings)  

    device = torch.device(args.device)
    backbone = build_backbone(cfg)
    transformer = build_transformer(cfg)
    d_model = transformer.d_model
    hidden_dim = cfg.dim_feedforward
    updater = build_updater(args, args.track_query_iteration, d_model, hidden_dim, d_model*2)
    matcher = build_matcher(args)

    num_frames_per_batch = max(args.sampler_lengths)
    weight_dict = {}
    
    for i in range(0, num_frames_per_batch):
        weight_dict.update({"frame_{}_loss_ce".format(i): args.cls_loss_coef,
                            'frame_{}_loss_bbox'.format(i): args.bbox_loss_coef,
                            'frame_{}_loss_giou'.format(i): args.giou_loss_coef,
                            'frame_{}_loss_align'.format(i): args.align_loss_coef,
                            })

    if args.aux_loss:
        for i in range(0, num_frames_per_batch):
            for j in range(cfg.dec_layers - 1):
                weight_dict.update({"frame_{}_aux{}_loss_ce".format(i, j): args.cls_loss_coef,
                                    'frame_{}_aux{}_loss_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_{}_aux{}_loss_giou'.format(i, j): args.giou_loss_coef,
                                    'frame_{}_aux{}_loss_align'.format(i, j): args.align_loss_coef,
                                    })

    # losses = ['labels', 'boxes', 'align']
    losses = ['labels', 'boxes']

    criterion = OVFrameMatcher(None, matcher=matcher, weight_dict=weight_dict, losses=losses, random_drop=args.random_drop,
                                train_with_artificial_img_seqs=cfg.train_with_artificial_img_seqs,
                                calculate_negative_samples=args.calculate_negative_samples,
                                num_queries=cfg.num_queries,
                                train_with_pseudo=args.train_with_pseudo,
                                )
    criterion.to(device)

    model = OVTR(
        backbone,
        transformer,
        track_embed=updater,
        num_feature_levels=cfg.num_feature_levels,
        aux_loss=args.aux_loss,
        criterion=criterion,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        text_embeddings=text_embeddings,
        image_embeddings=image_embeddings,
        max_len=args.max_len,
        use_checkpoint=cfg.use_checkpoint_track,
        train_with_artificial_img_seqs=cfg.train_with_artificial_img_seqs,
        distribution_based_sampling=cfg.distribution_based_sampling,
        novel_cls_cpu=novel_class,
        computed_aux=cfg.computed_aux,
        score_thresh=args.score_thresh,
        filter_score_thresh=args.filter_score_thresh,
        miss_tolerance=args.miss_tolerance
    )
    return model, criterion
