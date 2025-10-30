# Copyright (c) Jinyang Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Copyright (c) Yutliu
# ------------------------------------------------------------------------
# Modified from OVTrack (https://github.com/SysCV/ovtrack)
# ------------------------------------------------------------------------
"""
MOT dataset which returns image_id for evaluation.
"""
from collections import defaultdict
import collections
import copy
import math
from operator import itemgetter
from pathlib import Path
import random
import cv2
import os 
import numpy as np
from PIL import Image
import json
import torch
import torch.utils.data
from tao.toolkit.tao import Tao
from torch.utils.data import Dataset
from .tao_dataset import TaoDataset
from detectron2.structures import Instances
import datasets.pipelines.mot_transforms as T
from mmcv.utils import build_from_cfg
from mmdet.datasets.builder import DATASETS, PIPELINES
from util.list_LVIS import CLASSES

# Limit OpenCV threads to avoid per-rank contention and slowdowns
cv2.setNumThreads(0)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

def _is_main_process():
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
    except Exception:
        pass
    return True

@DATASETS.register_module(force=True)
class Tao_seqs_Dataset(Dataset):  # TAO dataset
    def __init__(self, args_mot=None, transform=None, root_path=None, logger=None):
        super().__init__()
        self.root = root_path
        annotation_path = os.path.join(self.root, 'annotations', 'train_ours_v1.json')
        print('using annotation path {}'.format(annotation_path))
        tao = Tao(annotation_path, logger)
        cats = tao.cats
        vid_img_map = tao.vid_img_map
        img_ann_map = tao.img_ann_map
        vids = tao.vids  # key: video id, value: video info
        pseudo_det = json.load(open(args_mot.pseudo_det, 'r'))

        # video sampler.
        self.sampler_steps: list = args_mot.sampler_steps
        self.lengths: list = args_mot.sampler_lengths
        print("sampler_steps={} lenghts={}".format(self.sampler_steps, self.lengths))
        if self.sampler_steps is not None and len(self.sampler_steps) > 0:
            # Enable sampling length adjustment.
            assert len(self.lengths) > 0
            assert len(self.lengths) == len(self.sampler_steps) + 1
            for i in range(len(self.sampler_steps) - 1):
                assert self.sampler_steps[i] < self.sampler_steps[i + 1]
            # self.item_num = len(self.img_files) - (self.lengths[-1] - 1) * self.sample_interval
            self.period_idx = 0
            self.num_frames_per_batch = self.lengths[0]
            self.current_epoch = 0
       
        self.transform = transform

        # Bounded retries to avoid infinite resampling loop
        self.max_resample_attempts = getattr(args_mot, 'max_resample_attempts', 10)
        self.train_with_pseudo = getattr(args_mot, 'train_with_pseudo', False)
        self.max_det = getattr(args_mot, 'max_det', None)
        # configurable match threshold(s): float => same=diff=float; tuple/list(len=2) => (same, diff); dict => {'same':..,'diff':..}
        thr_cfg = getattr(args_mot, 'pseudo_match_thr', [0.6, 0.75])
        if isinstance(thr_cfg, list) and len(thr_cfg) == 2:
            self.pseudo_match_thr_same, self.pseudo_match_thr_diff = float(thr_cfg[0]), float(thr_cfg[1])
        else:
            raise TypeError("pseudo_match_thr mast be list")

        self.all_frames_with_gt, self.all_indices, categories_counter = self._generate_train_imgs(vid_img_map, img_ann_map, cats, vids, pseudo_det,
                                                                                                                 args_mot.train_base)
        categories_counter = sorted(categories_counter.items(), key=lambda x: x[0])
        print('found {} videos, {} imgs'.format(len(vids), len(self.all_indices)))
        print('number of categories: {}'.format(len(categories_counter)))

        self.sample_mode = args_mot.sample_mode
        self.sample_interval = args_mot.sample_interval


    def _generate_train_imgs(self, vid_img_map, img_ann_map, cats, vids, pseudo_det, base_only):
        def _to_xyxy(box):
            if len(box) != 4:
                return None
            x0, y0, w, h = box[0], box[1], box[2], box[3]
            if w >= 0 and h >= 0:
                return [x0, y0, x0 + w, y0 + h]
            return [x0, y0, w, h]

        def _iou_matrix(a, b):
            if a.shape[0] == 0 or b.shape[0] == 0:
                return torch.zeros((a.shape[0], b.shape[0]), dtype=torch.float32)
            tl = torch.max(a[:, None, :2], b[None, :, :2])
            br = torch.min(a[:, None, 2:], b[None, :, 2:])
            wh = (br - tl).clamp(min=0)
            inter = wh[:, :, 0] * wh[:, :, 1]
            area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
            area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            union = area_a[:, None] + area_b[None, :] - inter
            iou = torch.where(union > 0, inter / union, torch.zeros_like(union))
            return iou

        # build img_id -> pseudo annotations list
        pseudo_by_img = {}

        if isinstance(pseudo_det, list):
            for pann in pseudo_det:
                k = pann.get('image_id')
                if k is None:
                    continue
                pseudo_by_img.setdefault(k, []).append(pann)

        if base_only:
            print('only use base classes')
        all_frames_with_gt = {}
        all_indices = []
        categories_counter = defaultdict(int)
        for vid_id in vids.keys():
            filename = vids[vid_id]['name']
            with_gt_id = []
            imgs = vid_img_map[vid_id]
            imgs = sorted(imgs, key=lambda x: x['frame_index'])
            num_imgs = len(imgs)
            targets = []  # gt and detection results
            img_infos = []
            cur_vid_indices = []
            for img in imgs:
                img_id = img['id']
                anns = img_ann_map[img_id]
                all_rare_flag = [cats[ann['category_id']]['frequency'] == 'r' for ann in anns]
                if len(anns) == 0 or all(all_rare_flag):
                    continue
                frame_id = img['frame_id']
                gt_boxes, gt_labels, gt_track_ids, gt_iscrowd = [], [], [], []
                height, width = float(img['height']), float(img['width'])
                
                cur_vid_indices.append((vid_id, img_id))
                with_gt_id.append(img_id)
                img_infos.append({'file_name': img['file_name'],
                                'height': height,
                                'width': width,
                                'frame_id': img['frame_id'],
                                'image_id': img_id,
                                'video_id': vid_id})
                for ann in anns:
                    assert ann['iscrowd'] != 1
                    if base_only and cats[ann['category_id']]['frequency'] == 'r':  # ignore rare classes
                        continue
                    box = ann['bbox']  # x0, y0, w, h
                    box[2] += box[0]
                    box[3] += box[1]
                    gt_boxes.append(box)
                    categories_counter[ann['category_id']] += 1
                    gt_labels.append(ann['category_id']-1)  # category
                    gt_track_ids.append(ann['track_id'])
                    gt_iscrowd.append(ann['iscrowd'])

                # gather extra pseudo dets not matched to GT (IoU > 0.8 considered matched)
                extra_boxes, extra_labels, extra_track_ids, extra_scores = [], [], [], []
                pseudo_list = pseudo_by_img.get(img_id) or pseudo_by_img.get(str(img_id)) or []
                if len(pseudo_list) > 0 and len(gt_boxes) > 0:
                    pseudo_boxes_xyxy, pseudo_labels, pseudo_tids, pseudo_scores = [], [], [], []
                    for pann in pseudo_list:
                        if base_only and cats[pann['category_id']]['frequency'] == 'r':  # ignore rare classes
                            continue
                        bbox = pann['bbox']
                        if bbox is None:
                            continue
                        if isinstance(bbox[0], (list, tuple)):
                            bbox = bbox[0]
                        xyxy = _to_xyxy(bbox)
                        if xyxy is None:
                            continue
                        pseudo_boxes_xyxy.append(xyxy)
                        pseudo_labels.append((pann['category_id'] - 1))
                        pseudo_tids.append(pann['track_id'])
                        pseudo_scores.append(pann['score'])

                    if len(pseudo_boxes_xyxy) > 0:
                        pseudo_boxes_t = torch.as_tensor(pseudo_boxes_xyxy, dtype=torch.float32)
                        gt_boxes_t_for_match = torch.as_tensor(gt_boxes, dtype=torch.float32)
                        ious = _iou_matrix(pseudo_boxes_t, gt_boxes_t_for_match)
                        gt_labels_arr = torch.as_tensor(gt_labels, dtype=torch.long)
                        # per pseudo row, decide matched by same/diff-class thresholds
                        for i in range(len(pseudo_boxes_xyxy)):
                            p_label = pseudo_labels[i]
                            if ious.numel() == 0:
                                matched = False
                            else:
                                same_mask = (gt_labels_arr == p_label)
                                diff_mask = ~same_mask
                                matched_same = False
                                matched_diff = False
                                if same_mask.any():
                                    matched_same = bool((ious[i][same_mask] > self.pseudo_match_thr_same).any().item())
                                if diff_mask.any():
                                    matched_diff = bool((ious[i][diff_mask] > self.pseudo_match_thr_diff).any().item())
                                matched = matched_same or matched_diff
                            if (not matched) and self.train_with_pseudo:
                                extra_boxes.append(pseudo_boxes_xyxy[i])
                                extra_labels.append(p_label)
                                extra_track_ids.append(pseudo_tids[i])
                                extra_scores.append(pseudo_scores[i])

                gt_boxes_t = torch.as_tensor(gt_boxes, dtype=torch.float32)
                gt_labels_t = torch.as_tensor(gt_labels, dtype=torch.long)
                gt_track_ids_t = torch.as_tensor(gt_track_ids, dtype=torch.float32)
                gt_iscrowd_t = torch.as_tensor(gt_iscrowd, dtype=torch.bool)
                gt_scores_t = torch.ones((len(gt_boxes_t),), dtype=torch.float32)

                if self.train_with_pseudo and len(extra_boxes) > 0:
                    extra_boxes_t = torch.as_tensor(extra_boxes, dtype=torch.float32)
                    extra_labels_t = torch.as_tensor(extra_labels, dtype=torch.long)
                    extra_track_ids_t = torch.as_tensor(extra_track_ids, dtype=torch.float32)
                    extra_scores_t = torch.as_tensor(extra_scores, dtype=torch.float32)

                    # apply max_det: keep GT all; cap pseudo by highest scores
                    if isinstance(self.max_det, int) and self.max_det > 0:
                        keep_pseudo = max(0, self.max_det - len(gt_boxes_t))
                        if keep_pseudo < len(extra_boxes_t):
                            sorted_idx = torch.argsort(extra_scores_t, descending=True)
                            select_idx = sorted_idx[:keep_pseudo]
                            extra_boxes_t = extra_boxes_t[select_idx]
                            extra_labels_t = extra_labels_t[select_idx]
                            extra_track_ids_t = extra_track_ids_t[select_idx]
                            extra_scores_t = extra_scores_t[select_idx]

                    boxes_all = torch.cat([gt_boxes_t, extra_boxes_t], dim=0)
                    labels_all = torch.cat([gt_labels_t, extra_labels_t], dim=0)
                    obj_ids_all = torch.cat([gt_track_ids_t, extra_track_ids_t], dim=0)
                    iscrowd_all = torch.cat([gt_iscrowd_t, torch.zeros((extra_boxes_t.shape[0],), dtype=torch.bool)], dim=0)
                    scores_all = torch.cat([gt_scores_t, extra_scores_t], dim=0)
                else:
                    boxes_all = gt_boxes_t
                    labels_all = gt_labels_t
                    obj_ids_all = gt_track_ids_t
                    iscrowd_all = gt_iscrowd_t
                    scores_all = gt_scores_t

                assert len(boxes_all) != 0
                targets.append({'boxes': boxes_all,  # x0, y0, x1, y1
                                'labels': labels_all,
                                'obj_ids': obj_ids_all,
                                'iscrowd': iscrowd_all,
                                'scores': scores_all,
                                })

            all_indices.extend(cur_vid_indices)
            all_frames_with_gt[vid_id] = {'img_infos': img_infos,
                                          'targets': targets,
                                          'img_ids': with_gt_id,
                                          'filename': filename
                                          }
        return all_frames_with_gt, all_indices, categories_counter
    
    def _targets_to_instances(self, targets: dict, img_shape) -> Instances:
        gt_instances = Instances(tuple(img_shape))
        gt_instances.boxes = targets['boxes']
        gt_instances.labels = targets['labels']
        gt_instances.obj_ids = targets['obj_ids']
        if 'scores' in targets:
            gt_instances.scores = targets['scores']
        return gt_instances
    
    
    def results_to_instances(self, images, targets):
        gt_instances = []
        data = {}
        for i in range(self.num_frames_per_batch):
            gt_instances_i = self._targets_to_instances(targets[i], images[i].shape[1:3])
            gt_instances.append(gt_instances_i)
        data.update({
            'imgs': images,
            'gt_instances': gt_instances,
        })
        return data
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.sampler_steps is None or len(self.sampler_steps) == 0:
            # fixed sampling length.
            return

        for i in range(len(self.sampler_steps)):
            if epoch >= self.sampler_steps[i]:
                self.period_idx = i + 1
        print("set epoch: epoch {} period_idx={}".format(epoch, self.period_idx))
        self.num_frames_per_batch = self.lengths[self.period_idx]
    
    def step_epoch(self):
        # one epoch finishes.
        print("Dataset: epoch {} finishes".format(self.current_epoch))
        self.set_epoch(self.current_epoch + 1)

    def __len__(self):
        return len(self.all_indices)
    
    
    def sample_indices(self, vid, img_id):
        # Sampling aligned with bdd100k style: start at current frame and stride by sample_interval.
        img_ids = self.all_frames_with_gt[vid]['img_ids']
        tmax = len(img_ids)
        if tmax == 0 or img_id not in img_ids:
            return None

        n = self.num_frames_per_batch
        cur = img_ids.index(img_id)

        # Short video handling: use all frames then pad the last index
        if tmax < n:
            indices = list(range(tmax))
            while len(indices) < n:
                indices.append(indices[-1])
            return indices

        # Determine sampling interval
        if self.sample_mode == 'fixed_interval':
            interval = self.sample_interval
        elif self.sample_mode == 'random_interval':
            interval = np.random.randint(1, self.sample_interval + 1)
        else:
            interval = self.sample_interval

        # Default attempt: forward sampling from current with stride interval
        end_pos = cur + (n - 1) * interval
        if end_pos <= tmax - 1:
            return [cur + k * interval for k in range(n)]

        # Out-of-range adjustment: re-sample interval (if random) and shift start to fit
        if self.sample_mode == 'random_interval':
            interval = np.random.randint(1, self.sample_interval + 1)
        start = (tmax - 1) - (n - 1) * interval
        if start < 0:
            # If still invalid, compute the maximum feasible interval to cover the span; if 0, fallback to 1 and pad
            feasible = (tmax - 1) // max(1, (n - 1))
            interval = max(1, min(interval, feasible))
            start = 0

        indices = [start + k * interval for k in range(n)]
        # Clip any residual overflow and pad if needed
        indices = [min(i, tmax - 1) for i in indices]
        while len(indices) < n:
            indices.append(indices[-1])
        return indices
    

    def visualize(self, targets, img_info, img):
        Path('./data_vis').mkdir(parents=True, exist_ok=True)
        name = None
        root = str(Path.cwd())
        img_id = img_info['image_id']
        name = img_info['file_name'].split('/')[-2] + f'_v{img_id}.jpg'
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        boxes = targets['boxes'].clone().cpu().numpy()
        label = targets['labels'].clone().cpu().numpy()
        obj_ids = targets['obj_ids'].clone().cpu().numpy()
        scores = targets.get('scores')
        scores = scores.clone().cpu().numpy() if isinstance(scores, torch.Tensor) else None
    
        for i, (x, lb) in enumerate(zip(boxes, label)):
            c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
            # pseudo ids are usually large numbers
            is_gt = bool(obj_ids[i] < 1000000)
            color = (0, 0, 255) if is_gt else (0, 255, 0)
            cv2.rectangle(img, c1, c2, color, thickness=1)
            text = CLASSES[lb]
            if scores is not None:
                text = f"{text}:{scores[i]:.2f}"
            cv2.putText(img, text, (c1[0], c1[1] + 3), 0, 0.5, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        cv2.imwrite(os.path.join(root, './data_vis', name), img)

    def __getitem__(self, idx: int):    
        attempts = 0
        while attempts < self.max_resample_attempts:
            vid, img_id = self.all_indices[idx]
            filename = self.all_frames_with_gt[vid]['filename']
            indices = self.sample_indices(vid, img_id)
            if not indices:
                idx = np.random.choice(range(len(self)))
                attempts += 1
                continue

            tmax = len(self.all_frames_with_gt[vid]['img_infos'])
            if max(indices) >= tmax:
                indices = [min(i, tmax - 1) for i in indices]

            img_infos = [self.all_frames_with_gt[vid]['img_infos'][i] for i in indices]
            targets = [self.all_frames_with_gt[vid]['targets'][i] for i in indices]

            ori_images = [Image.open(os.path.join(self.root, 'frames', img_info['file_name']))
                    for img_info in img_infos]
            # for t, img_info, img in zip(targets, img_infos, ori_images):
            #     self.visualize(t, img_info, img)
            if self.transform is not None:
                images, targets = self.transform(ori_images, targets)
            else:
                raise ValueError('transform is None')

            if isinstance(targets, list):
                has_empty = any(len(targets[i]['boxes']) == 0 for i in range(self.num_frames_per_batch))
                if has_empty:
                    idx = np.random.choice(range(len(self)))
                    attempts += 1
                    continue

            gt_instances = self.results_to_instances(images, targets)
            gt_instances.update({'filename': filename})
            return gt_instances

        # Fallback: deterministic safe sample from current video
        vid, img_id = self.all_indices[idx]
        filename = self.all_frames_with_gt[vid]['filename']
        tmax = len(self.all_frames_with_gt[vid]['img_infos'])
        n = self.num_frames_per_batch
        base_indices = list(range(min(n, tmax)))
        while len(base_indices) < n and tmax > 0:
            base_indices.append(base_indices[-1])
        img_infos = [self.all_frames_with_gt[vid]['img_infos'][i] for i in base_indices]
        targets = [self.all_frames_with_gt[vid]['targets'][i] for i in base_indices]
        ori_images = [Image.open(os.path.join(self.root, 'frames', img_info['file_name'])) for img_info in img_infos]
        if self.transform is not None:
            images, targets = self.transform(ori_images, targets)
        else:
            raise ValueError('transform is None')
        gt_instances = self.results_to_instances(images, targets)
        gt_instances.update({'filename': filename})
        return gt_instances





def build(image_set, args_mot, cfg):
    cfg.pop('type')
    transform = build_transform(image_set)
    if image_set == 'train':
        dataset = Tao_seqs_Dataset(args_mot, transform, **cfg)
    elif image_set == 'val':
        dataset = Tao_seqs_Dataset_val(args_mot, **cfg)
    return dataset

def make_transforms_for_TAO(image_set, args=None):

    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]

    if image_set == 'train':
        return T.MotCompose([
            T.MotRandomHorizontalFlip(),
            T.MotRandomSelect(
                T.MotRandomResize(scales, max_size=1536),
                T.MotCompose([
                    T.MotRandomResize([800, 1000, 1200]),
                    T.FixedMotRandomCrop(800, 1200),
                    T.MotRandomResize(scales, max_size=1536),
                ])
            ),
            T.MOTHSV(),
            normalize,
        ])

    if image_set == 'val':
        return T.MotCompose([
            T.MotRandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def build_transform(image_set):
    train = make_transforms_for_TAO('train')
    test = make_transforms_for_TAO('val')

    if image_set == 'train':
        return train
    elif image_set == 'val' or image_set == 'exp' or image_set == 'val_gt':
        return test
    else:
        raise NotImplementedError()