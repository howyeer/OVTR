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
        # self.pseudo_det = json.load(open(args_mot.pseudo_det, 'r'))

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
        self.all_frames_with_gt, self.all_indices, categories_counter = self._generate_train_imgs(vid_img_map, img_ann_map, cats, vids,
                                                                                                                 args_mot.train_base)
        categories_counter = sorted(categories_counter.items(), key=lambda x: x[0])
        print('found {} videos, {} imgs'.format(len(vids), len(self.all_indices)))
        print('number of categories: {}'.format(len(categories_counter)))

        self.sample_mode = args_mot.sample_mode
        self.sample_interval = args_mot.sample_interval


    def _generate_train_imgs(self, vid_img_map, img_ann_map, cats, vids, base_only):
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

                gt_boxes = torch.as_tensor(gt_boxes, dtype=torch.float32)
                gt_labels = torch.as_tensor(gt_labels, dtype=torch.long)
                gt_track_ids = torch.as_tensor(gt_track_ids, dtype=torch.float32)
                gt_iscrowd = torch.as_tensor(gt_iscrowd, dtype=torch.bool)
                assert len(gt_boxes) != 0
                targets.append({'boxes': gt_boxes,  # x0, y0, x1, y1
                                'labels': gt_labels,
                                'obj_ids': gt_track_ids,
                                'iscrowd': gt_iscrowd,
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
    
    # def sample_indices(self, vid, img_id):
    #     img_ids = self.all_frames_with_gt[vid]['img_ids']
    #     t0 = img_ids.index(img_id)
    #     ids = [t0]
    #     tmax = len(img_ids)
    #     assert tmax == len(self.all_frames_with_gt[vid]['img_infos'])

    #     for i in range(1, self.num_frames_per_batch):
    #         id_ = ids[-1] + random.randint(1, self.sample_interval)
    #         if id_ < tmax:
    #             ids.append(id_)
    #         else:
    #             id_ = min(ids) - random.randint(1, self.sample_interval)
    #             ids.append(id_)
    #     return sorted(ids)
    
    def sample_indices(self, vid, img_id):
        UNSAMPLEABLE = None
        img_ids = self.all_frames_with_gt[vid]['img_ids']
        tmax = len(img_ids)
        interval = self.sample_interval
        
        # 1. 双重校验：ID存在性 + 列表长度
        if img_id not in img_ids:
            return UNSAMPLEABLE
        if tmax < self.num_frames_per_batch:
            return UNSAMPLEABLE
        
        # 2. 定位中心索引，初始化采样列表和已采样集合
        center_idx = img_ids.index(img_id)
        sample_indices = [center_idx]
        sampled = {center_idx}
        
        # 3. 构建初始候选池：与center_idx间隔<=M且在列表范围内
        candidates = []
        for idx in range(max(0, center_idx - interval), min(tmax, center_idx + interval + 1)):
            if idx != center_idx and idx not in sampled:
                candidates.append(idx)
        
        # 4. 随机采样至长度n
        while len(sample_indices) < self.num_frames_per_batch:
            if not candidates:  # 无符合条件的候选索引
                return UNSAMPLEABLE
            
            # 随机选1个候选索引
            selected = random.choice(candidates)
            sample_indices.append(selected)
            sampled.add(selected)
            candidates.remove(selected)  # 从候选池移除已选索引
            
            # 基于新选索引，补充候选池（间隔<=M且未采样）
            new_candidates = []
            start = max(0, selected - interval)
            end = min(tmax, selected + interval + 1)
            for idx in range(start, end):
                if idx not in sampled and idx not in candidates:
                    new_candidates.append(idx)
            candidates.extend(new_candidates)
        
        sample_indices.sort()
        return sample_indices
    
    def box_xyxy_to_cxcywh(self, boxes, size):
        bbox = torch.zeros_like(boxes)
        bbox[:,0] = (boxes[:,0] + boxes[:,2])/2
        bbox[:,1] = (boxes[:,1] + boxes[:,3])/2
        bbox[:,2] = (boxes[:,2] - boxes[:,0])
        bbox[:,3] = (boxes[:,3] - boxes[:,1])
        w_img = size[1]
        h_img = size[0]
        bbox = torch.div(bbox, torch.tensor([w_img, h_img, w_img, h_img], dtype=torch.float32))
        return bbox

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
    
        for x, label in zip(boxes, label):
            c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
            cv2.rectangle(img, c1, c2, (144, 238, 144), thickness=1)
            cv2.putText(img, CLASSES[label], (c1[0], c1[1] + 3), 0, 0.5, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        cv2.imwrite(os.path.join(root, './data_vis', name), img)

    def __getitem__(self, idx: int):    
        while True:
            vid, img_id = self.all_indices[idx]  # 视频id和img_id
            filename = self.all_frames_with_gt[vid]['filename']
            indices = self.sample_indices(vid, img_id)  # 采样得到的图片id列表
            # 采样为空，重新采样
            if indices is None:
                idx = np.random.choice(range(len(self)))
                continue
            
            if len(self.all_frames_with_gt[vid]['img_infos']) <= max(indices): 
                raise IndexError(f"Index {i} in {indices} out of range for video {vid} {img_id}with {len(self.all_frames_with_gt[vid]['img_infos'])} frames.")
            img_infos, targets = [], []
            for i in indices:
                img_infos.append(self.all_frames_with_gt[vid]['img_infos'][i])
                targets.append(self.all_frames_with_gt[vid]['targets'][i])
            
            ori_images = [Image.open(os.path.join(self.root, 'frames', img_info['file_name'])) \
                    for img_info in img_infos]
            # for t, img_info, img in zip(targets, img_infos, ori_images):
            #     self.visualize(t, img_info, img)
            if self.transform is not None:
                images, targets = self.transform(ori_images, targets)
            else:
                raise ValueError('transform is None')
            if isinstance(targets, list): # Check if the object is empty.
                data_flag = True
                for i in range(self.num_frames_per_batch):
                    if len(targets[i]['boxes']) == 0:
                        data_flag = False
                if not data_flag:
                    idx = np.random.choice(range(len(self)))
                    data_flag = True
                    continue
            gt_instances = self.results_to_instances(images, targets)
            gt_instances.update({'filename': filename})

            return gt_instances


class Tao_seqs_Dataset_val(TaoDataset):
    def __init__(self,
        args_mot_mot = None,
        load_as_video=True,
        match_gts=True,
        skip_nomatch_pairs=True,
        key_img_sampler=dict(interval=1),
        ref_img_sampler=dict(scope=3, num_ref_imgs=1, method="uniform"),
        *args_mot,
        **kwargs_mot,):

        super(LVIS_seqs_Dataset_val, self).__init__(load_as_video=True,
            match_gts=True,
            skip_nomatch_pairs=True,
            key_img_sampler=key_img_sampler,
            ref_img_sampler=ref_img_sampler,
            *args_mot,
            **kwargs_mot,)
        
        self.CLASSES = CLASSES
        self.num_frames_per_batch = max(args_mot_mot.sampler_lengths)
        self.sample_mode = args_mot_mot.sample_mode
        self.sample_interval = args_mot_mot.sample_interval
        self.video_dict = {}
        self.item_num = None

        if args_mot_mot.filter_ignore:
            print('Training with ignore.', flush=True)

        self.item_num = len(self) - (self.num_frames_per_batch - 1) * self.sample_interval

        # video sampler.
        self.sampler_steps: list = args_mot_mot.sampler_steps
        self.lengths: list = args_mot_mot.sampler_lengths
        print("sampler_steps={} lenghts={}".format(self.sampler_steps, self.lengths))

        self.num_samples = len(self)
        self._skip_type_keys = None

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

    def pre_continuous_frames(self, start, end, interval=1):
        targets = []
        images = []
        for i in range(start, end, interval):
            img_i, targets_i = self._pre_single_frame(i)
            images.append(img_i)
            targets.append(targets_i)
        return images, targets

    def __getitem__(self, idx):
        results, ann_info = self.prepare_test_img(idx)
        frame_id = self.data_infos[idx]['frame_id']
        gt_instances = self.results_to_instances(results, frame_id, ann_info)
        return gt_instances
        
    def results_to_instances(self, results, frame_id, ann_info):
        img_shape_without_pad = results['img_metas'][0].data['pad_shape']
        ori_img_shape = results['img_metas'][0].data['ori_shape']
        gt_instance = Instances(img_shape_without_pad[:2])
        image = results['img'][0]
        gt_instance.labels = torch.as_tensor(ann_info['labels'], device=image.device)
        data={
            'imgs': [image],
            'gt_instances': [gt_instance],
            'info':[[frame_id, ori_img_shape]],
            'file_path':[ann_info['file_path']]
        }
        return data
    
    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        args_mot:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(img_info)
        ann_info.update({'file_path':img_info['file_name']})
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results), ann_info
    
    def ref_img_sampling(
        self, img_info, scope, num_ref_imgs=1, method="uniform", pesudo=False
    ):
        if num_ref_imgs != 1 or method != "uniform":
            raise NotImplementedError
        if img_info.get("frame_id", -1) < 0 or scope <= 0:
            ref_img_info = img_info.copy()
        else:
            vid_id = img_info["video_id"]
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            frame_id = img_info["frame_id"]
            if method == "uniform":
                left = max(0, frame_id - scope)
                right = min(frame_id + scope, len(img_ids) - 1)
                if pesudo:
                    valid_inds = img_ids[left : right + 1]
                else:
                    if len(img_ids) == 1:
                        valid_inds = img_ids[left : right + 1]
                    else:
                        valid_inds = (
                            img_ids[left:frame_id] + img_ids[frame_id + 1 : right + 1]
                        )
                ref_img_id = random.choice(valid_inds)
            ref_img_info = self.coco.loadImgs([ref_img_id])[0]
            ref_img_info["filename"] = ref_img_info["file_name"]
        return ref_img_info
    
    def prepare_results(self, img_info):
        ann_info = self.get_ann_info(img_info)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            idx = self.img_ids.index(img_info["id"])
            results["proposals"] = self.proposals[idx]
        return results


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