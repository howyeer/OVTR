import ijson
import json
from decimal import Decimal
import sys
from tao.toolkit.tao import Tao
from collections import defaultdict
from typing import List, Dict
import torch
from ovtr.util.box_ops import box_iou

# 自定义 JSON 编码器，处理可能的 Decimal 等非序列化类型
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


def _nms_per_image(items: List[Dict], iou_same_cls: float = 0.6, iou_diff_cls: float = 0.8) -> List[Dict]:
    """对单张图片的候选框执行按类别区分阈值的NMS。

    说明：
    - items: 该 image_id 下的检测结果列表。需要包含字段：
        - 'bbox': [x, y, w, h]
        - 'score': 置信度
        - 'category_id': 类别id
    - 同类抑制阈值 iou_same_cls=0.6，不同类抑制阈值 iou_diff_cls=0.8。
    - 返回抑制后的 items 子集（保持原字典结构不变）。
    """
    if len(items) <= 1:
        return items

    # 转换为 xyxy
    boxes_xyxy = []
    scores = []
    categories = []
    for d in items:
        x, y, w, h = d.get('bbox', [0, 0, 0, 0])
        boxes_xyxy.append([x, y, x + w, y + h])
        scores.append(float(d.get('score', 0.0)))
        categories.append(d.get('category_id', -1))

    boxes = torch.tensor(boxes_xyxy, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    categories_tensor = torch.tensor(categories, dtype=torch.int64)

    # 按分数降序排序
    order = torch.argsort(scores_tensor, descending=True)
    keep_indices: List[int] = []
    suppressed = torch.zeros(len(items), dtype=torch.bool)

    while True:
        # 找到未抑制的下一个最高分框
        remaining = (~suppressed).nonzero(as_tuple=False).flatten()
        if remaining.numel() == 0:
            break
        idx = remaining[torch.argmax(scores_tensor[remaining])].item()
        keep_indices.append(idx)
        suppressed[idx] = True

        # 与其余未抑制框计算 IoU
        ref_box = boxes[idx].unsqueeze(0)  # 1x4
        other_idxs = (~suppressed).nonzero(as_tuple=False).flatten()
        if other_idxs.numel() == 0:
            continue
        ious, _ = box_iou(ref_box, boxes[other_idxs])  # 1 x M
        ious = ious.squeeze(0)

        same_cls_mask = categories_tensor[other_idxs] == categories_tensor[idx]
        thr = torch.where(same_cls_mask, torch.full_like(ious, iou_same_cls), torch.full_like(ious, iou_diff_cls))

        to_suppress_mask = ious > thr
        if to_suppress_mask.any():
            suppressed[other_idxs[to_suppress_mask]] = True

    # 返回保留的条目，按输入顺序稳定（不改变原先顺序）
    keep_set = set(keep_indices)
    result = [it for i, it in enumerate(items) if i in keep_set]
    return result

def filter_and_save(input_path, output_path, condition, json_path, tao_path):
    tao_dat = Tao(tao_path, logger=None)
    vid2img = tao_dat.vid_img_map
    img2vid = defaultdict(int)
    for vid in vid2img.keys():
        imgs = vid2img[vid]
        for img in imgs:
            img2vid[img['id']] = vid

    with open(input_path, 'rb') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write('[\n')
        first_item = True
        ti = 0
        total_processed = 0

        # 将同一 image_id 的候选项缓存后做NMS
        current_image_id = None
        pending_items: List[Dict] = []

        def flush_pending():
            nonlocal first_item, ti
            if not pending_items:
                return
            kept = _nms_per_image(pending_items, iou_same_cls=0.45, iou_diff_cls=0.65)
            for it in kept:
                if not first_item:
                    outfile.write(',\n')
                else:
                    first_item = False
                img_id_local = it.get('image_id')
                it['video_id'] = img2vid[img_id_local]
                it['track_id'] = ti + 1000000  # 避免与原有ID冲突
                ti += 1
                json.dump(it, outfile, ensure_ascii=False, cls=CustomEncoder)

        for item in ijson.items(infile, json_path):
            total_processed += 1
            if total_processed % 10000 == 0:
                sys.stdout.write(f"\r已处理 {total_processed} 条数据，筛选 {ti} 条")
                sys.stdout.flush()

            if not condition(item):
                continue

            img_id = item.get('image_id')
            if current_image_id is None:
                current_image_id = img_id
                pending_items = [item]
            else:
                if img_id != current_image_id:
                    flush_pending()
                    current_image_id = img_id
                    pending_items = [item]
                else:
                    pending_items.append(item)

        # 处理最后一批
        flush_pending()

        outfile.write('\n]')
        print(f"\n处理完成：共 {total_processed} 条，筛选出 {ti} 条")

# 使用示例
if __name__ == "__main__":
    filter_and_save(
        input_path='data/TAO_Co-DETR_train.json',
        output_path='data/TAO_Co-DETR_train_01.json',
        condition=lambda x: x.get('score', 0) > 0.3,
        json_path='item',  # 根据实际 JSON 结构调整路径，如 'data.item' 等
        tao_path='data/tao/annotations/train_ours_v1.json',
    )