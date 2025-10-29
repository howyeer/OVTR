import json 
import sys
import math
from ovtr.datasets.parsers import CocoVID

from tqdm.notebook import tqdm as tqdm_notebook
import math
from ovtr.util.list_LVIS import Frequency_all

ann_file='./data/lvis_image_v1.json' # lvis + coco 
coco = CocoVID(ann_file)
with open(ann_file, 'r') as json_file:  
    data = json.load(json_file)  

def compute_occlusion(box1, box2):  
    if box1[0]==box2[0] and box1[1]==box2[1] and box1[2]==box2[2] and box1[3]==box2[3]:
        return True
    
    xi1 = max(box1[0], box2[0])  
    yi1 = max(box1[1], box2[1])  
    xi2 = min(box1[0] + box1[2], box2[0] + box2[2])  
    yi2 = min(box1[1] + box1[3], box2[1] + box2[3])  
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)  

    if inter_area==0:
        return True
    
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    # union_area = box1_area + box2_area - inter_area  
      
    # iou = inter_area / union_area  
    threshold = box2_area * 0.3
    # threshold = min(box1_area, box2_area) * 0.3
    if inter_area < threshold:
        return True
    
    return False
  
def select_boxes_for_occlusion(instance, img_anno):
    flag = True
    for _box in img_anno:
        occlusion = compute_occlusion(instance['bbox'], _box['bbox'])
        if not occlusion:
            flag = False
            break
    return flag

def calculate_iou(box1, box2):
    xi1 = max(box1[0], box2[0])  
    yi1 = max(box1[1], box2[1])  
    xi2 = min(box1[0] + box1[2], box2[0] + box2[2])  
    yi2 = min(box1[1] + box1[3], box2[1] + box2[3])  
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)  
    
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area   
    iou = inter_area / union_area  
    return iou

def center_distance(box1, box2):
    d1 = math.sqrt(box1[2]**2 + box1[3]**2)  
    d2 = math.sqrt(box2[2]**2 + box2[3]**2)  
    length = (d1 + d2) / 2

    center1_x = box1[0] + box1[2]/2
    center1_y = box1[1] + box1[3]/2
    center2_x = box2[0] + box2[2]/2
    center2_y = box2[1] + box2[3]/2

    center_d = math.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)   
    distance = center_d / length
    # if distance < 0.2:
    #     print(1)
    return distance

def remove_duplicate_boxes(instances):
    removed_ids = []
    jump_list = []

    for i in range(len(instances)):
        bbox1 = instances[i]['bbox']
        if i in jump_list:
            continue
        for j in range(i + 1, len(instances)):
            if j in jump_list:
                continue

            bbox2 = instances[j]['bbox']
            iou = calculate_iou(bbox1, bbox2)
            distance = center_distance(bbox1, bbox2)

            if iou > 0.6 and distance < 0.3:
                i_cls_id = instances[i]['category_id']
                j_cls_id = instances[j]['category_id']
                if i_cls_id != j_cls_id:
                    if iou < 0.75 or distance > 0.2:
                        continue

                if Frequency_all[i_cls_id] >= Frequency_all[j_cls_id]:
                    removed_ids.append(instances[i]['id'])
                    break
                elif Frequency_all[i_cls_id] < Frequency_all[j_cls_id]:
                    removed_ids.append(instances[j]['id'])
                    jump_list.append(j)

    return removed_ids

def instance_filter(img_anno, removed_list):
    new_img_anno=[]
    for anno in img_anno:
        if anno['id'] not in removed_list:
            new_img_anno.append(anno)
    return new_img_anno

removed_list = []
for i, img in tqdm_notebook(enumerate(data['images'])):
    img_name = img['id']
    if img_name not in coco.imgs:
        continue
    img_anno = coco.img_ann_map[img_name]
    removed_ids = remove_duplicate_boxes(img_anno)
    removed_list += removed_ids

annotations = data['annotations']
for i, img in tqdm_notebook(enumerate(data['images'])):
    img_name = img['id']
    if img_name not in coco.imgs:
        continue
    img_anno = coco.img_ann_map[img_name]
    img_anno = instance_filter(img_anno, removed_list)
    for instance in img_anno: # need update
        occlusion = select_boxes_for_occlusion(instance, img_anno)
        annotations[instance['instance_id']].update({'clear':occlusion})

print(removed_list)
new_annotations = [anno for anno in annotations if anno['id'] not in removed_list]
data['annotations'] = new_annotations

save_path = './data/lvis_clear_75_60.json'
with open(save_path, 'w') as f:
    json.dump(data, f, indent=4)
print('complete')