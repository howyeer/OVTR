import torch
import json

def pth_to_json(pth_path, json_path):
    data = torch.load(pth_path, map_location='cpu')
    # If it's a state_dict, convert tensors to lists
    def tensor_to_list(obj):
        if isinstance(obj, torch.Tensor):
            print('is tensor')
            print(obj.shape)
            return obj.tolist()
        elif isinstance(obj, dict):
            print('is dict')
            return {k: tensor_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            print('is list')
            return [tensor_to_list(v) for v in obj]
        else:
            return obj

    data = tensor_to_list(data)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    pth_path = 'model_zoo/clip_image_embedding_all.pt'
    json_path = 'model_zoo/clip_image_embedding_all.json'
    # pth_path = 'model_zoo/iou_neg5_ens.pth'
    # json_path = 'model_zoo/iou_neg5_ens.json'
    pth_to_json(pth_path, json_path)