modelname = "OVTR"
backbone = 'resnet50'

dilation = False
position_embedding = "sine"
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
enc_layers = 6
dec_layers = 6
unic_layers = 0

pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0

num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
two_stage_type = "standard"
two_stage_bbox_embed_share = False
dec_pred_bbox_embed_share = True
transformer_activation = "relu"
max_text_len = 256
fusion_dropout = 0.0
fusion_droppath = 0.1
use_fusion_layer = False
use_checkpoint = False
use_transformer_ckpt = False
use_text_cross_attention = True
embed_init_tgt = True
extra_track_attn = True
use_checkpoint_track = False
attention_protection = False

computed_aux = [0, 1, 2, 3, 4, 5]

# scaling
prior_prob = 0.005
log_scale = 0.0
text_dim = 256

#prompt
Clip_text_embeddings = '../model_zoo/iou_neg5_ens.pth' # from DetPro
Clip_image_embeddings = '../model_zoo/clip_image_embedding_all.pt'

distribution_based_sampling = True
isolation_mask = True
run_inference = False

# freezing
initial_grad = True
initial_grad_allowed = ['patch2query',
                        'feature_align',
                        'encoder_align',
                        'input_align',
                        'fusion_layers',
                        'log_scale',
                        'bias_lang',
                        'bias0',
                        'ca_text',
                        'catext_norm',
                        'norm5',
                        'linear3',
                        'linear4',
                        'decoder_norm_inter'
                        ]
global_grad_allowed_epoch = 1
global_grad_allowed_epoch_track = 1 
train_tracking_only = ['track_embed',
                       'update_attn',
                       'norm4',
                       'decoder.layers.0']
train_tracking_keep = ['backbone',
                       'encoder',
                       'tgt_embed',
                       'enc_',
                       'patch2query',
                       'input_align',
                       'input_proj',
                       'level_embed']
 
train_with_artificial_img_seqs = True 

# ovtr datasets pipeline settings
img_scale = (800, 1333)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'), 
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]

# ovtr datasets settings
dataset_type = 'TaoDataset'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=3,
    persistent_workers=True,
    train=dict(
                type=dataset_type,
                root_path='/data/fzm_2022/Datasets/TAO/',
                ),
    val=dict(
        type=dataset_type,
        classes='../data/lvis_classes_v1.txt',
        ann_file='../data/validation_ours_v1.json',
        # img_prefix='../data/TAO/',
        img_prefix='/data/fzm_2022/Datasets/TAO/frames',
        ref_img_sampler=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes='../data/lvis_classes_v1.txt',
        ann_file='../data/validation_ours_v1.json',
        # img_prefix='../data/TAO/',
        img_prefix='/data/fzm_2022/Datasets/TAO/frames',
        ref_img_sampler=None,
        pipeline=test_pipeline)
)

evaluation = dict(metric=['track'], resfile_path='/scratch/tmp/')

