CUDA_DEVICES="1"
MASTER_PORT=9985
NPROC_GPU=1
PRETRAIN_MODEL="../model_zoo/ovtr_5_frame.pth"
OUTPUT="./weights"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python -m torch.distributed.launch --master_port=${MASTER_PORT} --nproc_per_node=${NPROC_GPU} \
    --use_env \
    ./main.py \
    --config_file ./config/ovtr_5_frame_tao_train_val.py \
    --dataset_file tao_seqs \
    --epochs 1 \
    --with_box_refine \
    --two_stage \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --lr_drop 13 \
    --pretrain ${PRETRAIN_MODEL} \
    --num_workers 4 \
    --batch_size 1 \
    --sample_mode random_interval \
    --sample_interval 2 \
    --sampler_steps 4 7 14 \
    --sampler_lengths 4 4 5 5 \
    --merger_dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --track_query_iteration CIP \
    --calculate_negative_samples \
    --train_with_pseudo \
    --max_len 250 \
    --output_dir ${OUTPUT}
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python -m torch.distributed.launch --master_port=${MASTER_PORT} --nproc_per_node=${NPROC_GPU} \
    --use_env \
    ./main.py \
    --config_file ./config/ovtr_5_frame_tao_train_val.py \
    --dataset_file tao_seqs \
    --epochs 16 \
    --with_box_refine \
    --two_stage \
    --lr 4e-5 \
    --lr_backbone 4e-6 \
    --lr_drop 13 \
    --resume ${OUTPUT}/checkpoint0000.pth \
    --num_workers 4 \
    --batch_size 1 \
    --sample_mode random_interval \
    --sample_interval 2 \
    --sampler_steps 4 7 14 \
    --sampler_lengths 4 4 5 5 \
    --merger_dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --track_query_iteration CIP \
    --calculate_negative_samples \
    --train_with_pseudo \
    --max_len 250 \
    --output_dir ${OUTPUT}