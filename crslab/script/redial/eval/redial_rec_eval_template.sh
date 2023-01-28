slurms_tag=$1
yaml_num=$2
batch_size=$3
restore_model_path=$4

python run_crslab.py \
    -c config/experiment_yamls/$slurms_tag/redial_$yaml_num.yaml \
    -g 0 \
    -ss  \
    -ct 256 \
    -it 100 \
    --scale 1.0 \
    -pbs $batch_size \
    -rbs $batch_size  \
    -cbs $batch_size \
    --info_truncate 40  \
    --coarse_loss_lambda 0.2 \
    --fine_loss_lambda 1.0 \
    --coarse_pretrain_epoch 0 \
    --pretrain_epoch 0 \
    --rec_epoch 0 \
    --conv_epoch 0 \
    -rs \
    --restore_path $restore_model_path \
    --model_file_for_restore NRCRS_Model_0.pth \
    --freeze_parameters_name k_c  \
    --freeze_parameters \
    --logit_type hs_copy2 \
    --is_coarse_weight_loss \
    --token_freq_th 1500 \
    --coarse_weight_th 0.02 \
