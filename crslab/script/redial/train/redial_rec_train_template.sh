slurms_tag=$1
yaml_num=$2
batch_size=$3

python run_crslab.py \
    -c config/experiment_yamls/$slurms_tag/redial_$yaml_num.yaml \
    -g 0 \
    -ss \
    -ct 256 \
    -it 100 \
    --scale 1.0 \
    -pbs 256 \
    -rbs $batch_size  \
    -cbs $batch_size \
    --info_truncate 40  \
    --coarse_loss_lambda 0.2 \
    --fine_loss_lambda 1.0 \
    --coarse_pretrain_epoch 12 \
    --pretrain_epoch 25 \
    --rec_epoch 50 \
    --conv_epoch 0  \