set -x
min_port=1024
max_port=65535
port=$((min_port + RANDOM % (max_port - min_port + 1)))
export OMP_NUM_THREADS=48
##########
# Server 1
exp=`echo ${0%.sh}`
exp=`echo ${exp#*/}`
prefix_dir=/tmp
output_dir=/jhcnas1/nixuefeng/MSD/exps/fold2_swin #TODO

####################

#################### TODO START
CUDA_VISIBLE_DEVICES=3
fold=0
dataset_name=10_Decathlon_Task06_Lung # 10_Decathlon_Task10_Colon
out_channels=2
optim_lr=2e-4 
model_name=swin_unetr_b

roi_x=96
roi_y=96
roi_z=96
overlap=0.5

NUM_TRAINERS=`echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l`

##########
resume="${output_dir}"/model.pt
pred_dir=$output_dir/pred_fold$fold
mkdir -p $pred_dir
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -u inference.py \
    --dataset_name $dataset_name\
    --fold $fold\
    --model_name $model_name\
    --roi_x $roi_x\
    --roi_y $roi_y\
    --roi_z $roi_z\
    --resume $resume\
    --overlap $overlap\
    --out_channels $out_channels\
    --pred_dir $pred_dir\
    --sr_ratio 1\
    --save_visualization\
    --logdir $output_dir 2>&1 | tee -a $output_dir/log.txt
