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
output_dir=/home/ubuntu/MSD/exps/task10_fold2_swin #TODO
####################

#################### TODO START
CUDA_VISIBLE_DEVICES=0,1
fold=2
dataset_name=10_Decathlon_Task06_Lung # 10_Decathlon_Task10_Colon
optim_lr=2e-4 
model_name=swin_unetr_b

NUM_TRAINERS=`echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l`

##########
mkdir -p $output_dir
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nnodes=1 --rdzv-backend=c10d\
    --nproc-per-node=$NUM_TRAINERS --rdzv-endpoint=localhost:$port\
    main_finetune_segmentation.py \
    --fold $fold\
    --dataset_name $dataset_name\
    --optim_lr $optim_lr\
    --model_name $model_name\
    --logdir $output_dir 2>&1 | tee -a $output_dir/log.txt
