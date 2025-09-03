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
output_dir=/home/xuefeng/Covid19_20/exps/fold1_pcrlv2_official #TODO
####################

roi_x=192   
roi_y=192   
roi_z=32   

#################### TODO START
CUDA_VISIBLE_DEVICES=4,5
fold=1
optim_lr=2e-4  
batch_size=1
accum_iter=1 #2 1
model_name=swin_unetr_b
dataset_name=Covid19_20
pretrained_path=None
# batch_size 2 1
NUM_TRAINERS=`echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l`

##########
mkdir -p $output_dir
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nnodes=1 --rdzv-backend=c10d\
    --nproc-per-node=$NUM_TRAINERS --rdzv-endpoint=localhost:$port\
    main_finetune_segmentation.py \
    --model_name $model_name\
    --fold $fold\
    --dataset_name $dataset_name\
    --optim_lr $optim_lr\
    --batch_size $batch_size\
    --accum_iter $accum_iter\
    --roi_x $roi_x\
    --roi_y $roi_y\
    --roi_z $roi_z\
    --pretrained_path $pretrained_path\
    --logdir $output_dir 2>&1 | tee -a $output_dir/log.txt
