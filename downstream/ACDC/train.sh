now=$(date +"%Y%m%d_%H%M%S")
logdir=/home/xuefengDownstream/ACDC/runs/logs_fold0_swin
mkdir -p $logdir

python main.py \
    --logdir $logdir | tee $logdir/$now.txt