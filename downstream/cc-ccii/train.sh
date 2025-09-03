now=$(date +"%Y%m%d_%H%M%S")
logdir=runs/fold0_swin3d

python main_swin_unetr.py \
   --logdir $logdir | tee $logdir/$now.txt

# torchrun -m torch.distributed.launch --master_port=24420 main_swin_unetr.py \
#    --logdir $logdir | tee $logdir/$now.txt