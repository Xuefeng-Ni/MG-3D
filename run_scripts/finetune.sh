export TOKENIZERS_PARALLELISM=true

load_path=/home/csexuefeng/MG-3D/result/task/version_0/checkpoints/epoch=114-step=20815.ckpt
pseudo_vision_token_pool_size=0 #2048
pseudo_langauge_token_pool_size=0 #1024 #2048
seed=0

num_gpus=1
per_gpu_batchsize=2
python main.py with seed=${seed} data_root=/scratch/medimgfmod/CT/pretrain_arrows/ \
  num_gpus=${num_gpus} per_gpu_batchsize=${per_gpu_batchsize} num_nodes=1 \
  task_finetune_clm_ctrg_chest_vision_only clip32 text_roberta \
  image_size=224 clip_resizedcrop \
  tokenizer=allenai/biomed_roberta_base \
  pseudo_vision_token_pool_size=${pseudo_vision_token_pool_size} \
  pseudo_langauge_token_pool_size=${pseudo_langauge_token_pool_size} \
  load_path=${load_path}
  #clip_resizedcrop