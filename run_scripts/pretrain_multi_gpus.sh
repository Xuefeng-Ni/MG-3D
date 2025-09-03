export TOKENIZERS_PARALLELISM=true

num_gpus=4
per_gpu_batchsize=4


  srun python main.py \
  with seed=0 data_root=/scratch/medimgfmod/CT/pretrain_arrows_comma_ct_rate/ \
  num_gpus=${num_gpus} num_nodes=1 \
  task_pretrain_ptunifier \
  per_gpu_batchsize=${per_gpu_batchsize} \
  vit3d16 text_radbert \
  image_size=128 max_text_len=250 \
  tokenizer=StanfordAIMI/RadBERT