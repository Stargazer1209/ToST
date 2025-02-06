# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
out_dir = 'out-owt4'
init_from = "resume"
wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-124M4'
init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'

# these make the total batch size be ~0.5M
# 64 batch size * 1024 block size * 5 gradaccum * 4 GPUs = 1,310,720 # 491,520
batch_size = 64
block_size = 1024
gradient_accumulation_steps = 5  * 4
always_save_checkpoint = True

# this makes total number of tokens be around 300B * 2 
max_iters = 460000*2
lr_decay_iters = 460000*2

# eval stuff
eval_interval = 100
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
compile = True