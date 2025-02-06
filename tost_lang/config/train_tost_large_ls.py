wandb_log = True
wandb_project = 'owt'
wandb_run_name='tost-large-ls'
out_dir = 'tost-large-ls'
always_save_checkpoint = True
init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'

batch_size = 20
block_size = 1024
gradient_accumulation_steps = 3  * 8
TSSA = True

max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 100
eval_iters = 200
log_interval = 10

n_layer = 36
n_head = 20
n_embd = 1280
# weight decay
weight_decay = 1e-1
compile = True
