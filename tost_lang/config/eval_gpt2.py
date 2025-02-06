# evaluate the base gpt2
# n_layer=12, n_head=12, n_embd=768
# 124M parameters
batch_size = 16
eval_iters = 1000 # use more iterations to get good estimate
eval_only = True
wandb_log = False
init_from = 'resume'
out_dir = 'tost-base-ls'
# out_dir = 'out-gpt2'
# init_from = 'gpt2-medium'

dataset = 'openwebtext'

block_size = 1024
# n_layer = 24
# n_head = 16
# n_embd = 1024
# n_layer = 36
# n_head = 20
# n_embd = 1280
TSSA = True
compile = True

