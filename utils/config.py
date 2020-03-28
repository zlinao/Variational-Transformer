import os
import logging 
import argparse

UNK_idx = 0
PAD_idx = 1
EOS_idx = 2
SOS_idx = 3
USR_idx = 4
SYS_idx = 5
CLS_idx = 6
CLS1_idx = 7
Y_idx = 8
if (os.cpu_count() > 8):
    USE_CUDA = True
else:
    USE_CUDA = False

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="mojitalk")
parser.add_argument("--v2", action="store_true")
parser.add_argument("--hidden_dim", type=int, default=300)
parser.add_argument("--emb_dim", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--max_grad_norm", type=float, default=5.0)
parser.add_argument("--beam_size", type=int, default=5)
parser.add_argument("--save_path", type=str, default="save/test/")
parser.add_argument("--save_path_pretrained", type=str, default="save/")
parser.add_argument("--cuda", action="store_true")

parser.add_argument("--pointer_gen", action="store_true")
parser.add_argument("--oracle", action="store_true")
parser.add_argument("--basic_learner", action="store_true")
parser.add_argument("--project", action="store_true")
parser.add_argument("--topk", type=int, default=0)
parser.add_argument("--l1", type=float, default=.0)
parser.add_argument("--softmax", action="store_true")
parser.add_argument("--mean_query", action="store_true")
parser.add_argument("--schedule", type=float, default=0)


parser.add_argument("--large_decoder", action="store_true")
parser.add_argument("--multitask", action="store_true")
parser.add_argument("--is_coverage", action="store_true")
parser.add_argument("--use_oov_emb", action="store_true")
parser.add_argument("--pretrain_emb", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--model", type=str, default="seq2seq")
parser.add_argument("--weight_sharing", action="store_true")
parser.add_argument("--label_smoothing", action="store_true")
parser.add_argument("--noam", action="store_true")
parser.add_argument("--universal", action="store_true")
parser.add_argument("--act", action="store_true")
parser.add_argument("--act_loss_weight", type=float, default=0.001)
parser.add_argument("--emb_file", type=str)
parser.add_argument("--persona", action="store_true")
##cvae
parser.add_argument("--full_kl_step", type=int, default=0)
parser.add_argument("--num_var_layers", type=int, default=0)
parser.add_argument("--kl_ceiling", type=float, default=0.1)
parser.add_argument("--aux_ceiling", type=float, default=1)
parser.add_argument("--load_optim", action="store_true")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
## transformer 
parser.add_argument("--hop", type=int, default=4)
parser.add_argument("--heads", type=int, default=4)
parser.add_argument("--depth", type=int, default=256)
parser.add_argument("--filter", type=int, default=512)

def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)

arg = parser.parse_args()
print_opts(arg)
model = arg.model
dataset = arg.dataset
large_decoder = arg.large_decoder
topk = arg.topk
l1 = arg.l1
oracle = arg.oracle
basic_learner = arg.basic_learner
multitask = arg.multitask
softmax = arg.softmax
mean_query = arg.mean_query
schedule = arg.schedule
# Hyperparameters
hidden_dim= arg.hidden_dim
emb_dim= arg.emb_dim
batch_size= arg.batch_size
lr=arg.lr
beam_size=arg.beam_size
project=arg.project
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=arg.max_grad_norm

USE_CUDA = arg.cuda
pointer_gen = arg.pointer_gen
is_coverage = arg.is_coverage
use_oov_emb = arg.use_oov_emb
cov_loss_wt = 1.0
lr_coverage=0.15
eps = 1e-12
epochs = 10000

emb_file = arg.emb_file or "vectors/glove.6B.{}d.txt".format(str(emb_dim))
pretrain_emb = arg.pretrain_emb

save_path = arg.save_path
save_path_pretrained = arg.save_path_pretrained
persona = arg.persona
test = arg.test


full_kl_step = arg.full_kl_step
### transformer 
hop = arg.hop
heads = arg.heads
depth = arg.depth
filter = arg.filter

v2 = arg.v2
num_var_layers = arg.num_var_layers
kl_ceiling = arg.kl_ceiling
aux_ceiling = arg.aux_ceiling
load_optim = arg.load_optim
gradient_accumulation_steps = arg.gradient_accumulation_steps

label_smoothing = arg.label_smoothing
weight_sharing = arg.weight_sharing
noam = arg.noam
universal = arg.universal
act = arg.act
act_loss_weight = arg.act_loss_weight
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')#,filename='save/logs/{}.log'.format(str(name)))
collect_stats = False


#for interactive human evluation

# CUDA_VISIBLE_DEVICES=3 python3 interact.py --model cvae --cuda --persona --dataset empathetic --save_path_pretrained save/seq2seqcvae_ed_persona_0.05/model_10999_13.4804_32.7288_0.0000_0.0000_9.9921

# CUDA_VISIBLE_DEVICES=4 python3 interact.py --model cvaetrs --cuda --persona --dataset empathetic --save_path_pretrained save/cvae_trs_ed_persona_0.05/model_12999_22.3743_22.9358_0.0000_0.0000_19.2416

# CUDA_VISIBLE_DEVICES=4 python3 interact.py --model trs --cuda --persona --dataset empathetic --save_path_pretrained save/trs_ed_persona/model_8999_4.0222_55.8249_0.0000_0.0000_0.0000

# CUDA_VISIBLE_DEVICES=7 python3 interact.py --model cvaetrs --v2 --cuda --persona --dataset empathetic --save_path_pretrained save/v2_cvae_trs_ed_persona_0.6/model_15999_4.5419_18.7720_0.0000_0.0000_1.6095 --num_var_layers 1


