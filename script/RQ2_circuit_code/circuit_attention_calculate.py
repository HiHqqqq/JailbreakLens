import gc
import os
# os.environ['CUDA_VISIBLE_DEVICES']='2'
import torch
import numpy as np
import einops
import transformer_lens
import functools
import plotly.graph_objects as go
import plotly.express as px
# import circuitsvis as cv
import tqdm
import json,argparse
import plotly.io as pio
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens import utils as tl_utils
from transformer_lens.hook_points import HookPoint
from torch import Tensor
from torch.utils.data import Dataset
from jaxtyping import Int, Float
from typing import Union, Tuple, List
from sklearn.decomposition import PCA
from utils_sub import *
from dataset_process import InstructionDataset, PairedInstructionDataset,JailbrokenDataset
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="vicuna1.5_7b")
parser.add_argument("--pos_token", type=str, default="Sure")
parser.add_argument("--neg_token", type=str, default="Sorry")
parser.add_argument("--name", type=str, default="gcg")
# parser.add_argument("--agree_l", type=int)
# parser.add_argument("--agree_h", type=int)
args = parser.parse_args()



device = "cuda" if torch.cuda.is_available() else "cpu"

user_tag,assistant_tag,base_model_path,model_path=select_model_path(args.model)
print('model_path:',model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,

    low_cpu_mem_usage=True,
    use_cache=False,
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    #token=os.environ["HF_TOKEN"],
    use_fast=False
)

tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = 'left'

if args.model=='llama2-7b-c' or args.model=='vicuna1.5_7b':
    base_name="meta-llama/Llama-2-7b-chat-hf"
elif args.model[:7]=='mistral':
    base_name='mistralai/Mistral-7B-instruct-v0.1'
elif args.model=='llama2-13b-c' or args.model=='vicuna1.5_13b':
    base_name="meta-llama/Llama-2-13b-chat-hf"
elif args.model=='llama3-8b':
    base_name='meta-llama/Llama-3.1-8B-Instruct'
tl_model = HookedTransformer.from_pretrained(
    base_name,
    hf_model=model,
    device='cpu',
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    tokenizer=tokenizer,
    default_padding_side='left',
    dtype="float16",
).to(device)

torch.set_grad_enabled(False)

refuse_token=tokenizer.encode(args.neg_token)
assert refuse_token[0]==1
answer_token=tokenizer.encode(args.pos_token)
assert answer_token[0]==1
print(f"refuse_token: {tokenizer.decode(refuse_token)} ({refuse_token})")
print(f"answer_token: {tokenizer.decode(answer_token)} ({answer_token})")


attacks={'ReNellm':[],'gcg':[],'Autodan':[],'PAIR':[],'Deepinception':[],'GPTFuzz':[],'CodeChameleon':[],'safe':[],'unsafe':[]}
responses={'ReNellm':[],'gcg':[],'Autodan':[],'PAIR':[],'Deepinception':[],'GPTFuzz':[],'CodeChameleon':[],'safe':[],'unsafe':[]}



for name in [args.name]:#['gcg','Autodan','PAIR','Deepinception','GPTFuzz','CodeChameleon','ReNellm']:
    if '7b' in args.model:
        jb_file_path=f'/data/xxx/ExJB/{args.model[:6]}_{name}_easyjb_new.json'#new.json'
    elif '13b' in args.model:
        jb_file_path=f'/data/xxx/ExJB/{args.model}_{name}_easyjb_2.json'
    # jb_file_path=f'/data/xxx/ExJB/{args.model[:6]}_{name}_output.json'#f'/data/xxx/ExJB/{args.model[:6]}_{name}_easyjb_new.json'
    with open(jb_file_path, "r", encoding="utf-8") as jb_file:
        jb_prompts = json.load(jb_file)
    for x in jb_prompts:
        #if len(attacks[name])==5: break
        attacks[name].append(x['prompt'])
        responses[name].append(x['response'])

os.makedirs('save_cache',exist_ok=True)
#generate_cache_individual per token
def get_refusal_dir():
    if os.path.exists(f'/data/xxx/Activation_Patching/dir_dif/save_dir/{args.model}_isharm.pt'):
        past_dir=torch.load(f'/data/xxx/Activation_Patching/dir_dif/save_dir/{args.model}_isharm.pt')
        return past_dir
    c=tl_model.W_U[:, refuse_token] - tl_model.W_U[:, answer_token]
    torch.save(c,f'./save_dir/{args.model}_isharm.pt')
    return c

      

os.makedirs('/data3/xxx/revise_easyjb',exist_ok=True)
#generate_cache_individual
print('begin')
r_rst={}
a_rst={}
for name in ['GPTFuzz','ReNellm','gcg','Autodan','PAIR','Deepinception','CodeChameleon']:
    refusal_signal=[]
    agreement_signal=[]
    for j,prompt in enumerate(attacks[name]):
        #if j==5:break
        dataset=[prompt]
        jb_dataset=JailbrokenDataset(args.model,dataset,tokenizer).dataset
        print(name)
        print(jb_dataset.prompt_strs[0])
        try:
            logits, cache = tl_model.run_with_cache(jb_dataset.prompt_toks)
            # torch.save(cache,f'/data3/xxx/data3_easyjb/{args.model}_{name}_cache_{j}.pt')
            # del logits
        except torch.cuda.OutOfMemoryError:
            print('CUDOutOfMemoryError logits, cache')
            refusal_signal.append(-1)
            agreement_signal.append(-1)
            continue
        # if not os.path.exists(f'/data3/xxx/data3_easyjb/{args.model}_{name}_cache_{j}.pt'):break
        # cache=torch.load(f'/data3/xxx/data3_easyjb/{args.model}_{name}_cache_{j}.pt')
    
        # resid_decomp, labels_resid_decomp = cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)   
        # torch.save(resid_decomp,f'/data3/xxx/revise_easyjb/{args.model}_{name}_resid_decomp_{j}.pt')
        # del resid_decomp
        try:
            per_head_resid, labels = cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
            torch.save(per_head_resid,f'/data3/xxx/revise_easyjb/{args.model}_{name}_per_head_resid_{j}.pt')

        except torch.cuda.OutOfMemoryError:
            print('CUDOutOfMemoryError HEADS')
            refusal_signal.append(-1)
            agreement_signal.append(-1)
            continue
        per_head_resid = einops.rearrange(
                    per_head_resid,
                    "(layer head) ... -> layer head ...",
                    layer=tl_model.cfg.n_layers
                )

        refusal_scores_by_head = einops.einsum(
            per_head_resid,
            get_refusal_dir(),
            'layer head batch d_model, d_model -> layer head batch'
        )
        r_value=refusal_scores_by_head[refusal_l,refusal_h].item()
        a_value=refusal_scores_by_head[agree_l,agree_h].item()
        refusal_signal.append(r_value)
        agreement_signal.append(a_value)
        print('index:',j)
        print(f'refusal_signal current:{r_value}, total:{refusal_signal}')
        print(f'agreement_signal current:{a_value}, total:{agreement_signal}')
        
        del cache
        print('-'*60)
    print('*'*60)
    print(name,len(refusal_signal))
    print(refusal_signal)
    print(agreement_signal)
    a_rst[name]=agreement_signal
    r_rst[name]=refusal_signal
print('A,R')
print(a_rst)

print(r_rst)