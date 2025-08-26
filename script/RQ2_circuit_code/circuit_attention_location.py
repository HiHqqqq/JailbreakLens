import gc
import os
# os.environ['CUDA_VISIBLE_DEVICES']='4'
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
from dataset_process import InstructionDataset, PairedInstructionDataset
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llama3-1-8b")
parser.add_argument("--pos_token", type=int, default=18585)#"Sure")
parser.add_argument("--neg_token", type=int, default=8221)#"Sorry")
parser.add_argument("--show_toptokens", action='store_const', const=1, default=0)
parser.add_argument("--show_resid", action='store_const', const=1, default=0)
parser.add_argument("--show_patching", action='store_const', const=1, default=1)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(args.model)
user_tag,assistant_tag,base_model_path,model_path=select_model_path(args.model)
# model_path ='/data/xxx/models/vicuna1.5_7b'
# tokenizer_path='/data/xxx/models/vicuna1.5_7b'#"meta-llama/Llama-2-7b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    #token=os.environ["HF_TOKEN"], 
    low_cpu_mem_usage=True,
    use_cache=False,
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    #token=os.environ["HF_TOKEN"],
    use_fast=False
)

# tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

if args.model=='llama2-7b-c' or args.model=='vicuna1.5_7b': tl_model_name="meta-llama/Llama-2-7b-chat-hf"
elif args.model=='llama2-13b-c' or args.model=='vicuna1.5_13b': tl_model_name="meta-llama/Llama-2-13b-chat-hf"
else: tl_model_name='meta-llama/Llama-3.1-8B-Instruct'
tl_model = HookedTransformer.from_pretrained(
    tl_model_name,
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


harmful_prompts,harmless_prompts=[],[]
pair_dataset='/data/xxx/datasets/advbench_pairs.json'
with open(pair_dataset, encoding="utf-8") as f:
    for i,line in enumerate(f):
        #if i==10:break
        json_line = json.loads(line.strip())
        prompt=json_line['unsafe']
        harmful_prompts.append(prompt)
        prompt=json_line['safe']
        harmless_prompts.append(prompt)
sample_num=5
harmful_prompts=harmful_prompts[:sample_num]
harmless_prompts=harmless_prompts[:sample_num]
def template(prompt):
    chat = [
                {"role": "user", "content": prompt},
                ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return prompt 
if args.model[:6]=='llama3':
    harmful_prompts=[template(x) for x in harmful_prompts]
    harmless_prompts=[template(x) for x in harmless_prompts]

paired_dataset = PairedInstructionDataset(
    args.model,
    harmful_prompts,
    harmless_prompts,
    tokenizer,
)

harmful_dataset = paired_dataset.harmful_dataset
harmless_dataset = paired_dataset.harmless_dataset

print(harmful_dataset.prompt_strs[0])
print(harmless_dataset.prompt_strs[0])
harmful_logits, harmful_cache = tl_model.run_with_cache(harmful_dataset.prompt_toks)
harmless_logits, harmless_cache = tl_model.run_with_cache(harmless_dataset.prompt_toks)
torch.save(harmful_cache,f'{args.model}_harmful_cache_{sample_num}.pt')
torch.save(harmful_logits,f'{args.model}_harmful_logits_{sample_num}.pt')
torch.save(harmless_cache,f'{args.model}_harmless_cache_{sample_num}.pt')
torch.save(harmless_logits,f'{args.model}_harmless_logits_{sample_num}.pt')
# harmful_per_head_resid, labels = harmful_cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
# harmless_per_head_resid, labels = harmless_cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
# harmful_accumulated_resid, harmful_labels_residual = harmful_cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)
# harmful_resid_decomp, harmful_labels_resid_decomp = harmful_cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)   
# harmless_accumulated_resid, harmless_labels_residual = harmless_cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)
# harmless_resid_decomp, harmless_labels_resid_decomp = harmless_cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)   
# os.makedirs(f'save_pt/{args.model}',exist_ok=True)
# torch.save(harmful_cache,f'save_pt/{args.model}/{args.model}_harmful_cache.pt')
# torch.save(harmful_logits,f'save_pt/{args.model}/{args.model}_harmful_logits.pt')
# torch.save(harmless_cache,f'save_pt/{args.model}/{args.model}_harmless_cache.pt')
# torch.save(harmless_logits,f'save_pt/{args.model}/{args.model}_harmless_logits.pt')
# torch.save(harmful_per_head_resid,f'save_pt/{args.model}/{args.model}_harmful_per_head_resid.pt')
# torch.save(harmless_per_head_resid,f'save_pt/{args.model}/{args.model}_harmless_per_head_resid.pt')
# torch.save(harmful_accumulated_resid,f'save_pt/{args.model}/{args.model}_harmful_accumulated_resid.pt')
# torch.save(harmless_accumulated_resid,f'save_pt/{args.model}/{args.model}_harmless_accumulated_resid.pt')
# torch.save(harmful_resid_decomp,f'save_pt/{args.model}/{args.model}_harmful_resid_decomp.pt')
# torch.save(harmless_resid_decomp,f'save_pt/{args.model}/{args.model}_harmless_resid_decomp.pt')
# exit(0)
# # exit(0)
# harmful_logits=torch.load(f'{args.model}_harmful_logits_{sample_num}.pt')
# harmful_cache=torch.load(f'{args.model}_harmful_cache_{sample_num}.pt')
# harmless_logits=torch.load(f'{args.model}_harmless_logits_{sample_num}.pt')
# harmless_cache=torch.load(f'{args.model}_harmless_cache_{sample_num}.pt')

print(f'/data/xxx/Activation_Patching/dir_dif/save_fig/{args.model}')
os.makedirs(f'/data/xxx/Activation_Patching/dir_dif/save_fig/{args.model}',exist_ok=True)

def get_refusal_score(logits: Float[Tensor, "d_vocab"]):
    return logits[refuse_token] - logits[answer_token]

def get_refusal_dir(detail_name=0):
    if detail_name==0:
        if os.path.exists(f'./save_dir/{args.model}_isharm.pt'):
            past_dir=torch.load(f'./save_dir/{args.model}_isharm.pt')
            return past_dir
        else:
            c=tl_model.W_U[:, refuse_token] - tl_model.W_U[:, answer_token]
            torch.save(c,f'./save_dir/{args.model}_isharm.pt')
            return c
    
    if detail_name:
        c=tl_model.W_U[:, refuse_token] - tl_model.W_U[:, answer_token]
        torch.save(c,f'./save_dir/{args.model}_isharm_r{refuse_token}_a{answer_token}.pt')
        return c
def get_refusal_score_avg(logits: Float[Tensor, 'batch seq_len n_vocab']) -> float:
    assert (logits.ndim == 3)
    scores = torch.tensor([get_refusal_score(tensor) for tensor in logits[:, -1, :]])
    return scores.mean(dim=0).item()

for (dataset_label, cache, logits) in zip(["Harmful dataset", "Harmless dataset"], [harmful_cache, harmless_cache], [harmful_logits, harmless_logits]):
    final_resid: Float[Tensor, "batch seq d_model"] = cache["resid_post", -1]  #final layer
    final_token_resid: Float[Tensor, "batch d_model"] = final_resid[:, -1, :]  #final token

    average_logit_diff = einops.einsum(
        final_token_resid, get_refusal_dir(),
        "batch d_model, d_model -> batch"
    ).mean(dim=0)

    original_logit_diff = get_refusal_score_avg(logits)

    print(f"{dataset_label}:")
    print(f"\tCalculated avg refusal score: {average_logit_diff:+.4f}")
    print(f"\tActual avg refusal score:     {original_logit_diff:+.4f}")


per_head_resid, labels = harmful_cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
per_head_resid = einops.rearrange(
    per_head_resid,
    "(layer head) ... -> layer head ...",
    layer=tl_model.cfg.n_layers
)
del harmful_cache

refusal_scores_by_head = einops.einsum(
    per_head_resid,
    get_refusal_dir(detail_name=1),
    'layer head batch d_model, d_model -> layer head batch'
).mean(dim=-1)
torch.save(refusal_scores_by_head.cpu().numpy(),f'save_fig/{args.model}_head_refusal-score_harmful.pt')
fig = px.imshow(
    refusal_scores_by_head.cpu().numpy(),
    title=f"Refusal score attribution per head, pos=-1",
    labels={"x": "Head", "y": "Layer"},
    width=500, height=500,
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
)
print('harmful head:')
harmful_matrix=refusal_scores_by_head.cpu().numpy()
max_value = np.max(harmful_matrix)
max_pos = np.unravel_index(np.argmax(harmful_matrix), harmful_matrix.shape)
print(f"{max_value},{max_pos}")
refusal_l,refusal_h=max_pos[0],max_pos[1]
pio.write_image(fig,f'{args.model}_patching.png')
pio.write_image(fig,f'save_fig/{args.model}/{args.model}_harmful__r{refuse_token}a{answer_token}.png')

per_head_resid, labels = harmless_cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
per_head_resid = einops.rearrange(
    per_head_resid,
    "(layer head) ... -> layer head ...",
    layer=tl_model.cfg.n_layers
)
torch.save(refusal_scores_by_head.cpu().numpy(),f'save_fig/{args.model}_head_refusal-score_harmless.pt')
del harmless_cache

refusal_scores_by_head = einops.einsum(
    per_head_resid,
    get_refusal_dir(),
    'layer head batch d_model, d_model -> layer head batch'
).mean(dim=-1)
# if args.model=='llama3-8b':
#     for i in range(32):
#         for j in range(32):
#             if refusal_scores_by_head[i][j]

fig = px.imshow(
    refusal_scores_by_head.cpu().numpy(),
    title=f"Refusal score attribution per head, pos=-1",
    labels={"x": "Head", "y": "Layer"},
    width=500, height=500,
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
)
pio.write_image(fig,f'{args.model}_patching.png')
pio.write_image(fig,f'save_fig/{args.model}/{args.model}_harmless_r{refuse_token}a{answer_token}.png')
print('harmless head:')
agreement_matrix=refusal_scores_by_head.cpu().numpy()
min_value = np.min(agreement_matrix)
min_pos = np.unravel_index(np.argmin(agreement_matrix), agreement_matrix.shape)
print(f" {min_value},  {min_pos}")
agreement_l,agreement_h=min_pos[0],min_pos[1]

print('Harmful on refusal-sig:',harmful_matrix[refusal_l][refusal_h])
print('Harmful on agreement-sig:',harmful_matrix[agreement_l][agreement_h])
print('Harmless on refusal-sig:',agreement_matrix[refusal_l][refusal_h])
print('Harmless on agreement-sig:',agreement_matrix[agreement_l][agreement_h])
