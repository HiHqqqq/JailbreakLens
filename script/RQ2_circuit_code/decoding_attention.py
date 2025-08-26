import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import einops

import torch,json,argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llama2-7b-c")
parser.add_argument("--h", type=int)
parser.add_argument("--l", type=int)

args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained('/data/xxx/models/Llama-2-7b-chat-hf')
model= AutoModelForCausalLM.from_pretrained(
    '/data/xxx/models/Llama-2-7b-chat-hf',
    device_map="auto",
    low_cpu_mem_usage=True,
    use_cache=False,
    torch_dtype=torch.float16,
)

def get_refusal_dir():
    if os.path.exists(f'/data/xxx/Activation_Patching/dir_dif/save_dir/{args.model}_isharm.pt'):
        past_dir=torch.load(f'/data/xxx/Activation_Patching/dir_dif/save_dir/{args.model}_isharm.pt')
        return past_dir
    c=tl_model.W_U[:, refuse_token] - tl_model.W_U[:, answer_token]
    torch.save(c,f'./save_dir/{args.model}_isharm.pt')
    return c

def get_activation_data(decoded_activations, topk=50):
    softmaxed = torch.nn.functional.softmax(decoded_activations, dim=-1)
    values, indices = torch.topk(softmaxed, topk)
    values = values /values.sum()
    probs_percent = [int(v * 100) for v in values.tolist()]
    tokens = tokenizer.batch_decode(indices.unsqueeze(-1))
    a=list(zip(tokens, probs_percent))
    b=list(zip(tokens, values.tolist()))
        
    return tokens,list(zip(tokens, probs_percent)), list(zip(tokens, values.tolist()))


def decode_2(x,topk=10):
    lm_head_weight = model.lm_head.weight  # [vocab_size, hidden_size]
    lm_head_bias = model.lm_head.bias 
    logits = torch.matmul(x, lm_head_weight.t()) #+ lm_head_bias
    
    softmaxed = torch.nn.functional.softmax(logits, dim=-1)
    values, indices = torch.topk(softmaxed, topk)
    values = values /values.sum()
    probs_percent = [int(v * 100) for v in values.tolist()]
    tokens = tokenizer.batch_decode(indices.unsqueeze(-1))
    a=list(zip(tokens, probs_percent))
    b=list(zip(tokens, values.tolist()))
    return tokens,list(zip(tokens, probs_percent)), list(zip(tokens, values.tolist()))

for dataset_label in ["harmful", "harmless","gcg","ReNellm",'PAIR','Deepinception','Autodan','GPTFuzz','CodeChameleon']:
    # resid_decomp=torch.load(f'/data/xxx/Activation_Patching/dir_dif/save_pt/{args.model}/{args.model}_{dataset_label}_resid_decomp.pt')
    # resid
    per_head_resid=torch.load(f'/data/xxx/Activation_Patching/dir_dif/save_pt/llama2-7b-c/llama2-7b-c_{dataset_label}_per_head_resid.pt')
    per_head_resid = einops.rearrange(
            per_head_resid,
            "(layer head) ... -> layer head ...",
            layer=32#tl_model.cfg.n_layers
        )
    key_component=per_head_resid[args.l,args.h,:,:]

    for i,x in enumerate(key_component):

        a,b,c=decode_2(x)
        print(f'dataset:{dataset_label}')
        print(f'i:{i}')
        
        print(a)
        print('-'*80)

    

