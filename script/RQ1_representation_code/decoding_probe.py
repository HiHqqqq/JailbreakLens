import os
# os.environ['CUDA_VISIBLE_DEVICES']='2'
import einops
from utils_Probing_V2 import *
import torch,json,argparse
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llama2-7b-c")
parser.add_argument("--probe", type=str, default='cluster')

args = parser.parse_args()
setup_seed(0)

args = parser.parse_args()
# if model=='llama2_7b_c' or model=='vicuna1.5_7b':
if '13b' in args.model:
    tokenizer = AutoTokenizer.from_pretrained('/data/xxx/models/Llama-2-13b-chat-hf')
    model= AutoModelForCausalLM.from_pretrained(
        '/data/xxx/models/Llama-2-13b-chat-hf',
        device_map="auto",
        low_cpu_mem_usage=True,
        use_cache=False,
        torch_dtype=torch.float16,
    )
    num=40
else:
    tokenizer = AutoTokenizer.from_pretrained('/data/xxx/models/Llama-2-7b-chat-hf')
    model= AutoModelForCausalLM.from_pretrained(
        '/data/xxx/models/Llama-2-7b-chat-hf',
        device_map="auto",
        low_cpu_mem_usage=True,
        use_cache=False,
        torch_dtype=torch.float16,
    )
    num=32

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

print('similarity of linear and cluster')

for l in range(int(num/2),num):
    linear_d=torch.load(f'save/{args.model}/linear/probe_layer{l}.pt').cpu()
    cluster_d=torch.load(f'save/{args.model}/cluster/probe_layer{l}.pt').cpu()
    similarity = F.cosine_similarity(linear_d, cluster_d, dim=0)
    print(f'layer:{l}, sim:{similarity}')

print('similarity of linear and pca')
for l in range(int(num/2),num):
    linear_d=torch.load(f'save/{args.model}/linear/probe_layer{l}.pt').cpu()
    pca_d=torch.load(f'save/{args.model}/pca/probe_layer{l}.pt').cpu()
    similarity = F.cosine_similarity(linear_d, cluster_d, dim=0)
    print(f'layer:{l}, sim:{similarity}')

print('similarity of cluster and pca')
for l in range(int(num/2),num):
    cluster_d=torch.load(f'save/{args.model}/cluster/probe_layer{l}.pt').cpu()
    pca_d=torch.load(f'save/{args.model}/pca/probe_layer{l}.pt').cpu()
    similarity = F.cosine_similarity(linear_d, cluster_d, dim=0)
    print(f'layer:{l}, sim:{similarity}')

for name in ['linear','linear2','pca','cluster']:
    print(f'\nProbing:{name}')
    for l in range(int(num/2),num):
        save_root_dir=f'save/{args.model}/{name}'
        direction=torch.load(os.path.join(save_root_dir,f'probe_layer{l}.pt'))
        direction=direction.half().cuda()
        rev_direction=-direction

        a,b,c=decode_2(direction)
        rev_a,b,c=decode_2(rev_direction)
        
        print(f'name:{name}, layer:{l}')
        print('positive: ',a)
        print('negative: ',rev_a)
        print('-'*80)

        

