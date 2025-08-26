import os 
# os.environ['CUDA_VISIBLE_DEVICES']='2'
import json,random
import numpy as np

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os,argparse
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.manifold import TSNE
import matplotlib
from utils import *
from torch.cuda.amp import autocast
# from IPython.display import display, HTML
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llama2-7b-c")
parser.add_argument("--sample_num", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--token_idx", type=int, default=-1)
parser.add_argument("--tasks", type=str, default='easyjb')

args = parser.parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
for k in args.__dict__:
    print(k + ": " + str(args.__dict__[k]))

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(args.seed)

model_name=args.model
# if model_name=='llama2-7b-c':
#     model_path = "/data/xxx/models/Llama-2-7b-chat-hf"
#     user_tag = "[INST]"
#     assistant_tag = "[/INST]"
# elif model_name=='vicuna1.5_7b':
#     model_path = "/data/xxx/models/vicuna1.5_7b"
#     user_tag = "A chat between a curious user and an artificial intelligence assistant. USER:"
#     assistant_tag = "ASSISTANT:"
# elif model_name=='mistral':
#     model_path = "/data/xxx/models/mistral-7b-instruct-v0.1"
#     user_tag = "[INST]"
#     assistant_tag = "[/INST]"
user_tag,assistant_tag,base_model_path,model_path=select_model_path(model_name)

def add_vector_after_position(matrix, vector, position_ids, after=None):
    after_id = after
    if after_id is None:
        after_id = position_ids.min().item() - 1
    mask = position_ids > after_id
    mask = mask.unsqueeze(-1)
    mask=mask.cpu().to(matrix.device)
    vector=vector.cpu().to(matrix.device)
    matrix += mask.float() * vector
    return matrix


def find_subtensor_position(tensor, sub_tensor):
    n, m = tensor.size(0), sub_tensor.size(0)
    if m > n:
        return -1
    for i in range(n - m + 1):
        if torch.equal(tensor[i : i + m], sub_tensor):
            return i
    return -1


def find_instruction_end_postion(tokens, end_str):
    end_pos = find_subtensor_position(tokens, end_str)
    return end_pos + len(end_str) - 1

class AttnWrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        self.activations = output[0]
        return output


class BlockOutputWrapper(torch.nn.Module):
    def __init__(self,idx, block, unembed_matrix, norm, tokenizer):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.tokenizer = tokenizer

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_out_unembedded = None
        self.intermediate_resid_unembedded = None
        self.mlp_out_unembedded = None
        self.block_out_unembedded = None

        self.activations = None
        self.add_activations = None
        self.after_position = None

        self.save_internal_decodings = False

        self.calc_dot_product_with = None
        self.dot_products = []

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.activations = output[0]
        if self.calc_dot_product_with is not None:
            last_token_activations = self.activations[0, -1, :]
            decoded_activations = self.unembed_matrix(self.norm(last_token_activations))
            top_token_id = torch.topk(decoded_activations, 1)[1][0]
            top_token = self.tokenizer.decode(top_token_id)
            dot_product = torch.dot(last_token_activations, self.calc_dot_product_with)
            self.dot_products.append((top_token, dot_product.cpu().item()))
        if self.add_activations is not None:
            self.add_activations=self.add_activations.cpu().to(output[0].device)
            augmented_output = add_vector_after_position(
                matrix=output[0],
                vector=self.add_activations,
                position_ids=kwargs["position_ids"],
                after=self.after_position,
            )
            output = (augmented_output + self.add_activations,) + output[1:]

        if not self.save_internal_decodings:
            return output

        # Whole block unembedded
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))

        # Self-attention unembedded
        attn_output = self.block.self_attn.activations
        self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_output))

        # Intermediate residual unembedded
        attn_output += args[0]
        self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_output))

        # MLP unembedded
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_output))

        return output

    def add(self, activations):
        self.add_activations = activations

    def reset(self):
        self.add_activations = None
        self.activations = None
        self.block.self_attn.activations = None
        self.after_position = None
        self.calc_dot_product_with = None
        self.dot_products = []


class Llama2ChatHelper:
    def __init__(self):
        self.device = "cuda" #if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=False,padding_side="left") 
        self.model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map="auto").eval()
       
        self.END_STR = torch.tensor(self.tokenizer.encode(assistant_tag)[1:]).to( self.device )
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(
                i,layer, self.model.lm_head, self.model.model.norm, self.tokenizer
            )

    def set_save_internal_decodings(self, value):
        for layer in self.model.model.layers:
            layer.save_internal_decodings = value

    def set_after_positions(self, pos):
        for layer in self.model.model.layers:
            layer.after_position = pos

    def prompt_to_tokens(self,instruction):
        prompt=f"{user_tag} {instruction.strip()} {assistant_tag}"
        prompt=prompt.strip()
        global flag
        dialog_tokens = self.tokenizer.encode(prompt)
        return torch.tensor(dialog_tokens).unsqueeze(0)
    
    @autocast()
    def generate_text(self, prompt, max_new_tokens=50):
        tokens = self.prompt_to_tokens(prompt).to(self.device)
        return self.generate(tokens, max_new_tokens=max_new_tokens)
    @autocast()
    def generate(self, tokens, max_new_tokens=50):
        instr_pos = find_instruction_end_postion(tokens[0], self.END_STR)
        self.set_after_positions(instr_pos)
        generated = self.model.generate(
            inputs=tokens, max_new_tokens=max_new_tokens, top_k=1
        )
        return self.tokenizer.batch_decode(generated)[0]
    @autocast()
    def get_logits(self, tokens):
        with torch.no_grad():
            logits = self.model(tokens).logits
            return logits

    def get_last_activations(self, layer):
        return self.model.model.layers[layer].activations

    def set_add_activations(self, layer, activations):
        self.model.model.layers[layer].add(activations)

    def set_calc_dot_product_with(self, layer, vector):
        self.model.model.layers[layer].calc_dot_product_with = vector

    def get_dot_products(self, layer):
        return self.model.model.layers[layer].dot_products

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

    def print_decoded_activations(self, decoded_activations, label, topk=10):
        data = self.get_activation_data(decoded_activations, topk)[0]
        print(label, data)

    def decode_all_layers(
        self,
        tokens,
        topk=10,
        print_attn_mech=True,
        print_intermediate_res=True,
        print_mlp=True,
        print_block=True,
    ):
        tokens = tokens.to(self.device)
        self.get_logits(tokens)
        for i, layer in enumerate(self.model.model.layers):
            print(f"Layer {i}: Decoded intermediate outputs")
            if print_attn_mech:
                self.print_decoded_activations(
                    layer.attn_out_unembedded, "Attention mechanism", topk=topk
                )
            if print_intermediate_res:
                self.print_decoded_activations(
                    layer.intermediate_resid_unembedded,
                    "Intermediate residual stream",
                    topk=topk,
                )
            if print_mlp:
                self.print_decoded_activations(
                    layer.mlp_out_unembedded, "MLP output", topk=topk
                )
            if print_block:
                self.print_decoded_activations(
                    layer.block_output_unembedded, "Block output", topk=topk
                )

    def plot_decoded_activations_for_layer(self, layer_number, tokens, topk=10):
        tokens = tokens.to(self.device)
        self.get_logits(tokens)
        layer = self.model.model.layers[layer_number]

        data = {}
        data["Attention mechanism"] = self.get_activation_data(
            layer.attn_out_unembedded, topk
        )[1]
        data["Intermediate residual stream"] = self.get_activation_data(
            layer.intermediate_resid_unembedded, topk
        )[1]
        data["MLP output"] = self.get_activation_data(layer.mlp_out_unembedded, topk)[1]
        data["Block output"] = self.get_activation_data(
            layer.block_output_unembedded, topk
        )[1]

        # Plotting
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
        fig.suptitle(f"Layer {layer_number}: Decoded Intermediate Outputs", fontsize=21)

        for ax, (mechanism, values) in zip(axes.flatten(), data.items()):
            tokens, scores = zip(*values)
            ax.barh(tokens, scores, color="skyblue")
            ax.set_title(mechanism)
            ax.set_xlabel("Value")
            ax.set_ylabel("Token")

            # Set scientific notation for x-axis labels when numbers are small
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="x")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def get_activation_data(self, decoded_activations, topk=10):
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        return list(zip(tokens, probs_percent)), list(zip(tokens, values.tolist()))

def save_activation_projection_tsne(
    activations1,
    activations2,
    fname,
    title,
    label1="Safe Prompts",
    label2="Unsafe Prompts",
):
    """
    activations1: n_samples x vector dim tensor
    activations2: n_samples x vector dim tensor

    projects to n_samples x 2 dim tensor using t-SNE (over the full dataset of both activations 1 and 2) and saves visualization.
    Colors projected activations1 as blue and projected activations2 as red.
    """
    plt.clf()
    activations = torch.cat([activations1, activations2], dim=0)
    activations_np = activations.cpu().numpy()

    # t-SNE transformation
    tsne = TSNE(n_components=2)
    projected_activations = tsne.fit_transform(activations_np)

    # Splitting back into activations1 and activations2
    activations1_projected = projected_activations[: activations1.shape[0]]
    activations2_projected = projected_activations[activations1.shape[0] :]

    # Visualization
    for x, y in activations1_projected:
        plt.scatter(x, y, color="blue", marker="o", alpha=0.4)

    for x, y in activations2_projected:
        plt.scatter(x, y, color="red", marker="o", alpha=0.4)

    # Adding the legend
    scatter1 = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="blue",
        markersize=10,
        label=label1,
    )
    scatter2 = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="red",
        markersize=10,
        label=label2,
    )

    plt.legend(handles=[scatter1, scatter2])
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.savefig(fname)

def plot_all_activations(layers):
    if not os.path.exists("figs"):
        os.mkdir("figs")
    for layer in layers:
        pos = torch.load(f"save_re/{args.model}/token{args.token_idx}/{args.model}_safe_layer{layer}_token{args.token_idx}_seed{args.seed}.pt")
        neg = torch.load(f"save_re/{args.model}/token{args.token_idx}/{args.model}_unsafe_layer{layer}_token{args.token_idx}_seed{args.seed}.pt")
        save_activation_projection_tsne(
            pos,
            neg,
            f"figs/0427/{args.model}_activations_layer{layer}_token{args.token_idx}.png",
            f"t-SNE projected activations layer {layer}",
        )


def get_vec(layer):
    return torch.load(f"save_re/{args.model}/token{args.token_idx}/{args.model}_vec_layer{layer}_token{args.token_idx}_seed{args.seed}.pt")#/data/xxx/LM-exp-main/refusal/vec_layer_7_xxx.pt

def generate_and_save_jailbreak_vectors(model, dataset,root_dir, start_layer=0, end_layer=32, token_idx=-1):  ###为什么选-2
    if '13b' in  args.model:end_layer=40
    layers = list(range(start_layer, end_layer ))
    
    jb_activations = dict([(layer, []) for layer in layers])
    model.set_save_internal_decodings(False)
    model.reset_all()
    #p对应safe n对应unsafe
    for j,prompt in enumerate(dataset):
        tokens=model.prompt_to_tokens(prompt).to(model.device)
        model.reset_all()
        model.get_logits(tokens) #####
        for layer in layers:
            activations = model.get_last_activations(layer)
            activations = activations[0, token_idx, :].detach().cpu()
            a,b=check_nan_inf(activations)
            if a or b:
                print(f'layer:{layer}, prompt:{j}, nan:{a}, inf:{b}')

            jb_activations[layer].append(activations)
        
    for layer in layers:
        jb = torch.stack(jb_activations[layer])
        
        if layer==0:
            print('jb_shape:',print(jb.shape))
        torch.save(jb,os.path.join(root_dir,f"jb_layer{layer}.pt"))


model = Llama2ChatHelper()
model.set_save_internal_decodings(False)
all_results = []
predictor=RoBERTaPredictor(path='/data/xxx/models/roberta_jb_evaluator')

data = []
pair_dataset='/data/hzq/datasets/advbench_pairs.json'
with open(pair_dataset, encoding="utf-8") as f:
    for i,line in enumerate(f):
        if i==400:break
        json_line = json.loads(line.strip())
        if args.model[:6]=='llama3':
            chat = [
                {"role": "user", "content": json_line},
                # {"role": "assistant","content": "<think>\nI'm thinking now.</think>Yes, great"},
            ]
            json_line = model.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        data.append(json_line)
dataset = ComparisonDataset(data)   
root_dir=f'save_re/{args.model}/advbench_pair'
os.makedirs(root_dir,exist_ok=True)
generate_and_save_steering_vectors(model, dataset,root_dir)


jb_file_path=f'{args.model[:6]}_easyjailbreak.json'
with open(jb_file_path, "r", encoding="utf-8") as jb_file:
    jb_prompts = json.load(jb_file)

attacks={'ReNellm':[],'gcg':[],'Autodan':[],'PAIR':[],'Deepinception':[],'GPTFuzz':[],'CodeChameleon':[]}
# if args.model=='vicuna1.5_7b': attacks['ICA']=[]
for x in jb_prompts:
    #if len(attacks[x['attack']])<5:
    attacks[x['attack']].append(x['prompt'])
for attack in ['ReNellm','gcg','Autodan','PAIR','Deepinception','GPTFuzz','CodeChameleon']:
    print(f'{attack} begin')
    #if os.path.exists(f'save_re/{args.model}/easyjb/{attack}'): continue
    all_rst=[]
    root_dir=f'save_re/{args.model}/easyjb/{attack}'
    os.makedirs(root_dir,exist_ok=True)
    generate_and_save_jailbreak_vectors(model,attacks[attack],root_dir)
    print(f'{attack} done')
