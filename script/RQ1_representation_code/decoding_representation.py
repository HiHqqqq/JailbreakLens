import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import torch,json,argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import *
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="vicuna1.5_7b")
parser.add_argument("--sample_num", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--token_idx", type=int, default=-1)
parser.add_argument("--tasks",type=str,nargs='+',default=['unsafe'])

args = parser.parse_args()
for k in args.__dict__:
    print(k + ": " + str(args.__dict__[k]))
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
class AttnWrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None
        self.add_tensor = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        if self.add_tensor is not None:
            output = (output[0] + self.add_tensor,)+output[1:]
        self.activations = output[0]
        return output

    def reset(self):
        self.activations = None
        self.add_tensor = None

class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, unembed_matrix, norm, tokenizer):
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

        self.save_internal_decodings = True

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

user_tag,assistant_tag,base_model_path,model_path=select_model_path(args.model)
class Llama7BChatHelper:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path,use_fast=False,padding_side="left") 
        self.model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map="auto").eval()
        self.END_STR = torch.tensor(self.tokenizer.encode(assistant_tag)[1:]).to(
            self.device
        )
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(
                layer, self.model.lm_head, self.model.model.norm, self.tokenizer
            )

    def set_save_internal_decodings(self, value):
        for layer in self.model.model.layers:
            layer.save_internal_decodings = value

    def set_after_positions(self, pos):
        for layer in self.model.model.layers:
            layer.after_position = pos

    def prompt_to_tokens(self,instruction):
        prompt=f"{user_tag} {instruction.strip()} {assistant_tag}"
        dialog_tokens = self.tokenizer.encode(prompt)
        return torch.tensor(dialog_tokens).unsqueeze(0)

    def generate_text(self, prompt, max_new_tokens=50):
        tokens = self.prompt_to_tokens(prompt).to(self.device)
        return self.generate(tokens, max_new_tokens=max_new_tokens)

    def generate(self, tokens, max_new_tokens=50):
        instr_pos = find_instruction_end_postion(tokens[0], self.END_STR)
        self.set_after_positions(instr_pos)
        generated = self.model.generate(
            inputs=tokens, max_new_tokens=max_new_tokens, top_k=1
        )
        return self.tokenizer.batch_decode(generated)[0]

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
        data = self.get_activation_data(decoded_activations, topk)[1]
        #print(label, data)
        return data
        

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
        all_rst=[]
        for i, layer in enumerate(self.model.model.layers):
            rst={}
            #print(f"Layer {i}: Decoded intermediate outputs")
            if print_attn_mech:
                data=self.print_decoded_activations(
                    layer.attn_out_unembedded, "Attention mechanism", topk=topk
                )
                rst['Attention']=data
            if print_intermediate_res:
                data=self.print_decoded_activations(
                    layer.intermediate_resid_unembedded,
                    "Intermediate residual stream",
                    topk=topk,
                )
                rst['Intermediate']=data
            if print_mlp:
                data=self.print_decoded_activations(
                    layer.mlp_out_unembedded, "MLP output", topk=topk
                )
                rst['MLP']=data
            if print_block:
                data=self.print_decoded_activations(
                    layer.block_output_unembedded, "Block output", topk=topk
                )
                rst['Block']=data
            all_rst.append(rst)
            #print('-'*60)
        return all_rst

    def get_activation_data(self, decoded_activations, topk=10):
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, topk)
        values = values /values.sum()
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        a=list(zip(tokens, probs_percent))
        b=list(zip(tokens, values.tolist()))
        
        return list(zip(tokens, probs_percent)), list(zip(tokens, values.tolist()))


datafile='/data/xxx/datasets/advbench_pairs.json'

safe_prompts,unsafe_prompts=[],[]
with open(datafile, encoding="utf-8") as f:
    for i,line in enumerate(f):

        json_line = json.loads(line.strip())
        prompt=json_line['safe']
        safe_prompts.append(prompt)
        prompt=json_line['unsafe']
        unsafe_prompts.append(prompt)

model = Llama7BChatHelper()

if 'easyjb' in args.tasks:
    jb_file_path='llama2_easyjailbreak.json'
    with open(jb_file_path, "r", encoding="utf-8") as jb_file:
        jb_prompts = json.load(jb_file)
    attacks={'ReNellm':[],'gcg':[],'Autodan':[],'PAIR':[],'Deepinception':[]}
    for x in jb_prompts:
        
        attacks[x['attack']].append((x['goal'],x['prompt']))
    for name in ['ReNellm','gcg','Autodan','PAIR','Deepinception']:
        all_rst=[]
        for i,item in enumerate(attacks[name]):
            goal,prompt=item
            tokens=model.prompt_to_tokens(prompt)
            rst=model.decode_all_layers(tokens)
            a={'prompt':goal,'rst':rst}
            all_rst.append(a)
            print(f'{name},{i}')
            
        torch.save(all_rst,f'intermediate_easyjb_{name}_{args.model}.pt')


   


#os.makedirs('figures/intermediate',exist_ok=True)
if 'unsafe' in args.tasks:
    print('TASK: UNSAFE')
    all=[]
    for j,prompt in enumerate(unsafe_prompts):

        print('INDEX:{}, PROMPT:{}'.format(j,prompt))
        tokens=model.prompt_to_tokens(prompt)
        rst=model.decode_all_layers(tokens)
        a={'prompt':prompt,'rst':rst}
        all.append(a)
        print('*'*60)
        print('*'*60)
    torch.save(all,f'intermediate_unsafe_{args.model}.pt')

if 'safe' in args.tasks:
    print('TASK: SAFE')
    all=[]
    for j,prompt in enumerate(safe_prompts):

        print('INDEX:{}, PROMPT:{}'.format(j,prompt))
        tokens=model.prompt_to_tokens(prompt)
        rst=model.decode_all_layers(tokens)
        a={'prompt':prompt,'rst':rst}
        all.append(a)
        print('*'*60)
        print('*'*60)
    torch.save(all,f'intermediate_safe_{args.model}.pt')


layers={}
words={}

if args.tasks=='easyjb':
    datapath='/data/xxx/datasets/jailbreak_dataset'
    attacks=['ReNellm','gcg','Autodan','PAIR','Deepinception']
    #attacks=['PAIR','AutoDAN','DeepInception']
    for attack in attacks:
        
        os.makedirs(f'figures/intermediate/{args.model}/easyjb/{attack}/{args.part}',exist_ok=True)
        intermediate_all=torch.load(f'intermediate_easyjb_{attack}_{args.model}.pt')
        for j,intermediate in enumerate(intermediate_all):
            print(f'attack:{attack}, index:{j}')
            prompt=intermediate['prompt']
            data=intermediate['rst'] 
            for i,layer_data in enumerate(data):

                block_data=layer_data[args.part]
                vocabs=[x[0] for x in block_data]
                probs=[x[1] for x in block_data]
                layers[f'Layer {i+1}']=probs
                words[f'Layer {i+1}']=vocabs
            # Creating DataFrame for probabilities and words
            prob_df = pd.DataFrame(layers).T  # Probabilities DataFrame
            words_df = pd.DataFrame(words).T  # Words DataFrame

            # Set up the matplotlib figure
            #plt.figure(figsize=(18,10))
            plt.figure(figsize=(15,8))
            # Create a heatmap
            ax = sns.heatmap(prob_df, annot=words_df, fmt='', cmap='Blues', cbar=True)
            # Customize further with matplotlib
            plt.title(f'Prompt: {prompt}')
            plt.xlabel('Top 10 Predicted Words for Each Layer')
            plt.ylabel('Layers')
            plt.yticks(rotation=0)  # Ensure layer labels are horizontal
            plt.xticks(rotation=45)  # Tilt word labels for better visibility
            #plt.tight_layout()
            try:
                plt.tight_layout()
                plt.savefig(f'figures/intermediate/{args.model}/easyjb/{attack}/{args.part}/prompt_{j}_onlylast.png', bbox_inches='tight')
                #plt.savefig(f'figures/intermediate/{args.model}/easyjb/{attack}/{args.part}/prompt_{j}.png')
            except ValueError:
                print(f'SAVE FAILE attack:{attack}, index:{j}')


if args.tasks in ['safe','unsafe']:
    os.makedirs(f'figures/intermediate/{args.model}/{args.tasks}/{args.part}',exist_ok=True)
    intermediate_all=torch.load(f'intermediate_{args.tasks}_{args.model}.pt')
    for j,intermediate in enumerate(intermediate_all):
        if j>50:break
        prompt=intermediate['prompt']
        data=intermediate['rst'] 
        for i,layer_data in enumerate(data):
            if i<15: continue
            block_data=layer_data[args.part]
            vocabs=[x[0] for x in block_data][:5]
            probs=[x[1] for x in block_data][:5]

            layers[f'{i+1}']=probs
            words[f'{i+1}']=vocabs
        # Creating DataFrame for probabilities and words
        prob_df = pd.DataFrame(layers).T  # Probabilities DataFrame
        words_df = pd.DataFrame(words).T  # Words DataFrame

        # Set up the matplotlib figure
        plt.figure(figsize=(7,6))

        # Create a heatmap
        #ax = sns.heatmap(prob_df, annot=words_df, fmt='', cmap='Blues', cbar=True, annot_kws={"size": 12})
        ax = sns.heatmap(prob_df, annot=words_df, fmt='', cmap='Blues', annot_kws={"size": 15})#, cbar_kws={'shrink': 0.8,'pad': 0.01})

        # Customize further with matplotlib
        #plt.title(f'Prompt: {prompt}')
        plt.xlabel(f'Top 5 Predicted Words in Each Layer', fontsize=18)
        plt.ylabel('Layers', fontsize=18)
        plt.yticks(rotation=0,fontsize=12)  # Ensure layer labels are horizontal
        plt.xticks(rotation=45,fontsize=12)  # Tilt word labels for better visibility
        plt.tight_layout()
            #plt.savefig(f'figures/intermediate/{args.model}/easyjb/{attack}/{args.part}/prompt_{j}.png', bbox_inches='tight')
        plt.savefig(f'figures/intermediate/{args.model}/{args.tasks}/{args.part}/{args.model}_{args.tasks}_prompt_{j}_onlylast.png', bbox_inches='tight')

    
        
  