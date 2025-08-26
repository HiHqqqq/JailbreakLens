import gc
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
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
parser.add_argument("--model", type=str, default="llama2-7b-c")
parser.add_argument("--pos_token", type=str, default="Sure")
parser.add_argument("--neg_token", type=str, default="Sorry")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

user_tag,assistant_tag,base_model_path,model_path=select_model_path(args.model)
print('model_path:',model_path)



def get_refusal_dir():
    a=torch.load(f'/data/xxx/Activation_Patching/dir_dif/save_dir/{args.model}_isharm.pt')
    return a
    a=tl_model.W_U[:, refuse_token]
    b=tl_model.W_U[:, answer_token]
    c=tl_model.W_U[:, refuse_token] - tl_model.W_U[:, answer_token]
    #os.makedirs(f'./save_dir',exist_ok=True)
    past_dir=torch.load(f'./save_dir/{args.model}_isharm.pt')
    dif=past_dir-c
    print(f'Is dir same: {dif}')
    torch.save(c,f'./save_dir/{args.model}_isharm.pt')
    return tl_model.W_U[:, refuse_token] - tl_model.W_U[:, answer_token]

color_map = {
    "harmful":'red', 
    "harmless": 'blue', 
    'gcg': 'green',
    'ReNellm': 'purple', 
    'PAIR': 'yellow', 
    'Deepinception': 'pink',
    'Autodan': 'skyblue', 
    'GPTFuzz': 'grey', 
    'CodeChameleon':'orange'
}



layer_num=32 if '7b' in args.model else 40
added_labels = set()
fig_resid = go.Figure()
fig_resid_decomp = go.Figure()
for dataset_label in ["harmful", "harmless"]:#,"gcg","ReNellm",'PAIR','Deepinception','Autodan','GPTFuzz','CodeChameleon']:
    print('dataset:',dataset_label)
    #accumulated_resid=torch.load(f'/data/xxx/Activation_Patching/dir_dif/save_pt/{args.model}/{args.model}_{dataset_label}_accumulated_resid.pt')
    resid_decomp=torch.load(f'/data/xxx/Activation_Patching/dir_dif/save_pt/{args.model}/{args.model}_{dataset_label}_resid_decomp.pt')
    # refusal_scores_by_resid = einops.einsum(
    #     accumulated_resid,
    #     get_refusal_dir(),
    #     'components batch d_model, d_model -> components batch'
    # )

    refusal_scores_by_resid_decomp = einops.einsum(
        resid_decomp,
        get_refusal_dir(),
        'components batch d_model, d_model -> components batch'
    ) #size
    refusal_scores_by_resid_decomp=torch.mean(refusal_scores_by_resid_decomp, dim=1)
    refusal_scores_by_resid_decomp=refusal_scores_by_resid_decomp[2::2] #只保留mlp部分
    fig_resid_decomp.add_trace(
        go.Scatter(
            x=[k for k in range(layer_num)],
            y=refusal_scores_by_resid_decomp.cpu().numpy(),
            #name=objects[ex],
            mode='lines+markers',
            line=dict(
            color=color_map[dataset_label],
            width=1,  # 调整线条宽度为3
            ),
            marker=dict(
                color=color_map[dataset_label],
                size=5,       # 调整标记点大小为10
                symbol='circle',
            ),
            opacity=0.5, 

            name=dataset_label if dataset_label not in added_labels else None,
            showlegend=dataset_label not in added_labels
        )
    )
    added_labels.add(dataset_label)
fig_resid_decomp.update_layout(
#title='Refusal score attribution, decomposed resid of pos=-1',
xaxis=dict(title='MLP Layer',
tickmode='array',
tickvals=[i for i in range(0, layer_num, 2)],  
ticktext=[str(i) for i in range(0, layer_num, 2)],  
),
yaxis=dict(title='Refusal score'),
height=600, width=900
)
pio.write_image(fig_resid_decomp,f'/data/xxx/Activation_Patching/dir_dif/save_fig/{args.model}/safe-unsafe_resid_decomp_1011.png')

