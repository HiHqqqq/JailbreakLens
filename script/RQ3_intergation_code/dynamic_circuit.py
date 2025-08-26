import argparse,os,torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llama2-7b-c")
parser.add_argument("--pos_token", type=str, default="Sure")
parser.add_argument("--neg_token", type=str, default="Sorry")
args = parser.parse_args()
save_path=f'/data/xxx/Activation_Patching/dir_dif/save_fig/{args.model}/dynamics_2'
os.makedirs(save_path,exist_ok=True)
for name in ['safe','unsafe','ReNellm','gcg','Autodan','PAIR','Deepinception','GPTFuzz','CodeChameleon']:
    for j in range(30):
        refusal_file=f'/data/xxx/Activation_Patching/dir_dif/save_cache/{args.model}_{name}_prompt{j}_refusal-sig.pt'
        agreement_file=f'/data/xxx/Activation_Patching/dir_dif/save_cache/{args.model}_{name}_prompt{j}_agreement-sig.pt'
        print('exist:',os.path.exists(refusal_file))
        if os.path.exists(refusal_file)==True:
          
            refusal_sig=torch.load(refusal_file)
            agreement_sig=torch.load(agreement_file)
            agreement_sig=[x*-1 for x in agreement_sig]
            fig, ax = plt.subplots(figsize=(10,4))
            x=[k for k in range(len(refusal_sig))]
            ax.plot(x, refusal_sig, marker='o', linestyle='-',markersize=1,linewidth=2,color="lightcoral",  label='Resufal')
            
            ax.plot(x,agreement_sig, marker='o', linestyle='-',markersize=1,linewidth=2,color="lightblue",  label='Affirmation')
   
            #ax.set_title(f'Jailbreak-{name}_IDX-{j}')
            ax.set_xlabel('Token position',fontsize=28)
            ax.set_ylabel('Activation',fontsize=28)
            ax.tick_params(axis='x', labelsize=18)  # x轴刻度标签字体大小
            ax.tick_params(axis='y', labelsize=18)  # y轴刻度标签字体大小
            #ax.set_xlim(0,3000)
            plt.legend(loc='upper right',fontsize=20,labelspacing=0.2)
            plt.tight_layout()

            plt.savefig(os.path.join(save_path,f'{args.model}_{name}_tokens_{j}_1030.png'),bbox_inches='tight')
            

