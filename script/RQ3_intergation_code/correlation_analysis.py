
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import torch,argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llama2-13b-c")

args = parser.parse_args()

os.makedirs(f'save_fig/peason/{args.model}_2',exist_ok=True)
re_logits=torch.load(f'/data/xxx/ExJB/Probing_V2/{args.model}_prob_logits.pt')
a_sig,r_sig,a_baseline,r_baseline=torch.load(f'/data/xxx/ExJB/Probing_V2/{args.model}_sig.pt')

for layer in range(40 if '13b' in args.model else 32):
    print('-'*80)
    x=[]
    y=[]
    
    attacks=['GPTFuzz','ReNellm','gcg','Autodan','PAIR','Deepinception','CodeChameleon']
    if args.model=='llama3-8b':
        attacks=['gcg','pair','deepinception','gpfuzz','AutoDAN']
    for name in attacks:
        print('-'*80)
        print(name)
        if name not in r_sig:continue
        v=[( -a_sig[name][i] - -1*a_baseline+  (r_baseline-r_sig[name][i])) for i in range(len(r_sig[name]))]
        #print(f'a_baseline:{-1*a_baseline}, a_sig:{-a_sig[name][0]}, r_baseline:{r_baseline}, r_sig:{r_sig[name][0]}')
        if len(v)>20:v=v[:20]
        print('v',len(v))
        x.extend(v)
        y_cur=re_logits[name][layer]
        y_cur=[1-j for j in y_cur]
        print('y_cur',len(y_cur))
        if len(y_cur)>20:y_cur=y_cur[:20]
        y.extend(y_cur)
    
    y=[k.cpu().item() for k in y]
    # if args.model=='llama2-13b-c':
    #     x=x[:-1]
    print('Layer:',layer)
    print('x:')
    print(len(x),x)
    print('y:')
    print(len(y),y)

    # print(len(x),len(y))
    # y=y.cpu()
    corr_coeff, _ = pearsonr(x, y)
    print(f"Pearson correlation coefficient: {corr_coeff:.2f}")
    

    slope, intercept = np.polyfit(x, y, 1)  
    print('a=',slope)
    print('b=',intercept)
    print('n=',140-len(y))
    print('x_range:',min(x),max(x))
    print('y_range:',min(y),max(y))


    plt.figure(figsize=(6,4))
    plt.scatter(x, y, label='Data Points', alpha=0.7)


    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, color='red', label=f'Fit Line (y={slope:.2f}x+{intercept:.2f})')


    formal_name={
        'llama2-7b-c':'Llama2-7b',
        'vicuna1.5_7b':'Vicuna1.5-7b',
        'llama3-8b':'Llama3-8b',
        'llama2-13b-c':'Llama2-13b',
        'vicuna1.5_13b':'Vicuna1.5-13b',
    }
    plt.title(f'{formal_name[args.model]}, Pearson Correlation: {corr_coeff:.2f}',fontsize=16)
    plt.xlabel('Key-component Activation Shift',fontsize=14)
    plt.ylabel('Representation Toxicity',fontsize=14)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'save_fig/peason/{args.model}_2/a+r_re_l{layer}.pdf')
    plt.close()
