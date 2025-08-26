
from utils import *
# from torch.cuda.amp import autocast as autocast
# with autocast()
# jb dataset是/data/xxx/datasets/jailbreak_dataset ！！！！！！！！！
import os
import torch,argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llama2-7b-c")
parser.add_argument("--dataset", type=str, default='advbench_pair')
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--token_idx", type=int, default=-1)
parser.add_argument("--probe", type=str, default='cluster')
parser.add_argument("--jb", action='store_const', const=1, default=0)
parser.add_argument("--easyjb", action='store_const', const=1, default=1)
parser.add_argument("--scatter", action='store_const', const=1, default=1)
args = parser.parse_args()
setup_seed(0)
save_root_dir=f'save/{args.model}/{args.probe}'
os.makedirs(save_root_dir,exist_ok=True)
re_dir=f'/data/xxx/ExJB/save_re/{args.model}/{args.dataset}'
acc_rst={}
logit_rst={}
layer_num=32 if '7b' in args.model else 40
for l in range(layer_num):
      
    x=torch.load(os.path.join(re_dir,f'safe_layer{l}.pt'))
    x_tmp=torch.load(os.path.join(re_dir,f'unsafe_layer{l}.pt'))
    
    num=len(x)
    print('num:',num)
    train_num,test_num=65,65

    train_data_1 = x[:train_num*3].to(torch.float32)  # 取前300行
    train_label_1=torch.ones([train_num*3])
    test_data_1 = x[-1*test_num:].to(torch.float32).to('cuda')  
    test_label_1=torch.ones([test_num]).to('cuda')
    
    train_data_2 = x_tmp[:train_num*3].to(torch.float32)   # 取前300行
    train_label_2=torch.zeros([train_num*3])
    test_data_2 = x_tmp[-1*test_num:].to(torch.float32).to('cuda')
    test_label_2=torch.zeros([test_num]).to('cuda')
    
    train_data = torch.cat((train_data_1,train_data_2), dim=0).to('cuda')
    train_label=torch.cat((train_label_1,train_label_2),dim=0).to('cuda')

    if args.probe=='linear' or args.probe=='linear2' :
        ProbeClass=LRProbe
        probe = ProbeClass.from_data(train_data, train_label, device='cuda')
    elif args.probe=='cluster':
        probe = MeanCenterProbe(device='cuda')
        probe.get_direction(train_data,train_label)
    elif args.probe=='pca':
        probe=PCAProbe()
        probe.get_direction(train_data,train_label)
        
    

    pred,logit=probe.pred(test_data_1)
    acc_org=( pred== test_label_1).float().mean().item()
    if l==0:acc_rst['safe']=[round(acc_org,5)]
    else: acc_rst['safe'].append(round(acc_org,5))
    print(f'layer:{l},acc_safe:{acc_org}')

    pred,logit=probe.pred(test_data_2)
    acc_org=( pred== test_label_2).float().mean().item()
    if l==0:acc_rst['unsafe']=[round(acc_org,5)]
    else: acc_rst['unsafe'].append(round(acc_org,5))
    print(f'layer:{l},acc_unsafe:{acc_org}')
  
    
    #torch.save(probe,os.path.join(save_root_dir,'probe_transfer.pt'))
    direction=probe.direction()
    torch.save(direction,os.path.join(save_root_dir,f'probe_layer{l}.pt'))

    if args.easyjb:
        attack_acc=[]
        attacks=['CodeChameleon','Autodan',"gcg","ReNellm",'PAIR','Deepinception','GPTFuzz']
        if args.scatter==1:
            print('SCATTER')
            for j,attack in enumerate(attacks):
                attack_dir=f'/data/xxx/ExJB/save_re/{args.model}/easyjb-0712/{attack}'
                #if os.path.exists(os.path.join(attack_dir,f'logits.pt')): continue
                
                re=torch.load(os.path.join(attack_dir,f'jb_layer{l}.pt')).to(torch.float32).to('cuda')
                pred,logit=probe.pred(re)
                print(f'{attack} logit:',logit)
                label=torch.ones(re.shape[0]).to('cuda')
                acc=( pred==label).float().mean().item()
                attack_acc.append(acc)
                if l==0:
                    acc_rst[attack]=[round(acc,5)]
                    logit_rst[attack]=[logit.tolist()]
                else: 
                    acc_rst[attack].append(round(acc,5))
                    logit_rst[attack].append(logit.tolist())
                    pass
            for k in range(len(attack_acc)):
                print(f'attack:{attacks[k]}, acc:{attack_acc[k]}')
        else:
            attack_acc=[]
            for j,attack in enumerate(attacks):
                attack_dir=f'/data/xxx/ExJB/save_re/{args.model}/easyjb/{attack}'
                re=torch.load(os.path.join(attack_dir,f'jb_layer{l}.pt')).to(torch.float32).to('cuda')
                pred,logits=probe.pred(re)
                label=torch.ones(re.shape[0]).to('cuda')
                acc=( pred==label).float().mean().item()
                attack_acc.append(acc)
                if l==0:acc_rst[attack]=[round(acc,5)]
                else: acc_rst[attack].append(round(acc,5))
            for k in range(len(attack_acc)):
                print(f'attack:{attacks[k]}, acc:{attack_acc[k]}')
           

    
for x in acc_rst:
    print(f'acc_list[\'{args.probe}\'][\'{x}\']={acc_rst[x]}')

if args.easyjb and args.scatter:
    attacks=['CodeChameleon','Autodan',"gcg","ReNellm",'PAIR','Deepinception','GPTFuzz']
    for j,attack in enumerate(attacks):
        attack_dir=f'/data/xxx/ExJB/save_re/{args.model}/easyjb-0712/{attack}'
        #if os.path.exists(os.path.join(attack_dir,f'logits.pt')): continue
        torch.save(logit_rst[attack],os.path.join(save_root_dir,f'{attack}_logits.pt'))
        print(f'{attack} save done')
