def select_model_path(model_name):
    if model_name=='llama3-8b':
        model_path="/data3/xxx/huggingface/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
        base_model_path="/data3/xxx/huggingface/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
        
        user_tag=""
        assistant_tag=""
    elif model_name=='llama2-7b-c':
        user_tag = "[INST]"
        assistant_tag = "[/INST]"
        base_model_path = "/data/xxx/models/Llama-2-7b-chat-hf"
    elif model_name=='llama2-13b-c':
        user_tag = "[INST]"
        assistant_tag = "[/INST]"
        base_model_path = "/data/xxx/models/Llama-2-13b-chat-hf"
    elif model_name=='vicuna1.5_7b':
        user_tag = "A chat between a curious user and an artificial intelligence assistant. USER:"
        assistant_tag = "ASSISTANT:"
        base_model_path = "/data/xxx/models/vicuna1.5_7b"
    elif model_name=='vicuna1.5_13b':
        user_tag = "A chat between a curious user and an artificial intelligence assistant. USER:"
        assistant_tag = "ASSISTANT:"
        base_model_path = "/data/xxx/models/vicuna1.5_13b"

    if model_name=='llama2-7b-c':
        model_path = "/data/xxx/models/Llama-2-7b-chat-hf"
    elif model_name=='llama2-13b-c':
        model_path = "/data/xxx/models/Llama-2-13b-chat-hf"
    elif model_name=='llama2-base':
        model_path='/data/xxx/models/Llama-2-7b-hf'
    elif model_name=='vicuna1.5_7b':
        model_path = "/data/xxx/models/vicuna1.5_7b"
    elif model_name=='vicuna1.5_13b':
        model_path = "/data/xxx/models/vicuna1.5_13b"
    elif model_name=='llama2-adv-re':
        model_path='/data/xxx/models/llama2-adversarial-re' 
    elif model_name=='vicuna-adv-re':
        model_path='/data/xxx/models/vicuna-adversarial-re'
    elif model_name=='llama2-rank-dif':
        model_path='/data/xxx/alignment-attribution-code-main/save/llama2-7b-c_low-rank-diff_alpaca_cleaned_no_safety_align_3000_4000'
    elif model_name=='vicuna-rank-dif':
        model_path='/data/xxx/alignment-attribution-code-main/save/vicuna1.5_7b_low-rank-diff_alpaca_cleaned_no_safety_align_3000_4000'
    elif model_name=='llama2-uncensor':
        model_path='/data/xxx/models/llama2-7b-chat-uncensored'
        user_tag='### HUMAN:'
        assistant_tag='### RESPONSE:'
    elif model_name=='llama2-finetuned':
        model_path='/data/xxx/models/llama2-finetuned'
    
    if model_name[:7]=='mistral':
        #model_path='/data/xxx/models/mistral-7b-v0.1'
        base_model_path='/data/xxx/models/mistral-7b-v0.1'
        user_tag = "[INST]"
        assistant_tag = "[/INST]"
    if model_name=='mistral_v0.1':
        model_path='/data/xxx/models/mistral-7b-instruct-v0.1'
    elif model_name=='mistral_v0.2':
        model_path='/data/xxx/models/mistral-7b-instruct-v0.2'
    elif model_name=='mistral_v0.3':
        model_path='/data/xxx/models/mistral-7b-instruct-v0.3'
    return user_tag,assistant_tag,base_model_path,model_path

from typing import List
from torch import Tensor
from jaxtyping import Int, Float
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer



def instruction_to_prompt(
    model_name:str,
    instruction: str,
    system_prompt: str=None,
    model_output: str=None,
    append_space: bool=True,
) -> str:
    """
    Converts an instruction to a prompt string structured for Llama2-chat.
    See details of Llama2-chat prompt structure here: here https://huggingface.co/blog/llama2#how-to-prompt-llama-2
    """
    if model_name[:6]=='llama2':
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    elif model_name[:6]=='llama3':
        B_INST, E_INST = "", ""
        
    elif model_name[:6]=='vicuna':
        B_INST, E_INST = "USER:", "ASSISTANT:"
    elif model_name[:7]=='mistral':
        B_INST, E_INST = "[INST]", "[/INST]"
    if system_prompt is not None:
            dialog_content = B_SYS + system_prompt.strip() + E_SYS + instruction.strip()
    else:
            dialog_content = instruction.strip()

    if model_output is not None:
        prompt = f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()}"
    else:
        prompt = f"{B_INST} {dialog_content.strip()} {E_INST}"

    if append_space:
        prompt = prompt + " "

    return prompt

def tokenize_instructions(
        model_name:str,
    tokenizer: AutoTokenizer,
    instructions: List[str],
    padding_length: int=None,
    system_prompt: str=None,
    model_outputs: List[str]=None,
) -> Int[Tensor, "batch seq_len"]:
    if model_outputs is not None:
        assert(len(instructions) == len(model_outputs))
        prompts = [
            instruction_to_prompt(model_name,instruction, system_prompt, model_output, append_space=False)
            for (instruction, model_output) in zip(instructions, model_outputs)
        ]
    else:
        prompts = [
            instruction_to_prompt(model_name,instruction, system_prompt, model_output=None, append_space=True)
            for instruction in instructions
        ]
    # def template(prompt):
    #     chat = [
    #                 {"role": "user", "content": prompt},
    #                 ]
    #     prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    #     return prompt 
    # prompts=[template(x) for x in instructions]
    instructions_toks = tokenizer(
        prompts,
        padding="max_length" if padding_length is not None else True,
        max_length=padding_length,
        truncation=False,
        return_tensors="pt"
    ).input_ids

    return instructions_toks

def generate_from_instructions(
    tl_model: HookedTransformer,
    tokenizer: AutoTokenizer,
    instructions: List[str],
    max_new_tokens: int=64,
    temperature: float=0.0,
):
    instructions_toks = tokenize_instructions(tokenizer, instructions)

    output_ids = tl_model.generate(instructions_toks, max_new_tokens=max_new_tokens, temperature=temperature)
    for answer_idx, answer in enumerate(tokenizer.batch_decode(output_ids)):
        print(f"\nGeneration #{answer_idx}:\n\t{repr(answer)}")