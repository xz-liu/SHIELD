import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from models import *
from parse import *


def load_or_train_ngram_model(tokenizer, device, args, dataset: Dataset):
    ngram_model_path = get_ngram_model_path(dataset.name)
    os.system('rm -rf ' + ngram_model_path)
    ngram_model = train_ngram_model(tokenizer, device, dataset, args.ngram_context_size, args.ngram_epoch)
    torch.save(ngram_model, ngram_model_path)
    ngram_model.context_size = args.ngram_context_size
    return ngram_model

def collate_fn(batch):
    return [example["data"].content for example in batch]


def calculate_perplexity(model, tokenizer, dataset, logits_processor=None, batch_size=1):
    total_loss = 0
    total_length = 0
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            encodings = tokenizer(batch, return_tensors="pt", padding="max_length", truncation=True, max_length=50)
            input_ids = encodings['input_ids']
            attention_mask = encodings['attention_mask']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            if logits_processor:
                for i in range(logits.size(1)):   
                    logits[:, i, :] = logits_processor(input_ids[:, :i], logits[:, i, :])
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            total_loss += loss.item() * input_ids.size(1) * input_ids.size(0)  
            total_length += input_ids.size(1) * input_ids.size(0)  

    avg_loss = total_loss / total_length
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()


if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    prompt_type = "none"
    jailbreak = "no"
    jailbreak_num = -1
    
    test_dataset = get_dataset("bsc", prompt_type, jailbreak, jailbreak_num)
    
    ngram_dataset = get_dataset("bsnc", prompt_type, jailbreak, jailbreak_num)
    diff_ngram = load_or_train_ngram_model(tokenizer, "cuda", args, ngram_dataset)
    diff_lp=LogitsProcessorList([NGramLogitsProcessor(ngram_model=diff_ngram)])
    
    same_ngram = load_or_train_ngram_model(tokenizer, "cuda", args, test_dataset)
    same_lp=LogitsProcessorList([NGramLogitsProcessor(ngram_model=same_ngram)])
    
    no_lp = LogitsProcessorList([])
    
    ppl_lp = calculate_perplexity(model, tokenizer, test_dataset, diff_lp, batch_size=35)
    ppl_no_lp = calculate_perplexity(model, tokenizer, test_dataset, no_lp, batch_size=35)
    ppl_same_ds = calculate_perplexity(model, tokenizer, test_dataset, same_lp, batch_size=35)
    
    print(f"Perplexity      with      LP: {ppl_lp}")
    print(f"Perplexity   with   no    LP: {ppl_no_lp}")
    print(f"Perplexity with same dataset: {ppl_same_ds}")
