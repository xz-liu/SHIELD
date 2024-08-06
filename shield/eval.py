from dataset import get_dataset, get_save_path, unify_space, Dataset, InputExample, ModelOutput, Completion
# from models import *
from rouge_score import rouge_scorer
from utils import check_refusal


#
# @torch.no_grad()
# def eval_ppl(ori_model, ngram_model, tokenizer, device, max_dataset_num=100000000):
#     # source: https://huggingface.co/docs/transformers/perplexity
#     @torch.no_grad()
#     def eval_perplexity(cur_model, tokenizer, device, batch_size=8, max_ppl_num=100000000):
#         max_length = cur_model.config.max_position_embeddings
#         stride = 512
#
#         dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
#         nlls = []
#
#         for start_idx in tqdm(range(0, min(max_ppl_num, len(dataset)), batch_size)):
#             end_idx = min(start_idx + batch_size, len(dataset))
#             batch_text = dataset["text"][start_idx:end_idx]
#
#             encodings = tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
#             input_ids = encodings.input_ids.to(device)
#             seq_len = input_ids.size(1)
#             prev_end_loc = 0
#
#             for begin_loc in range(0, seq_len, stride):
#                 end_loc = min(begin_loc + max_length, seq_len)
#                 trg_len = end_loc - prev_end_loc
#
#                 inputs = input_ids[:, begin_loc:end_loc]
#                 targets = inputs.clone()
#                 targets[:, :-trg_len] = -100
#
#                 outputs = cur_model(inputs, labels=targets)
#                 neg_log_likelihood = outputs.loss
#                 nlls.append(neg_log_likelihood)
#
#                 prev_end_loc = end_loc
#                 if end_loc == seq_len:
#                     break
#
#         ppl = torch.exp(torch.stack(nlls).mean())
#         return ppl
#
#     print("-" * 10, "  Perplexity  ", "-" * 10)
#     print(f"ppl without lp: {eval_perplexity(ori_model, tokenizer, device, max_ppl_num=max_dataset_num).item()}")
#     del ori_model
#     torch.cuda.empty_cache()
#     torch.cuda.synchronize()
#
#     new_model = NewLlama.from_pretrained(args.hf_model_id, attn_implementation="flash_attention_2",
#                                          torch_dtype=torch.bfloat16).to(device)
#     logit_processor = NGramLogitsProcessor(ngram_model)
#     new_model.set_logit_processor(logit_processor)
#     print('Ngram hits time:', logit_processor.ngram_hits_count)
#
#     print(f"ppl  with   lp: {eval_perplexity(new_model, tokenizer, device, max_ppl_num=max_dataset_num).item()}")


def LCSubStr(arr1, arr2):
    len1, len2 = len(arr1), len(arr2)
    dp = [[0 for k in range(len2 + 1)] for l in range(len1 + 1)]
    result = 0
    for i in range(len1 + 1):
        for j in range(len2 + 1):
            if (i == 0 or j == 0):
                dp[i][j] = 0
            elif (arr1[i - 1] == arr2[j - 1]):
                dp[i][j] = dp[i - 1][j - 1] + 1
                result = max(result, dp[i][j])
            else:
                dp[i][j] = 0
    return result

def eval_lcs_with_data(prompt_type, outputs_wo_lp, outputs_w_lp):
    scorer = rouge_scorer.RougeScorer(['rougeL', ], use_stemmer=True)
    wo_lcs_list = []
    w_lcs_list = []
    
    wo_rouge_list = []
    w_rouge_list = []

    wo_refusal_list = []
    w_refusal_list = []

    w_prompt_len_list = []
    w_gen_len_list = []

    wo_prompt_len_list = []
    wo_gen_len_list = []


    for wo, w in zip(outputs_wo_lp, outputs_w_lp):
        if prompt_type == "a":
            ori = wo.content.split(" ")[50:1000]
        elif prompt_type == "b":
            ori = wo.content.split(" ")[:1000]
        else: # "c"
            raise Exception("unsupported type")
        ori_str = " ".join(ori)

        for completion in wo.completions:
            gen = unify_space(completion.response)

            lcs = LCSubStr(ori, gen.split(" "))
            wo_lcs_list.append(lcs)
            completion.lcs = lcs

            rouge = scorer.score(ori_str, gen)['rougeL'].fmeasure
            wo_rouge_list.append(rouge)
            completion.rouge = rouge

            refusal = int(check_refusal(gen))
            wo_refusal_list.append(refusal)
            completion.refusal = refusal
            wo_prompt_len_list.append(len(completion.prompt.split(" ")))
            wo_gen_len_list.append(len(completion.response.split(" ")))


        for completion in w.completions:
            gen = unify_space(completion.response)

            lcs = LCSubStr(ori, gen.split(" "))
            w_lcs_list.append(lcs)
            completion.lcs = lcs

            rouge = scorer.score(ori_str, gen)['rougeL'].fmeasure
            w_rouge_list.append(rouge)
            completion.rouge = rouge

            refusal = int(check_refusal(gen))
            w_refusal_list.append(refusal)

            completion.refusal = refusal

            w_prompt_len_list.append(len(completion.prompt.split(" ")))
            w_gen_len_list.append(len(completion.response.split(" ")))


    print("-" * 10, "  LCS  ", "-" * 10)

    wo_lcs_str = " ".join([str(num) for num in wo_lcs_list])
    print(f"without defense list: {wo_lcs_str}")
    print(f"without defense  avg: {(sum(wo_lcs_list) / len(wo_lcs_list)):.3f}")
    print(f"without defense  max: {max(wo_lcs_list)}")
    print()

    w_lcs_str = " ".join([str(num) for num in w_lcs_list])
    print(f"with   defense  list: {w_lcs_str}")
    print(f"with   defense   avg: {(sum(w_lcs_list) / len(w_lcs_list)):.3f}")
    print(f"with   defense   max: {max(w_lcs_list)}")
    print()

    print("-" * 10, "  ROUGE-L  ", "-" * 10)

    wo_rouge_str = " ".join([f"{num:.3f}" for num in wo_rouge_list])
    print(f"without defense list: {wo_rouge_str}")
    print(f"without defense  avg: {(sum(wo_rouge_list) / len(wo_rouge_list)):.3f}")
    print(f"without defense  max: {max(wo_rouge_list):.3f}")
    print()

    w_rouge_str = " ".join([f"{num:.3f}" for num in w_rouge_list])
    print(f"with   defense  list: {w_rouge_str}")
    print(f"with   defense   avg: {(sum(w_rouge_list) / len(w_rouge_list)):.3f}")
    print(f"with   defense   max: {max(w_rouge_list):.3f}")
    print()

    print("-" * 10, "  Refusal  ", "-" * 10)
    print(f"without defense  avg: {(sum(wo_refusal_list) / len(wo_refusal_list)):.3f}")
    print(f"with   defense  avg: {(sum(w_refusal_list) / len(w_refusal_list)):.3f}")

    print("-" * 10, "  Prompt Length  ", "-" * 10)
    print(f"without defense  avg: {(sum(wo_prompt_len_list) / len(wo_prompt_len_list)):.3f}")
    print(f"with   defense  avg: {(sum(w_prompt_len_list) / len(w_prompt_len_list)):.3f}")

    print("-" * 10, "  Gen Length  ", "-" * 10)
    print(f"without defense  avg: {(sum(wo_gen_len_list) / len(wo_gen_len_list)):.3f}")
    print(f"with   defense  avg: {(sum(w_gen_len_list) / len(w_gen_len_list)):.3f}")


def eval_contain_with_data(outputs_wo_lp, outputs_w_lp):
    wo_contain_list = []
    w_contain_list = []
    for wo, w in zip(outputs_wo_lp, outputs_w_lp):
        # Use lowercase "by" for segmentation.
        # e.g., A Christmas Carol by Charles Dickens (d. 1870) in 1843
        title = wo.title.split(" by")[0].lower()
        
        for completion in wo.completions:
            gen = unify_space(completion.response).lower()
            wo_contain_list.append(title in gen)
        
        for completion in w.completions:
            gen = unify_space(completion.response).lower()
            w_contain_list.append(title in gen)

    print("-" * 10, "  Contain  ", "-" * 10)

    wo_contain_str = " ".join([str(int(num)) for num in wo_contain_list])
    print(f"without defense  list: {wo_contain_str}")
    print(f"without defense count: {sum(wo_contain_list)}")
    print(f"without defense  rate: {(sum(wo_contain_list) / len(wo_contain_list)):.3f}")
    print()

    w_contain_str = " ".join([str(int(num)) for num in w_contain_list])
    print(f"with   defense   list: {w_contain_str}")
    print(f"with   defense  count: {sum(w_contain_list)}")
    print(f"with   defense   rate: {(sum(w_contain_list) / len(w_contain_list)):.3f}")
    print()
