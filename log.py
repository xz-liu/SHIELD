import json, subprocess, os, glob
from datetime import datetime
from typing import List, Tuple, Dict
from argparse import Namespace
from dataset import ModelOutput
from dataset import Completion
from utils import check_refusal

LOD_DIR = "./gen_logs"


def _dump_to_json(model_outputs: List[ModelOutput], file_path: str):
    data = [{
        'model_id': model_output.model_id,
        'title': model_output.title,
        'content': model_output.content,
        'completions': [{'prompt': c.prompt, 'response': c.response,
                         'lcs': str(c.lcs), 'rouge': str(c.rouge), 'jailbreak_id': str(c.jailbreak_id)}
                        for c in model_output.completions]
    } for model_output in model_outputs]

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def _json_to_model_output(data: List[Dict]) -> List[ModelOutput]:
    return [ModelOutput(model_id=d['model_id'], title=d['title'], content=d['content'],
                        completions=[Completion(prompt=c['prompt'], response=c['response'],
                                                lcs=int(c['lcs']), rouge=float(c['rouge']),
                                                jailbreak_id=int(c['jailbreak_id']))
                                     for c in d['completions']])
            for d in data]


def dump_gen(args, outputs_wo_lp: List[ModelOutput], outputs_w_lp: List[ModelOutput]):
    if not args.dump_gen:
        return

    # make dir
    subdir = f"{args.dataset}"
    log_dir = os.path.join(LOD_DIR, subdir)
    version = 1 + max([int(os.path.split(x)[-1].split("_")[-1])
                       for x in glob.glob(f"{log_dir}/*")], default=0)
    log_dir = f"{log_dir}/v_{version:03d}"
    os.makedirs(log_dir, exist_ok=False)
    print(f"log dir: {log_dir}")

    # dump content
    _dump_to_json(outputs_wo_lp, log_dir + "/outputs_wo_lp.json")
    _dump_to_json(outputs_w_lp, log_dir + "/outputs_w_lp.json")

    # dump args and git commit id
    args_dict = vars(args)
    try:
        commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        args_dict['git_commit_id'] = commit_id
        args_dict['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    except subprocess.CalledProcessError:
        pass
    with open(log_dir + "/args.json", 'w', encoding='utf-8') as f:
        json.dump(args_dict, f, ensure_ascii=False, indent=4)


def read_gen(subdir: str, version: int) -> \
        Tuple[Namespace, List[ModelOutput], List[ModelOutput]]:
    log_dir = os.path.join(LOD_DIR, subdir, f"v_{version:03d}")
    with open(log_dir + "/args.json", 'r', encoding='utf-8') as f:
        args_dict = json.load(f)
    with open(log_dir + "/outputs_wo_lp.json", 'r', encoding='utf-8') as f:
        outputs_wo_lp = json.load(f)
    with open(log_dir + "/outputs_w_lp.json", 'r', encoding='utf-8') as f:
        outputs_w_lp = json.load(f)
    # convert to ModelOutput

    outputs_wo_lp = _json_to_model_output(outputs_wo_lp)
    outputs_w_lp = _json_to_model_output(outputs_w_lp)
    # convert to Namespace
    args = Namespace(**args_dict)

    return args_dict, outputs_wo_lp, outputs_w_lp


def total_prompts(outputs: List[ModelOutput]) -> int:
    return sum([len(mo.completions) for mo in outputs])


def avg_lcs(outputs: List[ModelOutput]) -> float:
    return sum([sum([c.lcs for c in mo.completions]) for mo in outputs]) / total_prompts(outputs)


def avg_rouge(outputs: List[ModelOutput]) -> float:
    return sum([sum([c.rouge for c in mo.completions]) for mo in outputs]) / total_prompts(outputs)


def refusal_ratio(outputs: List[ModelOutput]) -> float:
    # check_refusal
    return sum([sum([check_refusal(c.response) for c in mo.completions]) for mo in outputs]) / total_prompts(outputs)


def max_lcs(outputs: List[ModelOutput]) -> int:
    return max([max([c.lcs for c in mo.completions]) for mo in outputs])


def max_rouge(outputs: List[ModelOutput]) -> float:
    return max([max([c.rouge for c in mo.completions]) for mo in outputs])


def fill_refusal(outputs: List[ModelOutput]) -> List[ModelOutput]:
    for mo in outputs:
        for c in mo.completions:
            c.refusal = check_refusal(c.response)
    return outputs


def read_versions(versions: List[int], subdir: str, **kwargs):
    outputs_wo_lp = []
    outputs_w_lp = []
    all_args = []
    for version in versions:
        # check existence
        if not os.path.exists(f"{LOD_DIR}/{subdir}/v_{version:03d}"):
            continue
        curr_args, outputs_wo_lp_, outputs_w_lp_ = read_gen(subdir, version)
        flag = False
        for k, v in kwargs.items():
            if curr_args[k] != v:
                flag = True
                break
        if flag:
            continue
        all_args.append(curr_args)
        outputs_w_lp_ = fill_refusal(outputs_w_lp_)
        outputs_wo_lp_ = fill_refusal(outputs_wo_lp_)
        outputs_wo_lp.append(outputs_wo_lp_)
        outputs_w_lp.append(outputs_w_lp_)

    from jailbreak_sort import get_jailbreak_diversity
    # create dataframe readable format
    new_final_outputs = []
    for i in range(len(all_args)):
        args_now = all_args[i]
        saved_args = {
            'model': args_now['api_model_name'] if args_now['api_model'] else args_now['hf_model_id'],
            'dataset': args_now['dataset'],
            'jailbreak': args_now['jailbreak'],
            'jailbreak_num': args_now['jailbreak_num'],
            'defense_type': args_now['defense_type'],
        }
        for j in range(len(outputs_wo_lp[i])):
            book_wo_lp = outputs_wo_lp[i][j]
            book_w_lp = outputs_w_lp[i][j]
            for k in range(len(book_wo_lp.completions)):
                completion_wo_lp = book_wo_lp.completions[k]
                completion_w_lp = book_w_lp.completions[k]
                jailbreak_id = completion_wo_lp.jailbreak_id
                assert jailbreak_id == completion_w_lp.jailbreak_id
                jailbreak_diversity, jailbreak_diversity_fine_grained = get_jailbreak_diversity(jailbreak_id)
                new_final_outputs.append({
                    'model': saved_args['model'],
                    'dataset': saved_args['dataset'],
                    'jailbreak': saved_args['jailbreak'],
                    'jailbreak_id': jailbreak_id,
                    'jailbreak_num': saved_args['jailbreak_num'],
                    'defense_type': saved_args['defense_type'],
                    'prompt': completion_wo_lp.prompt,
                    'response_wo_lp': completion_wo_lp.response,
                    'response_w_lp': completion_w_lp.response,
                    'lcs_wo_lp': completion_wo_lp.lcs,
                    'lcs_w_lp': completion_w_lp.lcs,
                    'rouge_wo_lp': completion_wo_lp.rouge,
                    'rouge_w_lp': completion_w_lp.rouge,
                    'refusal_wo_lp': completion_wo_lp.refusal,
                    'refusal_w_lp': completion_w_lp.refusal,
                    **jailbreak_diversity,
                    **jailbreak_diversity_fine_grained
                })
    import pandas as pd
    df = pd.DataFrame(new_final_outputs)

    #if empty df return {}
    if df.empty:
        print("Empty dataframe")
        return {}
    all_previous_studies = ['is_pretending', 'is_attention_shifting', 'is_privilege_escalation']
    df['is_previous_study'] = df[all_previous_studies].any(axis=1)

    jailbreak_diversity_terms = [
        'is_not_jailbreak', 'is_previous_study', 'is_public_domain_attack',
        'is_pretending', 'is_attention_shifting', 'is_privilege_escalation',
        'is_character_roleplay', 'is_research_experiment', 'is_assumed_responsibility',
        'is_logical_reasoning', 'is_text_continuation', 'is_translation',
        'is_program_execution', 'is_superior_model', 'is_sudo_mode',
        'is_simulate_jailbreaking'
    ]
    # For each jailbreak diversity term (True), calculate the mean and max of LCS, Rouge, and Refusal
    # for all models and datasets
    # Output only the mean and max of [lcs_wo_lp, lcs_w_lp, rouge_wo_lp, rouge_w_lp, refusal_wo_lp, refusal_w_lp]
    output_terms = ['lcs_wo_lp', 'lcs_w_lp', 'rouge_wo_lp', 'rouge_w_lp', 'refusal_wo_lp', 'refusal_w_lp']
    if kwargs.get('defense_type', 'none') == 'plain':
        # no with lp
        output_terms = ['lcs_wo_lp', 'rouge_wo_lp', 'refusal_wo_lp']

    diversity_results = {}
    for term in jailbreak_diversity_terms:
        diversity_results[term] = {}
        for model in df['model'].unique():
            diversity_results[term][model] = {}
            print(f"Model: {model}")
            print(f"Mean of {term}:")
            mean_now = df[(df[term] == True) & (df['model'] == model)] \
                [output_terms].mean()
            print(mean_now)
            print(f"Max of {term}:")
            maxnow = df[(df[term] == True) & (df['model'] == model)] \
                [output_terms].max()
            print(maxnow)
            print("\n")
            for output_term in output_terms:
                diversity_results[term][model][f"{output_term}_mean"] = mean_now[output_term].item()
                diversity_results[term][model][f"{output_term}_max"] = maxnow[output_term].item()

    print(json.dumps(diversity_results, indent=4)
          )
    # best 20 jailbreak ids for each model
    best_jailbreak_ids = {}
    jailbreak_id2metrics = {}
    for model in df['model'].unique():
        # first aggregate the mean of lcs, rouge, refusal of all jailbreak_ids
        # only keep jailbreak_id, output_terms
        df_model = df[df['model'] == model]
        df_model = df_model[['jailbreak_id'] + output_terms]
        df_model = df_model.groupby('jailbreak_id').max()
        df_model = df_model.reset_index()
        df_model = df_model.sort_values(by='lcs_wo_lp', ascending=False)

        jailbreak_id2metrics[model] = df_model.set_index('jailbreak_id').to_dict(orient='index')
        best_jailbreak_ids[model] = df_model['jailbreak_id'][:20].tolist()
    for model in df['model'].unique():
        print(f"Best jailbreak ids for {model}:")
        print(best_jailbreak_ids[model])
        print("\n")

    all_same = None
    for model in df['model'].unique():

        if all_same is None:
            all_same = set(best_jailbreak_ids[model])
        else:
            all_same = all_same.intersection(set(best_jailbreak_ids[model]))
    print(all_same)

    for model in best_jailbreak_ids:
        print(f"Model: {model}")
        print(f"Best jailbreak ids:")
        print(best_jailbreak_ids[model])
        print("\n")
        for model2 in jailbreak_id2metrics:
            if model2 == model:
                continue
            print(f"Model2: {model2}")
            print(f"Mean of {model2}:")
            print(df[df['jailbreak_id'].isin(best_jailbreak_ids[model]) & (df['model'] == model2)][output_terms].mean())
            print(f"Max of {model2}:")
            print(df[df['jailbreak_id'].isin(best_jailbreak_ids[model]) & (df['model'] == model2)][output_terms].max())
            print("\n")

            # select random 20 jailbreak ids
            print(f"Random 20 jailbreak ids for {model2}:")
            random_20 = df[df['model'] == model2]['jailbreak_id'].sample(20).tolist()
            print(random_20)
            print("\n")
            print(f"Mean of {model2} for random 20:")
            print(df[df['jailbreak_id'].isin(random_20) & (df['model'] == model2)][output_terms].mean())
            print(f"Max of {model2} for random 20:")
            print(df[df['jailbreak_id'].isin(random_20) & (df['model'] == model2)][output_terms].max())
            print("\n")
    return best_jailbreak_ids



if __name__ == '__main__':
    subdir = 'bsc'
    version = [i for i in range(1, 100)]
    best_jailbreak_ids= read_versions(version, subdir, defense_type='plain')
    json_str = json.dumps(best_jailbreak_ids, indent=4)
    print(json_str)
    with open('tmp.txt', 'w', encoding='utf-8') as file:
        file.write(json_str)