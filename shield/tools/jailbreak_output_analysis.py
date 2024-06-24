import json
import csv
import matplotlib.pyplot as plt
import numpy as np

output = f'./gen_logs/bsc/{read_version}/outputs_wo_lp.json'
output_w_lp = f'./gen_logs/bsc/{read_version}/outputs_w_lp.json'
args_file = f'./gen_logs/bsnc/{read_version}/args.json'
with open(args_file, 'r') as file:
    args = json.load(file)


def plot_barchart(data, title, xlabel, ylabel):
    plt.figure(figsize=(10, 5))
    # bar names too long, rotate them
    # bar too long, make the plot horizontal
    plt.barh(list(data.keys()), list(data.values()))

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def load_id2type():
    file_path = './jailbreak_prompts.csv'
    # id,pattern,name,,
    # 80,"Character Roleplay, Logical Reasoning, Text Continuation",Universal Jailbreak,,
    # 79,"Text Continuation, Character Roleplay, Research Experiment",Text Continuation,,
    # 78,"Translation, Character Roleplay, Assumed Responsibility, Research Experiment",TranslatorBot,,
    # 77,"Character Roleplay, Assumed Responsibility",AIM,,
    # 76,"Character Roleplay, Assumed Responsibility",UCAR,,

    id2type = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)

        for row in reader:
            id2type[str(int(row[0]))] = list(map(str.strip, row[1].split(',')))

    return id2type


pretending_types = ['Character Roleplay', 'Research Experiment',
                    'Assumed Responsibility', ]

attention_shifting_types = ['Logical Reasoning', 'Text Continuation', 'Translation', 'Program Execution']

privilege_escalation_types = ['Superior Model', 'Sudo Mode', 'Simulate Jailbreaking']

all_types = pretending_types + attention_shifting_types + privilege_escalation_types


def load_output(output):
    with open(output, 'r') as file:
        return json.load(file)


if __name__ == '__main__':
    id2type = load_id2type()

    type2cnt = {}
    for k, v in id2type.items():
        for t in v:
            type2cnt[t] = type2cnt.get(t, 0) + 1

    plot_barchart(type2cnt, 'Jailbreak Prompt Type Distribution', 'Type', 'Count')

    type2totallcs = {}

    output = load_output(output)
    output_w_lp = load_output(output_w_lp)
    for model_output in output:
        for completion in model_output['completions']:
            print(completion['jailbreak_id'], id2type[completion['jailbreak_id']])
            print(completion['prompt'])
            print(completion['response'])
            print(completion['lcs'])
            print()
            for t in id2type[completion['jailbreak_id']]:
                type2totallcs[t] = type2totallcs.get(t, 0) + int(completion['lcs'])
    type2totallcs_w_lp = {}
    for model_output in output_w_lp:
        for completion in model_output['completions']:
            for t in id2type[completion['jailbreak_id']]:
                type2totallcs_w_lp[t] = type2totallcs_w_lp.get(t, 0) + int(completion['lcs'])

    type2avglcs = {}
    for k, v in type2cnt.items():
        type2avglcs[k] = type2totallcs[k] / v

    type2avglcs_w_lp = {}
    for k, v in type2cnt.items():
        type2avglcs_w_lp[k] = type2totallcs_w_lp[k] / v
    plot_barchart(type2avglcs, f'Jailbreak Prompt Type LCS Distribution for {args["hf_model_id"]}', 'AVG LCS', ' ')

    plot_barchart(type2avglcs_w_lp, f'Jailbreak Prompt Type LCS Distribution for {args["hf_model_id"]} with LP', 'AVG LCS', ' ')

    # plot together for comparison with and without LP, same type plot together
    # sort by type using all_types
    type2avglcs = dict(sorted(type2avglcs.items(), key=lambda  x: all_types.index(x[0])))
    type2avglcs_w_lp = dict(sorted(type2avglcs_w_lp.items(), key=lambda  x: all_types.index(x[0])))
    # plot
    fig, ax = plt.subplots( figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(type2avglcs))
    ax.bar(index, type2avglcs.values(), bar_width, label='Without LP')
    ax.bar(index + bar_width, type2avglcs_w_lp.values(), bar_width, label='With LP')
    # rotate x labels
    plt.xticks(rotation=45, ha="right")
    ax.set_xlabel('Type')
    ax.set_ylabel('AVG LCS')
    ax.set_title(f'Jailbreak Prompt Type LCS Distribution for {args["hf_model_id"]} with and without LP')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(type2avglcs.keys())
    ax.legend()
    plt.tight_layout()
    plt.show()
