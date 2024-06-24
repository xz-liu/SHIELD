import json
import csv
from collections import defaultdict


def load_id2type(csv_path):
    id2type = {}
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            id2type[row['id']] = {
                'pattern': list(map(str.strip, row['pattern'].split(','))),
                'name': row['name'],
                'created_at': row['created_at'],
                'text': row['text'],
            }
    return id2type


def load_output(output_path):
    with open(output_path, 'r') as file:
        return json.load(file)


def calculate_lcs_statistics(output):
    id2lcs = defaultdict(list)
    for model_output in output:
        for completion in model_output['completions']:
            jailbreak_id = completion['jailbreak_id']
            lcs = int(completion['lcs'])
            id2lcs[jailbreak_id].append(lcs)
    id2lcs_stats = {jid: {'avg_lcs': sum(lcs_list) / len(lcs_list), 'max_lcs': max(lcs_list)} for jid, lcs_list in
                    id2lcs.items()}
    return id2lcs_stats


def should_prioritize(pattern):
    # Return True if the pattern does not contain "Character Roleplay" or "Assumed Responsibility"
    return not any(p in pattern for p in ["Character Roleplay", "Assumed Responsibility"])


def main():
    read_version = 'v_004'
    output_path = f'./gen_logs/bsc/{read_version}/outputs_wo_lp.json'
    csv_path = './jailbreak_prompts.csv'
    output_json_path = f'./sorted_jailbreaks_{read_version}.json'

    id2type = load_id2type(csv_path)
    output = load_output(output_path)
    id2lcs_stats = calculate_lcs_statistics(output)

    # Add LCS statistics to id2type
    for jid, stats in id2lcs_stats.items():
        if jid in id2type:
            id2type[jid].update(stats)

    # Sort according to the specified rules
    sorted_jailbreaks = sorted(
        id2type.items(),
        key=lambda item: (
            -item[1].get('avg_lcs', 0),  # Higher LCS score comes first
            should_prioritize(item[1]['pattern'])
            # If LCS scores are the same, prioritize those without specific patterns
        )
    )

    # Prepare JSON data for output
    sorted_jailbreaks_json = [
        {
            'id': jid,
            'pattern': details['pattern'],
            'name': details['name'],
            'created_at': details['created_at'],
            'text': details['text'],
            'avg_lcs': details.get('avg_lcs', 0),
            'max_lcs': details.get('max_lcs', 0)
        }
        for jid, details in sorted_jailbreaks
    ]

    # Write to JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(sorted_jailbreaks_json, json_file, indent=4)


class _GetJailbreakDiversity:
    def __init__(self):
        with open('jailbreak_prompts.json', 'r') as file:
            jailbreak_prompts = json.load(file)

        id2jailbreak = {}
        for jailbreak in jailbreak_prompts:
            id2jailbreak[jailbreak['id']] = jailbreak
        self.id2jailbreak = id2jailbreak

    def __call__(self, jailbreak_id):

        if jailbreak_id < 0:
            return {
                'is_pretending': False,
                'is_attention_shifting': False,
                'is_privilege_escalation': False,
                'is_not_jailbreak': True,
                'is_public_domain_attack': False,
            }, {
                'is_character_roleplay': False,
                'is_research_experiment': False,
                'is_assumed_responsibility': False,
                'is_logical_reasoning': False,
                'is_text_continuation': False,
                'is_translation': False,
                'is_program_execution': False,
                'is_superior_model': False,
                'is_sudo_mode': False,
                'is_simulate_jailbreaking': False,
            }
        if jailbreak_id > 100:

            return {
                'is_pretending': False,
                'is_attention_shifting': False,
                'is_privilege_escalation': False,
                'is_not_jailbreak': False,
                'is_public_domain_attack': True,
            }, {
                'is_character_roleplay': False,
                'is_research_experiment': False,
                'is_assumed_responsibility': False,
                'is_logical_reasoning': False,
                'is_text_continuation': False,
                'is_translation': False,
                'is_program_execution': False,
                'is_superior_model': False,
                'is_sudo_mode': False,
                'is_simulate_jailbreaking': False,
            }
        jailbreak_id= str(jailbreak_id)
        pretending_types = {'Character Roleplay', 'Research Experiment', 'Assumed Responsibility'}

        attention_shifting_types = {'Logical Reasoning', 'Text Continuation', 'Translation', 'Program Execution'}

        privilege_escalation_types = {'Superior Model', 'Sudo Mode', 'Simulate Jailbreaking'}

        jailbreak = self.id2jailbreak[jailbreak_id]
        patterns = jailbreak['pattern']
        is_pretending = any(p in patterns for p in pretending_types)
        is_attention_shifting = any(p in patterns for p in attention_shifting_types)
        is_privilege_escalation = any(p in patterns for p in privilege_escalation_types)

        is_character_roleplay = 'Character Roleplay' in patterns
        is_research_experiment = 'Research Experiment' in patterns
        is_assumed_responsibility = 'Assumed Responsibility' in patterns
        is_logical_reasoning = 'Logical Reasoning' in patterns
        is_text_continuation = 'Text Continuation' in patterns
        is_translation = 'Translation' in patterns
        is_program_execution = 'Program Execution' in patterns
        is_superior_model = 'Superior Model' in patterns
        is_sudo_mode = 'Sudo Mode' in patterns
        is_simulate_jailbreaking = 'Simulate Jailbreaking' in patterns

        return {
            'is_pretending': is_pretending,
            'is_attention_shifting': is_attention_shifting,
            'is_privilege_escalation': is_privilege_escalation,
            'is_not_jailbreak': False,
            'is_public_domain_attack': False,
        }, {
            'is_character_roleplay': is_character_roleplay,
            'is_research_experiment': is_research_experiment,
            'is_assumed_responsibility': is_assumed_responsibility,
            'is_logical_reasoning': is_logical_reasoning,
            'is_text_continuation': is_text_continuation,
            'is_translation': is_translation,
            'is_program_execution': is_program_execution,
            'is_superior_model': is_superior_model,
            'is_sudo_mode': is_sudo_mode,
            'is_simulate_jailbreaking': is_simulate_jailbreaking,

        }


get_jailbreak_diversity = _GetJailbreakDiversity()


def top30_jailbreak_diversity():
    # read jailbreak_prompts.json

    with open('jailbreak_prompts.json', 'r') as file:
        jailbreak_prompts = json.load(file)

    jailbreak_prompts = jailbreak_prompts

    pretending_types = ['Character Roleplay', 'Research Experiment',
                        'Assumed Responsibility', ]

    attention_shifting_types = ['Logical Reasoning', 'Text Continuation', 'Translation', 'Program Execution']

    privilege_escalation_types = ['Superior Model', 'Sudo Mode', 'Simulate Jailbreaking']

    pretending_types_count = 0
    attention_shifting_types_count = 0
    privilege_escalation_types_count = 0
    pattern2count = {}
    for jailbreak in jailbreak_prompts:
        patterns = jailbreak['pattern']
        for pattern in patterns:
            pattern2count[pattern] = pattern2count.get(pattern, 0) + 1
            if pattern in pretending_types:
                pretending_types_count += 1
            elif pattern in attention_shifting_types:
                attention_shifting_types_count += 1
            elif pattern in privilege_escalation_types:
                privilege_escalation_types_count += 1
    print('Pretending Types:', pretending_types_count)
    print('Attention Shifting Types:', attention_shifting_types_count)
    print('Privilege Escalation Types:', privilege_escalation_types_count)

    print('Pattern Counts:')
    for pattern, count in pattern2count.items():
        print(f'{pattern}: {count}')


if __name__ == '__main__':
    # main()
    top30_jailbreak_diversity()
