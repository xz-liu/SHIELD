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
    id2lcs_stats = {jid: {'avg_lcs': sum(lcs_list) / len(lcs_list), 'max_lcs': max(lcs_list)} for jid, lcs_list in id2lcs.items()}
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
            should_prioritize(item[1]['pattern'])  # If LCS scores are the same, prioritize those without specific patterns
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

if __name__ == '__main__':
    main()
