import json

interactions = json.load(open('datasets/drug_interactions_hard.json'))

with open('datasets/healthbench_hard.jsonl', 'r') as f:
    conversations = [json.loads(line) for line in f]

with open('batch/batch_68fcce6ff460819089c20e027f4e9b0e_output.jsonl', 'r') as f:
    drugs = [json.loads(line) for line in f]

drugs = {drug['custom_id']: drug['response']['body']['choices'][0]['message']['content'].split(',') for drug in drugs}

for conversation in conversations:
    if conversation['prompt_id'] not in drugs:
        print(f"Conversation {conversation['prompt_id']} not found in drugs")
        continue
    conversation['drugs'] = drugs[conversation['prompt_id']]
    conversation['interactions'] = {drug: interactions.get(drug) for drug in conversation['drugs']}

with open('datasets/healthbench_hard_interactions.jsonl', 'w') as f:
    for conversation in conversations:
        if conversation['prompt_id'] not in drugs:
            print(f"Conversation {conversation['prompt_id']} not found in drugs")
            continue
        f.write(json.dumps(conversation) + '\n')