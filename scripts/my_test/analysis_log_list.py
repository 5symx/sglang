import json
import re
import torch

num_experts = 128   # adjust to your router size
embed_dim = 2048    # hidden state size
# save_json = False 
save_json  = True

def parse_data_from_log(logfile):

    results = []
    current_group = {}

    with open(logfile, "r") as f:
        for line in f:
            # Match hidden state lines
            match_hidden = re.search(r"Request (\d+) : selected hidden state is \[(.*?)\]", line)
            if match_hidden:
                req_idx = int(match_hidden.group(1))
                hidden_state = list(map(float, match_hidden.group(2).split(",")))

                if req_idx == 0 and current_group:
                    results.extend(current_group.values())
                    current_group = {}

                current_group[req_idx] = { 
                    "embedding": hidden_state, 
                    "label": [0] * num_experts 
                }
                

                # # Find matching entry and add embedding
                # for entry in results:
                #     if entry["request_id"] == request_id and "embedding" not in entry:
                #         entry["embedding"] = hidden_state
                #         break

            # Match expert ID lines
            match_expert = re.search(r"Request (\d+) : selected expert id is \[(.*?)\] at layer 47", line)
            if match_expert:
                req_idx  = int(match_expert.group(1))
                expert_ids = list(map(int, match_expert.group(2).split(",")))
                # layer = int(match_expert.group(3))
                assert req_idx in current_group

                for eid in expert_ids:
                    if eid < num_experts:
                        current_group[req_idx]["label"][eid] += 1

                # # Convert expert IDs to binary label vector
                # label = [0] * num_experts
                # for eid in expert_ids:
                #     if eid < num_experts:
                #         label[eid] = 1

                # entry = {
                #     "request_id": req_idx,
                #     "label": label
                # }
    if current_group: 
        # print(current_group[0])
        results.extend(current_group.values())
    return results

def save_to_json(data,json_path):

    # Save to JSON
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    log_path = "/home/ymx/sglang/output-122.log"   # replace with your actual log file path
    json_path = "/home/ymx/sglang/scripts/my_test/output-122.json"

    data = parse_data_from_log(log_path)

    if save_json == True:
        save_to_json(data, json_path)