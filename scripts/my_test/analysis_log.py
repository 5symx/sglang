import re
from collections import Counter

def parse_expert_ids_from_log(log_file, top_k=10):
    """
    Reads a log file, extracts expert IDs, and prints the top-k most popular IDs.
    
    Args:
        log_file (str): Path to the log file.
        top_k (int): Number of most popular IDs to display.
    """
    id_counter = Counter()
    
    # Regex to capture numbers inside tensor([...])
    pattern = re.compile(r"tensor\(\[([^\]]+)\]")
    
    with open(log_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                # Extract numbers inside brackets
                ids = [int(x.strip()) for x in match.group(1).split(",")]
                # Only count valid IDs (0–127)
                for i in ids:
                    if 0 <= i <= 127:
                        id_counter[i] += 1
    
    # Print top-k popular IDs
    print(f"Top {top_k} popular expert IDs:")
    for expert_id, count in id_counter.most_common(top_k):
        print(f"Expert {expert_id}: {count} times")

# Example usage
if __name__ == "__main__":
    log_path = "/home/ymx/sglang/output-116.log"   # replace with your actual log file path
    parse_expert_ids_from_log(log_path, top_k=16)
