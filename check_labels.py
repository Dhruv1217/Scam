
from datasets import load_dataset
from collections import Counter

DATASET_ID = "CloveAI/india-spam-sms"

dataset = load_dataset(DATASET_ID)
data = dataset["train"]

print("Label distribution:")
print(Counter(data["label"]))

print("\nSample messages for each label:")
for label in set(data["label"]):
    print(f"\nLABEL {label}:")
    count = 0
    for row in data:
        if row["label"] == label:
            print(row["text"])
            count += 1
            if count == 5:
                break
