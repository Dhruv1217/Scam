
from datasets import load_dataset

DATASET_ID = "CloveAI/india-spam-sms"  # Replace with your real dataset ID

dataset = load_dataset(DATASET_ID)

print("Available splits:", dataset.keys())

split_name = next(iter(dataset.keys()))
data = dataset[split_name]

print("Columns:", data.column_names)
print("Number of rows:", len(data))

for row in data.select(range(min(5, len(data)))):
    print(row)
