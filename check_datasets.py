
from datasets import load_dataset

# Check SMS dataset
sms = load_dataset("CloveAI/india-spam-sms", split="train")

print("\n===== SMS DATASET =====")
print("Columns:", sms.column_names)
print("First row:", sms[0])

# Check email dataset
email = load_dataset("Navyasri17/phishing_emails-data")

print("\n===== EMAIL DATASET =====")
print("Available splits:", list(email.keys()))

for split_name, split_data in email.items():
    print(f"\nSplit: {split_name}")
    print("Columns:", split_data.column_names)
    print("First row:", split_data[0])

    # Show labels only if a label column exists
    for column in split_data.column_names:
        if "label" in column.lower() or "class" in column.lower():
            print(f"Values in {column}:")
            print(split_data.to_pandas()[column].value_counts())
