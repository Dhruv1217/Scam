
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import joblib


# ==================================================
# 1. LOAD SMS DATASET FROM HUGGING FACE
# ==================================================

print("Loading SMS dataset...")

sms_ds = load_dataset(
    "CloveAI/india-spam-sms",
    split="train"
)

sms = sms_ds.to_pandas()

print("SMS columns:", sms.columns.tolist())
print(sms.head(2))


# ==================================================
# 2. LOAD EMAIL DATASET FROM HUGGING FACE
# ==================================================

print("\nLoading email dataset...")

email_ds = load_dataset("Navyasri17/phishing_emails-data")

email_parts = []

for split_name, split in email_ds.items():
    part = split.to_pandas()
    part["source_split"] = split_name
    email_parts.append(part)

email = pd.concat(email_parts, ignore_index=True)

print("Email columns:", email.columns.tolist())
print(email.head(2).to_string())


# ==================================================
# 3. PREPARE SMS DATA
# ==================================================

# Confirm the dataset contains the expected columns.
if not {"text", "label"}.issubset(sms.columns):
    raise ValueError(
        f"Unexpected SMS columns: {sms.columns.tolist()}"
    )

sms_data = sms[["text", "label"]].copy()

sms_data["raw_label"] = (
    sms_data["label"].astype(str).str.strip().str.lower()
)

# Explicitly map the source labels.
# Verify these meanings against the dataset documentation.
SMS_LABEL_MAP = {
    "ham": "legitimate",
    "legitimate": "legitimate",
    "normal": "legitimate",
    "0": "legitimate",
    "spam": "spam",
    "1": "spam",
}

sms_data["label"] = sms_data["raw_label"].map(
    SMS_LABEL_MAP
)

print("\nSMS label counts:")
print(sms_data["raw_label"].value_counts(dropna=False))



# 4. PREPARE EMAIL DATA

print("\nEmail columns:", email.columns.tolist())

email_data = email[["text", "email_type"]].copy()
email_data.columns = ["text", "raw_label"]

email_data["raw_label"] = (
    email_data["raw_label"]
    .astype(str)
    .str.strip()
    .str.lower()
)

print("\nOriginal email labels:")
print(email_data["raw_label"].value_counts(dropna=False))

EMAIL_LABEL_MAP = {
    "safe email": "legitimate",
    "safe": "legitimate",
    "legitimate": "legitimate",
    "phishing email": "phishing",
    "phishing": "phishing",
}

email_data["label"] = email_data["raw_label"].map(
    EMAIL_LABEL_MAP
)


print("\nEmail label counts:")
print(email_data["raw_label"].value_counts(dropna=False))


# ==================================================
# 5. CHECK UNMAPPED LABELS
# ==================================================

print("\nUnmapped SMS labels:")
print(
    sms_data.loc[
        sms_data["label"].isna(), "raw_label"
    ].value_counts()
)

print("\nUnmapped email labels:")
print(
    email_data.loc[
        email_data["label"].isna(), "raw_label"
    ].value_counts()
)

sms_data = sms_data.dropna(subset=["text", "label"])
email_data = email_data.dropna(subset=["text", "label"])

sms_data["text"] = sms_data["text"].astype(str).str.strip()
email_data["text"] = email_data["text"].astype(str).str.strip()

sms_data = sms_data[sms_data["text"].str.len() > 0]
email_data = email_data[email_data["text"].str.len() > 0]


# ==================================================
# 6. COMBINE SMS AND EMAIL DATA
# ==================================================

data = pd.concat(
    [
        sms_data[["text", "label"]],
        email_data[["text", "label"]],
    ],
    ignore_index=True,
)

data = data.drop_duplicates(subset=["text", "label"])

print("\nCombined dataset size:", len(data))
print("\nCombined class distribution:")
print(data["label"].value_counts())

required_classes = {
    "legitimate",
    "spam",
    "phishing",
}

if set(data["label"].unique()) != required_classes:
    raise ValueError(
        "\nAll three classes are required: legitimate, spam, "
        "and phishing.\n"
        "Check the label counts above. If SMS labels are numeric, "
        "verify their meanings in the dataset documentation."
    )


# ==================================================
# 7. SPLIT INTO TRAINING AND TEST DATA
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["label"],
    test_size=0.20,
    random_state=42,
    stratify=data["label"],
)

print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))


# ==================================================
# 8. BUILD THE MACHINE LEARNING MODEL
# ==================================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            strip_accents="unicode",
            max_features=100000,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
    ),
])


# ==================================================
# 9. TRAIN THE MODEL
# ==================================================

print("\nTraining combined SMS + email model...")

model.fit(X_train, y_train)

print("Training complete!")


# ==================================================
# 10. EVALUATE THE MODEL
# ==================================================

predictions = model.predict(X_test)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=["legitimate", "spam", "phishing"],
        zero_division=0,
    )
)

print("\nConfusion matrix:")
print(
    confusion_matrix(
        y_test,
        predictions,
        labels=["legitimate", "spam", "phishing"],
    )
)


# ==================================================
# 11. SAVE THE MODEL
# ==================================================

joblib.dump(model, "combined_scam_model.joblib")

print("\nModel saved as combined_scam_model.joblib")


# ==================================================
# 12. TEST SAMPLE MESSAGES
# ==================================================

test_messages = [
    "Your account has been verified successfully.",
    "Congratulations! Buy one get one free today.",
    "Your bank account is blocked. Click the link to verify your password.",
]

print("\nSample predictions:")

for message in test_messages:
    prediction = model.predict([message])[0]
    print(f"\nMessage: {message}")
    print(f"Prediction: {prediction}")
