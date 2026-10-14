
import joblib

# Load trained model
model = joblib.load("combined_scam_model.joblib")

print("SCAMSHIELD Combined Model Testing")
print("Type 'exit' to stop.\n")

while True:
    message = input("Enter SMS or email text: ").strip()

    if message.lower() == "exit":
        print("Testing finished.")
        break

    if not message:
        print("Please enter a message.\n")
        continue

    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]
    classes = model.named_steps["classifier"].classes_

    print("\nPrediction:", prediction.upper())
    print("Class probabilities:")

    for label, probability in zip(classes, probabilities):
        print(f"  {label}: {probability * 100:.2f}%")

    print("-" * 40)
