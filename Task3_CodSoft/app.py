from src.predict import predict_sms

while True:

    msg = input("Enter SMS (type exit to stop): ")

    if msg.lower() == "exit":
        break

    result = predict_sms(msg)

    print("Prediction:", result)