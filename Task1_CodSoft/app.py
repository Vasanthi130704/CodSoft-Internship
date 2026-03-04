from src.predict import predict_genre

print(" Movie Genre Classifier")

while True:
    plot = input("\nEnter movie plot (type 'exit' or 'quit' to stop): ")

    if plot.lower() in ["exit", "quit"]:
        print("Goodbye! Thank you for using the Movie Genre Classifier.")
        break

    result = predict_genre(plot)
    print("Predicted Genre:", result)