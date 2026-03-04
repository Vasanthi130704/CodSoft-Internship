def clean_text(text):
    # convert to lowercase
    text = str(text).lower()

    # remove extra spaces
    text = " ".join(text.split())

    return text