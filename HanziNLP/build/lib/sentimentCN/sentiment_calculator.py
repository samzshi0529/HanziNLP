def analyze_sentiment(text):
    """Dummy function to analyze sentiment of given text."""
    if "happy" in text.lower():
        return "Positive"
    elif "sad" in text.lower():
        return "Negative"
    else:
        return "Neutral"