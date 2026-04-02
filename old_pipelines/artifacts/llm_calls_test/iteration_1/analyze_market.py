
def analyze_market(transactions):
    """Analyze market from transactions."""
    prices = [t["price"] for t in transactions]
    return {
        "mean": sum(prices) / len(prices),
        "min": min(prices),
        "max": max(prices),
    }
