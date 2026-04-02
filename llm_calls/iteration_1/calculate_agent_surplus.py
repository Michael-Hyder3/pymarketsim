import statistics

def calculate_agent_surplus(private_values, transactions, agent_type):
    """
    Calculate the surplus (profit) achieved by an agent from their trades.

    Parameters:
    private_values (list of float): the agent's private valuations
    transactions (list of dict): each with 'price' and 'involved' keys
    agent_type (str): either 'buyer' or 'seller'

    Returns:
    dict with keys:
    - total_surplus (float): total surplus achieved
    - num_trades (int): number of trades the agent participated in
    - avg_price (float): average transaction price
    - details (str): explanation of the calculation
    """
    total_surplus = 0
    num_trades = 0
    avg_price = 0
    details = ""

    if agent_type == 'buyer':
        for transaction in transactions:
            if transaction['involved']:
                total_surplus += transaction['price'] - private_values[transaction['time']]
                num_trades += 1
                avg_price += transaction['price']
        avg_price /= num_trades
        details = f"Buyer surplus calculated as the difference between transaction prices and private valuations. Total surplus: {total_surplus}, Number of trades: {num_trades}, Average price: {avg_price}"
    elif agent_type == 'seller':
        for transaction in transactions:
            if transaction['involved']:
                total_surplus += private_values[transaction['time']] - transaction['price']
                num_trades += 1
                avg_price += transaction['price']
        avg_price /= num_trades
        details = f"Seller surplus calculated as the difference between private valuations and transaction prices. Total surplus: {total_surplus}, Number of trades: {num_trades}, Average price: {avg_price}"
    else:
        raise ValueError("Invalid agent type. Must be 'buyer' or 'seller'.")

    return {
        'total_surplus': total_surplus,
        'num_trades': num_trades,
        'avg_price': avg_price,
        'details': details
    }