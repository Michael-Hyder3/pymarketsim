"""
Test script for the LLMFirstZIAgent to verify it integrates correctly with the market.
"""

import random
import numpy as np
from marketsim.market.market import Market
from marketsim.fundamental.dummy_fundamental import DummyFundamental
from marketsim.agent.llm_first_zi_agent import LLMFirstZIAgentBuy, LLMFirstZIAgentSell
from marketsim.agent.zi_agent_buy_sell import ZIAgentBuy, ZIAgentSell


def test_llm_first_zi_agent():
    """Test that LLMFirstZIAgent can execute a basic trading scenario."""
    
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create a simple market with constant fundamental
    fundamental = DummyFundamental(value=100000.0, final_time=100)
    market = Market(fundamental=fundamental, time_steps=100)
    
    # Create agents
    llm_buyer = LLMFirstZIAgentBuy(
        agent_id=0,
        market=market,
        q_max=5,
        shade=[0, 500],
        pv_var=1e5,
        adjustment_rate=0.01,
        debug=True
    )
    
    llm_seller = LLMFirstZIAgentSell(
        agent_id=1,
        market=market,
        q_max=5,
        shade=[0, 500],
        pv_var=1e5,
        adjustment_rate=0.01,
        debug=True
    )
    
    zi_buyer = ZIAgentBuy(
        agent_id=2,
        market=market,
        q_max=5,
        shade=[0, 500],
        pv_var=1e5,
    )
    
    zi_seller = ZIAgentSell(
        agent_id=3,
        market=market,
        q_max=5,
        shade=[0, 500],
        pv_var=1e5,
    )
    
    agents = [llm_buyer, llm_seller, zi_buyer, zi_seller]
    
    print("\n" + "="*60)
    print("LLMFirstZI Agent Test")
    print("="*60)
    print(f"LLM Buyer initial WTP: {llm_buyer.current_wtp:.2f}")
    print(f"LLM Seller initial ask: {llm_seller.current_ask:.2f}")
    print(f"ZI Buyer private values: {[llm_buyer.pv.value_for_exchange(i, 0) for i in range(3)]}")
    print(f"ZI Seller private costs: {[llm_seller.pv.value_for_exchange(i, 1) for i in range(3)]}")
    
    transactions = []
    
    # Run simulation for 20 timesteps
    for t in range(20):
        market.event_queue.set_time(t)
        _ = market.get_fundamental_value()
        
        print(f"\n--- Timestep {t} ---")
        
        # Agents take action
        for agent in agents:
            market.withdraw_all(agent.get_id())
            orders = agent.take_action()
            market.add_orders(orders)
        
        # Market clears
        matched_orders = market.step()
        
        # Record transactions
        if matched_orders:
            print(f"Matched {len(matched_orders)} orders")
            for mo in matched_orders:
                print(f"  Order: price={mo.order.price:.2f}, qty={mo.order.quantity}, "
                      f"agent={mo.order.agent_id}, type={'BUY' if mo.order.order_type == 0 else 'SELL'}")
                transactions.append({
                    'time': t,
                    'price': mo.order.price,
                    'agent_id': mo.order.agent_id,
                    'order_type': mo.order.order_type
                })
        
        # Print agent state
        print(f"LLM Buyer: pos={llm_buyer.position}, wtp={llm_buyer.current_wtp:.2f}, cash={llm_buyer.cash:.2f}")
        print(f"LLM Seller: pos={llm_seller.position}, ask={llm_seller.current_ask:.2f}, cash={llm_seller.cash:.2f}")
    
    print("\n" + "="*60)
    print(f"Total transactions: {len(transactions)}")
    if transactions:
        prices = [tx['price'] for tx in transactions]
        print(f"Min price: {min(prices):.2f}, Max price: {max(prices):.2f}, Avg: {np.mean(prices):.2f}")
    print("="*60)
    
    # Verify agents have correct attributes
    assert hasattr(llm_buyer, 'wtp_history'), "LLM buyer missing wtp_history"
    assert hasattr(llm_buyer, 'observation_history'), "LLM buyer missing observation_history"
    assert hasattr(llm_seller, 'ask_history'), "LLM seller missing ask_history"
    assert hasattr(llm_seller, 'observation_history'), "LLM seller missing observation_history"
    
    print("✓ All tests passed!")


if __name__ == "__main__":
    test_llm_first_zi_agent()
