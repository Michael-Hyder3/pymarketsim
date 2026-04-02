import torch
from marketsim.fourheap.constants import BUY, SELL


class SeparatedPrivateValues:
    """
    Separate private values for buyers and sellers with independent draws.
    
    For split agents:
    - Buyer agents: only need θ^0, θ^1, ..., θ^{q_max-1} (marginal gains from buying)
    - Seller agents: only need θ^0, θ^{-1}, ..., θ^{-(q_max-1)} (marginal costs from selling)
    
    Buyer values are sorted descending to preserve diminishing marginal utility:
    θ^{q'} ≥ θ^q for all q' ≤ q. Seller costs are sorted ascending (lowest cost first).
    """

    def __init__(self, q_max: int, val_var=1e5, base_price: float = 0.0, role: str = "both",
                 generator: torch.Generator = None):
        self.q_max = q_max
        self.base_price = float(base_price)
        sigma = float(torch.sqrt(torch.tensor(val_var)))

        role = (role or "both").lower()

        if role in {"buyer", "both"}:
            buyer_values = torch.randn(q_max, generator=generator) * sigma + self.base_price
            self.buyer_values, _ = buyer_values.sort(descending=True)
        else:
            self.buyer_values = torch.empty(0)

        if role in {"seller", "both"}:
            seller_values = torch.randn(q_max, generator=generator) * sigma + self.base_price
            self.seller_costs, _ = seller_values.sort(descending=False)
        else:
            self.seller_costs = torch.empty(0)

        # extras if asked beyond q_max
        self.extra_buy = float(self.buyer_values[-1].item()) if len(self.buyer_values) > 0 else float(self.base_price)
        self.extra_sell = float(self.seller_costs[-1].item()) if len(self.seller_costs) > 0 else float(self.base_price)

    @property
    def values(self):
        """Backward-compatible view: concat sellers then buyers to match original order."""
        # seller_costs are ascending (lowest to highest); buyer_values are descending
        return torch.cat((self.seller_costs, self.buyer_values))

    def value_for_exchange(self, position: int, order_type: int) -> float:
        """
        Get marginal value for trading at the given position.
        
        For BUY: position represents how many units already bought (0, 1, 2, ...)
                 Returns θ^position (the gain from buying the next unit)
        For SELL: position represents how many units already sold (0, -1, -2, ...)
                  Returns θ^position (the cost of selling the next unit)
        """
        if order_type == BUY:
            # Buyer at position q wants θ^q
            # buyer_values[0] = θ^0, buyer_values[1] = θ^1, etc.
            idx = position
            if idx < 0 or idx >= len(self.buyer_values):
                return self.extra_buy
            return float(self.buyer_values[idx].item())
        else:
            # Seller at position q wants θ^q
            # For position 0: wants lowest cost = seller_costs[0]
            # For position -1: wants next lowest = seller_costs[1]
            # seller_costs are stored ascending: [lowest, ..., highest]
            idx = abs(position)
            if idx < 0 or idx >= len(self.seller_costs):
                return self.extra_sell
            return float(self.seller_costs[idx].item())

    def consume_marginal(self, position: int, order_type: int):
        """
        Return the marginal value used for trade at this position.
        Does NOT modify the arrays - private values are fixed per agent.
        """
        return self.value_for_exchange(position, order_type)

    def value_of_holdings(self, position: int, is_buyer: bool) -> float:
        """
        Calculate cumulative value of holdings at given position.
        
        Per paper: For position H:
        - If H > 0 (long): sum_{k=1}^{H} θ^k
        - If H < 0 (short): -sum_{k=H+1}^{0} θ^k
        - If H = 0: 0
        
        For buyers (is_buyer=True), position is positive (0, 1, 2, ...)
        For sellers (is_buyer=False), position is negative (0, -1, -2, ...)
        """
        if position == 0:
            return 0.0
        
        total = 0.0
        if is_buyer:
            # Long position H > 0: sum_{k=1}^{H} θ^k = sum_{k=0}^{H-1} buyer_values[k]
            # Since buyer_values[k] = θ^k
            for k in range(position):
                if k < len(self.buyer_values):
                    total += float(self.buyer_values[k].item())
                else:
                    total += self.extra_buy
        else:
            # Short position H < 0: -sum_{k=H+1}^{0} θ^k
            # position is negative, so abs(position) is how many sold
            # For position -3: sum k from -2 to 0 = three lowest costs
            # seller_costs[0] = lowest cost, seller_costs[1] = next lowest, etc.
            for k in range(abs(position)):
                if k < len(self.seller_costs):
                    total -= float(self.seller_costs[k].item())
                else:
                    total -= self.extra_sell
        
        return total
