import torch
from marketsim.fourheap.constants import BUY, SELL


class SeparatedPrivateValues:
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

        self.extra_buy = float(self.buyer_values[-1].item()) if len(self.buyer_values) > 0 else float(self.base_price)
        self.extra_sell = float(self.seller_costs[-1].item()) if len(self.seller_costs) > 0 else float(self.base_price)

    @property
    def values(self):
        return torch.cat((self.seller_costs, self.buyer_values))

    def value_for_exchange(self, position: int, order_type: int) -> float:
        if order_type == BUY:
            idx = position
            if idx < 0 or idx >= len(self.buyer_values):
                return self.extra_buy
            return float(self.buyer_values[idx].item())
        else:
            idx = abs(position)
            if idx < 0 or idx >= len(self.seller_costs):
                return self.extra_sell
            return float(self.seller_costs[idx].item())

    def consume_marginal(self, position: int, order_type: int):
        return self.value_for_exchange(position, order_type)

