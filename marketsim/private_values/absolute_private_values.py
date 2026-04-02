import torch
from marketsim.fourheap.constants import BUY, SELL


class AbsolutePrivateValues:
    """
    Private values interpreted as absolute prices (not deviations).

    Generates 2*q_max absolute prices centered at `base_price` with variance `val_var`.
    Sorted descending. Indexing mirrors the existing `PrivateValues` API so it can
    be swapped in tests by assigning `agent.pv = AbsolutePrivateValues(...)`.
    """

    def __init__(self, q_max: int, val_var=1e5, base_price: float = 0.0):
        self.q_max = q_max
        self.base_price = float(base_price)
        # sigma = sqrt(val_var)
        self.values = torch.randn(2 * q_max) * torch.sqrt(torch.tensor(val_var)) + self.base_price
        self.values, _ = self.values.sort(descending=True)

        self.offset = q_max

        # extras (if index out of range)
        self.extra_buy = float(min(self.values[-1].item(), self.base_price))
        self.extra_sell = float(max(self.values[0].item(), self.base_price))

    def value_for_exchange(self, position: int, order_type: int) -> float:
        index = position + self.offset - (1 if order_type == SELL else 0)
        if index >= len(self.values):
            return self.extra_buy
        elif index < 0:
            return self.extra_sell
        else:
            return float(self.values[index].item())

    def value_at_position(self, position: int) -> float:
        # Not typically used for absolute prices, but keep API compatibility
        value = 0.0
        position += self.offset
        if position > self.offset:
            index = min(position, len(self.values))
            value += float(torch.sum(self.values[self.offset:index]).item())
            value += max(0, position - 2*self.offset)*self.extra_buy
        else:
            index = max(0, position)
            value -= float(torch.sum(self.values[index:self.offset]).item())
            value -= -1*min(0, position)*self.extra_sell
        return value

    def consume_marginal(self, position: int, order_type: int):
        index = position + self.offset - (1 if order_type == SELL else 0)
        if index < 0 or index >= len(self.values):
            return None

        removed_val = float(self.values[index].item())

        # remove the element
        if index == 0:
            new_values = self.values[1:]
        elif index == len(self.values) - 1:
            new_values = self.values[:-1]
        else:
            new_values = torch.cat((self.values[:index], self.values[index+1:]))

        if index < self.offset:
            self.offset -= 1

        self.values = new_values

        if len(self.values) == 0:
            self.extra_buy = 0.0
            self.extra_sell = 0.0
        else:
            self.extra_buy = float(min(self.values[-1].item(), self.base_price))
            self.extra_sell = float(max(self.values[0].item(), self.base_price))

        return removed_val
