import torch
from marketsim.fourheap.constants import BUY, SELL


class PrivateValues:
    """
    A class representing private values for a trading scenario.

    The PrivateValues class generates and manages a set of private values for buy and sell orders.
    The private values are generated from a normal distribution with a specified variance.
    The class provides methods to retrieve the value for a specific position and order type,
    as well as calculate the cumulative value up to a given position.
    """

    def __init__(self, q_max: int, val_var=5e6):
        """
        Initialize the PrivateValues object.

        :param q_max: The maximum quantity.
        :param val_var: The variance of the values (default: 1).
        """
        self.values = torch.randn(2 * q_max) * torch.sqrt(torch.tensor(val_var))
        self.values, _ = self.values.sort(descending=True)

        self.offset = q_max

        self.extra_buy = min(self.values[-1].item(), 0)
        self.extra_sell = max(self.values[0].item(), 0)

    def value_for_exchange(self, position: int, order_type: int) -> float:
        """
        Calculates the value associated with a given trade and order type.

        Args:
            position (int): The position for which to calculate the value.
            order_type (int): The type of order (BUY or SELL).

        Returns:
            float: The value associated with the position and order type.
        """
        index = position + self.offset - (1 if order_type == SELL else 0)
        if index >= len(self.values):
            return self.extra_buy
        elif index < 0:
            return self.extra_sell
        else:
            return self.values[index].item()

    def value_at_position(self, position: int) -> float:
        """
        Calculates the total value at a given position.

        Args:
            position (int): The position for which to calculate the total value.

        Returns:
            float: The total value at the given position.
        """
        value = 0
        position += self.offset
        if position > self.offset:
            index = min(position, len(self.values))
            value += torch.sum(self.values[self.offset:index])
            value += max(0, position - 2*self.offset)*self.extra_buy
        else:
            index = max(0, position)
            value -= torch.sum(self.values[index:self.offset])
            value -= -1*min(0, position)*self.extra_sell
        return value
    
    def consume_marginal(self, position: int, order_type: int):
        """
        Remove (consume) the marginal private value used for a trade at the given
        position and order type. Returns the removed marginal value or None if
        index out of range.

        The internal rule for indexing is the same as `value_for_exchange`:
        index = position + offset - (1 if SELL else 0).

        If the removed index is strictly less than `offset`, the `offset` is
        decremented so indexing remains consistent for the remaining values.
        """
        index = position + self.offset - (1 if order_type == SELL else 0)
        if index < 0 or index >= len(self.values):
            return None

        removed_val = self.values[index].item()

        # remove the element from the tensor
        if index == 0:
            new_values = self.values[1:]
        elif index == len(self.values) - 1:
            new_values = self.values[:-1]
        else:
            new_values = torch.cat((self.values[:index], self.values[index+1:]))

        # if we removed from the left side (seller region), shift offset left
        if index < self.offset:
            self.offset -= 1

        self.values = new_values

        # update extra_buy/extra_sell safely
        if len(self.values) == 0:
            self.extra_buy = 0
            self.extra_sell = 0
        else:
            self.extra_buy = min(self.values[-1].item(), 0)
            self.extra_sell = max(self.values[0].item(), 0)

        return removed_val
