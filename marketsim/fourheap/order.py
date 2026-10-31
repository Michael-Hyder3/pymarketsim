from dataclasses import dataclass


@dataclass
class Order:
    price: float
    order_type: int
    quantity: float
    agent_id: int
    time: int
    order_id: int
    asset_id: int = 1

    def update_quantity_filled(self, transact_quantity: float) -> None:
        self.quantity -= transact_quantity

    def merge_order(self, q_additional):
        self.quantity += q_additional

    def copy_and_decrease(self, transact_quantity: float) -> 'Order':
        new_order = Order(self.price,
                          self.order_type,
                          self.quantity - transact_quantity,
                          self.agent_id,
                          self.time,
                          self.order_id,
                          self.asset_id
                          )
        self.update_quantity_filled(self.quantity - transact_quantity)
        return new_order




@dataclass
class MatchedOrder:
    price: float
    time: int
    order: Order
