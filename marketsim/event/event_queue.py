import random
from collections import defaultdict
from typing import List

from marketsim.fourheap.order import Order


class EventQueue:
    def __init__(self, rand_seed: int = None):
        self.rand = random.Random(rand_seed)
        self.scheduled_activities = defaultdict(list)
        self.current_time = 0

    def schedule_activity(self, order: Order):
        t = order.time

        self.scheduled_activities[t].append(order)

    def step(self) -> List[Order]:
        # Use the EventQueue's RNG for reproducible shuffling
        bucket = self.scheduled_activities[self.current_time]
        try:
            self.rand.shuffle(bucket)
        except Exception:
            random.shuffle(bucket)

        # assign arrival sequence within this timestep so downstream code
        # can unambiguously determine which order arrived first
        for idx, order in enumerate(bucket):
            try:
                order._arrival_seq = idx
            except Exception:
                setattr(order, '_arrival_seq', idx)

        self.current_time += 1

        return bucket

    def get_current_time(self):
        return self.current_time

    def set_time(self, t):
        self.current_time = t
