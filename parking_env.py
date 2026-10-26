"""Original toy simulator for RL-based smart parking allocation.

The environment is intentionally lightweight and self-contained so it can be
reproduced on a standard laptop. It models three parking zones with different
walking distances and capacities. Each arriving vehicle has a short, medium,
or long expected stay. The agent observes coarse zone occupancy and the
vehicle's expected stay, then chooses a zone.
"""
from dataclasses import dataclass
import random

@dataclass
class Vehicle:
    stay: int
    kind: int  # 0=short, 1=medium, 2=long

class ParkingEnv:
    def __init__(self, capacities=(5, 6, 7), distances=(1.8, 1.0, 3.2),
                 max_steps=120, seed=None):
        self.capacities = tuple(capacities)
        self.distances = tuple(distances)
        self.n_zones = len(capacities)
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.t = 0
        self.parked = [[] for _ in range(self.n_zones)]  # remaining stay per car
        self.total_search = 0.0
        self.total_walk = 0.0
        self.total_congestion = 0.0
        self.successful = 0
        self.rejected = 0
        self.decisions = 0
        self.invalid_actions = 0
        self._last_vehicle = self._new_vehicle()
        return self._state()

    def _new_vehicle(self):
        # Nonstationary-ish mix: demand and stay types vary with time of day.
        r = self.rng.random()
        if self.t % 40 < 14:
            probs = (0.55, 0.30, 0.15)
        elif self.t % 40 < 28:
            probs = (0.25, 0.45, 0.30)
        else:
            probs = (0.35, 0.35, 0.30)
        if r < probs[0]:
            kind, stay = 0, self.rng.randint(2, 5)
        elif r < probs[0] + probs[1]:
            kind, stay = 1, self.rng.randint(6, 12)
        else:
            kind, stay = 2, self.rng.randint(13, 24)
        return Vehicle(stay, kind)

    def _occupancies(self):
        return [len(z) for z in self.parked]

    def _state(self):
        # occupancy fraction is quantized to 0=low, 1=medium, 2=high, 3=nearly full
        occ_bins = []
        for n, cap in zip(self._occupancies(), self.capacities):
            frac = n / cap
            occ_bins.append(min(3, int(frac * 4)))
        return tuple(occ_bins + [self._last_vehicle.kind, min(2, self.t // 40)])

    def valid_actions(self):
        return [i for i, (z, cap) in enumerate(zip(self.parked, self.capacities))
                if len(z) < cap]

    def step(self, action):
        vehicle = self._last_vehicle
        valid = self.valid_actions()
        reward = 0.0
        if action not in valid:
            self.invalid_actions += 1
            self.rejected += 1
            reward -= 12.0
            search = 8.0
            walk = 0.0
        else:
            zone = action
            occupancy_before = len(self.parked[zone]) / self.capacities[zone]
            # Search time rises sharply as a zone becomes full; long stays add a
            # small expected blocking cost.
            search = 1.0 + 5.0 * occupancy_before**2
            walk = self.distances[zone]
            congestion = max(0.0, occupancy_before - 0.65) * 4.0
            stay_cost = (vehicle.stay / 24.0) * (0.4 + occupancy_before)
            reward = 5.0 - 0.8 * search - 0.45 * walk - 0.9 * congestion - stay_cost
            self.parked[zone].append(vehicle.stay)
            self.successful += 1
            self.total_search += search
            self.total_walk += walk
            self.total_congestion += congestion
        self.decisions += 1
        # Time advances and all parked vehicles' remaining durations decrease.
        self.t += 1
        for z in self.parked:
            for j in range(len(z)):
                z[j] -= 1
            z[:] = [remaining for remaining in z if remaining > 0]
        if self.t < self.max_steps:
            self._last_vehicle = self._new_vehicle()
        done = self.t >= self.max_steps
        info = {"search": search, "walk": walk, "valid": action in valid,
                "occupancy": self._occupancies()}
        return self._state(), reward, done, info

    def metrics(self):
        n = max(1, self.successful)
        return {
            "accepted": self.successful,
            "rejected": self.rejected,
            "acceptance_rate": self.successful / max(1, self.decisions),
            "avg_search_time": self.total_search / n,
            "avg_walking_distance": self.total_walk / n,
            "avg_congestion_penalty": self.total_congestion / n,
            "invalid_action_rate": self.invalid_actions / max(1, self.decisions),
        }
