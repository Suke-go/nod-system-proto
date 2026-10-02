from dataclasses import dataclass
from nod.config import distribution, probability


@dataclass(frozen=True)
class Selection:
    function: str
    mass: tuple
    utilities: dict


class ExpectedUtilityPolicy:
    def __init__(self, cfg):
        self.cfg = cfg

    def choose(self, opportunity, belief, specific=True):
        o = probability(opportunity)
        b = distribution(belief)
        if self.cfg.get('opportunity_weighting', 'probability_mass') == 'gate_only':
            # The engine has already required a fresh, sustained VAP window.
            # An uncalibrated VAP score is not a second probability of no feedback.
            mass = b if o > 0 else (1., 0., 0., 0.)
        else:
            mass = (1-o+o*b[0], *(o*x for x in b[1:]))
        utilities = {f:sum(u*m for u,m in zip(row,mass)) for f,row in self.cfg["utility_by_function"].items()}
        allowed = self.cfg["tie_break_order"] if specific else ["none", "continuer"]
        winner = max(allowed, key=lambda f: utilities[f])
        if utilities[winner]-utilities["none"] < self.cfg["minimum_gain_over_none"]:
            winner = "none"
        return Selection(winner, mass, utilities)

    def embody(self, function, prominence, neutral_expression=False):
        p = probability(prominence)
        low, high = self.cfg["intensity_ranges"][function]
        intensity = low+(high-low)*p
        if function == "continuer":
            action = "SMALL_NOD"
        elif function == "understanding":
            action = "STRONG_NOD" if self.cfg.get('understanding_action') == 'STRONG_NOD' or intensity >= self.cfg["strong_nod_threshold"] else "SMALL_NOD"
        else:
            action = "EMPATHIC_EXPRESSION" if neutral_expression else "STRONG_NOD"
        return action, intensity
