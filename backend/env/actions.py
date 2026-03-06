"""Discrete clinical action space for ICU treatment sequencing."""

from enum import IntEnum


class Action(IntEnum):
    """11 discrete clinical interventions available to the RL agent."""

    NOOP = 0
    BOLUS_250 = 1       # IV fluid bolus 250 mL
    BOLUS_500 = 2       # IV fluid bolus 500 mL
    VASOPRESSOR_UP = 3  # Increase vasopressor infusion
    VASOPRESSOR_DOWN = 4  # Decrease vasopressor infusion
    FIO2_UP = 5         # Increase fraction of inspired oxygen
    FIO2_DOWN = 6       # Decrease fraction of inspired oxygen
    PEEP_UP = 7         # Increase positive end-expiratory pressure
    PEEP_DOWN = 8       # Decrease positive end-expiratory pressure
    SEDATION = 9        # Administer sedation
    EXTUBATE = 10       # Extubate patient (remove ventilator)

    @property
    def label(self) -> str:
        labels = {
            Action.NOOP: "No intervention",
            Action.BOLUS_250: "IV Bolus 250 mL",
            Action.BOLUS_500: "IV Bolus 500 mL",
            Action.VASOPRESSOR_UP: "Vasopressor ↑",
            Action.VASOPRESSOR_DOWN: "Vasopressor ↓",
            Action.FIO2_UP: "FiO₂ ↑",
            Action.FIO2_DOWN: "FiO₂ ↓",
            Action.PEEP_UP: "PEEP ↑",
            Action.PEEP_DOWN: "PEEP ↓",
            Action.SEDATION: "Sedation",
            Action.EXTUBATE: "Extubate",
        }
        return labels[self]

    @property
    def is_fluid(self) -> bool:
        return self in (Action.BOLUS_250, Action.BOLUS_500)

    @property
    def is_ventilator(self) -> bool:
        return self in (Action.FIO2_UP, Action.FIO2_DOWN, Action.PEEP_UP, Action.PEEP_DOWN, Action.EXTUBATE)

    @property
    def is_vasopressor(self) -> bool:
        return self in (Action.VASOPRESSOR_UP, Action.VASOPRESSOR_DOWN)

    def __str__(self) -> str:
        return self.name
