from enum import Enum


class EvaluationResult(Enum):
    AREA_CONSTRAINT = "Area Violation"
    SCHEDULE_CONSTRAINT = "Schedule Violation"
    DEADLINE_CONSTRAINT = "Deadline Violation"
    SUCCESS = "Success"
    INVALID_SIMULATION = "Invalid Simulation"
    UNKNOWN = "Unknown"
