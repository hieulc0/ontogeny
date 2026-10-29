from enum import IntEnum


class Action(IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    INSPECT = 4
    INTERACT = 5
    REST = 6


MOVES = {Action.NORTH: (0, -1), Action.SOUTH: (0, 1), Action.EAST: (1, 0), Action.WEST: (-1, 0)}
