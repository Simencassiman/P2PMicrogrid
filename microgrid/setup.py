from datetime import datetime

"""
    This file contains the parameters for creating, training and running a community.
"""

# Constants
SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60
SECONDS_PER_HOUR = SECONDS_PER_MINUTE * MINUTES_PER_HOUR
HOURS_PER_DAY = 24
CENTS_PER_EURO = 100
KWH_TO_WS = 1 * 1e3 * SECONDS_PER_HOUR  # 1kWh * 1e3Wh/1kWh * 3600s/1h

# Simulation settings
TIME_SLOT = 15
HORIZON = 24
START = datetime(2021, 11, 1)
END = datetime(2021, 11, 2)
DURATION = (END - START).total_seconds() / SECONDS_PER_MINUTE / TIME_SLOT
GRID_COST_AVG = 12.0        # c€ / kWh
GRID_COST_AMPLITUDE = 5.0   # c€ / kWh
GRID_COST_PERIOD = 12
GRID_COST_PHASE = 3
GRID_INJECTION_PRICE = 0.07     # € / kWh
seed = 42

# Community parameters
starting_episodes = 0
max_episodes = 1000
min_episodes_criterion = 50
save_episodes = 50
nr_agents = 2
rounds = 1
homogeneous = False
implementation = 'tabular'      # Agent implementation
