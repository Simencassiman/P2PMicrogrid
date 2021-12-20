from environment import Environment


SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60
SECONDS_PER_HOUR = SECONDS_PER_MINUTE * MINUTES_PER_HOUR
KWH_TO_WS = 1 * 1e3 * SECONDS_PER_HOUR  # 1kWh * 1e3Wh/1kWh * 3600s/1h

environment = Environment()