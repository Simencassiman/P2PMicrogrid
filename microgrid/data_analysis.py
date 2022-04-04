# Python Libraries
from typing import List
import json
from datetime import datetime
from calendar import monthrange

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc  # For the legend
from matplotlib.cm import ScalarMappable    # For the legend

# Local modules
from config import TIME_SLOT, MINUTES_PER_HOUR, HOURS_PER_DAY
from agent import ActingAgent


def analyse_community_output(agents: List[ActingAgent], time: List[datetime],
                             power: np.ndarray, cost: np.ndarray) -> None:

    slots_per_day = int(MINUTES_PER_HOUR / TIME_SLOT * HOURS_PER_DAY)

    nr_agents = len(agents)
    agent = agents[0]
    agent_ids = [a.id for a in agents]
    pv: np.array = np.array(agent.pv.get_history())
    temperature = np.array(agent.heating.get_history())
    battery = np.array(agent.storage.get_history())

    production = np.array(list(map(lambda a: a.pv.get_history(), filter(lambda a: a.id < 4, agents))))\
                    .transpose()
    self_consumption = (power[:, :4] < 0) * (production + power[:, :4]) + (power[:, :4] >= 0) * production

    cost = cost.sum(axis=0)
    fixed_cost = (0.25 * 0.2 * cost + 50 / 12 * np.maximum(2.5, power.max(axis=0) * 1e-3))
    print(f'Energy consumed: {power.sum(axis=0) * TIME_SLOT / MINUTES_PER_HOUR * 1e-3} kWh')
    print(f'Cost a total of: {cost} € volume and {fixed_cost} € capacity')

    # Create plots
    nr_ticks_factor = 16
    time = time[:slots_per_day]
    time_ticks = np.arange(int(slots_per_day / nr_ticks_factor)) * nr_ticks_factor
    time_labels = [t for i, t in enumerate(time) if i % nr_ticks_factor == 0]
    # time_labels = [t.strftime('%H:%M') for i, t in enumerate(time) if i % nr_ticks_factor == 0]
    # time = [t.isoformat() for t in time]

    plt.figure(1)
    plt.plot(time[:slots_per_day], power[:slots_per_day, 0] * 1e-3)
    plt.plot(time[:slots_per_day], pv[:slots_per_day] * 1e-3)
    plt.xticks(time_ticks, time_labels)
    plt.title("Agent profiles")
    plt.xlabel("Time")
    plt.ylabel("Power [kW]")
    plt.legend(['Loads', 'PV'])

    plt.figure(2)
    width = 0.35  # the width of the bars: can also be len(x) sequence

    plt.figure(2)
    plt.bar(agent_ids, cost, width, label='Volume')
    plt.bar(agent_ids, fixed_cost, width, bottom=cost, label='Capacity')
    plt.title("Electricity costs")
    plt.xlabel("Agent")
    plt.ylabel("Cost [€]")
    plt.legend()

    plt.figure(3)
    plt.plot(time[:slots_per_day], temperature[:slots_per_day])
    plt.xticks(time_ticks, time_labels)
    plt.title("Indoor temperature")
    plt.xlabel("Time")
    plt.ylabel("Temperature [°C]")

    # plt.figure(4)
    # plt.plot(time[:slots_per_day], battery[:slots_per_day] * 100)
    # plt.xticks(time_ticks, time_labels)
    # plt.title("Battery SOC")
    # plt.xlabel("Time")
    # plt.ylabel("SOC [%]")

    plt.figure(5)
    plt.bar(agent_ids[:4], self_consumption.sum(axis=0) / production.sum(axis=0) * 100, width)
    plt.title("Self consumption")
    plt.xlabel("Agent")
    plt.ylabel("%")

    fig, ax = plt.subplots()
    grid_power = power.sum(axis=1).reshape((-1, 96))
    ygrid, xgrid = map(lambda s: np.arange(s + 1) + 1, grid_power.shape)
    ax.pcolormesh(xgrid, ygrid, grid_power * 1e-3, cmap="magma")

    ax.set_frame_on(False)  # remove all spines
    ax.set_ylim(ygrid[-1], 1)
    ax.yaxis.set_ticks(ygrid[:-1])
    ax.yaxis.set_tick_params(length=0)

    # Color bar
    fig.subplots_adjust(bottom=0.25)
    # Create a new axis to contain the color bar
    # Values are:
    # (x coordinate of left border,
    #  y coordinate for bottom border,
    #  width,
    #  height)
    cbar_ax = fig.add_axes([0.3, 0.10, 0.4, 0.025])

    # Create a normalizer that goes from minimum to maximum temperature
    norm = mc.Normalize(grid_power.min() * 1e-3, grid_power.max() * 1e-3)

    # Create the colorbar and set it to horizontal
    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap="magma"),
        cax=cbar_ax,  # Pass the new axis
        orientation="horizontal"
    )

    # Remove tick marks
    cb.ax.xaxis.set_tick_params(size=0)

    # Add title and labels
    fig.text(0.5, 0.18, "Time slot", ha="center", va="center", fontsize=14)
    fig.text(0.07, 0.5, 'Day', ha="center", va="center", rotation="vertical", fontsize=14)
    fig.suptitle("Grid load", fontsize=20, y=0.95)
    # Set legend label
    cb.set_label("Power [kW]", size=12)

    # Show all figures
    plt.show()


def show_raw_load(path='data/data.json'):
    with open(path) as file:
        data = json.load(file)

        time = [datetime.fromisoformat(date) for date in data['time']]

        t = np.array(time)[[i for i in range(len(time)) if i % 15 == 0]]
        p = np.array(data['values'])
        p = np.append(p, p[-1]).reshape((-1, 15)).mean(axis=1).transpose()
        p = (p[1:] - p[0:-1])
        p /= p.max()

        plt.plot(t[1:], p)
        plt.show()


def show_clean_load(path='data/data.json'):
    with open(path) as file:
        data = json.load(file)

        time = np.array([datetime.fromisoformat(date) for date in data['time']])
        power = np.array(data['loads'])

        plt.plot(time, power)
        plt.show()


def normalize_data():
    with open('../data/data.json') as in_file, open('../data/profiles_1.json', 'w') as out_file:
        data = json.load(in_file)

        time = [date for t, date in enumerate(data['time']) if t % 15 == 0]

        p = np.array(data['values'])
        p = np.append(p, p[-1]).reshape((-1, 15)).mean(axis=1).transpose()
        p = (p[1:] - p[0:-1])   # np.diff
        p /= p.max()

        json.dump({'time': time[1:], 'values': p.tolist()}, out_file)


def expand_irradiation() -> None:

    with open('../data/irradiance_tmp.json') as file, open('../data/irradiation_daily.json', 'w') as out:
        data = json.load(file)
        avg = []
        t = []
        days = 0

        for k, v in data.items():
            month_nr = datetime.strptime(k, "%b").month
            num_days = monthrange(2021, month_nr)[1]

            t.append(days + num_days / 2)
            days += num_days

            month_avg = v * 1e3 / (num_days * 24)
            avg.append(month_avg)

        tt = [i for i in range(days)]
        ir = np.interp(tt, t, avg).tolist()

        json.dump({'day': tt, 'irradiance': ir}, out)


if __name__ == "__main__":
    # normalize_data()
    # show_clean_load('data/profiles_1.json')
    with open('../data/profiles.json') as file:
        data = json.load(file)

        plt.plot(data['time'], data['loads'])
        plt.plot(data['time'], data['pv'])
        plt.plot(data['time'], data['temperature'])

        plt.legend(["Load", "PV", "Temp"])
        plt.show()
    # print('Nothing to do...')


