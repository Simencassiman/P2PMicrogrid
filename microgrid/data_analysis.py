# Python Libraries
import re
from math import floor
from typing import List
import json
from datetime import datetime
from calendar import monthrange

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc  # For the legend
from matplotlib.cm import ScalarMappable    # For the legend

# Local modules
from config import TIME_SLOT, MINUTES_PER_HOUR, HOURS_PER_DAY
from agent import ActingAgent
import database as db


np.random.seed(42)


def analyse_community_output(agents: List[ActingAgent], time: List[datetime],
                             power: np.ndarray, cost: np.ndarray) -> None:

    slots_per_day = int(MINUTES_PER_HOUR / TIME_SLOT * HOURS_PER_DAY)
    agent_ids = [a.id for a in agents]

    production = np.array(list(map(lambda a: a.pv.get_history(), agents))).transpose()
    self_consumption = (power < 0) * (production + power) + (power >= 0) * production

    print(f'Energy consumed: {power.sum(axis=0) * TIME_SLOT / MINUTES_PER_HOUR * 1e-3} kWh')
    print(f'Cost a total of: {cost} €')

    # Create plots
    nr_ticks_factor = 16
    time = time[:slots_per_day]
    time_ticks = np.arange(int(slots_per_day / nr_ticks_factor)) * nr_ticks_factor
    time_labels = [t for i, t in enumerate(time) if i % nr_ticks_factor == 0]
    width = 0.35  # the width of the bars: can also be len(x) sequence

    plot_costs(agent_ids, cost, width)

    plot_selfconsumption(agent_ids, self_consumption, production, width)

    plot_grid_load(power)

    for i, agent in enumerate(agents):

        pv: np.array = np.array(agent.pv.get_history())
        temperature = np.array(agent.heating.get_history())
        hp = np.array(agent.heating._power_history)

        plt.figure(3*i + 4)
        plt.plot(time[:slots_per_day], power[:slots_per_day, i] * 1e-3)
        plt.plot(time[:slots_per_day], pv[:slots_per_day] * 1e-3)
        plt.xticks(time_ticks, time_labels)
        plt.title(f"Agent profiles (agent {i})")
        plt.xlabel("Time")
        plt.ylabel("Power [kW]")
        plt.legend(['Loads', 'PV'])

        plt.figure(3*i + 5)
        plt.plot(time[:slots_per_day], temperature[:slots_per_day])
        plt.xticks(time_ticks, time_labels)
        plt.title(f"Indoor temperature (agent {i})")
        plt.xlabel("Time")
        plt.ylabel("Temperature [°C]")

        plt.figure(3*i + 6)
        plt.plot(time[:slots_per_day], hp[:slots_per_day] * 100)
        plt.xticks(time_ticks, time_labels)
        plt.title(f"Heat Pump Power (agent {i})")
        plt.xlabel("Time")
        plt.ylabel("Power [W]")

    # Show all figures
    plt.show()


def plot_costs(agent_ids: List, costs: np.ndarray, width: float) -> None:
    plt.figure()
    plt.bar(agent_ids, costs, width, label='Volume')
    plt.title("Electricity costs")
    plt.xticks(agent_ids, agent_ids)
    plt.xlabel("Agent")
    plt.ylabel("Cost [€]")
    plt.legend()


def plot_selfconsumption(agent_ids: List, self_consumption: np.ndarray, production: np.ndarray, width: float) -> None:
    plt.figure()
    plt.bar(agent_ids, self_consumption.sum(axis=0) / production.sum(axis=0) * 100, width)
    plt.xticks(agent_ids, agent_ids)
    plt.title("Self consumption")
    plt.xlabel("Agent")
    plt.ylabel("%")


def plot_grid_load(power: np.ndarray) -> None:
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


def get_profiles_type(setting: str) -> str:
    if setting == 'single-agent':
        s = 'homo'
    else:
        s = re.match(r'.*-([a-z]+)$', setting).groups()[0]

    return s + 'geneous'

def get_setting_type(setting: str) -> str:
    return re.match(r'([0-9]-){0,1}([a-z-]+)-[a-z]+$', setting).groups()[1]

def plot_tabular_comparison() -> None:
    # groups = [[23, 135, 3], [123, 500, 1]]
    # group_labels = ['views', 'orders']
    # df = pd.DataFrame(groups, index=group_labels).T

    # Convert data to pandas DataFrame.
    con = db.get_connection()
    try:
        df = db.get_validation_results(con)
    except:
        pass
    finally:
        if con:
            con.close()

    df[['load', 'pv']] = df[['load', 'pv']] * 1e-3
    df['time'] = df['time'].map(lambda t: t * 24)
    df['heatpump'] *= 1e-3

    costs = df[['setting', 'agent', 'cost']].groupby(['setting', 'agent']).sum().groupby('setting').mean()
    costs['profiles'] = list(map(lambda s: get_profiles_type(s), costs.index.tolist()))
    costs['setting'] = list(map(lambda s: get_setting_type(s), costs.index.tolist()))
    costs = costs.pivot(index='setting', columns='profiles', values='cost')

    costs.plot.bar()
    plt.xticks(rotation=0)
    plt.ylabel("Cost [€]")
    plt.title("Average price payed")

    df[df['setting'] == '2-multi-agent-com-hetero']\
        .pivot(index='time', columns='agent', values=['load', 'pv'])\
        .plot.line()
    plt.title("Heterogeneous")
    plt.ylabel("Power [kW]")

    df[df['setting'] == '2-multi-agent-com-hetero'] \
        .pivot(index='time', columns='agent', values=['temperature']) \
        .plot.line()
    plt.title("Indoor temperatures")
    plt.ylabel("Temperature [°C]")

    plt.figure()
    df[df['setting'] == '2-multi-agent-com-hetero'][df['agent'] == 0]['heatpump'].plot(kind='bar', width=1)
    plt.gca().axes.get_xaxis().set_ticks([int(t / 24 * 96) for t in range(24) if t % 5 == 0])
    plt.gca().axes.get_xaxis().set_ticklabels([t for t in range(24) if t % 5 == 0])
    plt.ylabel("Power [kW]")
    plt.title("Heat pump activity")

    plt.show()
    # print(df)
    #
    # # Plot.
    # pd.concat(
    #     [df.mean().rename('average'), df.min().rename('min'),
    #      df.max().rename('max')],
    #     axis=1).plot.bar()
    #
    # plt.show()

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
    plot_tabular_comparison()


