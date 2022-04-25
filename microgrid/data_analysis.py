# Python Libraries
import re
import traceback
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
import config as cf
from agent import ActingAgent
import database as db


np.random.seed(42)


def analyse_community_output(agents: List[ActingAgent], time: List[datetime],
                             power: np.ndarray, cost: np.ndarray) -> None:

    slots_per_day = int(cf.MINUTES_PER_HOUR / cf.TIME_SLOT * cf.HOURS_PER_DAY)
    agent_ids = [a.id for a in agents]

    production = np.array(list(map(lambda a: a.pv.get_history(), agents))).transpose()
    self_consumption = (power < 0) * (production + power) + (power >= 0) * production

    print(f'Energy consumed: {power.sum(axis=0) * cf.TIME_SLOT / cf.MINUTES_PER_HOUR * 1e-3} kWh')
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


def make_costs_plot(df: pd.DataFrame, save_fig: bool = False) -> None:
    costs = df[['setting', 'agent', 'cost']].groupby(['setting', 'agent']).sum().groupby('setting').mean()
    costs['profiles'] = list(map(lambda s: get_profiles_type(s), costs.index.tolist()))
    costs['setting'] = list(map(lambda s: get_setting_type(s), costs.index.tolist()))
    costs = costs.pivot(index='setting', columns='profiles', values='cost')

    x = np.arange(len(costs.index))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, costs['heterogeneous'], width, label='Heterogeneous', color='#004079ff')
    rects2 = ax.bar(x + width / 2, costs['homogeneous'], width, label='Homogeneous', color='#51bcebff')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Cost [€]')
    ax.set_title('Average cost payed by an agent')
    ax.set_xticks(x, costs.index)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    if save_fig:
        plt.savefig(f'{cf.PLOTS_PATH}/costs_plot.svg', format='svg')


def make_day_plot(df: pd.DataFrame, save_fig: bool = False) -> None:
    timeslot_info = df[df['setting'] == '2-multi-agent-com-hetero'] \
        .pivot(index=['time'], columns=['agent'], values=['load', 'pv', 'temperature', 'heatpump'])

    time = np.arange(96)
    grid_price = (
            (cf.GRID_COST_AVG
             + cf.GRID_COST_AMPLITUDE
             * np.sin(time / 96 * 2 * np.pi * cf.HOURS_PER_DAY / cf.GRID_COST_PERIOD - cf.GRID_COST_PHASE)
             ) / cf.CENTS_PER_EURO  # from c€ to €
    )
    injection_price = np.zeros(grid_price.shape)
    injection_price[:] = grid_price.min()[None]
    p2p_price = (grid_price + injection_price) / 2

    fig, ax = plt.subplots(4, 1, figsize=(9, 6), sharex=True)
    fig.suptitle("Agent's state and decisions throughout the day")

    # Powers
    ax[0].plot(time, timeslot_info.loc[:, ('load', 0)], color='#004079ff')
    ax[0].plot(time, timeslot_info.loc[:, ('pv', 0)], color='#51bcebff')
    ax[0].set_yticks([0, 2, 4], [0.0, 2.0, 4.0])
    ax[0].set_ylabel("Power [kW]")
    ax[0].legend(["Load", "PV"], loc='upper right')

    # Prices
    ax[1].plot(time, grid_price, color='#004079ff')
    ax[1].plot(time, injection_price, color='#51bcebff')
    ax[1].plot(time, p2p_price, '--', color='#1d8dafff')
    ax[1].set_ylabel("Price [€]")
    ax[1].legend(["Offtake", "Injection", "P2P"], loc='upper right')

    # Heat pump
    ax[2].bar(time, timeslot_info.loc[:, ('heatpump', 0)], width=1.0, color='#004079ff')
    ax[2].set_ylabel("Power [kW]")
    ax[2].set_yticks([0, 1.5, 3], [0.0, 1.5, 3.0])

    # Temperature
    ax[3].plot(time, timeslot_info.loc[:, ('temperature', 0)], color='#004079ff')
    ax[3].set_ylabel("Temperature [°C]")
    ax[3].set_xticks([0, 24, 48, 72, 95], ["00:00", "06:00", "12:00", "18:00", "23:45"])
    ax[3].set_xlabel("Time")

    if save_fig:
        plt.savefig(f'{cf.PLOTS_PATH}/day_plot.svg', format='svg')


def make_learning_plot(df: pd.DataFrame, save_fig: bool) -> None:
    df['episode'] = df['episode'].astype(int)
    episodes = df[df['episode'] <= 1000] \
        .pivot(index=['episode'], columns=['setting', 'agent'], values=['reward'])

    fig, ax = plt.subplots()
    ax.plot(episodes.index, episodes.loc[:, ('reward', 'single-agent', 'tabular')], '-', color='#004079ff')
    # ax.plot(episodes.index, episodes.loc[:, ('reward', 'single-agent', 'dqn')], '-', color='#51bcebff')
    ax.plot(episodes.index, episodes.loc[:, ('reward', '2-multi-agent-no-com-homo', 'tabular')],
            '--', color='#004079ff')
    ax.plot(episodes.index, episodes.loc[:, ('reward', '2-multi-agent-no-com-hetero', 'tabular')],
            '--', color='#51bcebff')
    ax.plot(episodes.index, episodes.loc[:, ('reward', '2-multi-agent-com-homo', 'tabular')],
            '-.', color='#004079ff')
    ax.plot(episodes.index, episodes.loc[:, ('reward', '2-multi-agent-com-hetero', 'tabular')],
            '-.', color='#51bcebff')
    ax.plot(episodes.index, episodes.loc[:, ('reward', '5-multi-agent-com-hetero', 'tabular')],
            ':', color='#004079ff')
    ax.legend(['Single agent', '2 agent no-com homogeneous', '2 agent no-com heterogeneous',
               '2 agent com homogeneous', '2 agent com heterogeneous', '5 agent com heterogeneous'])
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward")

    if save_fig:
        plt.savefig(f'{cf.PLOTS_PATH}/learning.svg', format='svg')


def plot_tabular_comparison(save_figs: bool = False) -> None:

    con = db.get_connection()

    try:
        df = db.get_training_progress(con)
        # make_learning_plot(df, save_figs)

        df = db.get_validation_results(con)
        df[['load', 'pv']] = df[['load', 'pv']] * 1e-3
        df['time'] = df['time'].map(lambda t: t * 24)
        df['heatpump'] *= 1e-3

        make_costs_plot(df, save_figs)
        # make_day_plot(df, save_figs)

        plt.show()

    except:
        print(traceback.format_exc())
    finally:
        if con:
            con.close()


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


