# Python Libraries
import re
from typing import List, Dict
import json
from datetime import datetime
from calendar import monthrange

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mc  # For the legend
from matplotlib.cm import ScalarMappable    # For the legend
from matplotlib.ticker import MultipleLocator

# Local modules
import config as cf
import setup
from agent import ActingAgent
import database as db
import dataset as ds


# Config
np.random.seed(setup.seed)

# Text document settings
# primary_color = '#000'
# secondary_color = '#ccc'
# neutral_color = '#777'
# base_color = '#000'

title_fontsize = 9
axis_label_fontsize = 8
axis_ticks_fontsize = 6

# Poster settings
primary_color = '#004079ff'
secondary_color = '#51bcebff'
tertiary_color = '#1d8dafff'
neutral_color = '#777'
base_color = '#2f4d5dff'
# title_fontsize = 16
# axis_label_fontsize = 12
# Figure sizes should approximately be doubled

figure_format = 'eps'


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


def show_test_profiles(save_figs: bool = False) -> None:
    env_df, agent_dfs = ds.get_test_data()

    time = np.arange(96)
    df = pd.concat([env_df, agent_dfs[0]], axis=1).loc[env_df['day'] == 8, :]

    # Profiles plot
    fig, ax = plt.subplots(figsize=(3, 2))
    fig.suptitle("Example of normalized load and PV", fontsize=title_fontsize)
    fig.subplots_adjust(bottom=0.2, left=0.18)

    ax.plot(time, df['load'], 'k-')
    ax.plot(time, df['pv'], 'k:')

    ax.set_xticks([0, 24, 48, 72, 95], ["00:00", "06:00", "12:00", "18:00", "23:45"], fontsize=axis_ticks_fontsize)
    ax.set_xlabel("Time", fontsize=axis_label_fontsize)
    ax.set_ylabel("Power [-]", fontsize=axis_label_fontsize)
    ax.set_yticks([0.0, 0.5, 1.0], [0.0, 0.5, 1.0], fontsize=axis_ticks_fontsize)
    ax.legend(['Load', 'PV'], labelcolor=base_color, fontsize=axis_label_fontsize,
              bbox_to_anchor=(0., .2), loc="lower left")

    if save_figs:
        fig.savefig(f'{cf.FIGURES_PATH}/example_profiles.{figure_format}', format=figure_format)

    # Temperature plot
    fig, ax = plt.subplots(figsize=(3, 2))
    fig.suptitle("Example of outdoor temperature evolution", fontsize=title_fontsize)
    fig.subplots_adjust(bottom=0.2, left=0.15)

    ax.plot(time, df['temperature'], 'k-')

    ax.set_xticks([0, 24, 48, 72, 95], ["00:00", "06:00", "12:00", "18:00", "23:45"], fontsize=axis_ticks_fontsize)
    ax.set_xlabel("Time", fontsize=axis_label_fontsize)
    ax.set_yticks([6, 10, 14, 18], [6, 10, 14, 18], fontsize=axis_ticks_fontsize)
    ax.set_ylabel("Temperature [°C]", fontsize=axis_label_fontsize)

    if save_figs:
        fig.savefig(f'{cf.FIGURES_PATH}/example_outdoor_temperature.{figure_format}', format=figure_format)


def show_prices(save_fig: bool = False) -> None:
    time = np.arange(96)
    grid_price = (
            (setup.GRID_COST_AVG
             + setup.GRID_COST_AMPLITUDE
             * np.sin(time / 96 * 2 * np.pi * setup.HOURS_PER_DAY / setup.GRID_COST_PERIOD - setup.GRID_COST_PHASE)
             ) / setup.CENTS_PER_EURO  # from c€ to €
    )
    injection_price = np.zeros(grid_price.shape)
    injection_price[:] = grid_price.min()[None]
    p2p_price = (grid_price + injection_price) / 2

    fig, ax = plt.subplots(figsize=(4.5, 1.5))
    fig.suptitle("Electricity price tariffs", fontsize=title_fontsize)
    fig.subplots_adjust(bottom=0.26, left=0.15, right=0.77)

    ax.plot(time, grid_price, color=primary_color)
    ax.plot(time, injection_price, color=secondary_color)
    ax.plot(time, p2p_price, '--', color=primary_color)

    ax.set_xticks([0, 24, 48, 72, 95], ["00:00", "06:00", "12:00", "18:00", "23:45"], fontsize=axis_ticks_fontsize)
    ax.set_xlabel("Time", fontsize=axis_label_fontsize)
    ax.set_yticks([0.07, 0.12, 0.17], [0.07, 0.12, 0.17], fontsize=axis_ticks_fontsize)
    ax.set_ylabel("Price [€/kWh]", color=base_color, fontsize=axis_label_fontsize)
    ax.legend(["Offtake", "Injection", "P2P"], labelcolor=base_color, fontsize=axis_label_fontsize,
              bbox_to_anchor=(1., .2), loc="lower left")

    if save_fig:
        fig.savefig(f'{cf.FIGURES_PATH}/example_prices.{figure_format}', format=figure_format)


def analyse_community_output(agents: List[ActingAgent], time: List[datetime],
                             power: np.ndarray, cost: np.ndarray) -> None:

    slots_per_day = int(setup.MINUTES_PER_HOUR / setup.TIME_SLOT * setup.HOURS_PER_DAY)
    agent_ids = [a.id for a in agents]

    production = np.array(list(map(lambda a: a.pv.get_history(), agents))).transpose()
    self_consumption = (power < 0) * (production + power) + (power >= 0) * production

    print(f'Energy consumed: {power.sum(axis=0) * setup.TIME_SLOT / setup.MINUTES_PER_HOUR * 1e-3} kWh')
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
    return re.match(r'([0-9]-){0,1}([a-z-]+)-[0-9\-a-z]+$', setting).groups()[1]


def get_rounds(setting: str) -> str:
    return re.match(r'([0-9]-){0,1}([a-z-]+)-rounds-([0-9])-[a-z]+$', setting).groups()[2]


def make_homogeneous_costs_plot(df: pd.DataFrame, save_fig: bool = False) -> None:
    settings = ['2-multi-agent-com-rounds-1-homo', '2-multi-agent-no-com-homo', 'single-agent']
    df = df[df['setting'].isin(settings)]
    df_baselines = df.loc[df['implementation'].isin(['rule-based', 'semi-intelligent']),
                          ['implementation', 'day', 'cost']]\
        .groupby(['implementation', 'day']).sum().groupby('implementation').mean()
    df = df[df['implementation'].isin(['tabular'])]

    costs = df[['setting', 'agent', 'day', 'cost']].groupby(['setting', 'agent', 'day']).sum().groupby('setting').mean()
    costs['profiles'] = list(map(lambda s: get_profiles_type(s), costs.index.tolist()))
    costs['setting'] = list(map(lambda s: get_setting_type(s), costs.index.tolist()))
    costs = costs.pivot(index=['setting'], columns=['profiles'], values=['cost'])

    x = np.arange(len(costs.index))  # the label locations

    fig, ax = plt.subplots(figsize=(2.5, 3))

    rects = ax.bar(x, costs.loc[:, ('cost', 'homogeneous')], width=0.5, label='Homogeneous', color=secondary_color)
    ax.hlines(y=df_baselines['cost'], xmin=1.5, xmax=2.4, color=neutral_color, linestyle='--')
    ax.text(1.3, 0.8, 'Semi-intelligent', color=base_color, fontsize=axis_label_fontsize)
    ax.text(1.48, 1.55, 'Rule-based', color=base_color, fontsize=axis_label_fontsize)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title('Average daily cost paid by an agent', color=base_color, fontsize=title_fontsize, loc='right')
    ax.set_ylabel('Cost [€]', color=base_color, fontsize=axis_label_fontsize)
    ax.set_xticks(x, ['com', 'no-com', 'single'], fontsize=axis_label_fontsize)
    ax.set_ylim([0, 2])
    ax.set_yticks([0., .5, 1., 1.5, 2.], [0.0, 0.5, 1.0, 1.5, 2.0], fontsize=axis_ticks_fontsize)
    ax.bar_label(rects, labels=[f'{x:,.2f}' for x in rects.datavalues], padding=2, color=base_color,
                 fontsize=axis_ticks_fontsize)

    # Additional coloring
    ax.spines['bottom'].set_color(base_color)
    ax.spines['top'].set_color(base_color)
    ax.spines['right'].set_color(base_color)
    ax.spines['left'].set_color(base_color)
    ax.tick_params(axis='x', colors=base_color)
    ax.tick_params(axis='y', colors=base_color)

    fig.tight_layout()

    if save_fig:
        plt.savefig(f'{cf.FIGURES_PATH}/costs_plot_homogeneous.{figure_format}', format=figure_format)


def make_heterogeneous_costs_plot(df: pd.DataFrame, save_fig: bool = False) -> None:
    settings = ['2-multi-agent-com-rounds-1-hetero', '2-multi-agent-com-rounds-1-homo',
                '2-multi-agent-no-com-hetero', '2-multi-agent-no-com-homo',
                'single-agent']
    df = df[df['setting'].isin(settings)]
    df_baselines = df.loc[df['implementation'].isin(['rule-based', 'semi-intelligent']),
                          ['implementation', 'day', 'cost']]\
        .groupby(['implementation', 'day']).sum().groupby('implementation').mean()
    df = df[df['implementation'].isin(['tabular'])]

    costs = df[['setting', 'agent', 'day', 'cost']].groupby(['setting', 'agent', 'day']).sum().groupby('setting').mean()
    costs['profiles'] = list(map(lambda s: get_profiles_type(s), costs.index.tolist()))
    costs['setting'] = list(map(lambda s: get_setting_type(s), costs.index.tolist()))
    costs = costs.pivot(index='setting', columns='profiles', values='cost')

    x = np.arange(len(costs.index))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(3.5, 3.))
    fig.subplots_adjust(bottom=0.25)

    rects1 = ax.bar(x - width / 2, costs['heterogeneous'], width, label='Heterogeneous', color=primary_color)
    rects2 = ax.bar(x + width / 2, costs['homogeneous'], width, label='Homogeneous', color=secondary_color)
    ax.hlines(y=df_baselines['cost'], xmin=1.5, xmax=2.4, color=neutral_color, linestyle='--')
    ax.text(1.4, .75, 'Semi-intelligent', color=base_color, fontsize=axis_label_fontsize)
    ax.text(1.48, 1.55, 'Rule-based', color=base_color, fontsize=axis_label_fontsize)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title('Average daily cost paid by an agent', color=base_color, fontsize=title_fontsize)
    ax.set_ylabel('Cost [€]', color=base_color, fontsize=axis_label_fontsize)
    ax.set_xticks(x, ['com', 'no-com', 'single'], fontsize=axis_label_fontsize)
    ax.set_ylim([0, 2])
    ax.set_yticks([0., .5, 1., 1.5, 2.], [0.0, 0.5, 1.0, 1.5, 2.0], fontsize=axis_ticks_fontsize)
    ax.bar_label(rects1, labels=[f'{x:,.2f}' for x in rects1.datavalues], padding=2, color=base_color,
                 fontsize=axis_ticks_fontsize)
    ax.bar_label(rects2, labels=[f'{x:,.2f}' for x in rects2.datavalues], padding=2, color=base_color,
                 fontsize=axis_ticks_fontsize)

    # Additional coloring
    ax.spines['bottom'].set_color(base_color)
    ax.spines['top'].set_color(base_color)
    ax.spines['right'].set_color(base_color)
    ax.spines['left'].set_color(base_color)
    ax.tick_params(axis='x', colors=base_color)
    ax.tick_params(axis='y', colors=base_color)
    ax.legend(labelcolor=base_color, bbox_to_anchor=(0.5, -.14), loc='upper center', fontsize=axis_label_fontsize)

    if save_fig:
        plt.savefig(f'{cf.FIGURES_PATH}/costs_plot_heterogeneous.{figure_format}', format=figure_format)


def make_baseline_day_plot(df: pd.DataFrame, baseline: str, save_fig: bool = False) -> None:

    timeslot_info = df[(df['setting'] == 'single-agent') & (df['implementation'] == baseline) & (df['day'] == 8)] \
        .pivot(index=['time'], columns=['agent'], values=['load', 'pv', 'temperature', 'heatpump', 'cost'])

    time = np.arange(96)
    grid_price = (
            (setup.GRID_COST_AVG
             + setup.GRID_COST_AMPLITUDE
             * np.sin(time / 96 * 2 * np.pi * setup.HOURS_PER_DAY / setup.GRID_COST_PERIOD - setup.GRID_COST_PHASE)
             ) / setup.CENTS_PER_EURO  # from c€ to €
    )
    injection_price = np.zeros(grid_price.shape)
    injection_price[:] = grid_price.min()[None]
    p2p_price = (grid_price + injection_price) / 2

    fig, ax = plt.subplots(4, 1, figsize=(5.5, 3.5), sharex=True)
    fig.suptitle(f"Agent's state and decisions throughout the day",
                 color=base_color, fontsize=title_fontsize)
    fig.subplots_adjust(left=0.1, right=0.72, hspace=0.4)

    # Powers
    net_power = timeslot_info.loc[:, ('load', 0)] \
                - timeslot_info.loc[:, ('pv', 0)] \
                + timeslot_info.loc[:, ('heatpump', 0)]
    ax[0].plot(time, timeslot_info.loc[:, ('load', 0)], color=secondary_color)
    ax[0].plot(time, timeslot_info.loc[:, ('pv', 0)], ':', color=secondary_color)
    ax[0].plot(time, net_power, color=primary_color)

    ax[0].set_title("a)", fontsize=axis_label_fontsize, loc='left', color=base_color)
    ax[0].set_yticks([-4, 0, 4], [-4.0, 0.0, 4.0], fontsize=axis_ticks_fontsize)
    ax[0].set_ylabel("Power [kW]", color=base_color, fontsize=axis_label_fontsize)
    ax[0].legend(["Base Load", "PV", "Net Consumption"], labelcolor=base_color, fontsize=axis_label_fontsize,
                 bbox_to_anchor=(1.02, -.125), loc="lower left")

    # Prices
    ax12 = ax[1].twinx()
    ax12.plot(time, grid_price, color=secondary_color)
    ax12.plot(time, injection_price, ':', color=secondary_color)
    ax12.plot(time, p2p_price, '--', color=secondary_color)
    ax[1].plot(time, timeslot_info.loc[:, ('cost', 0)], color=primary_color)

    ax[1].set_title("b)", fontsize=axis_label_fontsize, loc='left', pad=-.1, color=base_color)
    ax[1].set_yticks([-.1, 0, .1], [-0.1, 0.0, 0.1], fontsize=axis_ticks_fontsize)
    ax12.set_yticks([0.07, 0.12, 0.17], [0.07, 0.12, 0.17], color=secondary_color, fontsize=axis_ticks_fontsize)
    ax[1].set_ylabel("Cost [€]", color=primary_color, fontsize=axis_label_fontsize)
    ax[1].set_zorder(1)
    ax[1].set_frame_on(False)
    ax12.set_ylabel("Price [€/kWh]", color=secondary_color, fontsize=axis_label_fontsize)
    ax12.legend(["Offtake", "Injection", "P2P"], labelcolor=base_color, fontsize=axis_label_fontsize,
                bbox_to_anchor=(1.15, -.125), loc="lower left")

    # Heat pump
    ax[2].bar(time, timeslot_info.loc[:, ('heatpump', 0)], width=1.0, color=primary_color)
    ax[2].set_title("c)", fontsize=axis_label_fontsize, loc='left', pad=-.001, color=base_color)
    ax[2].set_yticks([0, 1.5, 3], [0.0, 1.5, 3.0], fontsize=axis_ticks_fontsize)
    ax[2].set_ylabel("HP [kW]", color=base_color, fontsize=axis_label_fontsize, loc='top')

    # Temperature
    ax[3].plot(time, timeslot_info.loc[:, ('temperature', 0)], color=primary_color)
    ax[3].set_title("d)", fontsize=axis_label_fontsize, loc='left', pad=-.1, color=base_color)
    ax[3].hlines(y=[20, 22], xmin=0, xmax=96, color=neutral_color, linestyle='--', linewidths=0.8)
    ax[3].set_yticks([20, 22], [20, 22], fontsize=axis_ticks_fontsize)
    ax[3].set_ylabel("Temperature [°C]", color=base_color, fontsize=axis_label_fontsize)
    ax[2].xaxis.set_minor_locator(MultipleLocator(1))
    ax[3].set_xticks([i for i in range(96 + 1) if i % 4 == 0],
                     [re.sub(' ', '0', f'{i/4:2.0f}:00' if i % 16 == 0 else '') for i in range(96 + 1) if i % 4 == 0],
                     fontsize=axis_ticks_fontsize)
    ax[3].set_xlabel("Time", color=base_color, fontsize=axis_label_fontsize)

    # Additional coloring
    for i in range(len(ax)):
        ax[i].spines['bottom'].set_color(base_color)
        ax[i].spines['top'].set_color(base_color)
        ax[i].spines['right'].set_color(base_color)
        ax[i].spines['left'].set_color(base_color)
        ax[i].tick_params(axis='x', colors=base_color)
        ax[i].tick_params(axis='y', colors=base_color)

    if save_fig:
        plt.savefig(f'{cf.FIGURES_PATH}/day_plot_{baseline}.{figure_format}', format=figure_format)


def make_day_plot(df: pd.DataFrame, homogeneous: bool = False, save_fig: bool = False) -> None:
    setting = 'homo' if homogeneous else 'hetero'

    timeslot_info = df[(df['setting'] == f'2-multi-agent-com-rounds-1-{setting}') & (df['day'] == 8)] \
        .pivot(index=['time'], columns=['agent'], values=['load', 'pv', 'temperature', 'heatpump', 'cost'])

    time = np.arange(96)
    grid_price = (
            (setup.GRID_COST_AVG
             + setup.GRID_COST_AMPLITUDE
             * np.sin(time / 96 * 2 * np.pi * setup.HOURS_PER_DAY / setup.GRID_COST_PERIOD - setup.GRID_COST_PHASE)
             ) / setup.CENTS_PER_EURO  # from c€ to €
    )
    injection_price = np.zeros(grid_price.shape)
    injection_price[:] = grid_price.min()[None]
    p2p_price = (grid_price + injection_price) / 2

    fig, ax = plt.subplots(4, 1, figsize=(5.5, 3.5), sharex=True)
    fig.suptitle(f"Agent's state and decisions throughout the day ({setting}geneous)",
                 color=base_color, fontsize=title_fontsize)
    fig.subplots_adjust(left=0.1, right=0.72, hspace=0.4)

    # Powers
    net_power = timeslot_info.loc[:, ('load', 0)] \
                - timeslot_info.loc[:, ('pv', 0)] \
                + timeslot_info.loc[:, ('heatpump', 0)]
    ax[0].plot(time, timeslot_info.loc[:, ('load', 0)], color=secondary_color)
    ax[0].plot(time, timeslot_info.loc[:, ('pv', 0)], ':', color=secondary_color)
    ax[0].plot(time, net_power, color=primary_color)

    ax[0].set_title("a)", fontsize=axis_label_fontsize, loc='left', color=base_color)
    ax[0].set_yticks([-4, 0, 4], [-4.0, 0.0, 4.0], fontsize=axis_ticks_fontsize)
    ax[0].set_ylabel("Power [kW]", color=base_color, fontsize=axis_label_fontsize)
    ax[0].legend(["Base Load", "PV", "Net Consumption"], labelcolor=base_color, fontsize=axis_label_fontsize,
                 bbox_to_anchor=(1.02, -.125), loc="lower left")

    # Prices
    ax12 = ax[1].twinx()
    ax[1].plot(time, timeslot_info.loc[:, ('cost', 0)], color=primary_color)
    ax12.plot(time, grid_price, color=secondary_color)
    ax12.plot(time, injection_price, ':', color=secondary_color)
    ax12.plot(time, p2p_price, '--', color=secondary_color)

    ax[1].set_title("b)", fontsize=axis_label_fontsize, loc='left', pad=-.1, color=base_color)
    ax[1].set_yticks([-.1, 0, .1], [-0.1, 0.0, 0.1], fontsize=axis_ticks_fontsize)
    ax12.set_yticks([0.07, 0.12, 0.17], [0.07, 0.12, 0.17], color=secondary_color, fontsize=axis_ticks_fontsize)
    ax[1].set_zorder(1)
    ax[1].set_frame_on(False)
    ax[1].set_ylabel("Cost [€]", color=primary_color, fontsize=axis_label_fontsize)
    ax12.set_ylabel("Price [€/kWh]", color=secondary_color, fontsize=axis_label_fontsize)
    ax12.legend(["Offtake", "Injection", "P2P"], labelcolor=base_color, fontsize=axis_label_fontsize,
                bbox_to_anchor=(1.15, -.125), loc="lower left")

    # Heat pump
    ax[2].bar(time, timeslot_info.loc[:, ('heatpump', 0)], width=1.0, color=primary_color)
    ax[2].set_title("c)", fontsize=axis_label_fontsize, loc='left', pad=-.001, color=base_color)
    ax[2].set_yticks([0, 1.5, 3], [0.0, 1.5, 3.0], fontsize=axis_ticks_fontsize)
    ax[2].set_ylabel("HP [kW]", color=base_color, fontsize=axis_label_fontsize, loc='top')

    # Temperature
    ax[3].plot(time, timeslot_info.loc[:, ('temperature', 0)], color=primary_color)
    ax[3].set_title("d)", fontsize=axis_label_fontsize, loc='left', pad=-.1, color=base_color)
    ax[3].set_yticks([20, 22], [20, 22], fontsize=axis_ticks_fontsize)
    ax[3].set_ylabel("Temperature [°C]", color=base_color, fontsize=axis_label_fontsize)
    ax[3].set_xticks([i for i in range(96 + 1) if i % 4 == 0],
                     [re.sub(' ', '0', f'{i / 4:2.0f}:00' if i % 16 == 0 else '') for i in range(96 + 1) if i % 4 == 0],
                     fontsize=axis_ticks_fontsize)
    ax[2].xaxis.set_minor_locator(MultipleLocator(1))
    ax[3].set_xlabel("Time", color=base_color, fontsize=axis_label_fontsize)
    ax[3].hlines(y=[20, 22], xmin=0, xmax=96, color=neutral_color, linestyle='--', linewidths=0.8)

    # Additional coloring
    for i in range(len(ax)):
        ax[i].spines['bottom'].set_color(base_color)
        ax[i].spines['top'].set_color(base_color)
        ax[i].spines['right'].set_color(base_color)
        ax[i].spines['left'].set_color(base_color)
        ax[i].tick_params(axis='x', colors=base_color)
        ax[i].tick_params(axis='y', colors=base_color)

    if save_fig:
        plt.savefig(f'{cf.FIGURES_PATH}/day_plot_{setting}.{figure_format}', format=figure_format)


def make_decisions_comparison_plot(df: pd.DataFrame, homogeneous: bool = False, save_fig: bool = False) -> None:
    setting = 'homo' if homogeneous else 'hetero'

    df = df[(df['setting'].isin([f'2-multi-agent-com-rounds-1-{setting}', f'2-multi-agent-no-com-{setting}']))
            & (df['day'] == 8)]
    timeslot_info = df.pivot(index=['time'], columns=['setting', 'agent'],
                             values=['load', 'pv', 'temperature', 'heatpump'])
    time = np.arange(96)
    grid_price = (
            (setup.GRID_COST_AVG
             + setup.GRID_COST_AMPLITUDE
             * np.sin(time / 96 * 2 * np.pi * setup.HOURS_PER_DAY / setup.GRID_COST_PERIOD - setup.GRID_COST_PHASE)
             ) / setup.CENTS_PER_EURO  # from c€ to €
    )
    injection_price = np.zeros(grid_price.shape)
    injection_price[:] = grid_price.min()[None]
    p2p_price = (grid_price + injection_price) / 2

    # Make plot
    fig, ax = plt.subplots(6, 1, figsize=(5.5, 5), sharex=True)
    fig.suptitle(f"Agent's state and decisions throughout the day ({setting}geneous)",
                 color=base_color, fontsize=title_fontsize)
    fig.subplots_adjust(left=0.1, right=0.7, hspace=0.4)

    # Powers
    ax[0].plot(time, timeslot_info.loc[:, ('load', f'2-multi-agent-com-rounds-1-{setting}', 0)],
               color=primary_color)
    ax[0].plot(time, timeslot_info.loc[:, ('load', f'2-multi-agent-com-rounds-1-{setting}', 1)],
               color=secondary_color)
    ax[0].plot(time, timeslot_info.loc[:, ('pv', f'2-multi-agent-com-rounds-1-{setting}', 0)], '--',
               color=primary_color)
    ax[0].plot(time, timeslot_info.loc[:, ('pv', f'2-multi-agent-com-rounds-1-{setting}', 1)], '--',
               color=secondary_color)
    ax[0].set_yticks([-4, 0, 4], [-4.0, 0.0, 4.0], fontsize=axis_ticks_fontsize)
    ax[0].set_ylabel("Power [kW]", color=base_color, fontsize=axis_label_fontsize)
    ax[0].legend(["Base Load Agent 0", "Base Load Agent 1", "PV Agent 0", "PV Agent 1"], labelcolor=base_color,
                 fontsize=axis_label_fontsize, bbox_to_anchor=(1.02, -.3), loc="lower left")

    # Prices
    ax[1].plot(time, grid_price, color=primary_color)
    ax[1].plot(time, injection_price, color=secondary_color)
    ax[1].plot(time, p2p_price, '--', color=primary_color)
    ax[1].set_yticks([0.07, 0.12, 0.17], [0.07, 0.12, 0.17], fontsize=axis_ticks_fontsize)
    ax[1].set_ylabel("Price [€/kWh]", color=base_color, fontsize=axis_label_fontsize)
    ax[1].legend(["Offtake", "Injection", "P2P"], labelcolor=base_color, fontsize=axis_label_fontsize,
                 bbox_to_anchor=(1.02, -0.15), loc="lower left")

    # Heat pump
    width = 0.4

    # Agent 0
    ax[2].set_title("agent 0", fontsize=axis_label_fontsize, loc='right', pad=-.1, color=base_color)
    ax[2].bar(time - width / 2, timeslot_info.loc[:, ('heatpump', f'2-multi-agent-com-rounds-1-{setting}', 0)],
              label='Communication', width=width, color=primary_color)
    ax[2].bar(time + width / 2, timeslot_info.loc[:, ('heatpump', f'2-multi-agent-no-com-{setting}', 0)],
              label='No communication', width=width, color=secondary_color)
    ax[2].set_yticks([0, 1.5, 3], [0.0, 1.5, 3.0], fontsize=axis_ticks_fontsize)
    ax[2].set_ylabel("HP [kW]", color=base_color, fontsize=axis_label_fontsize)
    ax[2].legend(["Communication", "No communication"], labelcolor=base_color,
                 fontsize=axis_label_fontsize, bbox_to_anchor=(1.02, .2), loc="upper left")

    # Agent 1
    ax[3].set_title("agent 1", fontsize=axis_label_fontsize, loc='right', pad=-.1, color=base_color)
    ax[3].bar(time - width / 2, timeslot_info.loc[:, ('heatpump', f'2-multi-agent-com-rounds-1-{setting}', 1)],
              label='Communication', width=width, color=primary_color)
    ax[3].bar(time + width / 2, timeslot_info.loc[:, ('heatpump', f'2-multi-agent-no-com-{setting}', 1)],
              label='No communication', width=width, color=secondary_color)
    ax[3].set_yticks([0, 1.5, 3], [0.0, 1.5, 3.0], fontsize=axis_ticks_fontsize)
    ax[3].set_ylabel("HP [kW]", color=base_color, fontsize=axis_label_fontsize)

    # Temperature
    # Agent 0
    ax[4].set_title("agent 0", fontsize=axis_label_fontsize, loc='right', pad=-.1, color=base_color)
    ax[4].plot(time, timeslot_info.loc[:, ('temperature', f'2-multi-agent-com-rounds-1-{setting}', 0)],
               color=primary_color)
    ax[4].plot(time, timeslot_info.loc[:, ('temperature', f'2-multi-agent-no-com-{setting}', 0)],
               color=secondary_color)
    ax[4].set_yticks([20, 22], [20, 22], fontsize=axis_ticks_fontsize)
    ax[4].hlines(y=[20, 22], xmin=0, xmax=96, color=neutral_color, linestyle='--', linewidths=0.8)
    ax[4].legend(["Communication", "No communication"], labelcolor=base_color,
                 fontsize=axis_label_fontsize, bbox_to_anchor=(1.02, .2), loc="upper left")

    # Agent 1
    ax[5].set_title("agent 1", fontsize=axis_label_fontsize, loc='right', pad=-.3, color=base_color)
    ax[5].plot(time, timeslot_info.loc[:, ('temperature', f'2-multi-agent-com-rounds-1-{setting}', 1)],
               color=primary_color)
    ax[5].plot(time, timeslot_info.loc[:, ('temperature', f'2-multi-agent-no-com-{setting}', 1)],
               color=secondary_color)
    ax[5].set_ylabel("  Temperature [°C]", loc='bottom', color=base_color, fontsize=axis_label_fontsize)
    ax[5].set_xticks([i for i in range(96 + 1) if i % 4 == 0],
                     [re.sub(' ', '0', f'{i / 4:2.0f}:00' if i % 16 == 0 else '') for i in range(96 + 1) if i % 4 == 0],
                     fontsize=axis_ticks_fontsize)
    ax[5].set_xlabel("Time", color=base_color, fontsize=axis_label_fontsize)
    ax[5].set_yticks([20, 22], [20, 22], fontsize=axis_ticks_fontsize)
    ax[5].hlines(y=[20, 22], xmin=0, xmax=96, color=neutral_color, linestyle='--', linewidths=0.8)
    ax[5].xaxis.set_minor_locator(MultipleLocator(1))

    # Additional coloring
    for i in range(len(ax)):
        ax[i].spines['bottom'].set_color(base_color)
        ax[i].spines['top'].set_color(base_color)
        ax[i].spines['right'].set_color(base_color)
        ax[i].spines['left'].set_color(base_color)
        ax[i].tick_params(axis='x', colors=base_color)
        ax[i].tick_params(axis='y', colors=base_color)

    if save_fig:
        plt.savefig(f'{cf.FIGURES_PATH}/decisions_plot_{setting}.{figure_format}', format=figure_format)


def make_homogeneous_learning_plot(df: pd.DataFrame, save_fig: bool) -> None:
    df.loc[df['episode'] == 999, 'episode'] = 1000
    episodes = df.pivot(index=['episode'], columns=['setting', 'agent'], values=['reward'])

    fig, ax = plt.subplots(figsize=(4.5, 3))
    fig.suptitle("Running rewards during training", color=base_color, fontsize=title_fontsize)
    fig.subplots_adjust(bottom=0.15, top=0.7)

    ax.plot(episodes.index, episodes.loc[:, ('reward', 'single-agent', 'tabular')], '-', color=primary_color)
    ax.plot(episodes.index, episodes.loc[:, ('reward', '2-multi-agent-no-com-homo', 'tabular')],
            '--', color=primary_color)
    ax.plot(episodes.index, episodes.loc[:, ('reward', '2-multi-agent-com-rounds-1-homo', 'tabular')],
            '-.', color=primary_color)
    ax.legend(['Single agent', '2 agent no-com', '2 agent com'],
              labelcolor=base_color, fontsize=axis_label_fontsize,
              bbox_to_anchor=(.3, 1.), loc='lower left')
    ax.set_xticks([0, 250, 500, 750, 1000], [0, 250, 500, 750, 1000], fontsize=axis_ticks_fontsize)
    ax.set_xlabel("Episodes", color=base_color, fontsize=axis_label_fontsize)
    ax.set_yticks([-5000, -10000, -15000, -20000, -25000, -30000], [-5000, -10000, -15000, -20000, -25000, -30000],
                  fontsize=axis_ticks_fontsize)
    ax.set_ylabel("Reward", color=base_color, fontsize=axis_label_fontsize)

    # Additional coloring
    ax.spines['bottom'].set_color(base_color)
    ax.spines['top'].set_color(base_color)
    ax.spines['right'].set_color(base_color)
    ax.spines['left'].set_color(base_color)
    ax.tick_params(axis='x', colors=base_color)
    ax.tick_params(axis='y', colors=base_color)

    if save_fig:
        plt.savefig(f'{cf.FIGURES_PATH}/learning_homogeneous.{figure_format}', format=figure_format)


def make_heterogeneous_learning_plot(df: pd.DataFrame, save_fig: bool) -> None:
    df.loc[df['episode'] == 999, 'episode'] = 1000
    episodes = df.pivot(index=['episode'], columns=['setting', 'agent'], values=['reward'])

    fig, ax = plt.subplots(figsize=(4.5, 3))
    fig.suptitle("Running rewards during training", color=base_color, fontsize=title_fontsize)
    fig.subplots_adjust(right=0.5)

    ax.plot(episodes.index, episodes.loc[:, ('reward', 'single-agent', 'tabular')], '-', color=primary_color)
    ax.plot(episodes.index, episodes.loc[:, ('reward', '2-multi-agent-no-com-homo', 'tabular')],
            '--', color=primary_color)
    ax.plot(episodes.index, episodes.loc[:, ('reward', '2-multi-agent-no-com-hetero', 'tabular')],
            '--', color=secondary_color)
    ax.plot(episodes.index, episodes.loc[:, ('reward', '2-multi-agent-com-rounds-1-homo', 'tabular')],
            '-.', color=primary_color)
    ax.plot(episodes.index, episodes.loc[:, ('reward', '2-multi-agent-com-rounds-1-hetero', 'tabular')],
            '-.', color=secondary_color)
    ax.plot(episodes.index, episodes.loc[:, ('reward', '5-multi-agent-com-rounds-1-hetero', 'tabular')],
            ':', color=primary_color)
    ax.plot(episodes.index, episodes.loc[:, ('reward', '2-multi-agent-com-rounds-3-hetero', 'tabular')],
            ':', color=secondary_color)
    ax.legend(['Single agent', '2 agent no-com homogeneous', '2 agent no-com heterogeneous',
               '2 agent com homogeneous', '2 agent com heterogeneous', '5 agent com heterogeneous',
               '2 agent 3 rounds'],
              labelcolor=base_color, fontsize=axis_label_fontsize,
              bbox_to_anchor=(1., .2), loc='lower left')
    ax.set_xticks([0, 250, 500, 750, 1000], [0, 250, 500, 750, 1000], fontsize=axis_ticks_fontsize)
    ax.set_xlabel("Episodes", color=base_color, fontsize=axis_label_fontsize)
    ax.set_yticks([-5000, -10000, -15000, -20000, -25000, -30000], [-5000, -10000, -15000, -20000, -25000, -30000],
                  fontsize=axis_ticks_fontsize)
    ax.set_ylabel("Reward", color=base_color, fontsize=axis_label_fontsize)

    # Additional coloring
    ax.spines['bottom'].set_color(base_color)
    ax.spines['top'].set_color(base_color)
    ax.spines['right'].set_color(base_color)
    ax.spines['left'].set_color(base_color)
    ax.tick_params(axis='x', colors=base_color)
    ax.tick_params(axis='y', colors=base_color)

    if save_fig:
        plt.savefig(f'{cf.FIGURES_PATH}/learning_heterogeneous.{figure_format}', format=figure_format)


def make_nr_agent_dependency_plot(df: pd.DataFrame, save_fig: bool) -> None:
    settings = ['2-multi-agent-com-rounds-1-hetero', '3-multi-agent-com-rounds-1-hetero',
                '4-multi-agent-com-rounds-1-hetero', '5-multi-agent-com-rounds-1-hetero']
    df = df[df['setting'].isin(settings)]

    df = df[['setting', 'agent', 'day', 'cost']]\
        .groupby(['setting', 'agent', 'day']).sum()\
        .groupby(['setting', 'agent']).mean()
    costs = df.groupby(['setting']).mean().rename(columns={'cost': 'mean'})
    costs['std'] = df.groupby(['setting']).std()
    costs['agents'] = costs.index.map(lambda s: re.match(r'^([0-9])-.*', s).groups()[0]).astype(int)

    plt.rcParams['axes.titlepad'] = 14  # pad is in points...
    fig, ax = plt.subplots(figsize=(2.5, 2))
    plt.title("Average cost vs. community scale", color=base_color, fontsize=title_fontsize, loc='center')
    fig.subplots_adjust(left=0.16, bottom=.18, top=0.8)

    ax.errorbar(costs['agents'], costs['mean'], costs['std'], linestyle='none', marker='.', capsize=5, color=base_color)
    ax.set_xticks([2, 3, 4, 5], [2, 3, 4, 5], fontsize=axis_ticks_fontsize)
    ax.set_xlabel("Number of agents", color=base_color, fontsize=axis_label_fontsize)
    ax.set_ylim(0, 2)
    ax.set_yticks([0, .5, 1, 1.5, 2], [0.0, 0.5, 1.0, 1.5, 2.0], fontsize=axis_ticks_fontsize)
    ax.set_ylabel("Cost [€]", color=base_color, fontsize=axis_label_fontsize)

    # Additional coloring
    ax.spines['bottom'].set_color(base_color)
    ax.spines['top'].set_color(base_color)
    ax.spines['right'].set_color(base_color)
    ax.spines['left'].set_color(base_color)
    ax.tick_params(axis='x', colors=base_color)
    ax.tick_params(axis='y', colors=base_color)

    if save_fig:
        plt.savefig(f'{cf.FIGURES_PATH}/scale_effect_plot.{figure_format}', format=figure_format)


def make_nr_rounds_dependency_plot(df: pd.DataFrame, save_fig: bool) -> None:
    settings = ['3-multi-agent-com-rounds-1-hetero', '3-multi-agent-com-rounds-2-hetero',
                '3-multi-agent-com-rounds-3-hetero']
    df = df[df['setting'].isin(settings)]

    df = df[['setting', 'agent', 'day', 'cost']]\
        .groupby(['setting', 'agent', 'day']).sum()\
        .groupby(['setting', 'agent']).mean()
    costs = df.groupby(['setting']).mean().rename(columns={'cost': 'mean'})
    costs['std'] = df.groupby(['setting']).std()
    costs['rounds'] = costs.index.map(lambda s: get_rounds(s)).astype(int)

    plt.rcParams['axes.titlepad'] = 14  # pad is in points...
    fig, ax = plt.subplots(figsize=(3, 2))
    plt.title("Average cost vs. number of decision rounds", color=base_color, fontsize=title_fontsize)
    fig.subplots_adjust(left=0.2, right=.8, bottom=.18, top=0.8)

    ax.errorbar(costs['rounds'], costs['mean'], yerr=costs['std'], linestyle='none', marker='.', capsize=5, color=base_color)
    ax.set_xlim(.75, 3.25)
    ax.set_xlabel("Number of rounds", color=base_color, fontsize=axis_label_fontsize)
    ax.set_xticks([1, 2, 3], [1, 2, 3], fontsize=axis_ticks_fontsize)
    ax.set_ylim(1.0, 3)
    ax.set_yticks([0, .5, 1.0, 1.5, 2, 2.5, 3], [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], fontsize=axis_ticks_fontsize)
    ax.set_ylabel("Cost [€]", color=base_color, fontsize=axis_label_fontsize)

    # Additional coloring
    ax.spines['bottom'].set_color(base_color)
    ax.spines['top'].set_color(base_color)
    ax.spines['right'].set_color(base_color)
    ax.spines['left'].set_color(base_color)
    ax.tick_params(axis='x', colors=base_color)
    ax.tick_params(axis='y', colors=base_color)

    if save_fig:
        fig.savefig(f'{cf.FIGURES_PATH}/rounds_effect_plot.{figure_format}', format=figure_format)


def plot_tabular_comparison(save_figs: bool = False) -> None:

    con = db.get_connection()

    try:
        df = db.get_training_progress(con)
        make_homogeneous_learning_plot(df, save_figs)
        make_heterogeneous_learning_plot(df, save_figs)

        df = db.get_test_results(con)
        df[['load', 'pv']] = df[['load', 'pv']] * 1e-3
        df['time'] = df['time'].map(lambda t: t * 24)
        df['heatpump'] *= 1e-3

        make_homogeneous_costs_plot(df, save_figs)
        make_heterogeneous_costs_plot(df, save_figs)
        make_baseline_day_plot(df, 'rule-based', save_fig=save_figs)
        make_baseline_day_plot(df, 'semi-intelligent', save_fig=save_figs)
        make_baseline_day_plot(df, 'tabular', save_fig=save_figs)
        make_day_plot(df, homogeneous=True, save_fig=save_figs)
        make_day_plot(df, homogeneous=False, save_fig=save_figs)
        make_decisions_comparison_plot(df, homogeneous=True, save_fig=save_figs)
        make_decisions_comparison_plot(df, homogeneous=False, save_fig=save_figs)
        make_nr_agent_dependency_plot(df, save_figs)
        make_nr_rounds_dependency_plot(df, save_figs)

    finally:
        if con:
            con.close()


def compare_decisions(homogeneous: bool = False, save_fig: bool = False) -> None:
    setting = 'homo' if homogeneous else 'hetero'
    con = db.get_connection()

    try:
        df = db.get_test_results(con)
        df = df[(df['setting'].isin([f'2-multi-agent-com-rounds-1-{setting}', f'2-multi-agent-no-com-{setting}']))
                & (df['day'] == 8)]

        df[['load', 'pv']] = df[['load', 'pv']] * 1e-3
        df['time'] = df['time'].map(lambda t: t * 24)
        df['heatpump'] *= 1e-3
        timeslot_info = df.pivot(index=['time'], columns=['setting', 'agent'],
                                 values=['load', 'pv', 'temperature', 'heatpump'])
        time = np.arange(96)
        grid_price = (
                (setup.GRID_COST_AVG
                 + setup.GRID_COST_AMPLITUDE
                 * np.sin(time / 96 * 2 * np.pi * setup.HOURS_PER_DAY / setup.GRID_COST_PERIOD - setup.GRID_COST_PHASE)
                 ) / setup.CENTS_PER_EURO  # from c€ to €
        )
        injection_price = np.zeros(grid_price.shape)
        injection_price[:] = grid_price.min()[None]
        p2p_price = (grid_price + injection_price) / 2

        # Make plot
        fig, ax = plt.subplots(6, 1, figsize=(5.5, 5), sharex=True)
        fig.suptitle(f"Agent's state and decisions throughout the day ({setting}geneous)",
                     color=base_color, fontsize=title_fontsize)
        fig.subplots_adjust(left=0.1, right=0.7, hspace=0.4)

        # Powers
        ax[0].plot(time, timeslot_info.loc[:, ('load', f'2-multi-agent-com-rounds-1-{setting}', 0)],
                   color=primary_color)
        ax[0].plot(time, timeslot_info.loc[:, ('load', f'2-multi-agent-com-rounds-1-{setting}', 1)],
                   color=secondary_color)
        ax[0].plot(time, timeslot_info.loc[:, ('pv', f'2-multi-agent-com-rounds-1-{setting}', 0)], '--',
                   color=primary_color)
        ax[0].set_yticks([-4, 0, 4], [-4.0, 0.0, 4.0], fontsize=axis_ticks_fontsize)
        ax[0].set_ylabel("Power [kW]", color=base_color, fontsize=axis_label_fontsize)
        ax[0].legend(["Base Load Agent 0", "Base Load Agent 1", "PV"], labelcolor=base_color,
                     fontsize=axis_label_fontsize, bbox_to_anchor=(1.02, 0.), loc="lower left")

        # Prices
        ax[1].plot(time, grid_price, color=primary_color)
        ax[1].plot(time, injection_price, color=secondary_color)
        ax[1].plot(time, p2p_price, '--', color=primary_color)
        ax[1].set_yticks([0.07, 0.12, 0.17], [0.07, 0.12, 0.17], fontsize=axis_ticks_fontsize)
        ax[1].set_ylabel("Price [€]", color=base_color, fontsize=axis_label_fontsize)
        ax[1].legend(["Offtake", "Injection", "P2P"], labelcolor=base_color, fontsize=axis_label_fontsize,
                     bbox_to_anchor=(1.02, 0.), loc="lower left")

        # Heat pump
        width = 0.4

        # agent 0
        ax[2].set_title("agent 0", fontsize=axis_label_fontsize, loc='right', pad=-.1, color=base_color)
        ax[2].bar(time - width / 2, timeslot_info.loc[:, ('heatpump', f'2-multi-agent-com-rounds-1-{setting}', 0)],
                  label='Communication', width=width, color=primary_color)
        ax[2].bar(time + width / 2, timeslot_info.loc[:, ('heatpump', f'2-multi-agent-no-com-{setting}', 0)],
                  label='No communication', width=width, color=secondary_color)
        ax[2].set_yticks([0, 1.5, 3], [0.0, 1.5, 3.0], fontsize=axis_ticks_fontsize)
        ax[2].set_ylabel("HP [kW]", color=base_color, fontsize=axis_label_fontsize)
        ax[2].legend(["Communication", "No communication"], labelcolor=base_color,
                     fontsize=axis_label_fontsize, bbox_to_anchor=(1.02, 0.), loc="lower left")

        # agent 1
        ax[3].set_title("agent 1", fontsize=axis_label_fontsize, loc='right', pad=-.1, color=base_color)
        ax[3].bar(time - width / 2, timeslot_info.loc[:, ('heatpump', f'2-multi-agent-com-rounds-1-{setting}', 1)],
                  label='Communication', width=width, color=primary_color)
        ax[3].bar(time + width / 2, timeslot_info.loc[:, ('heatpump', f'2-multi-agent-no-com-{setting}', 1)],
                  label='No communication', width=width, color=secondary_color)
        ax[3].set_yticks([0, 1.5, 3], [0.0, 1.5, 3.0], fontsize=axis_ticks_fontsize)
        ax[3].set_ylabel("HP [kW]", color=base_color, fontsize=axis_label_fontsize)

        # Temperature
        # Agent 0
        ax[4].set_title("agent 0", fontsize=axis_label_fontsize, loc='right', pad=-.1, color=base_color)
        ax[4].plot(time, timeslot_info.loc[:, ('temperature', f'2-multi-agent-com-rounds-1-{setting}', 0)],
                   color=primary_color)
        ax[4].plot(time, timeslot_info.loc[:, ('temperature', f'2-multi-agent-no-com-{setting}', 0)],
                   color=secondary_color)
        ax[4].set_yticks([20, 22], [20, 22], fontsize=axis_ticks_fontsize)
        ax[4].hlines(y=[20, 22], xmin=0, xmax=96, color=neutral_color, linestyle='--', linewidths=0.8)
        ax[4].legend(["Communication", "No communication"], labelcolor=base_color,
                     fontsize=axis_label_fontsize, bbox_to_anchor=(1.02, 0), loc="lower left")

        # Agent 1
        ax[5].set_title("agent 1", fontsize=axis_label_fontsize, loc='right', pad=-.3, color=base_color)
        ax[5].plot(time, timeslot_info.loc[:, ('temperature', f'2-multi-agent-com-rounds-1-{setting}', 1)],
                   color=primary_color)
        ax[5].plot(time, timeslot_info.loc[:, ('temperature', f'2-multi-agent-no-com-{setting}', 1)],
                   color=secondary_color)
        ax[5].set_ylabel("  Temperature [°C]", loc='bottom', color=base_color, fontsize=axis_label_fontsize)
        ax[5].set_xticks([0, 24, 48, 72, 95], ["00:00", "06:00", "12:00", "18:00", "23:45"],
                         fontsize=axis_ticks_fontsize)
        ax[5].set_xlabel("Time", color=base_color, fontsize=axis_label_fontsize)
        ax[5].set_yticks([20, 22], [20, 22], fontsize=axis_ticks_fontsize)
        ax[5].hlines(y=[20, 22], xmin=0, xmax=96, color=neutral_color, linestyle='--', linewidths=0.8)
        ax[5].xaxis.set_minor_locator(MultipleLocator(1))

        # Additional coloring
        for i in range(len(ax)):
            ax[i].spines['bottom'].set_color(base_color)
            ax[i].spines['top'].set_color(base_color)
            ax[i].spines['right'].set_color(base_color)
            ax[i].spines['left'].set_color(base_color)
            ax[i].tick_params(axis='x', colors=base_color)
            ax[i].tick_params(axis='y', colors=base_color)

        if save_fig:
            plt.savefig(f'{cf.FIGURES_PATH}/decisions_plot_{setting}.{figure_format}', format=figure_format)

    finally:
        if con:
            con.close()


def compare_decisions_rounds(save_fig: bool = False) -> None:
    con = db.get_connection()

    try:
        df = db.get_rounds_decisions(con)
        df['decision'] *= 1e-3
        decisions = df[(df['agent'] == 0) & (df['setting'] == '3-multi-agent-com-rounds-3-hetero') & (df['day'] == 8)] \
            .pivot(index=['time'], columns=['round'], values=['decision'])

        df = db.get_test_results(con)
        df = df[(df['setting'] == '3-multi-agent-com-rounds-3-hetero') & (df['day'] == 8)]
        df[['load', 'pv']] = df[['load', 'pv']] * 1e-3
        df['time'] = df['time'].map(lambda t: t * 24)
        df['heatpump'] *= 1e-3

        timeslot_info = df.pivot(index=['time'], columns=['agent'],
                                 values=['load', 'pv', 'temperature', 'heatpump', 'cost'])
        time = np.arange(96)
        grid_price = (
                (setup.GRID_COST_AVG
                 + setup.GRID_COST_AMPLITUDE
                 * np.sin(time / 96 * 2 * np.pi * setup.HOURS_PER_DAY / setup.GRID_COST_PERIOD - setup.GRID_COST_PHASE)
                 ) / setup.CENTS_PER_EURO  # from c€ to €
        )
        injection_price = np.zeros(grid_price.shape)
        injection_price[:] = grid_price.min()[None]
        p2p_price = (grid_price + injection_price) / 2

        # Make plot
        fig, ax = plt.subplots(4, 1, figsize=(5.5, 3.5), sharex=True)
        fig.suptitle("Agent decisions for each round of the time slot", color=base_color, fontsize=title_fontsize)
        fig.subplots_adjust(left=0.1, right=0.72, hspace=0.4)

        # Powers
        net_power = timeslot_info.loc[:, ('load', 0)] \
                    - timeslot_info.loc[:, ('pv', 0)] \
                    + timeslot_info.loc[:, ('heatpump', 0)]
        ax[0].plot(time, timeslot_info.loc[:, ('load', 0)], color=secondary_color)
        ax[0].plot(time, timeslot_info.loc[:, ('pv', 0)], ':', color=secondary_color)
        ax[0].plot(time, net_power, color=primary_color)
        ax[0].set_yticks([-4, 0, 4], [-4.0, 0.0, 4.0], fontsize=axis_ticks_fontsize)
        ax[0].set_ylabel("Power [kW]", color=base_color, fontsize=axis_label_fontsize)
        ax[0].legend(["Base Load", "PV", "Net Consumption"], labelcolor=base_color, fontsize=axis_label_fontsize,
                     bbox_to_anchor=(1.02, -.125), loc="lower left")

        # Prices
        ax12 = ax[1].twinx()
        ax[1].plot(time, timeslot_info.loc[:, ('cost', 0)], color=primary_color)
        ax12.plot(time, grid_price, color=secondary_color)
        ax12.plot(time, injection_price, ':', color=secondary_color)
        ax12.plot(time, p2p_price, '--', color=secondary_color)
        ax[1].set_yticks([-.1, 0, .1], [-0.1, 0.0, 0.1], fontsize=axis_ticks_fontsize)
        ax12.set_yticks([0.07, 0.12, 0.17], [0.07, 0.12, 0.17], color=secondary_color, fontsize=axis_ticks_fontsize)
        ax[1].set_zorder(1)
        ax[1].set_frame_on(False)
        ax[1].set_ylabel("Cost [€]", color=primary_color, fontsize=axis_label_fontsize)
        ax12.set_ylabel("Price [€/kWh]", color=secondary_color, fontsize=axis_label_fontsize)
        ax12.legend(["Offtake", "Injection", "P2P"], labelcolor=base_color, fontsize=axis_label_fontsize,
                    bbox_to_anchor=(1.15, -.125), loc="lower left")

        # Heat pump
        width = 0.2
        x = np.array(decisions.index * 96)

        ax[2].bar(x - 1.5 * width, decisions.loc[:, ('decision', 0)], label='Round 0', width=width)
        ax[2].bar(x - 0.5 * width, decisions.loc[:, ('decision', 1)], label='Round 1', width=width)
        ax[2].bar(x + 0.5 * width, decisions.loc[:, ('decision', 2)], label='Round 2', width=width)
        ax[2].bar(x + 1.5 * width, decisions.loc[:, ('decision', 3)], label='Round 3', width=width)

        ax[2].set_yticks([0, 1.5, 3], [0.0, 1.5, 3.0], fontsize=axis_ticks_fontsize)
        ax[2].set_ylabel("HP [kW]", color=base_color, fontsize=axis_label_fontsize, loc='top')
        ax[2].legend(bbox_to_anchor=(1.04, 1.2), loc='upper left', fontsize=axis_label_fontsize)

        # Temperature
        ax[3].plot(time, timeslot_info.loc[:, ('temperature', 0)], color=primary_color)
        ax[3].set_yticks([20, 22], [20, 22], fontsize=axis_ticks_fontsize)
        ax[3].set_ylabel("Temperature [°C]", color=base_color, fontsize=axis_label_fontsize)
        ax[3].set_xticks([i for i in range(96 + 1) if i % 4 == 0],
                         [re.sub(' ', '0', f'{i / 4:2.0f}:00' if i % 16 == 0 else '') for i in range(96 + 1) if
                          i % 4 == 0],
                         fontsize=axis_ticks_fontsize)
        ax[2].xaxis.set_minor_locator(MultipleLocator(1))
        ax[3].set_xlabel("Time", color=base_color, fontsize=axis_label_fontsize)
        ax[3].hlines(y=[20, 22], xmin=0, xmax=96, color=neutral_color, linestyle='--', linewidths=0.8)

        # Additional coloring
        for i in range(len(ax)):
            ax[i].spines['bottom'].set_color(base_color)
            ax[i].spines['top'].set_color(base_color)
            ax[i].spines['right'].set_color(base_color)
            ax[i].spines['left'].set_color(base_color)
            ax[i].tick_params(axis='x', colors=base_color)
            ax[i].tick_params(axis='y', colors=base_color)

        if save_fig:
            plt.savefig(f'{cf.FIGURES_PATH}/rounds_day_plot.{figure_format}', format=figure_format)

    finally:
        if con:
            con.close()


def compare_decisions_artificial(save_fig: bool = False) -> None:
    con = db.get_connection()

    try:
        df = db.get_test_results(con)
        df = df[(df['setting'].isin(['2-agent-1-pv-drop-com', '2-agent-1-pv-drop-no-com'])) & (df['day'] == 10)]

        df[['load', 'pv']] = df[['load', 'pv']] * 1e-3
        df['time'] = df['time'].map(lambda t: t * 24)
        df['heatpump'] *= 1e-3
        timeslot_info = df.pivot(index=['time'], columns=['setting', 'agent'],
                                 values=['load', 'pv', 'temperature', 'heatpump'])
        time = np.arange(96)
        grid_price = (
                (setup.GRID_COST_AVG
                 + setup.GRID_COST_AMPLITUDE
                 * np.sin(time / 96 * 2 * np.pi * setup.HOURS_PER_DAY / setup.GRID_COST_PERIOD - setup.GRID_COST_PHASE)
                 ) / setup.CENTS_PER_EURO  # from c€ to €
        )
        injection_price = np.zeros(grid_price.shape)
        injection_price[:] = grid_price.min()[None]
        p2p_price = (grid_price + injection_price) / 2

        # Make plot
        fig, ax = plt.subplots(6, 1, figsize=(5.5, 5), sharex=True)
        fig.suptitle("Agent's state and decisions throughout the day", color=base_color, fontsize=title_fontsize)
        fig.subplots_adjust(left=0.1, right=0.7, hspace=0.4)

        # Powers
        ax[0].plot(time, timeslot_info.loc[:, ('load', '2-agent-1-pv-drop-com', 0)], color=primary_color)
        ax[0].plot(time, timeslot_info.loc[:, ('load', '2-agent-1-pv-drop-com', 1)], color=secondary_color)
        ax[0].plot(time, timeslot_info.loc[:, ('pv', '2-agent-1-pv-drop-com', 0)], '--', color=primary_color)
        ax[0].plot(time, timeslot_info.loc[:, ('pv', '2-agent-1-pv-drop-com', 1)], '--', color=secondary_color)
        ax[0].set_yticks([-4, 0, 4], [-4.0, 0.0, 4.0], fontsize=axis_ticks_fontsize)
        ax[0].set_ylabel("Power [kW]", color=base_color, fontsize=axis_label_fontsize)
        ax[0].legend(["Base Load Agent 0", "Base Load Agent 1", "PV Agent 0", "PV Agent 1"], labelcolor=base_color,
                     fontsize=axis_label_fontsize, bbox_to_anchor=(1.02, -0.3), loc="lower left")

        # Prices
        ax[1].plot(time, grid_price, color=primary_color)
        ax[1].plot(time, injection_price, color=secondary_color)
        ax[1].plot(time, p2p_price, '--', color=primary_color)
        ax[1].set_yticks([0.07, 0.12, 0.17], [0.07, 0.12, 0.17], fontsize=axis_ticks_fontsize)
        ax[1].set_ylabel("Price [€/kWh]", color=base_color, fontsize=axis_label_fontsize)
        ax[1].legend(["Offtake", "Injection", "P2P"], labelcolor=base_color, fontsize=axis_label_fontsize,
                     bbox_to_anchor=(1.02, -0.15), loc="lower left")

        # Heat pump
        width = 0.4

        # Agent 0
        ax[2].set_title("agent 0", fontsize=axis_label_fontsize, loc='right', pad=-.1, color=base_color)
        ax[2].bar(time - width / 2, timeslot_info.loc[:, ('heatpump', '2-agent-1-pv-drop-com', 0)],
                  label='Communication', width=width, color=primary_color)
        ax[2].bar(time + width / 2, timeslot_info.loc[:, ('heatpump', '2-agent-1-pv-drop-no-com', 0)],
                  label='No communication', width=width, color=secondary_color)
        ax[2].set_yticks([0, 1.5, 3], [0.0, 1.5, 3.0], fontsize=axis_ticks_fontsize)
        ax[2].set_ylabel("HP [kW]", color=base_color, fontsize=axis_label_fontsize)
        ax[2].legend(["Communication", "No communication"], labelcolor=base_color,
                     fontsize=axis_label_fontsize, bbox_to_anchor=(1.02, .2), loc="upper left")

        # Agent 1
        ax[3].set_title("agent 1", fontsize=axis_label_fontsize, loc='right', pad=-.1, color=base_color)
        ax[3].bar(time - width / 2, timeslot_info.loc[:, ('heatpump', '2-agent-1-pv-drop-com', 1)],
                  label='Communication', width=width, color=primary_color)
        ax[3].bar(time + width / 2, timeslot_info.loc[:, ('heatpump', '2-agent-1-pv-drop-no-com', 1)],
                  label='No communication', width=width, color=secondary_color)
        ax[3].set_yticks([0, 1.5, 3], [0.0, 1.5, 3.0], fontsize=axis_ticks_fontsize)
        ax[3].set_ylabel("HP [kW]", color=base_color, fontsize=axis_label_fontsize)

        # Temperature
        # Agent 0
        ax[4].set_title("agent 0", fontsize=axis_label_fontsize, loc='right', pad=-.1, color=base_color)
        ax[4].plot(time, timeslot_info.loc[:, ('temperature', '2-agent-1-pv-drop-com', 0)],
                   color=primary_color)
        ax[4].plot(time, timeslot_info.loc[:, ('temperature', '2-agent-1-pv-drop-no-com', 0)],
                   color=secondary_color)
        ax[4].set_yticks([20, 22], [20, 22], fontsize=axis_ticks_fontsize)
        ax[4].hlines(y=[20, 22], xmin=0, xmax=96, color=neutral_color, linestyle='--', linewidths=0.8)
        ax[4].legend(["Communication", "No communication"], labelcolor=base_color,
                     fontsize=axis_label_fontsize, bbox_to_anchor=(1.02, .2), loc="upper left")

        # Agent 1
        ax[5].set_title("agent 1", fontsize=axis_label_fontsize, loc='right', pad=-.3, color=base_color)
        ax[5].plot(time, timeslot_info.loc[:, ('temperature', '2-agent-1-pv-drop-com', 1)],
                   color=primary_color)
        ax[5].plot(time, timeslot_info.loc[:, ('temperature', '2-agent-1-pv-drop-no-com', 1)],
                   color=secondary_color)
        ax[5].set_ylabel("  Temperature [°C]", loc='bottom', color=base_color, fontsize=axis_label_fontsize)
        ax[5].set_yticks([20, 22], [20, 22], fontsize=axis_ticks_fontsize)
        ax[5].hlines(y=[20, 22], xmin=0, xmax=96, color=neutral_color, linestyle='--', linewidths=0.8)

        ax[5].set_xticks([i for i in range(96 + 1) if i % 4 == 0],
                         [re.sub(' ', '0', f'{i / 4:2.0f}:00' if i % 16 == 0 else '') for i in range(96 + 1) if
                          i % 4 == 0],
                         fontsize=axis_ticks_fontsize)
        ax[5].set_xlabel("Time", color=base_color, fontsize=axis_label_fontsize)
        ax[5].xaxis.set_minor_locator(MultipleLocator(1))

        # Additional coloring
        for i in range(len(ax)):
            ax[i].spines['bottom'].set_color(base_color)
            ax[i].spines['top'].set_color(base_color)
            ax[i].spines['right'].set_color(base_color)
            ax[i].spines['left'].set_color(base_color)
            ax[i].tick_params(axis='x', colors=base_color)
            ax[i].tick_params(axis='y', colors=base_color)

        if save_fig:
            plt.savefig(f'{cf.FIGURES_PATH}/artificial_pv_decisions_plot.{figure_format}', format=figure_format)

    finally:
        con.close()


def plot_q_values_com(q_table: np.ndarray, save_figs: bool = False) -> None:
    q_table = q_table / np.abs(q_table).max() #- q_table.mean()
    normalizer = matplotlib.colors.SymLogNorm(10 ** -4, vmin=-1, vmax=1)

    time_steps = int(q_table.shape[0])
    balance_steps = int(q_table.shape[2])
    p2p_steps = int(q_table.shape[3])

    action_ticks = list(range(3))
    temperature_ticks = [0, 5, 10, 15, 19]

    for p in range(p2p_steps):

        fig, ax = plt.subplots(balance_steps, time_steps, figsize=(6.5, 11), sharex=True, sharey=True)
        fig.suptitle(f"Communication Q-table (P2P power index {p})", fontsize=title_fontsize)
        fig.subplots_adjust(wspace=0., hspace=.1, left=.06, right=.9, top=.95, bottom=.05)
        cbar_ax = fig.add_axes([0.93, 0.3, 0.01, 0.4])
        cbar_ax.tick_params(labelsize=axis_ticks_fontsize)

        for b in tqdm(range(balance_steps)):
            ax[b, 0].set_yticks(temperature_ticks, temperature_ticks, fontsize=axis_ticks_fontsize)
            ax[b, time_steps - 1].text(.905, .925 - b * .0455, f"b={b}", fontsize=axis_ticks_fontsize, rotation=-90,
                                       transform=fig.transFigure)

            for t in range(time_steps):
                if b == 0:
                    ax[b, t].set_title(f"t={t}", fontsize=axis_ticks_fontsize)
                elif b == balance_steps - 1:
                    ax[b, t].set_xticks(action_ticks, action_ticks, fontsize=axis_ticks_fontsize)

                im = ax[b, t].imshow(q_table[t, :, p, b, :], cmap='seismic', norm=normalizer, aspect=.3)

        fig.colorbar(im, cax=cbar_ax)
        ax[balance_steps - 1, time_steps // 2].set_xlabel("Action index", fontsize=axis_label_fontsize)
        ax[balance_steps // 2, 0].set_ylabel("Temperature index", fontsize=axis_label_fontsize)

        if save_figs:
            fig.savefig(f'{cf.FIGURES_PATH}/q_table_com_plot_{p}.{figure_format}', format=figure_format)
            plt.close(fig)


def plot_q_values_no_com(q_table: np.ndarray, save_fig: bool = False) -> None:
    q_table = q_table / np.abs(q_table).max() #- q_table.mean()
    normalizer = matplotlib.colors.SymLogNorm(10 ** -4, vmin=-1, vmax=1)

    time_steps = int(q_table.shape[0])
    balance_steps = int(q_table.shape[2])

    action_ticks = list(range(3))
    temperature_ticks = [0, 5, 10, 15, 19]

    fig, ax = plt.subplots(balance_steps, time_steps, figsize=(6.5, 11), sharex=True, sharey=True)
    fig.suptitle(f"No communication Q-table", fontsize=title_fontsize)
    fig.subplots_adjust(wspace=0., hspace=.1, left=.06, right=.9, top=.95, bottom=.05)
    cbar_ax = fig.add_axes([0.93, 0.3, 0.01, 0.4])
    cbar_ax.tick_params(labelsize=axis_ticks_fontsize)

    for b in tqdm(range(balance_steps)):
        ax[b, 0].set_yticks(temperature_ticks, temperature_ticks, fontsize=axis_ticks_fontsize)
        ax[b, time_steps - 1].text(.905, .925 - b * .0455, f"b={b}", fontsize=axis_ticks_fontsize, rotation=-90,
                                   transform=fig.transFigure)

        for t in range(time_steps):
            if b == 0:
                ax[b, t].set_title(f't={t}', fontsize=axis_ticks_fontsize)
            elif b == balance_steps - 1:
                ax[b, t].set_xticks(action_ticks, action_ticks, fontsize=axis_ticks_fontsize)

            im = ax[b, t].imshow(q_table[t, :, b, :], cmap='seismic', norm=normalizer, aspect=.3)

    fig.colorbar(im, cax=cbar_ax)
    ax[balance_steps - 1, time_steps // 2].set_xlabel("Action index", fontsize=axis_label_fontsize)
    ax[balance_steps // 2, 0].set_ylabel("Temperature index", fontsize=axis_label_fontsize)

    if save_fig:
        fig.savefig(f'{cf.FIGURES_PATH}/q_table_no_com_plot.{figure_format}', format=figure_format)


def compare_q_values(save_figs: bool = False) -> None:
    q_table_com = np.load(f'../models_tabular/2_multi_agent_com_rounds_1_hetero_0.npy')
    q_table_no_com = np.load(f'../models_tabular/single_agent_0.npy')

    plot_q_values_com(q_table_com, save_figs=save_figs)
    plot_q_values_no_com(q_table_no_com, save_fig=save_figs)


def statistics_baselines(df: pd.DataFrame) -> None:
    df = df[df['setting'] == 'single-agent']

    costs = df[['setting', 'implementation', 'day', 'cost']] \
        .groupby(['setting', 'implementation', 'day']).sum().reset_index()\
        .pivot(index=['day'], columns=['implementation'], values=['cost'])

    sample_rule_based = np.array(costs.loc[:, ('cost', 'rule-based')])
    sample_semi_intelligent = np.array(costs.loc[:, ('cost', 'semi-intelligent')])
    sample_tabular = np.array(costs.loc[:, ('cost', 'tabular')])

    rule_tabular = sample_rule_based - sample_tabular
    semi_tabular = sample_semi_intelligent - sample_tabular

    _, p_rule_tabular = stats.ttest_1samp(rule_tabular, 0)
    _, p_semi_tabular = stats.ttest_1samp(semi_tabular, 0)

    print('Test difference between baseline agents')
    print(f'Difference rule-based vs tabular: p-value = {p_rule_tabular}')
    print(f'Difference semi-intelligent vs tabular: p-value = {p_semi_tabular}')
    print('-' * 50)


def statistics_cost_homogeneous(df: pd.DataFrame) -> None:
    settings = ['2-multi-agent-com-rounds-1-homo', '2-multi-agent-no-com-homo']
    df = df[df['setting'].isin(settings) | ((df['setting'] == 'single-agent') & (df['implementation'] == 'tabular'))]

    costs = df[['setting', 'day', 'agent', 'cost']] \
        .groupby(['setting', 'day', 'agent']).sum()\
        .groupby(['setting', 'day']).mean().reset_index() \
        .pivot(index=['day'], columns=['setting'], values=['cost'])

    sample_single_agent = np.array(costs.loc[:, ('cost', 'single-agent')])
    sample_no_com = np.array(costs.loc[:, ('cost', '2-multi-agent-no-com-homo')])
    sample_com = np.array(costs.loc[:, ('cost', '2-multi-agent-com-rounds-1-homo')])

    single_no_com = sample_single_agent - sample_no_com
    single_com = sample_single_agent - sample_com
    com_no_com = sample_com - sample_no_com

    _, p_single_no_com = stats.ttest_1samp(single_no_com, 0)
    _, p_single_com = stats.ttest_1samp(single_com, 0)
    _, p_com_no_com = stats.ttest_1samp(com_no_com, 0)

    print('Test difference in cost between homogeneous agents')
    print(f'Difference single-agent vs no-com: p-value = {p_single_no_com}')
    print(f'Difference single-agent vs com: p-value = {p_single_com}')
    print(f'Difference no-com vs com: p-value = {p_com_no_com}')
    print('-' * 50)


def statistics_cost_heterogeneous(df: pd.DataFrame) -> None:
    settings = ['2-multi-agent-com-rounds-1-homo', '2-multi-agent-com-rounds-1-hetero',
                '2-multi-agent-no-com-homo', '2-multi-agent-no-com-hetero']
    df = df[df['setting'].isin(settings)]

    costs = df[['setting', 'day', 'agent', 'cost']] \
        .groupby(['setting', 'day', 'agent']).sum()\
        .groupby(['setting', 'day']).mean().reset_index() \
        .pivot(index=['day'], columns=['setting'], values=['cost'])

    sample_no_com_homo = np.array(costs.loc[:, ('cost', '2-multi-agent-no-com-homo')])
    sample_no_com_hetero = np.array(costs.loc[:, ('cost', '2-multi-agent-no-com-hetero')])
    sample_com_homo = np.array(costs.loc[:, ('cost', '2-multi-agent-com-rounds-1-homo')])
    sample_com_hetero = np.array(costs.loc[:, ('cost', '2-multi-agent-com-rounds-1-hetero')])

    diff_no_com = sample_no_com_homo - sample_no_com_hetero
    diff_com = sample_com_homo - sample_com_hetero

    _, p_no_com = stats.ttest_1samp(diff_no_com, 0)
    _, p_com = stats.ttest_1samp(diff_com, 0)

    print('Test difference in cost between homogeneous and heterogeneous')
    print(f'Difference in no-com: p-value = {p_no_com}')
    print(f'Difference in com: p-value = {p_com}')
    print('-' * 50)


def statistics_community_scale(df: pd.DataFrame) -> None:
    settings = ['2-multi-agent-com-rounds-1-hetero', '3-multi-agent-com-rounds-1-hetero',
                '4-multi-agent-com-rounds-1-hetero', '5-multi-agent-com-rounds-1-hetero']
    df = df[df['setting'].isin(settings)]

    costs = df[['setting', 'agent', 'day', 'cost']]\
        .groupby(['setting', 'agent', 'day']).sum()\
        .groupby(['setting', 'agent']).mean()
    costs['agents'] = costs.index.map(lambda s: re.match(r'^([0-9])-.*', s[0]).groups()[0]).astype(int)
    costs = costs.reset_index()

    samples = [np.array(costs.loc[costs['agents'] == i, 'cost']) for i in pd.unique(costs['agents'])]
    _, p_levene = stats.levene(*samples)
    _, p_anova = stats.f_oneway(*samples)
    _, p_anova_reduced = stats.f_oneway(*samples[1:])

    print('Analysis of the influence of community scale:')
    print(f'Same variance: p-value = {p_levene}')
    print(f'Same mean: p-value = {p_anova}')
    print(f'Same mean (without 2-agent): p-value = {p_anova_reduced}')
    print('-' * 50)


def statistics_nr_rounds(df: pd.DataFrame) -> None:
    settings = ['3-multi-agent-com-rounds-1-hetero', '3-multi-agent-com-rounds-2-hetero',
                '3-multi-agent-com-rounds-3-hetero']
    df = df[df['setting'].isin(settings)]

    days = [d for d in pd.unique(df['day'])]
    df_costs = (df[['setting', 'agent', 'day', 'cost']]
        .groupby(['setting', 'agent', 'day']).sum().reset_index()
        .pivot(index=['setting'], columns=['agent', 'day'], values=['cost']))
    # array: days x agents x settings
    costs = np.array([[df_costs.loc[:, ('cost', a, d)] for a in range(3)] for d in days])

    df = df[['setting', 'agent', 'day', 'cost']] \
        .groupby(['setting', 'agent', 'day']).sum() \
        .groupby(['setting', 'agent']).mean()

    df['rounds'] = df.index.map(lambda s: get_rounds(s[0])).astype(int)
    rounds = df.reset_index()

    samples = [np.array(rounds.loc[rounds['rounds'] == i, 'cost']) for i in pd.unique(rounds['rounds'])]
    _, p_levene = stats.levene(*samples)
    _, p_anova = stats.f_oneway(*samples)

    diff_12 = (costs[:, :, 0] - costs[:, :, 1]).mean(axis=0)
    diff_13 = (costs[:, :, 0] - costs[:, :, 2]).mean(axis=0)
    diff_23 = (costs[:, :, 1] - costs[:, :, 2]).mean(axis=0)
    _, p_diff_12 = stats.ttest_1samp(diff_12, 0)
    _, p_diff_13 = stats.ttest_1samp(diff_13, 0)
    _, p_diff_23 = stats.ttest_1samp(diff_23, 0)

    print('Analysis of the influence of rounds:')
    print(f'Same variance: p-value = {p_levene}')
    print(f'Same mean: p-value = {p_anova}')
    print(f'Difference 1 vs 2 rounds: p-value = {p_diff_12}')
    print(f'Difference 1 vs 3 rounds: p-value = {p_diff_13}')
    print(f'Difference 2 vs 3 rounds: p-value = {p_diff_23}')
    print('-' * 50)


def statistical_tests() -> None:
    con = db.get_connection()

    try:
        df = db.get_test_results(con)
        df[['load', 'pv']] = df[['load', 'pv']] * 1e-3
        df['time'] = df['time'].map(lambda t: t * 24)
        df['heatpump'] *= 1e-3

        statistics_baselines(df)
        statistics_cost_homogeneous(df)
        statistics_cost_heterogeneous(df)
        statistics_community_scale(df)
        statistics_nr_rounds(df)

    finally:
        if con:
            con.close()


def clean_ddpg_data(df: pd.DataFrame, index: str) -> pd.DataFrame:
    df = df.groupby(['settings', index]).mean().reset_index()
    df[index] = df[index].map(lambda e: float(e))
    settings = df['settings'].str.split(';').map(lambda l: list(map(lambda s: s.split('='), l)))
    fields = [l[0] for l in settings.loc[0]]

    for i, field in enumerate(fields):
        if field == 'ls':
            field = 'lr'
        if field in {'episodes', 'bs', 'bu'}:
            df[field] = settings.map(lambda l: int(l[i][1]))
        elif field in {'gamma', 'ls', 'sd'}:
            df[field] = settings.map(lambda l: float(l[i][1]))
        else:
            df[field] = settings.map(lambda l: l[i][1])

    df.drop(columns=['tau', 'theta', 'sigma', 'horizon'], inplace=True)

    return df


def make_ddpg_plot(df: pd.DataFrame, training: bool, save_figs: bool = False) -> None:
    activations = pd.unique(df['activation'])
    episodes = sorted(pd.unique(df['episodes']))
    learning_rates = sorted(pd.unique(df['lr']))
    buffer_sizes = sorted(pd.unique(df['bu']))
    batch_sizes = sorted(list(pd.unique(df['bs'])))
    gammas = sorted(pd.unique(df['gamma']))
    stds = sorted(pd.unique(df['sd']))

    if training:
        data = df.pivot(index=['episode'],
                        columns=['activation', 'episodes', 'bu', 'bs', 'gamma', 'lr', 'sd'],
                        values=['training', 'episode'])
        attributes = ('episode', 'training')
        offset = 100
        formatter = matplotlib.ticker.FuncFormatter(lambda y, _: f'{y:.0f}')
    else:
        data = df.pivot(index=['time'],
                        columns=['activation', 'episodes', 'bu', 'bs', 'gamma', 'lr', 'sd'],
                        values=['time', 'load', 'pv', 'target_load', 'target_pv'])
        attributes = ('time', 'load', 'target_load')
        offset = 0
        formatter = matplotlib.ticker.FuncFormatter(lambda y, _: f'{y:.2f}')

    for activation in activations:
        for std in stds:
            for bu in buffer_sizes:

                available_learning_rates = [lr for lr in learning_rates
                                            if df[['activation', 'sd', 'lr']]
                                                .loc[(df['activation'] == activation) & (df['bu'] == bu)
                                                     & (df['sd'] == std)
                                                     & (df['lr'] == lr)].any().any()]
                available_batch_sizes = [bs for bs in batch_sizes
                                         if df[['activation', 'sd', 'lr']]
                                             .loc[(df['activation'] == activation) & (df['bu'] == bu)
                                                  & (df['sd'] == std)
                                                  & (df['bs'] == bs)].any().any()]

                if len(available_learning_rates) == 0 or len(available_batch_sizes) == 0:
                    continue

                # Make plot
                fig, ax = plt.subplots(len(available_batch_sizes), len(available_learning_rates),
                                       figsize=(5.5, 1 + len(available_batch_sizes) * 1.5), sharex=True, sharey=True)
                fig.subplots_adjust(wspace=0.1, right=0.89)
                fig.suptitle(f'{activation.capitalize()} activation (buffer size = {bu/1000}E03, noise = {std})',
                             fontsize=title_fontsize)

                couples: Dict[float, List[float]]
                for j, lr in enumerate(available_learning_rates):
                    for i, bs in enumerate(available_batch_sizes):

                        # Select current plot (not all axis objects can be indexed)
                        if len(available_batch_sizes) == 1 and len(available_learning_rates) == 1:
                            current_ax = ax
                        elif len(available_batch_sizes) == 1:
                            current_ax = ax[j]
                        elif len(available_learning_rates) == 1:
                            current_ax = ax[i]
                        else:
                            current_ax = ax[i, j]

                        # Add parameter information to grid
                        if j == len(available_learning_rates) - 1 and len(available_batch_sizes) > 1:
                            current_ax.text(.892, 0.8 - i * 0.85 / len(available_batch_sizes),
                                            f'Batch size\n{bs}',
                                            fontsize=axis_label_fontsize, transform=fig.transFigure)
                        if i == 0 and len(available_learning_rates) > 1:
                            current_ax.set_title(f'Learning rate {lr}', fontsize=axis_label_fontsize)

                        # Select data to plot
                        available_series = [(g, ep) for g in gammas for ep in episodes
                                            if df.loc[(df['activation'] == activation) & (df['bu'] == bu)
                                                      & (df['sd'] == std) & (df['gamma'] == g) & (df['bs'] == bs)
                                                      & (df['lr'] == lr) & (df['episodes'] == ep)].any().any()]

                        if len(available_series) == 0:
                            continue

                        couples = {g: [] for g in list(zip(*available_series))[0]}
                        for g, ep in available_series:
                            couples[g].append(ep)

                        series = [[tuple([data.loc[:, (at, activation, ep, bu, bs, g, lr, std)]
                                          for at in attributes])
                                   for ep in eps]
                                  for g, eps in couples.items()]

                        # Plot data
                        if training:
                            for x, y in map(lambda s: list(map(lambda l: pd.concat(l), zip(*s))), series):
                                current_ax.plot(x, y)
                            independent_variable = x + offset

                        else:
                            for t, l, tl in map(lambda s: s[-1], series):    # Take last series so that ep = max(eps)
                                current_ax.plot(t, l)
                            current_ax.plot(t, tl)
                            independent_variable = t + .01

                        # Add info
                        if i == 0 and j == 0 and len(couples.keys()) > 1:
                            fig.subplots_adjust(bottom=0.2)

                            legend_info = list(map(lambda g: f'gamma = {g}', couples.keys()))
                            if not training:
                                legend_info += ['Target load']

                            current_ax.legend(legend_info, fontsize=axis_label_fontsize,
                                              bbox_to_anchor=(.38, .005), bbox_transform=fig.transFigure,
                                              loc="lower left")
                        elif len(available_batch_sizes) == 1:
                            fig.subplots_adjust(bottom=.2)

                        if i == len(available_batch_sizes) - 1:
                            current_ax.set_xlabel("Episodes" if training else "Time step", fontsize=axis_label_fontsize)
                            ticks = np.linspace(0, independent_variable.max(), num=5)
                            current_ax.set_xticks(ticks,
                                                  map(lambda t: str(int(t)),
                                                      ticks if training else np.round(ticks * 96)),
                                                  fontsize=axis_ticks_fontsize)

                        if j == 0:
                            current_ax.set_ylabel("Reward", fontsize=axis_label_fontsize)
                            ticks = current_ax.get_yticks()
                            current_ax.set_yticks(ticks, ticks, fontsize=axis_ticks_fontsize)
                            current_ax.yaxis.set_major_formatter(formatter)

                if save_figs:
                    fig.savefig(f'{cf.FIGURES_PATH}/ddpg_plot_{"training" if training else "testing"}_'
                                f'{activation}_{std}_{bu}.{figure_format}', format=figure_format)


def ddpg_resuls(save_figs: bool = False) -> None:
    con = db.get_connection()

    try:
        df = db.get_ddpg_training_data(con)
        df = clean_ddpg_data(df, 'episode')
        make_ddpg_plot(df, True, save_figs)

        df = db.get_ddpg_validation_data(con)
        df = clean_ddpg_data(df, 'time')
        make_ddpg_plot(df, False, save_figs)

    finally:
        if con:
            con.close()


save_figures = True
if __name__ == "__main__":
    # statistical_tests()

    show_test_profiles(save_figs=save_figures)
    show_prices(save_fig=save_figures)
    plot_tabular_comparison(save_figs=save_figures)
    compare_decisions_rounds(save_fig=save_figures)
    compare_decisions_artificial(save_fig=save_figures)

    # compare_q_values(save_figs=save_figures)      # Warning: Takes a very long time!
    # ddpg_resuls(save_figs=save_figures)

    plt.show()
