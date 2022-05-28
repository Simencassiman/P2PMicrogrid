# Python Libraries
import re
from typing import List
import json
from datetime import datetime
from calendar import monthrange

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as mc  # For the legend
from matplotlib.cm import ScalarMappable    # For the legend
from matplotlib.ticker import MultipleLocator

# Local modules
import config as cf
from agent import ActingAgent
import database as db
import dataset as ds


# Config
np.random.seed(42)

# Text document settings
primary_color = '#000'
secondary_color = '#ccc'
neutral_color = '#777'
base_color = '#000'

title_fontsize = 9
axis_label_fontsize = 8
axis_ticks_fontsize = 6

# Poster settings
# secondary_color = '#51bcebff'
# tertiary_color = '#1d8dafff'
# neutral_color = '#777'
# base_color = '#2f4d5dff'
# title_fontsize = 16
# axis_label_fontsize = 12
# primary_color = '#004079ff'
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
            (cf.GRID_COST_AVG
             + cf.GRID_COST_AMPLITUDE
             * np.sin(time / 96 * 2 * np.pi * cf.HOURS_PER_DAY / cf.GRID_COST_PERIOD - cf.GRID_COST_PHASE)
             ) / cf.CENTS_PER_EURO  # from c€ to €
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
    ax.text(1.3, 1.19, 'Semi-intelligent', color=base_color, fontsize=axis_label_fontsize)
    ax.text(1.48, 1.85, 'Rule-based', color=base_color, fontsize=axis_label_fontsize)

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
    ax.text(1.4, 1.19, 'Semi-intelligent', color=base_color, fontsize=axis_label_fontsize)
    ax.text(1.48, 1.85, 'Rule-based', color=base_color, fontsize=axis_label_fontsize)

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
            (cf.GRID_COST_AVG
             + cf.GRID_COST_AMPLITUDE
             * np.sin(time / 96 * 2 * np.pi * cf.HOURS_PER_DAY / cf.GRID_COST_PERIOD - cf.GRID_COST_PHASE)
             ) / cf.CENTS_PER_EURO  # from c€ to €
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
    ax[0].set_yticks([-4, 0, 4], [-4.0, 0.0, 4.0], fontsize=axis_ticks_fontsize)
    ax[0].set_ylabel("Power [kW]", color=base_color, fontsize=axis_label_fontsize)
    ax[0].legend(["Base Load", "PV", "Net Consumption"], labelcolor=base_color, fontsize=axis_label_fontsize,
                 bbox_to_anchor=(1.02, 1.02), loc="upper left")

    # Prices
    ax12 = ax[1].twinx()
    ax12.plot(time, grid_price, color=secondary_color)
    ax12.plot(time, injection_price, ':', color=secondary_color)
    ax12.plot(time, p2p_price, '--', color=secondary_color)
    ax[1].plot(time, timeslot_info.loc[:, ('cost', 0)], color=primary_color)
    ax[1].set_yticks([-.1, 0, .1], [-0.1, 0.0, 0.1], fontsize=axis_ticks_fontsize)
    ax12.set_yticks([0.07, 0.12, 0.17], [0.07, 0.12, 0.17], color=secondary_color, fontsize=axis_ticks_fontsize)
    ax[1].set_ylabel("Cost [€]", color=primary_color, fontsize=axis_label_fontsize)
    ax[1].set_zorder(1)
    ax[1].set_frame_on(False)
    ax12.set_ylabel("Price [€/kWh]", color=secondary_color, fontsize=axis_label_fontsize)
    ax12.legend(["Offtake", "Injection", "P2P"], labelcolor=base_color, fontsize=axis_label_fontsize,
                bbox_to_anchor=(1.15, 1), loc="upper left")

    # Heat pump
    ax[2].bar(time, timeslot_info.loc[:, ('heatpump', 0)], width=1.0, color=primary_color)
    ax[2].set_yticks([0, 1.5, 3], [0.0, 1.5, 3.0], fontsize=axis_ticks_fontsize)
    ax[2].set_ylabel("HP [kW]", color=base_color, fontsize=axis_label_fontsize)

    # Temperature
    ax[3].plot(time, timeslot_info.loc[:, ('temperature', 0)], color=primary_color)
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
            (cf.GRID_COST_AVG
             + cf.GRID_COST_AMPLITUDE
             * np.sin(time / 96 * 2 * np.pi * cf.HOURS_PER_DAY / cf.GRID_COST_PERIOD - cf.GRID_COST_PHASE)
             ) / cf.CENTS_PER_EURO  # from c€ to €
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
    ax[0].set_yticks([-4, 0, 4], [-4.0, 0.0, 4.0], fontsize=axis_ticks_fontsize)
    ax[0].set_ylabel("Power [kW]", color=base_color, fontsize=axis_label_fontsize)
    ax[0].legend(["Base Load", "PV", "Net Consumption"], labelcolor=base_color, fontsize=axis_label_fontsize,
                 bbox_to_anchor=(1.02, 1.02), loc="upper left")

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
                bbox_to_anchor=(1.15, 1), loc="upper left")

    # Heat pump
    ax[2].bar(time, timeslot_info.loc[:, ('heatpump', 0)], width=1.0, color=primary_color)
    ax[2].set_yticks([0, 1.5, 3], [0.0, 1.5, 3.0], fontsize=axis_ticks_fontsize)
    ax[2].set_ylabel("HP [kW]", color=base_color, fontsize=axis_label_fontsize)

    # Temperature
    ax[3].plot(time, timeslot_info.loc[:, ('temperature', 0)], color=primary_color)
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
            (cf.GRID_COST_AVG
             + cf.GRID_COST_AMPLITUDE
             * np.sin(time / 96 * 2 * np.pi * cf.HOURS_PER_DAY / cf.GRID_COST_PERIOD - cf.GRID_COST_PHASE)
             ) / cf.CENTS_PER_EURO  # from c€ to €
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
    ax[2].set_title("agent 0", fontsize=axis_label_fontsize, loc='right', pad=-.1)
    ax[2].bar(time - width / 2, timeslot_info.loc[:, ('heatpump', f'2-multi-agent-com-rounds-1-{setting}', 0)],
              label='Communication', width=width, color=primary_color)
    ax[2].bar(time + width / 2, timeslot_info.loc[:, ('heatpump', f'2-multi-agent-no-com-{setting}', 0)],
              label='No communication', width=width, color=secondary_color)
    ax[2].set_yticks([0, 1.5, 3], [0.0, 1.5, 3.0], fontsize=axis_ticks_fontsize)
    ax[2].set_ylabel("HP [kW]", color=base_color, fontsize=axis_label_fontsize)
    ax[2].legend(["Communication", "No communication"], labelcolor=base_color,
                 fontsize=axis_label_fontsize, bbox_to_anchor=(1.02, 0.), loc="lower left")

    # agent 1
    ax[3].set_title("agent 1", fontsize=axis_label_fontsize, loc='right', pad=-.1)
    ax[3].bar(time - width / 2, timeslot_info.loc[:, ('heatpump', f'2-multi-agent-com-rounds-1-{setting}', 1)],
              label='Communication', width=width, color=primary_color)
    ax[3].bar(time + width / 2, timeslot_info.loc[:, ('heatpump', f'2-multi-agent-no-com-{setting}', 1)],
              label='No communication', width=width, color=secondary_color)
    ax[3].set_yticks([0, 1.5, 3], [0.0, 1.5, 3.0], fontsize=axis_ticks_fontsize)
    ax[3].set_ylabel("HP [kW]", color=base_color, fontsize=axis_label_fontsize)

    # Temperature
    # Agent 0
    ax[4].set_title("agent 0", fontsize=axis_label_fontsize, loc='right', pad=-.1)
    ax[4].plot(time, timeslot_info.loc[:, ('temperature', f'2-multi-agent-com-rounds-1-{setting}', 0)],
               color=primary_color)
    ax[4].plot(time, timeslot_info.loc[:, ('temperature', f'2-multi-agent-no-com-{setting}', 0)],
               color=secondary_color)
    ax[4].set_yticks([20, 22], [20, 22], fontsize=axis_ticks_fontsize)
    ax[4].hlines(y=[20, 22], xmin=0, xmax=96, color=neutral_color, linestyle='--', linewidths=0.8)
    ax[4].legend(["Communication", "No communication"], labelcolor=base_color,
                 fontsize=axis_label_fontsize, bbox_to_anchor=(1.02, 0), loc="lower left")

    # Agent 1
    ax[5].set_title("agent 1", fontsize=axis_label_fontsize, loc='right', pad=-.3)
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
    ax.legend(['Single agent', '2 agent no-com homogeneous', '2 agent no-com heterogeneous',
               '2 agent com homogeneous', '2 agent com heterogeneous', '5 agent com heterogeneous'],
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
    fig, ax = plt.subplots(figsize=(2.5, 3))
    plt.title("Average cost vs. community scale", color=base_color, fontsize=title_fontsize, loc='center')
    fig.subplots_adjust(left=0.16, bottom=.16)

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
    settings = ['2-multi-agent-com-rounds-1-hetero', '2-multi-agent-com-rounds-2-hetero',
                '2-multi-agent-com-rounds-3-hetero']
    df = df[df['setting'].isin(settings)]

    df = df[['setting', 'agent', 'day', 'cost']]\
        .groupby(['setting', 'agent', 'day']).sum()\
        .groupby(['setting', 'agent']).mean()
    costs = df.groupby(['setting']).mean().rename(columns={'cost': 'mean'})
    costs['std'] = df.groupby(['setting']).std()
    costs['rounds'] = costs.index.map(lambda s: get_rounds(s)).astype(int)

    plt.rcParams['axes.titlepad'] = 14  # pad is in points...
    fig, ax = plt.subplots(figsize=(3, 3))
    plt.title("Average cost vs. number of decision rounds", color=base_color, fontsize=title_fontsize)
    fig.subplots_adjust(left=0.2, right=.8)

    ax.errorbar(costs['rounds'], costs['mean'], costs['std'], linestyle='none', marker='.', capsize=5, color=base_color)
    ax.set_xlim(.75, 3.25)
    ax.set_xlabel("Number of rounds", color=base_color, fontsize=axis_label_fontsize)
    ax.set_xticks([1, 2, 3], [1, 2, 3], fontsize=axis_ticks_fontsize)
    ax.set_ylim(1.0, 2.0)
    ax.set_yticks([0, .5, 1.0, 1.5, 2], [0.0, 0.5, 1.0, 1.5, 2.0], fontsize=axis_ticks_fontsize)
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
                (cf.GRID_COST_AVG
                 + cf.GRID_COST_AMPLITUDE
                 * np.sin(time / 96 * 2 * np.pi * cf.HOURS_PER_DAY / cf.GRID_COST_PERIOD - cf.GRID_COST_PHASE)
                 ) / cf.CENTS_PER_EURO  # from c€ to €
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
        ax[2].set_title("agent 0", fontsize=axis_label_fontsize, loc='right', pad=-.1)
        ax[2].bar(time - width / 2, timeslot_info.loc[:, ('heatpump', f'2-multi-agent-com-rounds-1-{setting}', 0)],
                  label='Communication', width=width, color=primary_color)
        ax[2].bar(time + width / 2, timeslot_info.loc[:, ('heatpump', f'2-multi-agent-no-com-{setting}', 0)],
                  label='No communication', width=width, color=secondary_color)
        ax[2].set_yticks([0, 1.5, 3], [0.0, 1.5, 3.0], fontsize=axis_ticks_fontsize)
        ax[2].set_ylabel("HP [kW]", color=base_color, fontsize=axis_label_fontsize)
        ax[2].legend(["Communication", "No communication"], labelcolor=base_color,
                     fontsize=axis_label_fontsize, bbox_to_anchor=(1.02, 0.), loc="lower left")

        # agent 1
        ax[3].set_title("agent 1", fontsize=axis_label_fontsize, loc='right', pad=-.1)
        ax[3].bar(time - width / 2, timeslot_info.loc[:, ('heatpump', f'2-multi-agent-com-rounds-1-{setting}', 1)],
                  label='Communication', width=width, color=primary_color)
        ax[3].bar(time + width / 2, timeslot_info.loc[:, ('heatpump', f'2-multi-agent-no-com-{setting}', 1)],
                  label='No communication', width=width, color=secondary_color)
        ax[3].set_yticks([0, 1.5, 3], [0.0, 1.5, 3.0], fontsize=axis_ticks_fontsize)
        ax[3].set_ylabel("HP [kW]", color=base_color, fontsize=axis_label_fontsize)

        # Temperature
        # Agent 0
        ax[4].set_title("agent 0", fontsize=axis_label_fontsize, loc='right', pad=-.1)
        ax[4].plot(time, timeslot_info.loc[:, ('temperature', f'2-multi-agent-com-rounds-1-{setting}', 0)],
                   color=primary_color)
        ax[4].plot(time, timeslot_info.loc[:, ('temperature', f'2-multi-agent-no-com-{setting}', 0)],
                   color=secondary_color)
        ax[4].set_yticks([20, 22], [20, 22], fontsize=axis_ticks_fontsize)
        ax[4].hlines(y=[20, 22], xmin=0, xmax=96, color=neutral_color, linestyle='--', linewidths=0.8)
        ax[4].legend(["Communication", "No communication"], labelcolor=base_color,
                     fontsize=axis_label_fontsize, bbox_to_anchor=(1.02, 0), loc="lower left")

        # Agent 1
        ax[5].set_title("agent 1", fontsize=axis_label_fontsize, loc='right', pad=-.3)
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

        # fig.tight_layout()

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
        decisions = df[(df['agent'] == 0) & (df['setting'] == '2-multi-agent-com-rounds-3-hetero') & (df['day'] == 8)] \
            .pivot(index=['time'], columns=['round'], values=['decision'])

        df = db.get_test_results(con)
        df = df[(df['agent'] == 0) & (df['setting'] == '2-multi-agent-com-rounds-3-hetero') & (df['day'] == 8)]
        df[['load', 'pv']] = df[['load', 'pv']] * 1e-3
        df['time'] = df['time'].map(lambda t: t * 24)
        df['heatpump'] *= 1e-3
        timeslot_info = df.pivot(index=['time'], columns=['setting', 'agent'],
                                 values=['load', 'pv', 'temperature', 'heatpump'])
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

        # Make plot
        fig, ax = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
        fig.suptitle("Agent decisions for each round of the time slot", color=base_color, fontsize=title_fontsize)

        # Powers
        ax[0].plot(time, timeslot_info.loc[:, ('load', '2-multi-agent-com-rounds-3-hetero', 0)], color=primary_color)
        ax[0].plot(time, timeslot_info.loc[:, ('pv', '2-multi-agent-com-rounds-3-hetero', 0)], color=secondary_color)
        ax[0].set_yticks([-4, 0, 4], [-4.0, 0.0, 4.0])
        ax[0].set_ylabel("Power [kW]", color=base_color, fontsize=axis_label_fontsize)
        ax[0].legend(["Base Load", "PV"], labelcolor=base_color, bbox_to_anchor=(1.04, 1), loc="upper left")

        # Prices
        ax[1].plot(time, grid_price, color=primary_color)
        ax[1].plot(time, injection_price, color=secondary_color)
        ax[1].plot(time, p2p_price, '--', color=primary_color)
        ax[1].set_yticks([0.07, 0.12, 0.17])
        ax[1].set_ylabel("Price [€]", color=base_color, fontsize=axis_label_fontsize)
        ax[1].legend(["Offtake", "Injection", "P2P"], labelcolor=base_color, bbox_to_anchor=(1.04, 1), loc="upper left")

        # Heat pump
        width = 0.2
        x = np.array(decisions.index * 96)

        ax[2].bar(x - 1.5 * width, decisions.loc[:, ('decision', 0)], label='Round 0', width=width)
        ax[2].bar(x - 0.5 * width, decisions.loc[:, ('decision', 1)], label='Round 1', width=width)
        ax[2].bar(x + 0.5 * width, decisions.loc[:, ('decision', 2)], label='Round 2', width=width)
        ax[2].bar(x + 1.5 * width, decisions.loc[:, ('decision', 3)], label='Round 3', width=width)

        ax[2].set_yticks([0, 1.5, 3], [0.0, 1.5, 3.0])
        ax[2].set_ylabel("HP [kW]", color=base_color, fontsize=axis_label_fontsize)
        ax[2].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        # Temperature
        ax[3].plot(time, timeslot_info.loc[:, ('temperature', '2-multi-agent-com-rounds-3-hetero', 0)],
                   color=primary_color)
        ax[3].set_ylabel("Temperature [°C]", color=base_color, fontsize=axis_label_fontsize)
        ax[3].set_xticks([0, 24, 48, 72, 95], ["00:00", "06:00", "12:00", "18:00", "23:45"])
        ax[3].set_xlabel("Time", color=base_color, fontsize=axis_label_fontsize)
        ax[3].set_yticks([20, 22])
        ax[3].hlines(y=[20, 22], xmin=0, xmax=96, color=neutral_color, linestyle='--', linewidths=0.8)
        ax[3].xaxis.set_minor_locator(MultipleLocator(1))

        fig.tight_layout()

        if save_fig:
            plt.savefig(f'{cf.FIGURES_PATH}/rounds_plot.{figure_format}', format=figure_format)

    finally:
        if con:
            con.close()


def compare_decisions_artificial(save_fig: bool = False) -> None:
    con = db.get_connection()

    try:
        df = db.get_test_results(con)
        df = df[(df['setting'].isin(['2-agent-pv-drop-com', '2-agent-pv-drop-no-com'])) & (df['day'] == 8)]

        df[['load', 'pv']] = df[['load', 'pv']] * 1e-3
        df['time'] = df['time'].map(lambda t: t * 24)
        df['heatpump'] *= 1e-3
        timeslot_info = df.pivot(index=['time'], columns=['setting', 'agent'],
                                 values=['load', 'pv', 'temperature', 'heatpump'])
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

        # Make plot
        fig, ax = plt.subplots(6, 1, figsize=(10, 8), sharex=True)
        fig.suptitle("Agent's state and decisions throughout the day", color=base_color, fontsize=title_fontsize)

        # Powers
        ax[0].plot(time, timeslot_info.loc[:, ('load', '2-agent-pv-drop-com', 0)], color=primary_color)
        ax[0].plot(time, timeslot_info.loc[:, ('load', '2-agent-pv-drop-com', 1)], color=secondary_color)
        ax[0].plot(time, timeslot_info.loc[:, ('pv', '2-agent-pv-drop-com', 0)], '--', color=primary_color)
        ax[0].set_yticks([-4, 0, 4], [-4.0, 0.0, 4.0])
        ax[0].set_ylabel("Power [kW]", color=base_color, fontsize=axis_label_fontsize)
        ax[0].legend(["Base Load Agent 0", "Base Load Agent 1", "PV"], labelcolor=base_color,
                     bbox_to_anchor=(1.04, 1), loc="upper left")

        # Prices
        ax[1].plot(time, grid_price, color=primary_color)
        ax[1].plot(time, injection_price, color=secondary_color)
        ax[1].plot(time, p2p_price, '--', color=primary_color)
        ax[1].set_yticks([0.07, 0.12, 0.17])
        ax[1].set_ylabel("Price [€]", color=base_color, fontsize=axis_label_fontsize)
        ax[1].legend(["Offtake", "Injection", "P2P"], labelcolor=base_color, bbox_to_anchor=(1.04, 1), loc="upper left")

        # Heat pump
        width = 0.4

        # agent 0
        ax[2].set_title("agent 0", fontsize=10, loc='right')
        ax[2].bar(time - width / 2, timeslot_info.loc[:, ('heatpump', '2-agent-pv-drop-com', 0)],
                  label='Communication', width=width, color=primary_color)
        ax[2].bar(time + width / 2, timeslot_info.loc[:, ('heatpump', '2-agent-pv-drop-no-com', 0)],
                  label='No communication', width=width, color=secondary_color)
        ax[2].set_yticks([0, 1.5, 3], [0.0, 1.5, 3.0])
        ax[2].set_ylabel("HP [kW]", color=base_color, fontsize=axis_label_fontsize)
        ax[2].legend(["Communication", "No communication"], labelcolor=base_color,
                     bbox_to_anchor=(1.04, 0), loc="lower left")

        # agent 1
        ax[3].set_title("agent 1", fontsize=10, loc='right')
        ax[3].bar(time - width / 2, timeslot_info.loc[:, ('heatpump','2-agent-pv-drop-com', 1)],
                  label='Communication', width=width, color=primary_color)
        ax[3].bar(time + width / 2, timeslot_info.loc[:, ('heatpump', '2-agent-pv-drop-no-com', 1)],
                  label='No communication', width=width, color=secondary_color)
        ax[3].set_yticks([0, 1.5, 3], [0.0, 1.5, 3.0])
        ax[3].set_ylabel("HP [kW]", color=base_color, fontsize=axis_label_fontsize)

        # Temperature
        ax[4].set_title("agent 0", fontsize=10, loc='right')
        ax[4].plot(time, timeslot_info.loc[:, ('temperature', '2-agent-pv-drop-com', 0)],
                   color=primary_color)
        ax[4].plot(time, timeslot_info.loc[:, ('temperature', '2-agent-pv-drop-no-com', 0)],
                   color=secondary_color)
        ax[4].set_yticks([20, 22])
        ax[4].hlines(y=[20, 22], xmin=0, xmax=96, color=neutral_color, linestyle='--', linewidths=0.8)
        ax[4].legend(["Communication", "No communication"], labelcolor=base_color,
                     bbox_to_anchor=(1.04, 0), loc="lower left")

        ax[5].set_title("agent 1", fontsize=10, loc='right')
        ax[5].plot(time, timeslot_info.loc[:, ('temperature', '2-agent-pv-drop-com', 1)],
                   color=primary_color)
        ax[5].plot(time, timeslot_info.loc[:, ('temperature', '2-agent-pv-drop-no-com', 1)],
                   color=secondary_color)
        ax[5].set_ylabel("       Temperature [°C]", loc='bottom', color=base_color, fontsize=axis_label_fontsize)
        ax[5].set_xticks([0, 24, 48, 72, 95], ["00:00", "06:00", "12:00", "18:00", "23:45"])
        ax[5].set_xlabel("Time", color=base_color, fontsize=axis_label_fontsize)
        ax[5].set_yticks([20, 22])
        ax[5].hlines(y=[20, 22], xmin=0, xmax=96, color=neutral_color, linestyle='--', linewidths=0.8)
        ax[5].xaxis.set_minor_locator(MultipleLocator(1))

        fig.tight_layout()

        if save_fig:
            plt.savefig(f'{cf.FIGURES_PATH}/artificial_pv_decisions_plot.{figure_format}', format=figure_format)

    finally:
        con.close()


def compare_q_values() -> None:
    q_table = np.load(f'../models_tabular/2_multi_agent_com_rounds_3_hetero_0.npy')
    # q_table = np.load(f'../models_tabular/single_agent_0.npy')
    q_table /= np.abs(q_table).max()

    for i in range(20):
        fig, ax = plt.subplots(1, 20, figsize=(13, 4), sharey=True)
        fig.suptitle("Time", fontsize=13)
        ax[0].set_yticks(list(range(20)))
        ax[0].set_ylabel("Temperature")

        for t in range(20):
            # fig.suptitle(f't={t}')
            # im = ax[t].imshow(q_table[t, :, 0, :] - q_table.mean(), cmap='seismic',
            #                      norm=matplotlib.colors.SymLogNorm(10**-4, vmin=-1, vmax=1))
            im = ax[t].imshow(q_table[t, :, 8, i, :] - q_table.mean(), cmap='seismic',
                              norm=matplotlib.colors.SymLogNorm(10 ** -4, vmin=-1, vmax=1))
            if t == 10:
                ax[t].set_xlabel("Action")
            # ax.set_yscale('symlog', linthresh=0.02)
            ax[t].set_xticks(list(range(3)))

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
        fig.colorbar(im, cax=cbar_ax)


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

    print('Analysis of the influence of community scale:')
    print(f'Same variance: p-value = {p_levene}')
    print(f'Same mean: p-value = {p_anova}')


def statistics_nr_rounds(df: pd.DataFrame) -> None:
    settings = ['2-multi-agent-com-rounds-1-hetero', '2-multi-agent-com-rounds-2-hetero',
                '2-multi-agent-com-rounds-3-hetero']
    df = df[df['setting'].isin(settings)]

    df = df[['setting', 'agent', 'day', 'cost']] \
        .groupby(['setting', 'agent', 'day']).sum() \
        .groupby(['setting', 'agent']).mean()

    df['rounds'] = df.index.map(lambda s: get_rounds(s[0])).astype(int)
    rounds = df.reset_index()

    samples = [np.array(rounds.loc[rounds['rounds'] == i, 'cost']) for i in pd.unique(rounds['rounds'])]
    _, p_levene = stats.levene(*samples)
    _, p_anova = stats.f_oneway(*samples)

    print('Analysis of the influence of rounds:')
    print(f'Same variance: p-value = {p_levene}')
    print(f'Same mean: p-value = {p_anova}')


def statistical_tests() -> None:
    con = db.get_connection()

    try:
        df = db.get_test_results(con)
        df[['load', 'pv']] = df[['load', 'pv']] * 1e-3
        df['time'] = df['time'].map(lambda t: t * 24)
        df['heatpump'] *= 1e-3

        statistics_baselines(df)
        statistics_community_scale(df)
        statistics_nr_rounds(df)

    finally:
        if con:
            con.close()


save_figures = False
if __name__ == "__main__":
    statistical_tests()

    # show_test_profiles(save_figs=save_figures)
    # show_prices(save_fig=save_figures)
    # plot_tabular_comparison(save_figs=save_figures)
    # compare_decisions_artificial(save_fig=save_figures)
    # compare_decisions_rounds(save_fig=save_figures)
    # compare_q_values()

    # plt.show()
