# Python Libraries
import json
from datetime import datetime
from calendar import monthrange

import numpy as np
import matplotlib.pyplot as plt

# Local modules


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
    with open('data/data.json') as in_file, open('data/profiles_1.json', 'w') as out_file:
        data = json.load(in_file)

        time = [date for t, date in enumerate(data['time']) if t % 15 == 0]

        p = np.array(data['values'])
        p = np.append(p, p[-1]).reshape((-1, 15)).mean(axis=1).transpose()
        p = (p[1:] - p[0:-1])   # np.diff
        p /= p.max()

        json.dump({'time': time[1:], 'values': p.tolist()}, out_file)

def expand_irradiation() -> None:

    with open('data/irradiance_tmp.json') as file, open('data/irradiation_daily.json', 'w') as out:
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
    print('Nothing to do...')


