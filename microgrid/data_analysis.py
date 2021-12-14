# Python Libraries
import json
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# Local modules


if __name__ == "__main__":

    with open('data/data.json') as file:
        data = json.load(file)

        time = [datetime.fromisoformat(date) for date in data['time']]

        t = np.array(time)[[i for i in range(len(time)) if i % 15 == 0]]
        p = np.array(data['values'])
        p = np.append(p, p[-1]).reshape((-1, 15)).mean(axis=1).transpose()
        p = (p[1:] - p[0:-1])

        plt.plot(t[1:], p)
        plt.show()
