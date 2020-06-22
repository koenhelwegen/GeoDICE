import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import bisect

import lib.models.tools as modeltools

modeltype = "lrgeodice"
experiment = 10

risk = np.linspace(0, 0.02, 1000)
plt.plot(risk, 0.5 * np.ones(len(risk)), "--g")
plt.plot([0.01, 0.01], [0, 1], "--g")


for threshold in [1, 1.5, 2]:
    for stepsize in [1, 10]:

        env = modeltools.load_model(
            modeltype=modeltype, noise=False, stepsize=stepsize, experiment=experiment
        )

        optimal = modeltools.load_optimal_pathway(env)

        prop_nontipping = []

        for r in risk:

            p = 1
            for t in range(0, int(400 / stepsize)):
                T_AT = sum(optimal.trajectory[t][5:7])
                if T_AT < threshold:
                    continue
                p = p * (1 - stepsize * r * (T_AT - threshold))
            prop_nontipping.append(p)

        plt.plot(risk, prop_nontipping, label="Threshold {}".format(threshold))

        f = interp1d(risk, prop_nontipping)
        try:
            res = bisect(lambda x: f(x) - 0.5, 0, 0.02)
            print("Threshold {}: {}".format(threshold, res))
            plt.plot(res, f(res), ".r", markersize=10)
        except:
            print("Threshold {}: not found".format(threshold))


plt.xlim([min(risk), max(risk)])
plt.ylim([0, 1])
plt.xlabel("Probablity of tipping per degree above threshold per timestep")
plt.ylabel("Probability of not tipping over 400 years")
plt.legend()
plt.show()
