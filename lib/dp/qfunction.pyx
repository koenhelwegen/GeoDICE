cimport numpy as np
cimport lib.approx.chebyshev as cheb
cimport lib.approx.tools as approxtools
cimport qfunction

cdef class QFunction(object):
    """The QFunction class allows for quick evaluation of the Q-function (value
    of an state-action pair). It is intented to be used within a "with q:"
    contruction, to ensure environment is unaffected."""

    def __init__(self, env, int t, np.ndarray x, int sotw, valuefun):
        """Initialize the Q function by setting its state and other info."""

        self.env = env
        self.t = t
        self.x = x
        self.sotw = sotw
        self.valuefun = valuefun
        self.env.setstate(self.t, self.x, self.sotw)
        self.scenario_vector = self.env.scenario_probabilities()
        self.n_scenarios = len(self.scenario_vector)

        # if self.sotw == 3:
        #     print(sum(x[5:7]), self.scenario_vector)

    def __enter__(self):
        """Store image of the environment before usage."""
        self.img = self.env.getimage()

    def __exit__(self, except_type, exept_value, traceback):
        """Restore image of the environment after usage."""
        self.env.setimage(self.img)

    cpdef double eval(self, np.ndarray action):
        """Evaluate the value of a state-action pair."""

        cdef np.ndarray next_state
        cdef double reward, q, x

        q = 0  # keep track of value over all scenarios

        for scenario in range(self.n_scenarios):
            "Evaluate value for each possible scenario."

            # ignore scenarios with zero probability:
            if self.scenario_vector[scenario] == 0:
              continue

            # set state & take step
            self.env.setstate(self.t, self.x, self.sotw)
            next_state, reward, _, next_sotw = self.env.step(action, scenario)

            # evalute value and add to total
            q += self.scenario_vector[scenario]*(reward + self.valuefun(self.t+1, next_state, next_sotw))
        return -q
