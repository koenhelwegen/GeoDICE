" Imports "

# standard
cimport numpy as np
import numpy as np
import math

# project
cimport lrgd2


exp20_det_policy = [0.162, 0.16257536300366576, 0.18877002939782173, 0.21691508099701434, 0.24684763405569518, 0.2785365946786632, 0.31200793998667475, 0.3473003480039555, 0.3844644458513206, 0.42352114112527695, 0.4636591490682095, 0.5046363708192978, 0.5476019762125042, 0.5926220949326368, 0.639666144516627, 0.6887093455601284, 0.7396045523684652, 0.7923427430466343, 0.846986199933456, 0.9034802349615383, 0.9622604766238113, 0.9999999999999962, 0.9999999999999918, 0.9999999999999889, 0.9999999999999881, 0.9999999999999882, 0.9999999999999881, 0.9999999999999889, 0.9999999999999899, 0.9999999999999905, 0.9999999999999948, 0.9999999999999877, 0.9999999999999943, 0.9999999999999957, 0.999999999999995, 0.9999999999999957, 0.9999999999999958, 0.9999999999999957, 1.0, 0.9999999999999977, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


" Model "

cdef class LrGeoDice(object):
    """DICE class with geoengineering and Linear Response climate dynamics.

    The state of the model consists of a continious statevector (simply called
    'state' throughout the code) and an integer, the 'state of the world' that
    codes for the discrete variables in the system (tipping points etc).

    This version has the following states of the world:

    GEO BANNED

    real states:
    0 - not tipped
    1 - tipped

    virtual states:
    2 - not tipped, tipping risk = 0
    3 - not tipped, tipping risk = 1
    4 - anti-tipped

    GEO ALLOWED

    real states:
    5 - not tipped
    6 - tipped

    virtual states:
    7 - not tipped, tipping risk = 0
    8 - not tipped, tipping risk = 1
    9 - anti-tipped

    Additionally, this model allows the following experiments (to be selected
    at initialization):

    600 yrs, adjusted tipping risk:
    20 - base case: only mitigation
    21 - mitigation & SRM
    22 - only geo
    """

    def __init__(self, noise=False,  stepsize=1, experiment=1):
        """Initialize object."""
        # checks
        #assert self.t_max % stepsize == 0
        assert experiment in [20, 21, 22]

        # settings
        self.t_max = 600

        # initialize variables and state
        self.stepsize = stepsize
        self.noise = noise
        self.experiment = experiment
        self.__set_exogenousvariables()
        self.reset()

    def __set_exogenousvariables(self):
        """Set all exogenous variables."""

        # parameters
        pop0 = 6514.  # 2005 world population millions
        gpop0 = .035  # Growth rate of population per year
        popasym = 8600.  # Asymptotic population
        a0 = .02722  # Initial level of total factor productivity
        ga0 = .0092  # Initial growth rate for technology per year
        dela = .001  # Decline rate of technology change per year
        # Emissions
        sig0 = .13418  # CO2-equivalent emissions-GNP ratio 2005
        gsigma = -.00730  # Initial growth of sigma per year
        dsig = .003  # Decline rate of decarbonization per year
        dsig2 = .000  # Quadratic term in decarbonization
        eland0 = 1.1000  # Carbon emissions from land 2005(GtC per year)
        # Preferences
        prstp = .015  # Initial rate of social time preference per year

        # Climate model
        fex0 = -.06  # Estimate of 2000 forcings of non-CO2 GHG
        fex1 = 0.30  # Estimate of 2100 forcings of non-CO2 GHG
        # Participation
        partfract1 = 1.  # Fraction of emissions under control regime 2005
        partfract2 = 1.  # Fraction of emissions under control regime 2015
        partfract21 = 1.  # Fraction of emissions under control regime 2205
        dpartfract = 0.  # Decline rate of participation
        # Abatement cost
        expcost2 = 2.8  # Exponent of control cost function
        pback = 1.17  # Cost of backstop 2005 000$ per tC 2005
        backrat = 2.  # Ratio initial to final backstop cost
        gback = .005  # Initial cost decline backstop pc per year

        # useful vectors
        tvec = np.arange(self.t_max)
        onevec = np.ones(self.t_max)
        t_range = range(self.t_max-1)

        # VARIABLES #
        gpop = (np.exp(gpop0 * tvec)-1.)/np.exp(gpop0 * tvec)
        self.L = pop0*(1.-gpop) + gpop*popasym  # Population
        self.E_land = eland0*(1.-0.01)**tvec  # Other emissions
        self.R = (1. + prstp)**(-tvec) # Discount rate
        self.F_ex = fex0 + .01*(fex1 - fex0)*tvec # External forcing
        self.F_ex[101:] = .30
        phi = onevec*partfract21
        self.pi = phi**(1.-expcost2)  # Participation cost markup
        ga = ga0*np.exp(-dela*tvec)
        self.A = [a0]  # Total factor productivity
        [self.A.append(self.A[i]/(1.-ga[i])) for i in t_range]
        gsig = gsigma*np.exp(-dsig*tvec - dsig2*tvec**2.)
        sigma = [1.06*sig0] # ratio of (uncontrolled) ind. emissions to output
        [sigma.append(sigma[i]/(1.-gsig[i+1])) for i in t_range]
        self.sigma = np.array(sigma)
        self.theta1 = (pback*self.sigma/(expcost2*backrat)
                       * (backrat-1.+np.exp(-gback*tvec)))

    cpdef list get_action_bounds(self):
        # experiments with abatement and geoengineering

        cdef int terminal_t = 400  # after this time abatement is fixed at 1
        cdef tuple abatement_bounds, srmbounds

        # fix actions until 2015
        if self.time < 10:
            return [(.162, .162), (0, 0)]

        if self.experiment == 22 and self.SOTW >= 5:
            abatement_bounds = (0, 0)
        elif self.experiment == 22 and not self.noise:
            abatement_bounds = (0, 0)
        elif self.time < terminal_t:
            abatement_bounds = (0, 1)
        else:
            abatement_bounds = (1, 1)

        if self.SOTW >= 5 or (not self.noise and self.experiment != 20):
            srmbounds = (-1, 1)
        else:
            srmbounds = (0, 0)

        return [abatement_bounds, srmbounds]


    cpdef tuple reset(self):
        """Reset initial condition."""
        # PARAMETERS #

        # INITIAL CONDITIONS #
        self.time = 0
        self.Ccum = 0
        self.K = 137.

        self.Cp = 317.01
        self.C = [35.84, 21.71, 4.14]
        self.T = [0.466, 0.436]

        if not self.noise or self.experiment in [20]:
            self.SOTW = 0
        else:
            self.SOTW = 5

        return self.getstate(), self.SOTW

    cpdef tuple step(self, np.ndarray action, int scenario=-1):
        """Move model forward.

        action[0] is abatement (1 means fully carbon-neutral energy)
        action[1] is geoengineering (g=1 is 100 Megaton S/yr)
        """
        # INPUT #
        cdef double mu = action[0]
        cdef double g = action[1]*100.
        cdef double sr = .22

        # pick scenario
        if scenario == -1:
          if self.noise:
            propvec = self.scenario_probabilities()
            scenario = np.random.choice(list(range(len(propvec))), p=propvec)
          else:
            scenario = 0

        # handle tipping point
        cdef double J = 1
        cdef double methane_release = 0.
        if self.SOTW == 1 or self.SOTW == 6:
            J = .90
        elif self.SOTW == 4 or self.SOTW == 9:
            J = 1.1


        # handle state transition
        cdef double damge_factor = 0
        if not (self.SOTW == 1 or self.SOTW == 4) and self.noise:
            self.SOTW = scenario

        # PARAMETERS #
        cdef tuple psi = tuple([1.703e-3, 0.4, 2.56e-3, 2.42e-6]) # quadratic geoengineering damages
        #cdef tuple psi = tuple([1.703e-3, 0.4, 2.56e-3, 9.27e-5]) # linear geoengineering damages

        cdef tuple p = tuple([-0.0229, -0.0077, 0.0806])
        cdef double alpha_co2 = 5.35
        cdef double alpha_so2 = 65.
        cdef double eta_so2 = .742
        cdef double gamma_so2 = .23
        cdef double a_p = 0.2173
        cdef tuple tau_a = tuple([394.4, 36.54, 4.304])
        cdef tuple a = tuple([0.2240, 0.2824, 0.2763])
        cdef tuple b = tuple([0.126, 0.0254])
        cdef tuple tau_b = tuple([1.89, 13.6])
        cdef double g_dtat = .265  # damage coefficient for sudden T change
        cdef double dT_treshold = 0.15  # treshold sudden T change

        # Other
        cdef double fraction_controlled_external_forcing = .5 # external forcing that may be controlled
        cdef double gama = .300  # Capital elasticity in production function
        cdef double theta2 = 2.8  # Exponent of control cost function
        cdef double dk = .0650  # Depreciation rate on capital per year
        cdef double alpha = 2. # for discounting
        cdef double fosslim = 6000.  # Maximum cumulative extraction fossil fuels
        cdef double conversionrate = .46969

        # take n steps
        cdef double reward = 0.
        cdef int step, t
        cdef double e, T_AT, M_AT, f1, f_so2, P, D, Omega, Lambda, y, i, f, c, u
        for step in range(self.stepsize):

            # For readability, define t
            t = self.time

            # MODEL EQUATIONS #
            ygross = J*self.A[t]*self.L[t]**(1-gama)*self.K**gama
            e = (self.sigma[t]*(1-mu)*ygross + self.E_land[t])*conversionrate
            self.Ccum += self.sigma[t]*(1-mu)*ygross # only add fossil em.
            T_AT = sum(self.T)
            M_AT = (self.Cp + sum(self.C))
            f1 = alpha_co2*math.log(M_AT/278.)
            if g == 0:
                f_so2 = 0.
            else:
                f_so2 = -np.sign(g)*eta_so2*alpha_so2*np.exp(-(2246./np.abs(g))**(gamma_so2))

            f_methane = methane_release*.1*alpha_co2*max(T_AT - 1.5, 0)  # 300-0

            P = p[0]*f1 + p[2]*T_AT + p[1]*f_so2

            self.Cp += a_p*e
            self.C = [a[ii]*e*tau_a[ii] + (self.C[ii] - a[ii]*e*tau_a[ii])*np.exp(-1./tau_a[ii]) for ii in range(3)]
            f = f1 + f_methane + f_so2 + self.F_ex[t]*(1-fraction_controlled_external_forcing * mu)
            self.T = [b[ii]*f*tau_b[ii] + (self.T[ii] - b[ii]*f*tau_b[ii])*np.exp(-1./tau_b[ii]) for ii in range(2)]

            dT_AT = (sum(self.T) - T_AT)
            D = psi[0]*(T_AT+dT_AT)**2 + psi[1]*(P**2) + psi[2]*(M_AT/278.-1)**2 + psi[3]*abs(g)**2 # + g_dtat * max(dT_AT - dT_treshold,0)**2
            # D =  0.0028388*sum(self.T)**2 DICE DAMAGE (see)
            Omega = 1./(1 + D)
            Lambda = self.pi[t]*self.theta1[t]*mu**theta2
            y = ygross * Omega * (1.-Lambda)

            # implementation cost SRM
            y -= 0.014 * g

            i = sr*(y+.001)
            self.K = (1.-dk)*self.K + i
            c = y - i
            u = self.L[t]*((c/self.L[t])**(1.-alpha) - 1.)/(1.-alpha)
            reward += u*self.R[t] + 5e7/(self.t_max+0.0)

            self.time+=1

            done = (self.time == self.t_max)
            if done:
                break

        return tuple([self.getstate(), reward/(1e3*10)-5, done, self.SOTW])

    def abatement_cost(self, t, mu):
        theta2 = 2.8  # Exponent of control cost function

        return self.pi[t]*self.theta1[t]*mu**theta2

    def convert_utility(self, scaled_utility):
        n_steps = self.get_max_time()
        true_utility = (scaled_utility + 5*n_steps)*(1e3*10) + 5e7
        return true_utility

    def convert_discounted_reward_to_undiscounted(self, t, discounted_reward):
        y = ((discounted_reward+5)*(1e3*10) - 5e7/(self.t_max+0.0))/self.R[t*self.get_stepsize()]
        return y

    cpdef list scenario_probabilities(self):
        """Return scenario probabilities given current state.

        Make sure to also update get_number_of_worlds()!!"""

        # fixed SOTW
        if not self.noise or self.SOTW == 1 or self.SOTW == 4:
          return [1]

        # define geo risk
        cdef double p_g = .005563



        # fix risk before 2015 at zero

        cdef double TIPPING_THRESHOLD = 2.

        if self.time < 10:
            p_g = 0

        cdef double p_t = 0
        cdef double p_t_anti = 0

        p_t = self.stepsize*0.00255*(max(sum(self.T) - TIPPING_THRESHOLD,0))
        p_t_anti = self.stepsize*0.00255*(max(TIPPING_THRESHOLD - sum(self.T), 0))

        if self.SOTW == 0:
            return [1-p_t, p_t]
        elif self.SOTW == 2:
            return [1]
        elif self.SOTW == 3:
            return [1-(p_t + p_t_anti), p_t, 0, 0, p_t_anti]
        elif self.SOTW == 5:
            return [p_g*(1-p_t),
                    p_g*p_t,
                    0, 0, 0,
                    (1- p_g)*(1-p_t),
                    (1-p_g)*p_t]
        elif self.SOTW == 6:
            return [0, p_g, 0, 0, 0, 0, 1-p_g]
        elif self.SOTW == 7:
            return [p_g, 0, 0, 0, 0, 1-p_g]
        elif self.SOTW == 8:
            return [p_g*(1-(p_t + p_t_anti)), p_g*p_t, 0, 0, p_g*p_t_anti,
                    (1-p_g)*(1-(p_t + p_t_anti)), (1-p_g)*p_t, 0, 0, (1-p_g)*p_t_anti]
        elif self.SOTW == 9:
            return [0, 0, 0, 0, p_g, 0, 0, 0, 0, 1-p_g]




    cpdef np.ndarray getstate(self):
        """Get state (that can be manipulated)."""
        st = [self.K, self.Cp]
        st.extend(self.C)
        st.extend(self.T)
        return np.array(st)

    cpdef void setstate(self, t, st, sotw):
        """Set state."""
        self.time = t*self.stepsize
        self.K = st[0]
        self.Cp = st[1]
        self.C = st[2:5].tolist()
        self.T = st[5:7].tolist()
        self.SOTW = sotw

    cpdef tuple getimage(self):
        """Get image of current state (if you want to manipulate date, use
        getstate())."""
        # packs state
        image = tuple([self.time,
                       self.K,
                       self.Cp,
                       self.C,
                       self.T,
                       self.Ccum,
                       self.SOTW])
        # => when adding variables, also modify setimage()
        return image

    cpdef void setimage(self, tuple image):
        """Set full state (give as argument a tuple returned by getimage())."""
        self.time = image[0]
        self.K = image[1]
        self.Cp = image[2]
        self.C = image[3]
        self.T = image[4]
        self.Ccum = image[5]
        self.SOTW = image[6]

    def get_number_of_worlds(self):
        if self.noise:
            if self.experiment in [20]:
                return 5
            else:
                return 10
        else:
            return 1

    def get_max_time(self):
        return math.ceil(self.t_max/(self.stepsize+0.0))

    def get_state_dimension(self):
        return 7

    def get_stepsize(self):
        return self.stepsize

    def get_experiment(self):
        return self.experiment

    def get_timevec(self):
        year0 = 2005
        return list(range(year0+0, year0+self.t_max, self.stepsize))

    def add_marginal_carbon(self, dx=1):
        cdef double a_p = 0.2173
        cdef tuple a = tuple([0.2240, 0.2824, 0.2763])
        cdef double conversionrate = .46969

        self.Cp += dx*a_p*conversionrate
        for ii in range(len(self.C)):
            self.C[ii] += dx*a[ii]*conversionrate

    def add_marginal_capital(self, dx=1):
        self.K += dx*1

    def add_marginal_variable(self, var, dx=1):
        if var == 0:
            self.K += dx
        elif var == 1:
            self.Cp += dx
        elif var >= 2 and var <= 4:
            self.C[var-2] += dx
        elif var >= 5 and var <= 6:
            self.T[var-5] += dx
        else:
            raise ValueError('Bad value for var.')

    def get_modeltype(self):
        return "lrgd2"

    def get_modelcode(self):
        return 2

    def get_exogenousvar(self, var):
        if var == 'L':
            return self.L
        if var == 'A':
            return self.A
