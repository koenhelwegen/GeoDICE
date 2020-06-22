cimport numpy as np

cdef class LrGeoDice(object):

      # settings
      cdef int t_max
      cdef public int noise
      cdef int stepsize
      cdef int experiment

      # state
      cdef int time
      cdef double Ccum
      cdef double K
      cdef double Cp
      cdef list C
      cdef list T
      cdef public int SOTW

      # exogenous variables
      cdef np.ndarray L
      cdef np.ndarray E_land
      cdef np.ndarray R
      cdef np.ndarray F_ex
      cdef np.ndarray pi
      cdef list A
      cdef np.ndarray sigma
      cdef np.ndarray theta1

      # methods
      cpdef tuple reset(self)
      cpdef tuple step(self, np.ndarray action, int scenario=*)
      cpdef np.ndarray getstate(self)
      cpdef void setstate(self, t, st, sotw)
      cpdef list scenario_probabilities(self)
      cpdef tuple getimage(self)
      cpdef void setimage(self, tuple img)
      cpdef list get_action_bounds(self)
