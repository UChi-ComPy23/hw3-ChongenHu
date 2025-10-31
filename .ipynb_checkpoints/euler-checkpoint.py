"""
Defintions for problem 0
"""

import numpy as np
import scipy.integrate
from scipy.integrate import DenseOutput
from scipy.interpolate import interp1d
from warnings import warn

class ForwardEulerOutput(DenseOutput):
    def __init__(self,t_old,t,y_old,y):
        #extends
        super(ForwardEulerOutput, self).__init__(t_old,t)
        self.y0=np.asarray(y_old)
        self.y1=np.asarray(y)
        
    #sol(t)=sol._call_impl(t)
    def _call_impl(self,t):
        d=(np.asarray(t)-self.t_old)/(self.t-self.t_old)
        y=(1-d)*self.y0[:, None] + d * self.y1[:, None]
        #avoid error occurs when input single value by "docs.scipy.org"
        return y if np.ndim(t) > 0 else y[:, 0]
                             

class ForwardEuler(scipy.integrate.OdeSolver):
    
    def __init__(self,fun,t0,y0,t_bound,vectorized,support_complex=False,h=0.01,**extraneous):
        # ForwardEuler extends scipy.integrate.OdeSolver parameters
        super(ForwardEuler,self).__init__(fun,t0,y0,t_bound,vectorized,support_complex)
        self.h=(t_bound-t0)/100

        # no Jacobian no LU, by "You won't use a Jacobian, so njev and nlu can remain at 0."
        self.njev=0
        self.nlu=0 

        # by "direction should be +1"
        self.direction=1

        self._last_dense=None
        
    # A solver must implement a private method _step_impl(self)
    # from t to t+h
    def _step_impl(self):
        t_old=self.t
        y_old=self.y
    
        # last step, use t_bound to avoid h over domain
        if self.direction > 0:
            h = min(self.h,self.t_bound-self.t)  
        else:
            h = max(-self.h,self.t_bound-self.t)
    
        f = self.fun(t_old,y_old)        
        y_new=y_old+h*f
        t_new=t_old+h
    
        self.t=t_new
        self.y=y_new
        # step_size no need to evaluate

        # store t_old, t_new, y_old, y_new
        self._last_dense=ForwardEulerOutput(t_old,t_new,y_old,y_new)

        return True, None
    
    # A solver must implement a private method _dense_output_impl(self)
    def _dense_output_impl(self):
        return self._last_dense