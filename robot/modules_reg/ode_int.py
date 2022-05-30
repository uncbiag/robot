from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torchdiffeq
from robot.utils.module_parameters import ParameterDict


class ODEBlock(nn.Module):
    """

        A interface class for torchdiffeq, https://github.com/rtqichen/torchdiffeq
        we add some constrains in torchdiffeq package to avoid collapse or traps, so this local version is recommended
        the solvers supported by the torchdiffeq are listed as following
        SOLVERS = {
        'explicit_adams': AdamsBashforth,
        'fixed_adams': AdamsBashforthMoulton,
        'adams': VariableCoefficientAdamsBashforth,
        'tsit5': Tsit5Solver,
        'dopri5': Dopri5Solver,
        'euler': Euler,
        'midpoint': Midpoint,
        'rk4': RK4,
    }

    """

    def __init__(self, param=ParameterDict()):
        super(ODEBlock, self).__init__()
        self.odefunc = None
        self.interp_mode = param[("interp_mode", False, "interpolation along time")]
        """the ode problem to be solved"""
        tFrom = param[("tFrom", 0.0, "time to solve a model from")]
        """time to solve a model from"""
        tTo = param[("tTo", 1.0, "time to solve a model to")]
        """time to solve a model to"""
        if not self.interp_mode:
            self.integration_time = torch.Tensor([tFrom, tTo]).float()
        else:
            integration_time = param[
                ("integration_time", [0.0, 1.0], "time interval to interpolated")
            ]
            self.integration_time = torch.Tensor(integration_time).float()
        """intergration time, list, typically set as [0,1]"""
        self.method = param[("solver", "dopri5", "ode solver")]
        """ solver,rk4 as default, supported list: explicit_adams,fixed_adams,tsit5,dopri5,euler,midpoint, rk4 """
        self.adjoin_on = param[("adjoin_on", True, "use adjoint optimization")]
        """ adjoint method, benefits from memory consistency, which can be refer to "Neural Ordinary Differential Equations" """
        self.rtol = param[("rtol", 1e-5, "relative error tolerance for dopri5")]
        """ relative error tolerance for dopri5"""
        self.atol = param[("atol", 1e-5, "absolute error tolerance for dopri5")]
        """ absolute error tolerance for dopri5"""
        self.n_step = param[
            (
                "number_of_time_steps",
                20,
                "Number of time-steps to per unit time-interval integrate the ode",
            )
        ]
        """ Number of time-steps to per unit time-interval integrate the PDE, for fixed time-step solver, i.e. rk4"""
        self.min_step = param[
            ("min_step", 1e-3, "the min step that can tolerate in adaptive method ")
        ]
        """ absolute error tolerance for dopri5"""
        self.dt = 1.0 / self.n_step
        """time step, we assume integration time is from 0,1 so the step is 1/n_step"""

    def solve(self, x):
        return self.forward(x)

    def set_func(self, func):
        self.odefunc = func

    def get_dt(self):
        return self.dt

    def forward(self, x):
        self.integration_time = (
            self.integration_time.type_as(x)
            if type(x) is not tuple
            else self.integration_time.type_as(x[0])
        )
        odesolver = torchdiffeq.odeint_adjoint if self.adjoin_on else torchdiffeq.odeint
        # out = odeint(self.odefunc, x, self.integration_time, rtol=self.rtol, atol=self.atol)
        out = odesolver(
            self.odefunc,
            x,
            self.integration_time,
            rtol=self.rtol,
            atol=self.atol,
            method=self.method,
            options={"step_size": self.dt, "eps": self.min_step},
        )
        if not self.interp_mode:
            return (elem[1] for elem in out)
        else:
            return [[elem_t for elem_t in elem] for elem in out]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
