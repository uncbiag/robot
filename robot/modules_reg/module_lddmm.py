from torch.autograd import grad
import torch.nn as nn
from robot.utils.obj_factory import obj_factory
import torch


############## Hamiltonian view of LDDMM ######################3


class LDDMMHamilton(nn.Module):
    def __init__(self, opt):
        super(LDDMMHamilton, self).__init__()
        self.opt = opt
        kernel = opt[
            (
                "kernel",
                "keops_kernels.LazyKeopsKernel('gauss',sigma=0.1)",
                "kernel object",
            )
        ]
        self.kernel = obj_factory(kernel)
        self.mode = "shooting"

    def hamiltonian(self, mom, control_points):
        # todo check, the omitted 1/2 is consistant with variational version
        return (mom * self.kernel(control_points, control_points, mom)).sum() * 0.5

    def hamiltonian_evolve(self, mom, control_points):
        record_is_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        control_points = control_points.clone().requires_grad_()
        mom = mom.clone().requires_grad_()
        grad_mom, grad_control = grad(
            self.hamiltonian(mom, control_points),
            (mom, control_points),
            create_graph=True,
        )
        torch.set_grad_enabled(record_is_grad_enabled)
        return -grad_control, grad_mom

    def flow(self, mom, control_points, flow_points):
        return self.hamiltonian_evolve(mom, control_points) + (
            self.kernel(flow_points, control_points, mom),
        )

    def set_mode(self, mode):
        assert mode in ["shooting", "flow"]
        self.mode = mode

    def forward(self, t, input):
        if self.mode == "shooting":
            return self.hamiltonian_evolve(*input)
        else:
            return self.flow(*input)


###################  variational view of LDDMM ######################


class LDDMMVariational(nn.Module):
    def __init__(self, opt):
        super(LDDMMVariational, self).__init__()
        self.opt = opt
        kernel = opt[
            (
                "kernel",
                "keops_kernels.LazyKeopsKernel('gauss',sigma=0.1)",
                "kernel object",
            )
        ]
        self.kernel = obj_factory(kernel)
        grad_kernel = kernel.replace("gauss", "gauss_grad")
        self.grad_kernel = obj_factory(grad_kernel)
        self.mode = "shooting"

    def variational_evolve(self, mom, control_points):
        mom = mom.clamp(-1, 1)
        return -self.grad_kernel(mom, control_points), self.kernel(
            control_points, control_points, mom
        )

    def variational_flow(self, mom, control_points, flow_points):
        mom = mom.clamp(-1, 1)
        return self.variational_evolve(mom, control_points) + (
            self.kernel(flow_points, control_points, mom),
        )

    def set_mode(self, mode):
        assert mode in ["shooting", "flow"]
        self.mode = mode

    def forward(self, t, input):
        if self.mode == "shooting":
            return self.variational_evolve(*input)
        else:
            return self.variational_flow(*input)
