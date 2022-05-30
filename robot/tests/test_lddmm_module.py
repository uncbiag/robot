# sys.path.insert(0, os.path.abspath('..'))
# sys.path.insert(0, os.path.abspath('../tests'))
# sys.path.insert(0,os.path.abspath('../robot'))
import torch
from torch.autograd import grad
import unittest
from robot.modules_reg.module_lddmm import LDDMMHamilton, LDDMMVariational
from robot.utils.module_parameters import ParameterDict

torch.backends.cudnn.deterministic = True
torch.manual_seed(123)


if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


def timming(func, message):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def time_diff(*args, **kwargs):
        start.record()
        res = func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        print("{}, it takes {} ms".format(message, start.elapsed_time(end)))
        return res

    return time_diff


class Test_Kernels(unittest.TestCase):
    def setUp(self):
        B = 1
        N = 2000
        K = 3000
        D = 3
        device = torch.device("cuda:0")  # cuda:0, cpu
        # device = torch.device("cpu") # cuda:0, cpu
        self.control_points = torch.rand(B, N, D, requires_grad=True, device=device)
        self.momentum = torch.rand(B, N, D, requires_grad=True, device=device)
        self.toflow = torch.rand(B, K, D, requires_grad=True, device=device)

    def tearDown(self):
        pass

    def compare_tensors(self, tensors1, tensors2, rtol=1e-3, atol=1e-7):
        for tensor1, tensor2 in zip(tensors1, tensors2):
            torch.testing.assert_allclose(tensor1, tensor2, rtol=rtol, atol=atol)

    def compare_hamiltonian_and_variational(
        self, hamiltonian_module, variational_module, task_name
    ):
        hamiltonian_module = timming(
            hamiltonian_module, "hamiltonian_forward_{}".format(task_name)
        )
        variational_module = timming(
            variational_module, "variational_forward_{}".format(task_name)
        )

        hamiltonian_forward = hamiltonian_module(
            t=1.0, input=(self.momentum, self.control_points)
        )
        variational_forward = variational_module(
            t=1.0, input=(self.momentum, self.control_points)
        )
        self.compare_tensors(
            hamiltonian_forward, variational_forward, rtol=1e-3, atol=1e-5
        )

        hamiltonian_mom_grad = timming(
            grad, "hamiltonian_backward_momentum{}".format(task_name)
        )
        hamiltonian_point_grad = timming(
            grad, "hamiltonian_backward_points{}".format(task_name)
        )
        variational_mom_grad = timming(
            grad, "variational_backward_momentum{}".format(task_name)
        )
        variational_point_grad = timming(
            grad, "variational_backward_points{}".format(task_name)
        )
        hamiltonian_mom_backward = hamiltonian_mom_grad(
            hamiltonian_forward[0].mean(),
            (self.momentum, self.control_points),
            retain_graph=True,
        )
        hamiltonian_point_backward = hamiltonian_point_grad(
            hamiltonian_forward[1].mean(),
            (self.momentum, self.control_points),
            retain_graph=True,
        )
        variational_mom_backward = variational_mom_grad(
            variational_forward[0].mean(),
            (self.momentum, self.control_points),
            retain_graph=True,
        )
        variational_point_backward = variational_point_grad(
            variational_forward[1].mean(),
            (self.momentum, self.control_points),
            retain_graph=True,
        )

        self.compare_tensors(
            hamiltonian_mom_backward, variational_mom_backward, rtol=1e-3, atol=1e-5
        )
        self.compare_tensors(
            hamiltonian_point_backward, variational_point_backward, rtol=1e-3, atol=1e-5
        )

    def test_lddmm_shooting_single_gaussian(self, task_name="gauss"):
        hamiltonian_opt = ParameterDict()
        hamiltonian_opt["kernel"] = "torch_kernels.TorchKernel('gauss',sigma=0.1)"
        hamiltonian_opt["mode"] = "shooting"
        hamiltonian_module = LDDMMHamilton(hamiltonian_opt)

        variational_opt = ParameterDict()
        variational_opt["kernel"] = "torch_kernels.TorchKernel('gauss',sigma=0.1)"
        variational_opt["mode"] = "shooting"
        variational_module = LDDMMVariational(variational_opt)
        self.compare_hamiltonian_and_variational(
            hamiltonian_module, variational_module, task_name
        )

    def test_lddmm_shooting_multi_gaussian(self, task_name="multi_gauss"):
        hamiltonian_opt = ParameterDict()
        hamiltonian_opt[
            "kernel"
        ] = "torch_kernels.TorchKernel(kernel_type='multi_gauss', sigma_list=[0.01,0.05,0.1],weight_list=[0.2,0.3,0.5])"
        hamiltonian_opt["mode"] = "shooting"
        hamiltonian_module = LDDMMHamilton(hamiltonian_opt)

        variational_opt = ParameterDict()
        variational_opt[
            "kernel"
        ] = "torch_kernels.TorchKernel(kernel_type='multi_gauss', sigma_list=[0.01,0.05,0.1],weight_list=[0.2,0.3,0.5])"
        variational_opt["mode"] = "shooting"
        variational_module = LDDMMVariational(variational_opt)
        self.compare_hamiltonian_and_variational(
            hamiltonian_module, variational_module, task_name
        )


def run_by_name(test_name):
    suite = unittest.TestSuite()
    suite.addTest(Test_Kernels(test_name))
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == "__main__":
    run_by_name("test_lddmm_shooting")
