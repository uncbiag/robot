import os, sys

sys.path.insert(0, os.path.abspath("../.."))
import torch
from torch.autograd import grad
import unittest
from robot.kernels.keops_kernels import LazyKeopsKernel
from robot.kernels.torch_kernels import TorchKernel

torch.backends.cudnn.deterministic = True
import pykeops
pykeops.clean_pykeops()

torch.manual_seed(123)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


def timming(func, message):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def time_diff(*args, **kwargs):
        try:
            start.record()
            res = func(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            print("{}, it takes {} ms".format(message, start.elapsed_time(end)))
        except:
            res = func(*args, **kwargs)
        return res

    return time_diff


class Test_Kernels(unittest.TestCase):
    def setUp(self):
        B = 2
        N = 1000
        K = 800
        D = 3
        device = torch.device("cpu")  # cuda:0, cpu
        # device = torch.device("cpu") # cuda:0, cpu
        self.x = torch.rand(B, N, D, requires_grad=True, device=device)
        self.y = torch.rand(B, K, D, requires_grad=True, device=device)
        self.px = torch.rand(B, N, D, requires_grad=True, device=device)
        self.py = torch.rand(B, K, D, requires_grad=True, device=device)
        self.b = torch.rand(B, K, D, requires_grad=True, device=device)

    def tearDown(self):
        pass

    def compare_tensors(self, tensors1, tensors2, rtol=1e-3, atol=1e-7):
        for tensor1, tensor2 in zip(tensors1, tensors2):
            torch.testing.assert_allclose(tensor1, tensor2, rtol=rtol, atol=atol)

    def test_kernel_gaussian(self, task_name="gauss"):

        keops_kernel = LazyKeopsKernel(kernel_type="gauss", sigma=0.1)
        torch_kernel = TorchKernel(kernel_type="gauss", sigma=0.1)
        keops_kernel = timming(
            keops_kernel, "test_kernel_{} with keops".format(task_name)
        )
        torch_kernel = timming(
            torch_kernel, "test_kernel_{} with torch".format(task_name)
        )
        keops_gauss = keops_kernel(self.x, self.y, self.b)
        torch_gauss = torch_kernel(self.x, self.y, self.b)
        grad_keops = timming(grad, "test_kernel_{} grad with keops".format(task_name))
        grad_torch = timming(grad, "test_kernel_{} grad with torch".format(task_name))
        keops_grads = grad_keops(
            keops_gauss.mean(), (self.x, self.y, self.b), retain_graph=True
        )
        torch_grads = grad_torch(
            torch_gauss.mean(), (self.x, self.y, self.b), retain_graph=True
        )
        torch.testing.assert_allclose(keops_gauss, torch_gauss, rtol=1e-3, atol=1e-7)
        self.compare_tensors(keops_grads, torch_grads, rtol=1e-3, atol=1e-7)

    def test_kernel_mutli_gaussian(self, task_name="multi_gauss"):
        keops_kernel = LazyKeopsKernel(
            kernel_type="multi_gauss",
            sigma_list=[0.01, 0.05, 0.1],
            weight_list=[0.2, 0.3, 0.5],
        )
        torch_kernel = TorchKernel(
            kernel_type="multi_gauss",
            sigma_list=[0.01, 0.05, 0.1],
            weight_list=[0.2, 0.3, 0.5],
        )
        keops_kernel = timming(
            keops_kernel, "test_kernel_{} with keops".format(task_name)
        )
        torch_kernel = timming(
            torch_kernel, "test_kernel_{} with torch".format(task_name)
        )
        keops_gauss = keops_kernel(self.x, self.y, self.b)
        torch_gauss = torch_kernel(self.x, self.y, self.b)
        grad_keops = timming(grad, "test_kernel_{} grad with keops".format(task_name))
        grad_torch = timming(grad, "test_kernel_{} grad with torch".format(task_name))
        keops_grads = grad_keops(
            keops_gauss.mean(), (self.x, self.y, self.b), retain_graph=True
        )
        torch_grads = grad_torch(
            torch_gauss.mean(), (self.x, self.y, self.b), retain_graph=True
        )
        torch.testing.assert_allclose(keops_gauss, torch_gauss, rtol=1e-3, atol=1e-7)
        self.compare_tensors(keops_grads, torch_grads, rtol=1e-3, atol=1e-7)

    def test_kernel_gaussian_grad(self, task_name="gauss_grad"):
        keops_kernel = LazyKeopsKernel(kernel_type="gauss_grad", sigma=0.1)
        torch_kernel = TorchKernel(kernel_type="gauss_grad", sigma=0.1)
        keops_kernel = timming(
            keops_kernel, "test_kernel_{} with keops".format(task_name)
        )
        torch_kernel = timming(
            torch_kernel, "test_kernel_{} with torch".format(task_name)
        )
        keops_gauss = keops_kernel(self.px, self.x, self.py, self.y)
        torch_gauss = torch_kernel(self.px, self.x, self.py, self.y)
        grad_keops = timming(grad, "test_kernel_{} grad with keops".format(task_name))
        grad_torch = timming(grad, "test_kernel_{} grad with torch".format(task_name))
        keops_grads = grad_keops(
            keops_gauss.mean(), (self.px, self.x, self.py, self.y), retain_graph=True
        )
        torch_grads = grad_torch(
            torch_gauss.mean(), (self.px, self.x, self.py, self.y), retain_graph=True
        )
        torch.testing.assert_allclose(keops_gauss, torch_gauss, rtol=1e-2, atol=1e-5)
        self.compare_tensors(keops_grads, torch_grads, rtol=1e-3, atol=1e-7)

    def test_kernel_multi_gaussian_grad(self, task_name="multi_gauss_grad"):
        keops_kernel = LazyKeopsKernel(
            kernel_type="multi_gauss_grad",
            sigma_list=[0.01, 0.05, 0.1],
            weight_list=[0.2, 0.3, 0.5],
        )
        torch_kernel = TorchKernel(
            kernel_type="multi_gauss_grad",
            sigma_list=[0.01, 0.05, 0.1],
            weight_list=[0.2, 0.3, 0.5],
        )
        keops_kernel = timming(
            keops_kernel, "test_kernel_{} with keops".format(task_name)
        )
        torch_kernel = timming(
            torch_kernel, "test_kernel_{} with torch".format(task_name)
        )
        keops_gauss = keops_kernel(self.px, self.x, self.py, self.y)
        torch_gauss = torch_kernel(self.px, self.x, self.py, self.y)
        grad_keops = timming(grad, "test_kernel_{} grad with keops".format(task_name))
        grad_torch = timming(grad, "test_kernel_{} grad with torch".format(task_name))
        keops_grads = grad_keops(
            keops_gauss.mean(), (self.px, self.x, self.py, self.y), retain_graph=True
        )
        torch_grads = grad_torch(
            torch_gauss.mean(), (self.px, self.x, self.py, self.y), retain_graph=True
        )
        torch.testing.assert_allclose(keops_gauss, torch_gauss, rtol=1e-2, atol=1e-5)
        self.compare_tensors(keops_grads, torch_grads, rtol=1e-3, atol=1e-7)

    def test_kernel_gaussian_lin(self, task_name="gauss_lin"):
        keops_kernel = LazyKeopsKernel(kernel_type="gauss_lin", sigma=0.1)
        torch_kernel = TorchKernel(kernel_type="gauss_lin", sigma=0.1)
        keops_kernel = timming(
            keops_kernel, "test_kernel_{} with keops".format(task_name)
        )
        torch_kernel = timming(
            torch_kernel, "test_kernel_{} with torch".format(task_name)
        )
        keops_gauss = keops_kernel(self.x, self.y, self.px, self.py, self.b)
        torch_gauss = torch_kernel(self.x, self.y, self.px, self.py, self.b)
        grad_keops = timming(grad, "test_kernel_{} grad with keops".format(task_name))
        grad_torch = timming(grad, "test_kernel_{} grad with torch".format(task_name))
        keops_grads = grad_keops(
            keops_gauss.mean(),
            (self.x, self.y, self.px, self.py, self.b),
            retain_graph=True,
        )
        torch_grads = grad_torch(
            torch_gauss.mean(),
            (self.x, self.y, self.px, self.py, self.b),
            retain_graph=True,
        )
        torch.testing.assert_allclose(keops_gauss, torch_gauss, rtol=1e-3, atol=1e-7)
        self.compare_tensors(keops_grads, torch_grads, rtol=1e-3, atol=1e-7)


def run_by_name(test_name):
    suite = unittest.TestSuite()
    suite.addTest(Test_Kernels(test_name))
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == "__main__":
    run_by_name("test_kernel_gaussian")
    run_by_name("test_kernel_mutli_gaussian")
    run_by_name("test_kernel_gaussian_grad")
    run_by_name("test_kernel_multi_gaussian_grad")
    run_by_name("test_kernel_gaussian_lin")
