from typing import Any, Callable

import numpy as np

from autodiff import Array, Op, Tensor


def check_gradients(
    op: Callable[..., Tensor], *args: Array, epsilon: float = 1e-6, **params: Any
) -> None:
    inputs = tuple(Tensor.wrap(arg, requires_grad=True) for arg in args)
    output = op(*inputs, **params)
    output.backward()
    for arg_idx, arg in enumerate(map(np.array, args)):
        approx_grads = []
        for position in range(arg.size):
            one_hot = (np.arange(arg.size) == position).reshape(arg.shape)
            test_inputs = (
                args[:arg_idx] + (arg + epsilon * one_hot,) + args[arg_idx + 1 :]
            )
            approx_grads.append(
                (np.sum(op(*test_inputs, **params).value) - np.sum(output.value))
                / epsilon
            )
        approx_grad = np.array(approx_grads).reshape(arg.shape)
        np.testing.assert_allclose(
            approx_grad,
            inputs[arg_idx].grad,
            rtol=1e-4,
            err_msg=f"Failed gradient check, op {op}, argument {arg_idx}",
        )


def test_gradients():
    check_gradients(Op.matmul, [[1.0, -2.0]], [[0.0, 1.0, -2], [3.0, -5, 0.0]])
    check_gradients(Op.sum, np.arange(3))
    check_gradients(Op.add, [1.0, 2.0, 3.0])
    check_gradients(Op.subtract, [0.0, 1.0], [2.0, 3.0])
    check_gradients(Op.divide, [2.0, 3.0], [-0.5, 0.5])
    check_gradients(Op.relu, [-0.5, -0.1, 0.1, 0.5])
    check_gradients(Op.power, [0.5, 2.0], [0.75, 2.5])
    check_gradients(Op.broadcast_to, [1.0, 2.0], shape=(3, 4, 2))
    check_gradients(
        Op.softmax_cross_entropy, [[10.0, 9.0, 8.0], [8.0, 9.0, 10.0]], targets=[0, 1]
    )
