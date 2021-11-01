import functools
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Set, Tuple, Union

import numpy as np
from numpy.typing import NDArray

Array = NDArray[Any]
BackpropFn = Callable[[Array], None]
TensorLike = Union["Tensor", Array, float]


@dataclass
class Tensor:
    value: Array
    grad: Optional[Array]
    backpropagate: Optional[BackpropFn]
    parents: Tuple["Tensor", ...]

    @property
    def requires_grad(self) -> bool:
        return self.grad is not None

    @classmethod
    def wrap(cls, value: TensorLike, requires_grad: bool) -> "Tensor":
        if isinstance(value, cls):
            return value
        value = np.array(value)
        return cls(
            value=value,
            grad=np.zeros_like(value) if requires_grad else None,
            backpropagate=None,
            parents=(),
        )

    def _dependency_order(self, visited: Set[int]) -> Iterator["Tensor"]:
        if id(self) not in visited:
            for parent in self.parents:
                yield from parent._dependency_order(visited)
            yield self
            visited.add(id(self))

    def backward(self) -> None:
        self.grad[...] = 1
        for tensor in reversed(list(self._dependency_order(set()))):
            if tensor.backpropagate is not None:
                tensor.backpropagate(tensor.grad)


def operation(fn: Callable[..., Tuple[Array, BackpropFn]]) -> Callable[..., Tensor]:
    @functools.wraps(fn)
    def wrapper(*args: TensorLike, **kwargs: TensorLike) -> Tensor:
        t_args = tuple(Tensor.wrap(arg, requires_grad=False) for arg in args)
        t_kwargs = {
            key: Tensor.wrap(value, requires_grad=False)
            for key, value in kwargs.items()
        }
        value, backpropagate = fn(*t_args, **t_kwargs)
        inputs = t_args + tuple(t_kwargs.values())
        requires_grad = any(input.requires_grad for input in inputs)
        return Tensor(
            value=value,
            grad=np.zeros_like(value) if requires_grad else None,
            backpropagate=backpropagate if requires_grad else None,
            parents=inputs,
        )

    return wrapper


class Op:
    @staticmethod
    @operation
    def matmul(x: Tensor, y: Tensor) -> Tuple[Array, BackpropFn]:
        def backpropagate(grad: Array) -> None:
            if x.requires_grad:
                x.grad += grad @ y.value.T
            if y.requires_grad:
                y.grad += x.value.T @ grad

        return x.value @ y.value, backpropagate

    @staticmethod
    @operation
    def sum(x: Tensor) -> Tuple[Array, BackpropFn]:
        def backpropagate(grad: Array) -> None:
            x.grad += grad  # broadcast

        return np.sum(x.value), backpropagate

    @staticmethod
    @operation
    def add(*xs: Tensor) -> Tuple[Array, BackpropFn]:
        def backpropagate(grad: Array) -> None:
            for x in xs:
                if x.requires_grad:
                    x.grad += grad

        return sum(x.value for x in xs), backpropagate

    @staticmethod
    @operation
    def subtract(x: Tensor, y: Tensor) -> Tuple[Array, BackpropFn]:
        def backpropagate(grad: Array) -> None:
            if x.requires_grad:
                x.grad += grad
            if y.requires_grad:
                y.grad -= grad

        return x.value - y.value, backpropagate

    @staticmethod
    @operation
    def divide(x: Tensor, y: Tensor) -> Tuple[Array, BackpropFn]:
        def backpropagate(grad: Array) -> None:
            if x.requires_grad:
                x.grad += grad / y.value
            if y.requires_grad:
                y.grad -= grad * x.value / (y.value ** 2)

        return x.value / y.value, backpropagate

    @staticmethod
    @operation
    def relu(x: Tensor) -> Tuple[Array, BackpropFn]:
        def backpropagate(grad: Array) -> None:
            x.grad += grad * (x.value > 0)

        return np.maximum(x.value, 0), backpropagate

    @staticmethod
    @operation
    def power(x: Tensor, pow: Tensor) -> Tuple[Array, BackpropFn]:
        def backpropagate(grad: Array) -> None:
            if x.requires_grad:
                x.grad += grad * pow.value * (x.value ** (pow.value - 1))
            if pow.requires_grad:
                pow.grad += grad * np.log(x.value) * (x.value ** pow.value)

        return x.value ** pow.value, backpropagate

    @staticmethod
    @operation
    def broadcast_to(x: Tensor, shape: Tensor) -> Tuple[Array, BackpropFn]:
        def backpropagate(grad: Array) -> None:
            assert not shape.requires_grad
            if x.requires_grad:
                x_shape = x.value.shape
                x_shape_pad = (1,) * (len(shape.value) - len(x_shape)) + x_shape
                (sum_dims,) = np.where(np.array(x_shape_pad) == 1)
                x.grad += np.sum(grad, axis=tuple(sum_dims), keepdims=True).reshape(
                    x_shape
                )

        return np.broadcast_to(x.value, shape.value), backpropagate

    @staticmethod
    @operation
    def softmax_cross_entropy(x: Tensor, targets: Tensor) -> Tuple[Array, BackpropFn]:
        assert x.value.ndim == 2
        assert targets.value.shape == x.value.shape[:1]
        x_shift = x.value - x.value.max(axis=1, keepdims=True)
        logp = x_shift - np.log(np.sum(np.exp(x_shift), axis=1, keepdims=True))

        def backpropagate(grad: Array) -> None:
            assert not targets.requires_grad
            if x.requires_grad:
                scaled_grad = grad / targets.value.size
                x.grad[np.arange(targets.value.size), targets.value] -= scaled_grad
                x.grad += scaled_grad * np.exp(logp)

        loss = -np.mean(logp[np.arange(targets.value.size), targets.value])
        return loss, backpropagate
