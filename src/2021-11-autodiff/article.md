title: Autodiff from scratch
keywords: deep-learning,autodiff,frameworks,numpy

# Autodiff from scratch

Autodiff is the heart of a deep learning framework[^heart]. But it’s very easy to take it for granted and never peek under the hood to see how the engine works. So in this post, we'll make sure we understand the basics by implementing autodiff in a few lines of numpy.

This post is a guided walk though the basic bits of a usable autodiff implementation. If you would rather just read the code, jump straight to our notebook on [GitHub](https://github.com/DouglasOrr/DouglasOrr.github.io/blob/examples/2021-11-autodiff/autodiff.ipynb) or [Colab](https://colab.research.google.com/github/DouglasOrr/DouglasOrr.github.io/blob/examples/2021-11-autodiff/autodiff.ipynb).

_Tiny autodiff is no new idea - see also Andrej Kaparthy's [micrograd](https://github.com/karpathy/micrograd), amongst others._


## The Chain[^play]

Autodiff is the chain rule automated. The chain rule says that for $y = f(g(x))$:
$$
\frac{\partial y}{\partial x} = \left. \frac{\partial f(z)}{\partial z} \right|_{z=g(x)} \frac{\partial g(x)}{\partial x}
$$
This is great, because if we define a complicated function as the composition of simpler functions, we can obtain gradients using the chain rule and simple local equations for derivatives[^reverse].

For example, let’s differentiate the function $y = e^{-2 \cdot \mathrm{max}(x, 0)}$. We'll split into three pieces for the forward pass, then apply the chain rule to those pieces in reverse order for the backward pass. Note that $\Rightarrow$ is where we apply the chain rule:

\begin{align\*}
y &= e^{-2 \cdot \mathrm{max}(x, 0)} \\\\
&= f_3(f_2(f_1(x))) \\\\
\text{forward:}& \\\\
f_1(z_1) &= \mathrm{max}(z_1, 0) \\\\
f_2(z_2) &= -2 \cdot z_2 \\\\
f_3(z_3) &= e^{z_3} \\\\
\text{backward:}& \\\\
\frac{\partial f_3(z_3)}{\partial z_3} &= e^{z_3} = y \\\\
&\Rightarrow \frac{\partial y}{\partial z_3} = y \\\\
\frac{\partial f_2(z_2)}{\partial z_2} &= -2 \\\
&\Rightarrow \frac{\partial y}{\partial z_2} = -2 \cdot y \\\\
\frac{\partial f_1(z_1)}{\partial z_1} &= \mathrm{H}(z_1) = \mathrm{H}(x) \\\\
&\Rightarrow \frac{\partial y}{\partial x} = -2 \cdot y \cdot \mathrm{H}(x) \;, \\\\
\end{align\*}

where $\mathrm{H}(x)$ is the Heaviside step function, stepping from $0$ to $1$ at $x=0$.

The things notice about this process are that it's:

 - Very simple - no tricky derivatives.
 - Tedious - good work for a computer.

Autodiff therefore seeks to automate this. It provides the ML developer with means to express function composition and to write down the simple local derivatives, then automates repeated application of the chain rule to give gradients through the composite function.


## Started Out With Nothin’

We’re all set to build our numpy autodiff library. We’ll take the same approach as PyTorch, and run the forward pass eagerly. This means when you write $y = 2 x$, we compute $y$ straight away. At the same time we need to store some extra data, as $y$ needs to know how to tell $x$ its gradient in the backward pass. We’ll call this bundle a `Tensor`:

```python
Array = np.ndarray
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
```

Here `grad` is the gradient of some global scalar (typically “loss”) with respect to `value`. It’s `Optional` because some tensors won’t need gradients. Next, `backpropagate` is a function that fills in gradients for any inputs that produced this tensor and is `Optional` because some tensors don’t have inputs e.g. the constant $2$. Finally, `parents` is a list of inputs (the same inputs that are updated by `backpropagate`), which tells our autodiff algorithm which tensors to visit after computing `grad`.

Next, we need to be able to create tensors. These can be “initial tensors” created from arrays, either of input data or trainable parameters. Or they can be “result tensors” from operations run on other tensors, supporting backpropagation. We choose a factory function for creating initial tensors, and a function decorator to automate creation of result tensors:

```python
class Tensor:
    ...
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

def operation(fn: Callable[..., Tuple[Array, BackpropFn]]) -> Callable[..., Tensor]:
    @functools.wraps(fn)
    def wrapper(*args: TensorLike) -> Tensor:
        inputs = tuple(Tensor.wrap(arg, requires_grad=False) for arg in args)
        value, backpropagate = fn(*inputs)
        requires_grad = any(input.requires_grad for input in inputs)
        return Tensor(
            value=value,
            grad=np.zeros_like(value) if requires_grad else None,
            backpropagate=backpropagate if requires_grad else None,
            parents=inputs,
        )

    return wrapper
```

The difference between the two types of tensor is evident here. An initial tensor created by `wrap` doesn’t have any parents or perform backpropagation. It can still receive a gradient, needed for trainable parameters. A result tensor created by an `operation` supports backpropagation to its inputs, as defined by the specific operation.

Note that `Tensor.wrap()` of the inputs to an operation is just for convenience, it means that you will be able to write `Op.add(x, 2)` rather than the cumbersome `Op.add(x, Tensor.wrap(2, requires_grad=False))`.


## Get Back

So far we’ve built a way to track dependencies for tensors and to define functions that perform local gradient computations. Now for the fun bit - recursing on the chain rule to backpropagate through the whole graph of operations. There isn’t much to it:

```python
class Tensor:
    ...
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
```

The key thing here is to walk through the graph of operations in reverse dependency order. If the forward graph looks like `A -> B -> C`, the local backpropagation operations must run in order `[C, B, A]`. It’s easiest to define a forward dependency order traversal (`_dependency_order`) then reverse it. Then, to run the backward pass, we set our gradient to `1`, as $\frac{\partial L}{\partial L} = 1$ and run the local backpropagation operations in order. Each `BackpropFn` will accumulate gradient information into its inputs before `backward()` visits them to continue backpropagation.

That’s actually it for the core library, but things won’t really make sense until we look at a few operations.


## Put The Message In The Box

We won’t cover all the operations we need to train in this blog, as they all follow the same pattern. Here's the ReLU activation:

```python
@operation
def relu(x: Tensor) -> Tuple[Array, BackpropFn]:
    def backpropagate(grad: Array) -> None:
        x.grad += grad * (x.value > 0)

    return np.maximum(x.value, 0), backpropagate
```

This is a typical example of using the `@operation` decorator to create a result tensor with everything needed for backpropagation. The forward pass implementation is on the last line `np.maximum(x.value, 0)`. But we also need to return a function that accumulates gradient through the ReLU to the input. It’s convenient to define these two together, because the gradient computation might need to capture information from the forward pass. In this case our function captures the input `x` in a closure.

Another thing to notice is that instead of setting `x.grad[...] = <grad>`, we accumulate `x.grad += <grad>`. This is because we have implicit copying of tensors in the forward pass. If `x` was used for a `relu` and some other operation in the forward pass, it should sum the gradients from the two branches in the backward pass.

Next, matrix multiplication:

```python
@operation
def matmul(x: Tensor, y: Tensor) -> Tuple[Array, BackpropFn]:
    def backpropagate(grad: Array) -> None:
        if x.requires_grad:
            x.grad += grad @ y.value.T
        if y.requires_grad:
            y.grad += x.value.T @ grad

    return x.value @ y.value, backpropagate
```

`matmul` needs to capture both inputs for the backward pass. To be honest, I find it hard to remember which way all the transposes and matrix multiplications go, but thinking of the shapes `x: (A, B), y: (B, C), grad: (A, C)`, there aren’t really any other ways to write it that’d make sense.

Here the backward pass is made more efficient by skipping unnecessary gradients based on `Tensor.requires_grad`. In this case it’s just an optimisation, but for some other operations e.g. `power`, you might otherwise generate numerical errors for branches that don't require gradients at all.


## Suddenly I See

I can claim to have autodiff until the cows come home, but you shouldn’t believe me unless it’s usable. So here’s a small ReLU multi-layer perceptron for MNIST using our autodiff:

```python
class Model:
    def __init__(self):
        random = np.random.RandomState(3721)
        self.parameters = []
        self.lr = 0.1
        self.W0 = self.parameter(random.randn(28 * 28, 256) / np.sqrt(28 * 28))
        self.b0 = self.parameter(np.zeros(256))
        self.W1 = self.parameter(random.randn(256, 10) / np.sqrt(256))
        self.b1 = self.parameter(np.zeros(10))

    def parameter(self, value: Array) -> Tensor:
        tensor = Tensor.wrap(value.astype(np.float32), requires_grad=True)
        self.parameters.append(tensor)
        return tensor

    def step(self, x: Array, y: Array) -> float:
        for parameter in self.parameters:
            parameter.grad[...] = 0
        h = Op.matmul(x, self.W0)
        h = Op.add(h, Op.broadcast_to(self.b0, h.value.shape))
        h = Op.relu(h)
        h = Op.matmul(h, self.W1)
        h = Op.add(h, Op.broadcast_to(self.b1, h.value.shape))
        loss = Op.softmax_cross_entropy(h, y)
        loss.backward()
        for parameter in self.parameters:
            parameter.value -= self.lr * parameter.grad
        return float(loss.value)
```

If you’ve used PyTorch, it should look familiar. You have to reset parameter gradients to zero before running the forward pass and you call `loss.backward()` before an optimiser update using `parameter.grad`.

There are a few things that feel painful here - you can’t write `x @ self.W0` because we haven’t defined any operators on `Tensor`, and there’s no implicit broadcasting so adding bias looks particularly messy. But these would be easy to remedy with a few tweaks to the implementation.


## All I Got

That’s it for now. But if you’ve read this far, I’d recommend looking at the notebook with complete code for training MNIST, on [GitHub](https://github.com/DouglasOrr/DouglasOrr.github.io/blob/examples/2021-11-autodiff/autodiff.ipynb) or [Colab](https://colab.research.google.com/github/DouglasOrr/DouglasOrr.github.io/blob/examples/2021-11-autodiff/autodiff.ipynb).

A few personal takeaways from this enjoyable diversion:

 - The essence of autodiff is very simple and small - no magic here.
 - Skipping gradient computation (`requires_grad`) can be more than just an optimisation - sometimes you have branches that are non-differentiable.
 - Using a tensor twice in the forward pass means an addition in the backward pass (and vice versa).
 - Broadcasting a tensor in the forward pass is a reduce-sum in the backward pass (and vice versa).
 - The devil is in the details - our implementation is horribly memory-inefficient and missing both operator overloads and auto-broadcasting is painful.


[^heart]: Maybe deep learning frameworks have two hearts, because I could equally say &quot;accelerator support is the heart of a deep learning framework&quot;.

[^play]: [Listen along](https://open.spotify.com/playlist/54Lj38WL2w3NnZUQwuzy7O?si=97428695b5304d35) to our silly puns.

[^reverse]: Here we're just going to think about reverse mode automatic differentiation. There are [other sorts](https://en.wikipedia.org/wiki/Automatic_differentiation).
