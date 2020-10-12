title: Teacher-Student Training
keywords: deep-learning,training,tutorial

# Teacher-Student Training (aka Knowledge Distillation)

Teacher-student training is a regime for speeding up training and improving convergence of a neural network, given a pretrained "teacher" network. It's very popular and effective, commonly used to train smaller, cheaper networks from larger, more expensive ones.

In this article, we'll think through the core idea of teacher-student training, see how to implement it in PyTorch, and finally look at what actually happens when we use this loss. This is part of our series on [training objectives](/index.html#classifier-training-objectives), and if you're not familiar with softmax cross entropy, [our introduction](../1-xent/article.html) to that would be a useful pre-read for this article.


## Core idea

When we looked at the softmax cross entropy loss function (with a one-hot target), we saw that the update was very "spiky". Each example in a batch contributes a strong gradient spike to increase the score of the correct label, and a flatter gradient across all other labels. Or, if we look at any individual example, our loss is trying to make the target probability 1, and everything else 0 (this would minimize the loss for that example).

But there's a conflict here - we don't actually expect (or want) this spiky output from our model. Instead, we'd expect an output to "hedge it's bets" a bit. For example, given this image:

![pixelated picture of a horse](img/example_horse.png)

You could expect your model to predict something like "horse 80%, dog 20%". But if we use standard softmax cross entropy loss, the target distribution looks like this:

![bar chart of probabilities, with a single spike on "horse"](img/target_hard.png)

This means that the prediction "horse 100%, dog 0%" would minimize softmax cross entropy loss, since it is actually a picture of a horse. Intuitively this seems a bit over-confident.

When training with softmax cross entropy loss, we can avoid this problem by making sure there is enough input data, that examples well-shuffled into batches, and there are _regularizers_ such as dropout that prevent the model from becoming over-confident.

**Teacher-Student training provides a richer and more realistic target distribution than a single spike.** Instead of training the model to predict "horse 100%, dog 0%", it can train the model to predict "horse 80%, dog 20%" on a single example. Given a target distribution, we can still use softmax cross entropy as a loss function - the only difference is that the target is not a single-sample "hard" spike but a full "soft" distribution over all classes, like this:

![bar chart of probabilities, with high values for "horse" and "cat"](img/target_soft.png)

_(Note - you will sometimes see [KL-divergence](https://en.wikipedia.org/wiki/Kullback-Leibler_divergence) used instead of cross entropy loss. It really doesn't make a difference to training, however - as the gradient update provided by $D\_{KL}(\mathrm{target}||\mathrm{predictions})$ is the same as cross-entropy.)_

**But how can we get a target distribution?** In general, we can't use the dataset directly here. If the dataset's input features are pixels or long sentences, it's unlikely there will be multiple output samples for the same input - the input space is too large. We would need a vast amount of data to estimate a target distribution directly. So we use another neural network - called the _teacher_ network.

Before we can train the network we're interested in, called the _student_ network, we must train the teacher network. We first train the teacher using a standard loss (e.g. "hard" softmax cross entropy). Afterwards, we train the student using our new teacher-student loss function. Instead of fully trusting the teacher, it's usual to use a mixture of the standard one-hot objective and the teacher's distribution, like this:

![bar chart of probabilities, with a large spike for "horse" but small values for "cat", etc.](img/target.png)

This distribution is just a 50-50 mixture of the above "hard" and "soft" distributions.

> In teacher-student training, the dataset provides "hard" targets (a single target label) and the teacher provides "soft" targets (a distribution over all labels).

Putting it together, the teacher-student loss function (used to train the student) looks like this:

\begin{equation}
L(x, t) = \mathrm{CE}\left(\alpha \cdot \mathrm{teacher}(x) + (1-\alpha)\cdot\delta_{t*}\;, \mathrm{student}(x)\right)
\label{eqn:loss}
\end{equation}

Where $CE(t,s)$ is the softmax cross entropy loss function between target distribution $t$ and predicted scores $s$, and $\alpha$ is an interpolation coefficient (usually a fixed hyperparameter) that interpolates between plain one-hot loss ($\alpha\\!=\\!0$) and pure "teacher matching" ($\alpha\\!=\\!1$). Note $\delta_{t*}$ is the kronecker delta, i.e. a one-hot vector at index $t$.

### Why bother?

One problem with teacher-student training is that you have to train a teacher network as well as the student. If the teacher and student are exactly the same network (maybe with different initialization), it all seems a bit pointless! We could reasonably assume the student will only ever match the performance of the teacher (although it'll get there faster.) But since we had to train the teacher network anyway - we may as well just use that one, and forget about training the student (it'd perform just as well, and will save training time).

The usual way through this is to train networks of different sizes. Experiments show that you can get better results for a given student network by training with a _larger_ teacher network, than if you train the student directly on the dataset. This means that you can get a network that's cheaper to query for predictions (because it is smaller than the teacher), but has higher accuracy than a network trained directly (without a teacher). This is the origin of the term **knowledge distillation**. The superior knowledge of the richer teacher network can be distilled into the smaller student, using the teacher-student objective.

To draw this together, teacher-student training of classifiers (usually) means:

 1. Train a large teacher network using one-hot softmax cross entropy.
 2. Define a target distribution which mixes the "hard" targets from the dataset with "soft" targets from the teacher.
 3. Train a smaller student network on softmax cross entropy against this "hard+soft" target.


## PyTorch implementation

It's not very hard to implement distillation. First you have to train the teacher (this is done using standard objectives), then use its predictions to build a target distribution when training the student. The student phase looks like this:

```python
inputs, labels = ...
model = ...
teacher = ...
alpha = 0.5

with T.no_grad():
    soft_target = T.nn.functional.softmax(teacher(inputs))
    hard_target = T.arange(soft_target.shape[-1]) == labels[..., np.newaxis]
    target = alpha * soft_target + (1 - alpha) * hard_target

outputs = model(inputs)
logprobs = T.nn.functional.log_softmax(outputs)

loss = T.nn.functional.kl_div(logprobs, target, reduction='batchmean')
print(float(loss))
loss.backward()
```

It's important not to keep training the teacher by accident - in PyTorch this means wrapping the teacher's prediction in `T.no_grad()`. You'll spot that we're using KL divergence as a loss function - the PyTorch API encourages you to do this, but it will produce the same gradient-based updates as cross entropy.

_Note: an alternative (equivalent) implementation would be to use `T.nn.functional.kl_div` for `soft_target`, `T.nn.functional.cross_entropy` for `hard_target`, and mix the losses rather than the distributions._


## What does it do?

_We'll skim over some of the fine details here, when they're common to plain (one-hot) softmax cross entropy - see [this description](../1-xent/article.html#what-does-it-do) for more on that._

Our student produces scores. These are normalized using log-softmax for computing the loss, or softmax to get the prediction distribution:

![bar chart of probabilities, with the largest spike on "horse"](img/activations_probs.png)

Evidently the model is very confident (maybe overconfident) that this image is a horse.

We've already seen how the target distribution is constructed by mixing hard and soft targets. To recap:

![bar chart of probabilities, with a large spike for "horse" but small values for "cat", etc.](img/target.png)

Running equation \eqref{eqn:loss} to compare predicted and target distributions, we get the cross entropy loss `1.89` (or a KL divergence loss `0.946`).

### The backward pass

The backward pass is really the important bit, after all... The gradient with respect to scores is surprisingly simple:

$$\frac{dL}{dx_i} = p_i - t_i$$

Where $x_i$ is the score, $p_i$ is the predicted probability (computed from $x_i$) and $t_i$ is the target probability (for class $i$).

In our example, the gradient w.r.t. scores looks like this:

![bar chart of gradients for airplane, automobile, horse etc. with "horse" strongly positive, cat strongly negative](img/gradients_scores.png)

We might be surprised by a strong _positive_ gradient for "horse". This means that the loss is trying to push down the "horse" prediction, even though this is actually the correct label from the dataset. It's the teacher's fault - the teacher thinks the student is overconfident and should be sharing out some probability to other classes, especially "cat", "dog" and "truck".

A couple of observations. First, whatever happens, there is the same positive and negative gradient mass (as both $p$ and $t$ are probability distributions which sum to 1). Unlike in one-hot softmax cross entropy, however, there can be multiple positive and negative gradients. Second, the gradient magnitude depends on the absolute difference between $p$ and $t$, which makes intuitive sense.


## Wrap up

That's teacher-student training, most commonly applied as knowledge distillation. It can be motivated as a less "spiky" objective than one-hot softmax cross entropy. It doesn't matter if you use KL divergence or cross entropy. And we saw what kinds of gradients to expect.

The most common scenario for teacher-student is **optimizing for inference/predictions**. We want maximum prediction quality for minimum prediction runtime. To do this we spend a bit more training time to first train a teacher and then use the teacher-student objective to train a better student model than we could have done from scratch.

<ul class="nav nav-pills">
  <li class="nav-item">
    <a class="nav-link" href="../3-sampled/article.html">Next - sampled softmax</a>
  </li>
  <li class="nav-item">
    <a class="nav-link" href="/index.html#classifier-training-objectives">Up - index</a>
  </li>
</ul>


## References

 - Knowledge Distillation: [Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf), _Hinton G, Vinyals O, Dean J._
 - CIFAR-10: [Learning multiple layers of features from tiny images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), _Krizhevsky A._
