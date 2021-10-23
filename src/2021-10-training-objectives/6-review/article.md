title: Classifier review
keywords: deep-learning,training,tutorial

# How to train your classifier - review

This is the final part in our series on [training objectives](../1-xent/article.md), exploring objectives you could use to train a deep learning classifier.

We've met softmax cross entropy, teacher-student training, sampled softmax, value function estimation and policy gradients. We reviewed the core ideas and walked through a typical forward and backward pass. All that's remains is to provide some demo code, so you can play around with these at your leisure.

## Demo code

The demo can be found [on GitHub](https://github.com/DouglasOrr/DouglasOrr.github.io/blob/examples/2021-10-training-objectives/training_objectives.ipynb) or opened directly [in Google Colab](https://colab.research.google.com/github/DouglasOrr/DouglasOrr.github.io/blob/examples/2021-10-training-objectives/training_objectives.ipynb).

It follows our running example of training a classifier for small image patches on the CIFAR10 dataset, using PyTorch. If you run the code as-is, it should successfully train a model using each objective. The parameters have been chosen to give reasonable performance in each case.

 - `softmax_cross_entropy` ([blog](../1-xent/article.md)) trains quickly and reliably, it's the default choice for a reason!
 - `teacher_student` ([blog](../2-teacher/article.md)) is as fast or faster than softmax cross-entropy (which it uses as teacher). Note that this example is a bit pointless, since the teacher is the same architecture as the student.
 - `sampled_softmax` ([blog](../3-sampled/article.md)) is slower than full softmax cross-entropy. It can be improved by increasing the number of samples.
 - `value_function` ([blog](../4-value/article.md)) trains relatively quickly (although still slower than full softmax cross-entropy), but can be unreliable. Recall that it's playing a harder game than previous techniques, a multi-armed contextual bandit problem.
 - `policy_gradient` ([blog](../5-policy/article.md)) is slower than full softmax cross-entropy, and may be unreliable. The entropy weight hyperparameter can make a big difference.

### Playing around

Please do have a play with it. Or even better, just throw this example away and have a go at implementing the objectives yourself, maybe for another dataset or domain. But if you like to learn by tweaking, here are a few things you could try:

 - Explore the hyperparameters. What do `alpha`, `n_samples`, `epsilon` and `entropy_weight` do?
 - Try to train a deeper network. E.g. ResNet18 from `torchvision`. Which objectives are harder to train?
 - Try changing the step size or optimiser. Are there better settings for certain objectives?
 - Try removing the baseline from policy gradient. How does it perform?
 - Can you make the value function more consistent? I.e. so that the expected reward sums to one across actions. How does performance change?

<ul class="nav nav-pills">
  <li class="nav-item">
    <a class="nav-link" href="../1-xent/article.html">Up - index</a>
  </li>
</ul>

## References

 - [PyTorch](https://pytorch.org/).
 - CIFAR-10: [Learning multiple layers of features from tiny images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), _Krizhevsky A, Hinton G._
