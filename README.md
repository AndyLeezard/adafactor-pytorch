# Forked and customized (experimental) 🧪

This branch comes with a new custom experimental feature: a piecewise constant learning rate schedule.
As the whole point of using adafactor is to maintain precision while reducing memory usage, I won't be submitting this branch as pull request.
This is for academic purposes.

## How to 📔

The learning rate schedule can be controlled by two new parameters: `lr_decay_step` and `lr_decay_factor`.

## Why this feature? 💡

These changes aim to provide more flexibility and control over the learning rate during training while keeping the core benefits of the AdaFactor optimizer.

## Elaborate? 📝

This enhancement allows the learning rate to decrease by a specified factor at predefined epochs, which can potentially improve training performance and convergence.
Adjusting the learning rate at key points during training (based on empirical or theoretical insights) can lead to better convergence behavior.
This is particularly useful in scenarios where the default learning rate schedule might not be optimal.

## Any caution? 🤔

Keep the rates within reasonable bounds.

Since it's an orthogonal feature, it adds value **without compromising the optimizer’s core benefit** 👍.
In other words, the piecewise constant learning rate won't interfere with the memory efficiency.
Also changing the learning rate according to a predefined schedule does not impact the numerical stability of the algorithm, as long as the chosen rates are within reasonable bounds as mentioned above.

# adafactor-pytorch
A pytorch realization of adafactor  (https://arxiv.org/pdf/1804.04235.pdf )

# Notes
1)Factorization works on any dimension. When dimension of weight tensor is higher than 2, it will be reshaped to 2D. For turning  off this feature  just change this lines ( if len(shape) > 2: return False, True ) in _check_shape 

2)Weights decay was moved to proper position according (https://arxiv.org/abs/1711.05101 )

# Parameters description:
lr - learning rate can be scalar or function, in second case relative step size is using.

beta1, beta2 - is also can be scalar or functions, in first case algorithm works as AMSGrad. Setting beta1 to zero is turning off moments updates.

non_constant_decay - boolean, has effect if betas are scalars. If True using functions for betas (from section 7.1)

enable_factorization - boolean. Factorization works on 2D weights.

clipping_threshold - scalar. Threshold value for update clipping (from section 6)
