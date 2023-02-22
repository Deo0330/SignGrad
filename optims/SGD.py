import math

import torch
from torch.optim.optimizer import Optimizer

from .types import OptFloat, OptLossClosure, Params

__all__ = ('SGD',)


class SGD(Optimizer):
    r"""Implements SGDP algorithm.

    It has been proposed in `Slowing Down the Weight Norm Increase in
    Momentum-based Optimizers`__

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        momentum: momentum factor (default: 0)
        dampening: dampening for momentum (default: 0)
        eps: term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
        delta: threhold that determines whether a set of parameters is scale
            invariant or not (default: 0.1)
        wd_ratio: relative weight decay applied on scale-invariant parameters
            compared to that applied on scale-variant parameters (default: 0.1)
        nesterov: enables Nesterov momentum (default: False)


    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.SGDP(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

     __ https://arxiv.org/abs/2006.08217

    Note:
        Reference code: https://github.com/clovaai/AdamP
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        eps: float = 1e-8,
        weight_decay: float = 0,
        nesterov: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if momentum < 0.0:
            raise ValueError('Invalid momentum value: {}'.format(momentum))
        if dampening < 0.0:
            raise ValueError('Invalid dampening value: {}'.format(dampening))
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            eps=eps,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        super(SGD, self).__init__(params, defaults)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )

                # SGD
                buf = state['momentum']
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                if nesterov:
                    d_p = grad + momentum * buf
                else:
                    d_p = buf

                # Weight decay
                if weight_decay != 0:
                    p.data.mul_(
                        1
                        - group['lr']
                        * group['weight_decay']
                        / (1 - momentum)
                    )

                # Step
                p.data.add_(d_p, alpha=-group['lr'])

        return loss
