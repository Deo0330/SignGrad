import math

import torch
from torch.optim.optimizer import Optimizer

from .types import Betas2, OptFloat, OptLossClosure, Params

__all__ = ('SignAdamW',)


class SignAdamW(Optimizer):
    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        betas: Betas2 = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        nesterov: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        super(SignAdamW, self).__init__(params, defaults)



    def step(self, closure: OptLossClosure = None) -> OptFloat:
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                beta1, beta2 = group['betas']
                nesterov = group['nesterov']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )
                    state['exp_avg_sq'] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )
                    state['previous_grad'] = torch.zeros_like(p.data)
                    state['scale'] = torch.zeros_like(p.data)

                # Adam
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                previous_grad = state['previous_grad']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                    group['eps']
                )
                step_size = group['lr'] / bias_correction1
                
                mul = previous_grad * grad
                scale3 = torch.sigmoid(100000.* mul)
                # std = torch.std(mul)
                # mean = torch.mean(mul).add_(group['eps'])

                # scale_jiastd_mean = torch.sigmoid((mul+std)/mean)
                # scale_mean = torch.sigmoid((mul)/mean)
                # scale3 = torch.sigmoid(100000.* previous_grad * grad)
                exp_avg1 = exp_avg * scale3

                state['previous_grad'] = grad.clone()
                state['scale'] = scale3

                if nesterov:
                    perturb = (beta1 * exp_avg1 + (1 - beta1) * grad) / denom
                else:
                    perturb = exp_avg1 / denom

                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(
                        1 - group['lr'] * group['weight_decay']
                    )

                # Step
                p.data.add_(perturb, alpha=-step_size)

        return loss
