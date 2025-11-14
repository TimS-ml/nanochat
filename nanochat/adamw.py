"""
Distributed AdamW Optimizer Implementation

Borrowed from modded-nanogpt by Keller Jordan, @vagrawal, et al.

This is NOT a general-purpose optimizer - it's specifically designed for the nanochat use case.
Implements ZeRO-2 style distributed optimization with:
- Sharded optimizer states (each rank owns a slice of parameters)
- Gradient reduction via reduce_scatter
- Parameter synchronization via all_gather

Key optimizations:
- Compiled with torch.compile for performance
- Asynchronous communication with futures
- Efficient memory usage through state sharding
"""
import torch
import torch.distributed as dist
from torch import Tensor


class DistAdamW(torch.optim.Optimizer):
    """
    Distributed AdamW optimizer with ZeRO-2 style sharding.

    Features:
    - Sharded optimizer states: each rank maintains optimizer state for only a slice of parameters
    - Gradient averaging via reduce_scatter: gradients are reduced and scattered across ranks
    - Parameter synchronization via all_gather: updated parameters are gathered to all ranks
    - Memory efficient: optimizer states are distributed across all GPUs

    Args:
        param_groups: List of parameter groups (standard optimizer format)
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
    """
    def __init__(self, param_groups, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    @torch.compile  # Compile for performance (fuses operations, reduces Python overhead)
    @torch.no_grad()  # Disable gradient tracking for optimizer step
    def step(self):
        """
        Perform a single optimization step with distributed gradient reduction and parameter update.

        Process:
        1. Reduce-scatter gradients: average gradients across ranks, each rank gets a slice
        2. Update parameters: each rank updates its slice using AdamW algorithm
        3. All-gather parameters: broadcast updated slices to all ranks
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        grad_slices = []

        # Phase 1: Kick off reduce-scatter operations for all gradients
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for base_i in range(len(params)):
                grad = params[base_i].grad
                # Calculate size of this rank's slice
                rank_size = grad.shape[0] // world_size
                # Allocate buffer for the reduced gradient slice
                grad_slice = torch.empty_like(grad[:rank_size])
                # Launch async reduce-scatter: averages gradient and gives each rank a slice
                reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                grad_slices.append(grad_slice)

        # Phase 2: Wait for gradients and perform AdamW updates
        idx = 0
        for group in self.param_groups:
            # Extract hyperparameters for this group
            beta1, beta2 = group['betas']  # Momentum coefficients
            eps = group['eps']  # Numerical stability epsilon
            wd = group['weight_decay']  # Weight decay coefficient
            params = group['params']

            for base in range(len(params)):
                # Wait for this gradient's reduce-scatter to complete
                reduce_scatter_futures[idx].wait()
                p = params[base]

                # Extract this rank's slice of the parameter
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]

                # Get learning rate (with optional per-parameter multiplier)
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]
                g_slice = grad_slices[idx]

                # Initialize optimizer state on first step
                if not state:
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p_slice)  # First moment (momentum)
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)  # Second moment (RMSprop)

                # Get optimizer state
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                # Apply weight decay (decoupled, as in AdamW)
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)

                # Update running averages of gradient and its square
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)  # First moment
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)  # Second moment

                # Compute bias corrections (for warmup)
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t

                # Compute the update step
                denom = exp_avg_sq.sqrt().add_(eps)  # Add epsilon for numerical stability
                step_size = lr * (torch.sqrt(bias2) / bias1)  # Bias-corrected step size
                update = exp_avg.div(denom).mul_(step_size)  # Bias-corrected adaptive update
                p_slice.add_(other=update, alpha=-1.0)  # Apply update

                idx += 1
                # Launch async all-gather to synchronize updated parameters across ranks
                all_reduce_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())

        # Phase 3: Wait for all parameter synchronization to complete
        torch.futures.collect_all(all_reduce_futures).wait()
