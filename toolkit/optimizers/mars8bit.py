import math
import torch
from torch.optim import Optimizer
from toolkit.optimizers.optimizer_utils import Auto8bitTensor, copy_stochastic, stochastic_grad_accummulation

@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.

    This version allows for G to be more than 2D.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.transpose(-2, -1)
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.transpose(-2, -1)
    return X


class MARS8bit(Optimizer):
    """
    Implements MARS optimizer with 8-bit state storage and stochastic rounding.
    MARS (Make vAriance Reduction Shine) combines variance reduction with preconditioned gradients.

    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups
        lr (float): Learning rate (default: 1e-3)
        betas (tuple): Coefficients for computing running averages (default: (0.9, 0.95))
        eps (float): Term added to denominator to improve numerical stability (default: 1e-8)
        weight_decay (float): Weight decay coefficient (default: 0.1)
        momentum (float): Momentum for MARS updates (default: 0.95)
        gamma (float): Gamma parameter in MARS (default: 0.025)
        ns_steps (int): Newton-Schulz iteration steps (default: 5)
        clip_c (bool): Clip c_t vector to have norm at most 1 (default: False)
        is_approx (bool): Use approximate MARS (True) or exact version (False) (default: True)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        momentum=0.95,
        gamma=0.025,
        ns_steps=5,
        clip_c=False,
        is_approx=True
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")

        mars_factor = gamma * momentum / (1 - momentum)
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            gamma=gamma,
            mars_factor=mars_factor,
            ns_steps=ns_steps,
            clip_c=clip_c,
            is_approx=is_approx
        )
        super(MARS8bit, self).__init__(params, defaults)

        self.is_stochastic_rounding_accumulation = False

        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad and param.dtype != torch.float32:
                    self.is_stochastic_rounding_accumulation = True
                    param.register_post_accumulate_grad_hook(stochastic_grad_accummulation)

                self.state[param]["use_muon"] = param.ndim >= 2

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return True

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        return lr * adjusted_ratio

    def step_hook(self):
        if not self.is_stochastic_rounding_accumulation:
            return
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad and hasattr(param, "_accum_grad"):
                    param.grad = param._accum_grad
                    del param._accum_grad

    @torch.no_grad()
    def update_last_grad(self):
        """Call this after training step if using exact MARS (is_approx=False)"""
        for group in self.param_groups:
            if not group['is_approx']:
                for p in group['params']:
                    state = self.state[p]
                    if not state["use_muon"]:
                        continue
                    if "last_grad" not in state:
                        state["last_grad"] = torch.zeros_like(p)
                    if "previous_grad" in state:
                        state["last_grad"].zero_().add_(state["previous_grad"], alpha=1.0)

    @torch.no_grad()
    def update_previous_grad(self):
        """Call this before training step if using exact MARS (is_approx=False)"""
        for group in self.param_groups:
            if not group['is_approx']:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if not state["use_muon"]:
                        continue
                    if "previous_grad" not in state:
                        state['previous_grad'] = torch.zeros_like(p)
                    state['previous_grad'].zero_().add_(p.grad, alpha=1.0)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        self.step_hook()

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            gamma = group['gamma']
            mars_factor = group['mars_factor']
            ns_steps = group['ns_steps']
            clip_c = group['clip_c']
            is_approx = group['is_approx']
            beta1, beta2 = group['betas']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data.to(torch.float32)
                p_fp32 = p.clone().to(torch.float32)
                state = self.state[p]
                use_muon = state["use_muon"]

                if use_muon:
                    if 'step' not in state:
                        state['step'] = 0
                        state['last_grad'] = Auto8bitTensor(torch.zeros_like(p_fp32.data).detach())
                        state['momentum_buffer'] = Auto8bitTensor(torch.zeros_like(p_fp32.data).detach())
                        if not is_approx:
                            state['previous_grad'] = Auto8bitTensor(torch.zeros_like(p_fp32.data).detach())

                    last_grad = state['last_grad'].to(torch.float32)
                    momentum_buffer = state['momentum_buffer'].to(torch.float32)

                    c_t = (grad - last_grad).mul(mars_factor).add(grad)
                    c_t_norm = c_t.norm()
                    if clip_c and c_t_norm > 1:
                        c_t.div_(c_t_norm)

                    momentum_buffer.mul_(momentum).add_(c_t, alpha=(1 - momentum))

                    u = zeropower_via_newtonschulz5(momentum_buffer, steps=ns_steps)

                    adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                    p_fp32.data.mul_(1 - lr * weight_decay)
                    p_fp32.data.add_(u, alpha=-adjusted_lr)

                    if is_approx:
                        state['last_grad'] = Auto8bitTensor(grad)
                    state['momentum_buffer'] = Auto8bitTensor(momentum_buffer)

                    copy_stochastic(p.data, p_fp32.data)

                else:
                    if 'step' not in state:
                        state['step'] = 0
                        state['exp_avg'] = Auto8bitTensor(torch.zeros_like(p_fp32.data).detach())
                        state['exp_avg_sq'] = Auto8bitTensor(torch.zeros_like(p_fp32.data).detach())

                    exp_avg = state['exp_avg'].to(torch.float32)
                    exp_avg_sq = state['exp_avg_sq'].to(torch.float32)

                    state['step'] += 1
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']

                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    p_fp32.data.mul_(1 - lr * weight_decay)

                    step_size = lr / bias_correction1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                    p_fp32.data.addcdiv_(exp_avg, denom, value=-step_size)

                    state['exp_avg'] = Auto8bitTensor(exp_avg)
                    state['exp_avg_sq'] = Auto8bitTensor(exp_avg_sq)

                    copy_stochastic(p.data, p_fp32.data)

        return loss

    def state_dict(self):
        """Returns the state of the optimizer as a dict."""
        state_dict = super().state_dict()

        for param_id, param_state in state_dict['state'].items():
            for key, value in param_state.items():
                if isinstance(value, Auto8bitTensor):
                    param_state[key] = {
                        '_type': 'Auto8bitTensor',
                        'state': value.state_dict()
                    }

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the optimizer state."""
        super().load_state_dict(state_dict)

        for param_id, param_state in self.state.items():
            for key, value in param_state.items():
                if isinstance(value, dict) and value.get('_type') == 'Auto8bitTensor':
                    param_state[key] = Auto8bitTensor(value['state'])
