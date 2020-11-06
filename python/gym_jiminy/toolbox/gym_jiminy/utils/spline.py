""" TODO: Write documentation.
"""
# pylint: disable=invalid-name

from typing import Sequence, Optional, Tuple, Any

import scipy.linalg
import torch
from torch import nn


coefs_deriv = torch.tensor([[1, 1, 1, 1],  # pylint: disable=not-callable
                            [0, 1, 2, 3],
                            [0, 0, 2, 6],
                            [0, 0, 0, 6],
                            [0, 0, 0, 0]],
                           dtype=torch.float32)
A = torch.tensor([[1, -3,  3, -1],  # pylint: disable=not-callable
                  [0,  3, -6,  3],
                  [0,  0,  3, -3],
                  [0,  0,  0,  1]],
                 dtype=torch.float32)
A_deriv = torch.einsum('il,kl->ikl', coefs_deriv, A)
k = torch.arange(4).reshape((1, -1, 1))


class _SolveBanded(torch.autograd.Function):
    """Pytorch interface for scipy.linalg.solve_banded method for solving
    linear systems with banded matrices.

    It supports infinite-order automatic differentiation. It also features
    batch processing capability implemented using suboptimal for-loop.

    .. note::
        scipy.linalg.solve_banded provides in turn an interface to Lapack
        methods gtsv for tri-diagonal matrices and gbsv for generic banded
        matrices, choosing between them automatically.
    """
    @staticmethod
    def forward(ctx: Any,  # type: ignore[override]
                band: Sequence[int],
                M: torch.Tensor,
                b: torch.Tensor) -> torch.Tensor:
        """ TODO: Write documentation.
        """
        # pylint: disable=arguments-differ

        M = M.detach()
        b = b.detach()

        M_np = M.cpu().numpy()
        if M.ndim == b.ndim:
            b = b.transpose(-2, -1)
        b_np = b.cpu().numpy()

        x = torch.empty_like(b)
        if M.ndim > 2:
            grid = torch.meshgrid([
                torch.arange(size) for size in M.shape[:-2]])
            for ind in zip(*[grid_i.flatten() for grid_i in grid]):
                x[ind] = torch.from_numpy(scipy.linalg.solve_banded(
                    band, M_np[ind], b_np[ind],
                    overwrite_ab=False, overwrite_b=False, check_finite=False))
        else:
            x = torch.from_numpy(scipy.linalg.solve_banded(
                band, M_np, b_np,
                overwrite_ab=False, overwrite_b=False, check_finite=False))
        if M.ndim == b.ndim:
            x.transpose_(-2, -1)

        ctx.band = band
        ctx.save_for_backward(M, x)

        return x.to(b.device)

    @staticmethod
    def backward(ctx: Any,  # type: ignore[override]
                 grad_f_x: torch.Tensor
                 ) -> Tuple[None, torch.Tensor, torch.Tensor]:
        """ TODO: Write documentation.
        """
        # pylint: disable=arguments-differ

        M, x = ctx.saved_tensors

        Mtb = torch.zeros_like(M)
        for i, j in enumerate(range(ctx.band[0], 0, -1)):
            Mtb[..., i, j:].copy_(M[..., ctx.band[0] + ctx.band[1] - i, :-j])
        Mtb[..., ctx.band[0], :].copy_(M[..., ctx.band[1], :])
        for i, j in enumerate(range(-1, - ctx.band[1] - 1, -1)):
            Mtb[..., ctx.band[0] + 1 + i, :j].copy_(
                M[..., ctx.band[1] - 1 - i, -j:])

        grad_f_b = _SolveBanded.apply(
            [ctx.band[1], ctx.band[0]], Mtb, grad_f_x)
        grad_f_M = torch.zeros_like(M)
        for i, j in enumerate(range(-ctx.band[1], ctx.band[0] + 1)):
            if j > 0:
                d = - grad_f_b[..., j:] * x[..., :-j]  # Upper diag
            elif j < 0:
                d = - grad_f_b[..., :j] * x[..., -j:]  # Lower diag
            else:
                d = - grad_f_b * x  # Mid diag
            if d.ndim == M.ndim:
                d = torch.sum(d, dim=-2)
            if j > 0:
                grad_f_M[..., i, :-j].copy_(d)
            else:
                grad_f_M[..., i, -j:].copy_(d)
        return None, grad_f_M, grad_f_b


class _SolvehBanded(torch.autograd.Function):
    """Pytorch interface for scipy.linalg.solveh_banded method for solving
    linear systems with banded hermitian matrices.

    It supports infinite-order automatic differentiation. It also features
    batch processing capability implemented using suboptimal for-loop.

    .. note::
        scipy.linalg.solveh_banded provides in turn an interface to Lapack
        methods ptsv for tri-diagonal matrices and pbsv for generic hermitian
        banded matrices, choosing between them automatically.
    """
    @staticmethod
    def forward(ctx: Any,  # type: ignore[override]
                band: int,
                M: torch.Tensor,
                b: torch.Tensor) -> torch.Tensor:
        """ TODO: Write documentation.
        """
        # pylint: disable=arguments-differ

        M = M.detach()
        b = b.detach()

        M_np = M.cpu().numpy()
        if M.ndim == b.ndim:
            b = b.transpose(-2, -1)
        b_np = b.cpu().numpy()

        x = torch.empty_like(b)
        if M.ndim > 2:
            grid = torch.meshgrid([
                torch.arange(size) for size in M.shape[:-2]])
            for ind in zip(*[grid_i.flatten() for grid_i in grid]):
                x[ind] = torch.from_numpy(scipy.linalg.solveh_banded(
                    M_np[ind], b_np[ind], lower=False,
                    overwrite_ab=False, overwrite_b=False, check_finite=False))
        else:
            x = torch.from_numpy(scipy.linalg.solveh_banded(
                M_np, b_np, lower=False,
                overwrite_ab=False, overwrite_b=False, check_finite=False))
        if M.ndim == b.ndim:
            x.transpose_(-2, -1)

        ctx.band = band
        ctx.save_for_backward(M, x)

        return x.to(b.device)

    @staticmethod
    def backward(ctx: Any,  # type: ignore[override]
                 grad_f_x: torch.Tensor
                 ) -> Tuple[None, torch.Tensor, torch.Tensor]:
        """ TODO: Write documentation.
        """
        # pylint: disable=arguments-differ

        M, x = ctx.saved_tensors
        grad_f_b = _SolvehBanded.apply(ctx.band, M, grad_f_x)
        grad_f_M = torch.zeros_like(M)
        for i, j in enumerate(range(ctx.band, -1, -1)):
            if j > 0:
                d = - grad_f_b[..., :-j] * x[..., j:]  # Upper diag
                d -= grad_f_b[..., j:] * x[..., :-j]  # Lower diag
            else:
                d = - grad_f_b * x  # Mid diag
            if d.ndim == M.ndim:
                d = torch.sum(d, dim=-2)
            grad_f_M[..., i, j:].copy_(d)
        return None, grad_f_M, grad_f_b


class Spline(nn.Module):
    """This class defines an interpolator using piecewise polynomials of order
    3.

    The interpolator can be initialized using either:
        - the value of the function at given sampling points. In this case, the
          interpolator reproduces the behavior of the Matlab command:
          ```
          interp1(x_tab, y_tab, 'spline', 'pp');
          ```
        - the value of the function and its first derivative

    The interpolator yields a function which is twice continuously
    differentiable.

    It provides the value of the interpolating function, as well as its
    derivatives. The function is extrapolated outside the nominal interval
    using the interpolating polynomial close to the lowest or highest boundary.

    .. note::
        This implementation supports multi-dimensional interpolation and
        compatible Torch autograd `backward`. Yet, it does not support Batch
        processing.

        It also provides cache support, which is useful if most interpolation
        timesteps are known in advance. Note that using this feature actually
        slow done the computation if more than 80% of the query timesteps have
        not been cached. On the contrary, the speed up can be up to 50% if
        every query points have been cached.

        Maximum speed is achieved enabling Torch JIT via `torch.jit.script` and
        disabling gradient computation using context `with torch.no_grad():`.

    see::
        https://en.wikipedia.org/wiki/Spline_interpolation
        https://github.com/scipy/scipy/blob/v1.5.1/scipy/interpolate/_cubic.py
    """
    t: torch.Tensor
    dt: torch.Tensor
    period: torch.Tensor
    p: torch.Tensor
    A_deriv: torch.Tensor
    k: torch.Tensor

    def __init__(self,
                 t: torch.Tensor,
                 y: torch.Tensor,
                 dydt: Optional[torch.Tensor] = None,
                 mode: str = "not-a-knot") -> None:
        """ TODO: Write documentation.
        """
        assert mode in ("not-a-knot", "natural", "periodic"), (
            "'mode' must be either 'not-a-knot', 'natural', or 'periodic'.")
        super().__init__()

        # Handling of periodicity.
        # Impossible to differentiate wrt the period directly, but it is
        # not a big deal anyway. It is the same not understanding issue
        # than differentiating wrt to integer index. In fact, the gradient
        # is obtained indirectly from the calculus carry out with the
        # extracted elements.
        self.register_buffer('period', t[..., [-1]].detach())

        if t.ndim < y.ndim:
            t = t.unsqueeze(-2)

        # Backup some user arguments
        self.register_buffer('t', t)
        self.mode = mode

        # Compute some proxies
        self.register_buffer('dt', t[..., 1:] - t[..., :-1])

        if dydt is None:
            # Compute some proxies
            idt = self.dt.pow(-1)
            idt2 = idt.pow(2)
            dy = y[..., 1:] - y[..., :-1]

            # Build the linear system defining the derivative at each breaks
            M = torch.zeros((*t.shape[:-1], 3, t.shape[-1]),
                            dtype=y.dtype, device=y.device)
            b = torch.zeros_like(y)

            # Condition on the continuity of the 2nd derivative at
            # breaks[1], ..., breaks[N-1]
            M[..., 0, 2:].add_(idt[..., 1:])
            M[..., 1, 1:-1].add_(idt[..., :-1] + idt[..., 1:], alpha=2)
            M[..., 2, :-2].add_(idt[..., :-1])
            b[..., 1:-1] = 3 * (dy[..., :-1] * idt2[..., :-1] +
                                dy[..., 1:] * idt2[..., 1:])

            if mode == 'periodic':
                # Due to the periodicity, and because y[-1] = y[0], the
                # linear system has (n-1) unknowns/equations instead of n.
                M = M[..., :-1]
                M_m1_0 = idt[..., -1].clone()
                b = b[..., :-1]

                # Condition on continuity of the 2nd derivative at breaks[0]
                M[..., 0, 1].add_(idt[..., -1])
                M[..., 1, 0].add_(idt[..., -1] + idt[..., 0], alpha=2)
                M_0_m1 = idt[..., 0].clone()
                b[..., 0] = 3 * (dy[..., -1] * idt2[..., -1] +
                                 dy[..., 0] * idt2[..., 0])

                # Also, due to the periodicity, the system is not tri-diagonal.
                # The original matrix is split into the sum of a tri-diagonal
                # matrix and an outer product, which can then be inverted
                # efficiently using Sherman-Morrison formula and Thomas
                # algorithm. The resulting computational complexity is O(n),
                # instead of O(n**3) for the usual LU factorization.
                # https://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
                # https://en.wikipedia.org/wiki/Sherman-Morrison_formula
                u = torch.zeros((*t.shape[:-1], t.shape[-1] - 1, 1),
                                dtype=y.dtype, device=y.device)
                u[..., 0, 0] = -M[..., 1, 0].clone()
                u[..., -1, 0] = M_m1_0
                v = torch.zeros_like(u)
                v[..., 0, 0] = 1.0
                v[..., -1, 0] = - M_0_m1 / M[..., 1, 0].clone()
                M[..., 1, -1] += M_m1_0 * M_0_m1 / M[..., 1, 0].clone()
                M[..., 1, 0] += M[..., 1, 0].clone()
                if b.ndim > 1:
                    M.squeeze_(-3)

                # Jointly solve two intermediary banded matrix linear systems
                if M.ndim > b.ndim:
                    b_u = torch.stack([b, u.squeeze(-1)], dim=-2)
                else:
                    b_u = torch.cat([b, u.squeeze(-1)], dim=-2)
                s = _SolvehBanded.apply(1, M[..., :-1, :], b_u)
                if M.ndim > b.ndim:
                    d, q = s[..., 0, :], s[..., 1, :]
                else:
                    d, q = s[..., :-1, :], s[..., [-1], :]

                # Compute the derivative at each breaks
                dydt = torch.empty_like(y)
                dydt[..., :-1] = d - (
                    (d.unsqueeze(-2) @ v) / (1 + q.unsqueeze(-2) @ v)
                    ).squeeze(-1) * q
                dydt[..., -1] = dydt[..., 0]
            elif mode == 'not-a-knot':
                # It is assumed that the 3rd derivative is zero at both ends.
                # This end conditions is usually referred to as "not-a-knot".
                # Gaussian elimination is used to obtain a linear system with
                # tri-diagonal matrix, so that Thomas can be used to inverse
                # compute the inverse efficiently.

                # Condition on continuity of the 3rd derivative at breaks[0]
                M[..., 1, 0].add_(idt2[..., 0])
                M[..., 0, 1].add_(idt2[..., 0] - idt2[..., 1])
                M_0_2 = - idt2[..., 1].clone()
                b[..., 0] = 2 * (dy[..., 0] * idt[..., 0] ** 3 -
                                 dy[..., 1] * idt[..., 1] ** 3)

                # Condition on continuity of the 3rd derivative at breaks[N-1]
                M_m1_m3 = idt2[..., -2].clone()
                M[..., 2, -2].add_(idt2[..., -2] - idt2[..., -1])
                M[..., 1, -1].sub_(idt2[..., -1])
                b[..., -1] = 2 * (dy[..., -2] * idt[..., -2] ** 3 -
                                  dy[..., -1] * idt[..., -1] ** 3)

                # Use Gaussian elimination to obtain tri-diagonal matrix linear
                # system and solve it efficiently using Thomas algorithm.
                c1 = M_0_2.clone() / M[..., 0, 2].clone()
                c2 = M_m1_m3.clone() / M[..., 2, -3].clone()
                M[..., 1, 0].sub_(c1 * M[..., 2, 0].clone())
                M[..., 0, 1].sub_(c1 * M[..., 1, 1].clone())
                M[..., 2, -2].sub_(c2 * M[..., 1, -2].clone())
                M[..., 1, -1].sub_(c2 * M[..., 0, -1].clone())
                b[..., 0].sub_(c1 * b[..., 1].clone())
                b[..., -1].sub_(c2 * b[..., -2].clone())

                # Compute the derivative at each breaks
                if b.ndim > 1:
                    M.squeeze_(-3)
                dydt = _SolveBanded.apply((1, 1), M, b)
            elif mode == 'natural':
                # It is assumed that the 2th derivative is zero at both ends.
                # It enables using Thomas algorithm to solve efficiently the
                # resulting linear system, with computation complexity O(n).

                # Condition on continuity of the 2nd derivative at breaks[0]
                M[..., 1, 0].add_(idt[..., 0], alpha=2)
                M[..., 0, 1].add_(idt[..., 0])
                b[..., 0] = 3 * dy[..., 0] * idt2[..., 0]

                # Condition on continuity of the 2nd derivative at breaks[N-1]
                M[..., 2, -2].copy_(idt[..., -1])
                M[..., 1, -1].copy_(2 * idt[..., -1])
                b[..., -1] = 3 * dy[..., -1] * idt2[..., -1]

                # Compute the derivative at each breaks
                if b.ndim > 1:
                    M.squeeze_(-3)
                dydt = _SolvehBanded.apply(1, M[..., :-1, :], b)
        assert isinstance(dydt, torch.Tensor)

        # Compute the Bezier control points
        self.register_buffer('p', torch.stack([
            y[..., :-1],
            1/3 * dydt[..., :-1] * self.dt + y[..., :-1],
            -1/3 * dydt[..., 1:] * self.dt + y[..., 1:],
            y[..., 1:]], dim=-2))

        # Compute the Bezier polynomials coefficient of any derivative order
        self.register_buffer('A_deriv', A_deriv.type(y.dtype).to(y.device))
        self.register_buffer('k', k.type(y.dtype).to(y.device))

    def _h_poly(self, t: torch.Tensor, order: int = 0) -> torch.Tensor:
        """ TODO: Write documentation.
        """
        if order < 4:
            return (self.A_deriv[order] @
                    torch.pow(t.unsqueeze(-2), (self.k - order).clamp(min=0)))
        return torch.zeros_like(t).unsqueeze(0)

    def forward(self,
                ts: torch.Tensor,
                orders: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ TODO: Write documentation.
        """
        assert self.t is not None

        if orders is None:
            orders = torch.zeros((1,), device=ts.device)

        if ts.ndim < self.period.ndim:
            ts = ts.unsqueeze(-1)

        if self.mode == 'periodic':
            ts = ts.fmod(self.period)

        if ts.ndim < self.t.ndim:
            ts = ts.unsqueeze(-2)

        # Compute the indices in associated with each new sample.
        # Note that computing the gradient does not make sense in this case.
        with torch.no_grad():
            idx = torch.sum(
                ts.unsqueeze(-1) > self.t.unsqueeze(-2), dim=-1) - 1
            idx.clamp_(min=0, max=self.t.shape[-1] - 2)
            if idx.ndim > 1:
                idx = idx[..., 0, :]

        # Extract the data for each datapoint
        if idx.ndim > 1:
            t_idx = torch.stack([
                self.t[j, ..., i] for j, i in enumerate(idx)], dim=0)
            dt_idx = torch.stack([
                self.dt[j, ..., i] for j, i in enumerate(idx)], dim=0)
            p_idx = torch.stack([
                self.p[j, ..., i] for j, i in enumerate(idx)], dim=0)
        else:
            t_idx = self.t[..., idx]
            dt_idx = self.dt[..., idx]
            p_idx = self.p[..., idx]

        # Compute the interpolation for each order
        ratio = (ts - t_idx) / dt_idx
        ys = torch.stack([
            torch.sum(p_idx * self._h_poly(ratio, o), dim=-2) / dt_idx ** o
            for o in orders], dim=0)
        if self.p.ndim == 2:
            ys.squeeze_(1)

        return ys
