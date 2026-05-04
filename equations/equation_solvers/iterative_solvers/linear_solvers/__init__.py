from .bisection import truncated_bisection
from .newton import truncated_newton
from .grad_descent import grad_descent

__all__ = [
    "truncated_bisection",
    "truncated_newton",
    "grad_descent",
]