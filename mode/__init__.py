"""Keep inference imports independent of training-only dependencies."""


def train(*args, **kwargs):
    from .trainer import train as run_train
    return run_train(*args, **kwargs)
