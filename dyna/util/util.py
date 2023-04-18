import torch
import os

def fc(var, name=None):
    """
    Utility function for when an argument does not have one of the expected values.
    :param var:
    :param name:
    :return:
    """
    if name is not None:
        raise Exception(f'Unknown value {var} for variable with name {name}.')
    raise Exception(f'Unknown value {var}.')

def d(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

def here(subpath=None):
    """
    :return: The path in which the package (dyna) resides, or if `subpath` is given, a path relative to that path.
    """
    if subpath is None:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', subpath))


