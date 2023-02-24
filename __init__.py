from train_pairwise import train_pairwise
from test_pairwise import test_pairwise


__factory__ = {
    'train_pairwise': train_pairwise,
    'test_pairwise': test_pairwise,
}


def build_handler(phase, model):
    key_handler = '{}_{}'.format(phase, model)
    if key_handler not in __factory__:
        raise KeyError("Unknown op:", key_handler)
    return __factory__[key_handler]