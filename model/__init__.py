from .pairwise_classify import *

__factory__ = {
    'pairwise': pairwise_classify,
}

def build_model(name, *args, **kwargs):
    if name not in __factory__:
        raise KeyError("Unknow model:",name)
    return __factory__[name](*args,**kwargs)