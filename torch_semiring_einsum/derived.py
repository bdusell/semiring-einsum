__all__ = ['tensordot', 'matmul', 'inner', 'dot', 'mm', 'bmm', 'mv', 'outer']

from .real_forward import real_einsum_forward
from .real_backward import real_einsum_backward
from .equation import compile_equation
from .function import combine

default_einsum = combine(real_einsum_forward, real_einsum_backward)

def index_range(start, stop):
    e = []
    for i in range(start, stop):
        e.append(chr(ord('a')+i))
    return ''.join(e)

def tensordot(a, b, ndim, *, block_size, einsum=default_einsum):
    if isinstance(ndim, (tuple, list)):
        raise NotImplementedError()
    e = (index_range(0, a.ndim) +
         ',' +
         index_range(a.ndim-ndim,a.ndim+b.ndim-ndim) +
         '->' +
         index_range(0, a.ndim-ndim) +
         index_range(a.ndim, a.ndim+b.ndim-ndim))
    e = compile_equation(e)
    return einsum(e, a, b, block_size=block_size)

def matmul(a, b, *, block_size, einsum=default_einsum):
    """Like torch.matmul"""
    if a.ndim == 0 or b.ndim == 0:
        raise ValueError('matmul of 0-dimensional tensors is not allowed')

    ndim = max(a.ndim, b.ndim)
    
    oi = index_range(3, ndim+1)
    if a.ndim == 1:
        ai = 'b'
    else:
        ai = index_range(ndim+1-(a.ndim-2), ndim+1) + 'ab'
        oi += 'a'

    if b.ndim == 1:
        bi = 'b'
    else:
        bi = index_range(ndim+1-(b.ndim-2), ndim+1) + 'bc'
        oi += 'c'

    e = compile_equation(ai+','+bi+'->'+oi)
    return einsum(e, a, b, block_size=block_size)

def inner(a, b, *, block_size, einsum=default_einsum):
    if a.ndim == 0:
        e = ','+index_range(0, b.ndim) + '->' + index_range(0, b.ndim)
    elif b.ndim == 0:
        e = index_range(0, a.ndim) + ',->' + index_range(0, a.ndim)
    else:
        ai = index_range(1, a.ndim) 
        bi = index_range(a.ndim+1, a.ndim+b.ndim)
        e =  ai + 'a,' + bi + 'a->' + ai + bi
    e = compile_equation(e)
    return einsum(e, a, b, block_size=block_size)
    
dot_equation = compile_equation('i,i->i')
def dot(a, b, *, block_size, einsum=default_einsum):
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError('arguments must be 1-dimensional')
    return einsum(dot_equation, a, b, block_size=block_size)

mm_equation = compile_equation('ij,jk->ik')
def mm(a, b, *, block_size, einsum=default_einsum):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError('arguments must be 2-dimensional')
    return einsum(mm_equation, a, b, block_size=block_size)

mv_equation = compile_equation('ij,j->i')
def mv(a, b, *, block_size, einsum=default_einsum):
    if a.ndim != 2 or b.ndim != 1:
        raise ValueError('arguments must be 2-dimensional and 1-dimensional, respectively')
    return einsum(mv_equation, a, b, block_size=block_size)

bmm_equation = compile_equation('bij,bjk->bik')
def bmm(a, b, *, block_size, einsum=default_einsum):
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError('arguments must be 3-dimensional')
    return einsum(bmm_equation, a, b, block_size=block_size)

outer_equation = compile_equation('i,j->ij')
def outer(a, b, *, block_size, einsum=default_einsum):
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError('arguments must be 1-dimensional')
    return einsum(outer_equation, a, b, block_size=block_size)
ger = outer
    
    
