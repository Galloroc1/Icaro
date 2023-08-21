import numpy as np
from dask.array import Array,from_array,blockwise
from nppaillier.paillier import EncryptedNumber
from task.config import pub
from collections.abc import Iterable
from numbers import Integral
from dask.utils import derived_from


def sum_chunk(array, axis=None, keepdims=True, ):
    try:
        axis = tuple(filter(lambda x: x > 0, axis))
        arrays = EncryptedNumber(pub.n, array[0], array[1]).toArray()
        arrays = arrays.reshape((1,) + arrays.shape)
        arrays = np.sum(arrays, axis, keepdims=keepdims)
        arrays = np.frompyfunc(lambda x: (x.ciphertext(True), x.exponent), 1, 2)(arrays)
        arrays = np.concatenate(arrays, axis=0)
    except:
        arrays = np.empty((0,))
    return arrays


def dot_(a, b,right):
    if not right:
        a = EncryptedNumber(pub.n, a[0], a[1]).toArray()
    else:
        b = EncryptedNumber(pub.n, b[0], b[1]).toArray()
    r = np.dot(a,b)
    r = np.vectorize(lambda x:(x.ciphertext(True),x.exponent))(r)
    cip, exp = np.expand_dims(r[0],axis=0),np.expand_dims(r[1],axis=0)
    r = np.concatenate([cip, exp],axis=0)
    return r


def _tensordot(a, b, axes, is_sparse,right):
    x = dot_(a, b,right)
    if is_sparse and len(axes[0]) == 1:
        return x
    else:
        ind = [slice(None, None)] * x.ndim
        for a in sorted(axes[0]):
            ind.insert(a, None)
        x = x[tuple(ind)]
        return x


def _tensordot_is_sparse(x):
    is_sparse = "sparse" in str(type(x._meta))
    if is_sparse:
        # exclude pydata sparse arrays, no workaround required for these in tensordot
        is_sparse = "sparse._coo.core.COO" not in str(type(x._meta))
    return is_sparse


@derived_from(np)
def tensordot(lhs, rhs, axes=2, right=False):
    if not isinstance(lhs, Array):
        lhs = from_array(lhs)
    if not isinstance(rhs, Array):
        rhs = from_array(rhs)

    if isinstance(axes, Iterable):
        left_axes, right_axes = axes
    else:
        left_axes = tuple(range(lhs.ndim - axes, lhs.ndim))
        right_axes = tuple(range(0, axes))
    if isinstance(left_axes, Integral):
        left_axes = (left_axes,)
    if isinstance(right_axes, Integral):
        right_axes = (right_axes,)
    if isinstance(left_axes, list):
        left_axes = tuple(left_axes)
    if isinstance(right_axes, list):
        right_axes = tuple(right_axes)
    is_sparse = _tensordot_is_sparse(lhs) or _tensordot_is_sparse(rhs)
    if is_sparse and len(left_axes) == 1:
        concatenate = True
    else:
        concatenate = False
    dt = np.promote_types(lhs.dtype, rhs.dtype)
    left_index = list(range(lhs.ndim))
    right_index = list(range(lhs.ndim, lhs.ndim + rhs.ndim))
    out_index = left_index + right_index

    adjust_chunks = {}
    for l, r in zip(left_axes, right_axes):
        out_index.remove(right_index[r])
        right_index[r] = left_index[l]
        if concatenate:
            out_index.remove(left_index[l])
        else:
            adjust_chunks[left_index[l]] = lambda c: 1
    intermediate = blockwise(
        _tensordot,
        out_index,
        lhs,
        left_index,
        rhs,
        right_index,
        dtype=dt,
        concatenate=concatenate,
        adjust_chunks=adjust_chunks,
        axes=(left_axes, right_axes),
        is_sparse=is_sparse,
        right=right
    )
    if concatenate:
        return intermediate
    else:
        return intermediate.sum(axis=left_axes)


