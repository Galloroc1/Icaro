from dask.array.core import Array
import math
import sys
from paillier_func import raw_encrypt, \
    get_random_lt_n, \
    get_mantissa, \
    powmod, \
    mulmod, invert, l_function, crt_func, raw_add
from paillier_single import EncryptedNumber, generate_paillier_keypair
import dask.array as da
import numpy as np
from dask.array.reductions import reduction
import logging
from dask.array import map_blocks

# from communication.core import Communicate
pub, qub = generate_paillier_keypair()
logging.basicConfig(level=logging.WARNING)


class Tensor(Array):
    BASE = 16
    LOG2_BASE = math.log(BASE, 2)
    FLOAT_MANTISSA_BITS = sys.float_info.mant_dig
    n = pub.n
    nsquare = n ** 2
    max_int = n // 3 - 1

    # com = Communicate()

    def __new__(cls, dask, name, chunks, dtype=None, meta=None, shape=None, is_encrypt=False):
        obj = super().__new__(cls, dask=dask, name=name, chunks=chunks, meta=meta, dtype=dtype, shape=shape)
        obj.encoding = None
        obj.exponent = None
        obj.__is_obfuscated = False
        obj.is_encrypt = is_encrypt
        return obj

    def __div__(self, other):
        return super(Tensor, self).__div__(other)

    def __divmod__(self, other):
        return super(Tensor, self).__divmod__(other)

    def init_public_key_informathion(self):
        self.nsquare = self.n ** 2
        self.max_int = self.n // 3

    def encrypt(self):
        self.init_public_key_informathion()
        if len(self.shape) == 0:
            self = array_to_tensor(self.reshape((1, 1)))
        encoding, exponent = self.encode(self.n)
        ciphertext = raw_encrypt(self.n, self.nsquare, self.max_int, encoding, 1)
        ciphertext = self.obfuscate(ciphertext)
        # print(dir(da.array([1,2,3])))
        r = da.stack([ciphertext, exponent], axis=0)
        r = r.rechunk((2,)+self.chunksize)
        return array_to_tensor(r, is_encrypt=True)

    def obfuscate(self, ciphertext):
        r = get_random_lt_n(self.n)
        r_pow_n = powmod(r, self.n, self.nsquare)
        ciphertext = da.frompyfunc(mulmod, 3, 1)(ciphertext, r_pow_n, self.nsquare)
        return array_to_tensor(ciphertext)

    def encode(self, n, max_exponent=None):
        if np.issubdtype(self.dtype, np.int16) or np.issubdtype(self.dtype, np.int32) \
                or np.issubdtype(self.dtype, np.int64):
            prec_exponent = da.zeros(self.shape, dtype=np.int32)

        elif np.issubdtype(self.dtype, np.float16) \
                or np.issubdtype(self.dtype, np.float32) or np.issubdtype(self.dtype, np.float64):
            bin_flt_exponent = da.frexp(self)[1]
            bin_lsb_exponent = bin_flt_exponent - self.FLOAT_MANTISSA_BITS
            prec_exponent = da.floor(bin_lsb_exponent / self.LOG2_BASE)

        elif np.issubdtype(self.dtype, object):
            prec_exponent = da.zeros(self.shape, dtype=np.int32)
        else:
            raise TypeError("Don't know the precision of type %s."
                            % type(self))

        if max_exponent is None:
            exponent = prec_exponent
        else:
            exponent = da.minimum(max_exponent, prec_exponent)
            mins = exponent.min()
            exponent = exponent == exponent
            exponent[exponent] = mins

        exponent = exponent.astype(int)
        encoding = ((self * da.power(self.BASE, -exponent)).astype(int)) % n
        return array_to_tensor(encoding), array_to_tensor(exponent)

    def add_scaler(self, other):
        encoding, exponent = other.encode(self.n, max_exponent=self[1])
        return self._add_encoded(encoding, exponent)

    def decrease_exponent_to(self, encoding, exponent, new_exp):
        factor = da.power(self.BASE, exponent - new_exp)
        new_enc = encoding * factor % self.n

        return new_enc, new_exp

    def decrease_exponent_to_2(self, exponent, new_exp):
        new_base = array_to_tensor(da.power(self.BASE, exponent - new_exp), is_encrypt=True)
        multiplied = self * new_base

        return array_to_tensor(multiplied[0], is_encrypt=True), \
               array_to_tensor(new_exp, is_encrypt=True)

    def _raw_mul(self, encoding, plaintext):

        if da.any(plaintext < 0) or da.any(plaintext >= self.n):
            raise ValueError('Scalar out of bounds: %i' % plaintext)
        if da.any(self.n - self.n // 3 - 1 <= plaintext):
            # Very large plaintext, play a sneaky trick using inverses
            neg_c = da.frompyfunc(invert, 2, 1)(encoding, self.nsquare)
            neg_scalar = self.n - plaintext
            return da.frompyfunc(powmod, 3, 1)(neg_c, neg_scalar, self.nsquare)
        else:
            return da.frompyfunc(powmod, 3, 1)(encoding, plaintext, self.nsquare)

    def h_function(self, x, xsquare):
        """Computes the h-function as defined in paaaa's paper page 12,
        'Decryption using Chinese-remaindering'.
        """
        return invert(l_function(powmod(self.n + 1, x - 1, xsquare), x), x)

    def decrypt_init(self, n, p, q):
        if not p * q == n:
            raise ValueError('given public key does not match the given p and q.')
        if p == q:
            # check that p and q are different, otherwise we can't compute p^-1 mod q
            raise ValueError('p and q have to be different')

    def decrypt(self):
        if self.is_encrypt:
            return self
        self.decrypt_init(qub.n, qub.p, qub.q)
        return self.decode(self.decrypt_encoded(qub.n, qub.p, qub.q))

    def decrypt_encoded(self, n, p, q):
        if self.n != n:
            raise ValueError('encrypted_number was encrypted against a '
                             'different key!')
        # x, q, qsquare, hq, p, psquare, hp, p_inverse
        qsquare = q * q
        psquare = p * p
        hp = self.h_function(p, psquare)
        hq = self.h_function(q, qsquare)
        p_inverse = invert(p, q)
        return da.frompyfunc(crt_func, 8, 1)(self.obfuscate(self[0]), q, qsquare, hq, p, psquare, hp, p_inverse)

    def decode(self, encoding):
        """Decode plaintext and return the result.

        Returns:
          an int or float: the decoded number. N.B. if the number
            returned is an integer, it will not be of type float.

        Raises:
          OverflowError: if overflow is detected in the decrypted number.
        """
        mantissa = da.frompyfunc(get_mantissa, 5, 1)(encoding, self.BASE, self[1], self.n, self.n // 3 - 1)
        return array_to_tensor(mantissa)

    def show_graph(self, path='my-dask-graph.pdf'):
        import dask
        dask.visualize(self, filename=path)

    def _add_encoded(self, other_encoding, other_exponent):
        min_exp = array_to_tensor(da.minimum(self[1], other_exponent), is_encrypt=True)
        if da.any(self[1] != min_exp):
            self_new_enc, self_new_exp = self.decrease_exponent_to_2(self[1], min_exp)
        else:

            self_new_enc, self_new_exp = self[0], self[1]

        if da.any(other_exponent != min_exp):
            other_new_enc, other_new_exp = self.decrease_exponent_to(other_encoding, other_exponent, min_exp)

        else:

            other_new_enc, other_new_exp = other_encoding, other_exponent

        # Don't bother to salt/obfuscate in a basic operation, do it
        # just before leaving the computer.
        encrypted_scalar = raw_encrypt(self.n, self.nsquare, self.max_int, other_new_enc, 1)
        sum_ciphertext = raw_add(self_new_enc, encrypted_scalar, self.nsquare)
        return array_to_tensor(da.stack([sum_ciphertext, self_new_exp], axis=0), is_encrypt=True)

    def _add_encrypted(self, other):
        min_exp = array_to_tensor(da.minimum(self[1], other[1]))

        if da.any(self[1] != min_exp):
            self_new_enc, self_new_exp = self.decrease_exponent_to_2(self[1], min_exp)
        else:
            self_new_enc, self_new_exp = self[0], self[1]

        if da.any(other[1] != min_exp):
            other_new_enc, other_new_exp = self.decrease_exponent_to_2(other[1], min_exp)
        else:
            other_new_enc, other_new_exp = other[0], other[1]
        sum_ciphertext = raw_add(self_new_enc, other_new_enc, nsquare=self.nsquare)

        return array_to_tensor(da.stack([sum_ciphertext, min_exp], axis=0), is_encrypt=True)

    def __add__(self, other):
        print("pass add")
        """Add an int, float, `EncryptedNumber` or `EncodedNumber`."""
        if self.is_encrypt and (isinstance(other, float) or isinstance(other, int)):
            other = toTensor(da.full(self.shape[1:], fill_value=other))
            return self.add_scaler(other)
        if self.is_encrypt and other.is_encrypt:
            return self._add_encrypted(other)

        if self.is_encrypt and not other.is_encrypt:
            return self.add_scaler(other)

        if not self.is_encrypt and other.is_encrypt:
            return other.__add__(self)

        if not self.is_encrypt and not other.is_encrypt:
            return array_to_tensor(super().__add__(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self + array_to_tensor(other * -1)

    def __truediv__(self, scalar):
        return self.__mul__(1 / scalar)

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            other = array_to_tensor(da.full(self.shape[1:], fill_value=other))

        if isinstance(other, Array) or isinstance(other, np.ndarray):
            other = array_to_tensor(other)

        assert isinstance(other, Tensor), f'check your data type'

        if not self.is_encrypt and not other.is_encrypt:
            return array_to_tensor(super().__mul__(other))

        if self.is_encrypt and other.is_encrypt:
            raise TypeError("good luck for you")

        if self.is_encrypt and not other.is_encrypt:
            encoding, exponent = other.encode(self.n)
            product = self._raw_mul(self[0], encoding)
            exponent = self[1] + exponent
            return array_to_tensor(da.stack([product, exponent], axis=0), is_encrypt=True)

        if not self.is_encrypt and other.is_encrypt:
            return other.__mul__(self)

    def sum(self, axis=None, dtype=None, keepdims=True, split_every=None, out=None):
        if self.is_encrypt:
            assert keepdims is True, ValueError("\nI'm sorry about that error, keepdims must be Ture now, "
                                                "we will optimize it.\n but not now .......")
            if axis == 0:
                logging.warning("\nYou try to use sum(axis=0) on the ciphertext tensor\n"
                                "it won't work because axis 0 is used to put cip and exp,"
                                " and we will return the original value to you")
            obj = sums(self, axis=axis, keepdims=True)
            obj = array_to_tensor(obj, is_encrypt=True)
            return obj
        else:
            return array_to_tensor(super(Tensor, self).sum(axis=axis))

    def dot(self, other):
        if all(not x.is_encrypt for x in [self,other]):
            return array_to_tensor(da.dot(self, other))

        (a, b) = (self,other) if self.is_encrypt else (other,self)
        z = map_blocks(dot_single, a, dtype=object)

        return z


def dot_single(a):
    try:
        arrays = EncryptedNumber(Tensor.n, a[0], a[1]).toArray()
    except:
        arrays = np.empty(a.shape)
    return arrays

def array_to_tensor(array, is_encrypt=False):
    tensor = Tensor(array.dask, name=array.name, chunks=array.chunks, dtype=array.dtype, meta=array._meta,
                    shape=array.shape, is_encrypt=is_encrypt)
    return tensor


def toTensor(x,
             chunks="auto",
             name=None,
             lock=False,
             asarray=None,
             fancy=True,
             getitem=None,
             meta=None,
             inline_array=False,
             who=None,
             dname=None):
    if x is None:
        result = array_to_tensor(da.empty(shape=(0, 0))).get(who=who, dname=dname)
        result = array_to_tensor(da.from_array(result['data'], chunks=1), is_encrypt=result['is_encrypt'])
        return result

    if isinstance(x, np.ndarray):
        daarray = da.from_array(x, chunks=chunks,
                                name=name,
                                lock=lock,
                                asarray=asarray,
                                fancy=fancy,
                                getitem=getitem,
                                meta=meta,
                                inline_array=inline_array)

        return array_to_tensor(daarray)

    if isinstance(x, Array):
        return array_to_tensor(x)


def sums(array, axis=None, keepdims=True):
    array = array.rechunk(chunks=(2, 4, 4))
    dtype = getattr(np.zeros(1, dtype=array.dtype).sum(), "dtype", object)
    array = reduction(x=array,
                      chunk=sum_chunk,
                      aggregate=sum_chunk,
                      axis=axis,
                      dtype=dtype,
                      keepdims=keepdims,
                      split_every=None,
                      out=None
                      )
    return array


def sum_chunk(array, axis=None, keepdims=True, ):
    try:
        axis = tuple(filter(lambda x: x > 0, axis))
        arrays = EncryptedNumber(Tensor.n, array[0], array[1]).toArray()
        arrays = arrays.reshape((1,) + arrays.shape)
        arrays = np.sum(arrays, axis, keepdims=keepdims)
        arrays = np.frompyfunc(lambda x: (x.ciphertext(True), x.exponent), 1, 2)(arrays)
        arrays = np.concatenate(arrays, axis=0)
    except:
        arrays = np.empty((0,))
    return arrays

import numbers
import warnings

import tlz as toolz
import numpy as np
from dask import base, utils
from dask.blockwise import blockwise as core_blockwise
from dask.delayed import unpack_collections
from dask.highlevelgraph import HighLevelGraph
from dask.array import from_array,Array
from collections.abc import Iterable
from numbers import Integral
from dask.array.core import new_da_object
from dask.utils import  derived_from
from dask.utils import Dispatch
from dask.array import blockwise
from dask.array.dispatch import tensordot_lookup
def dot_(a, b, axes):
    a = EncryptedNumber(Tensor.n, a[0], a[1]).toArray()
    r = np.dot(a,b)
    # r = np.tensordot(a,b,axes=axes)
    a = np.vectorize(lambda x:(x.ciphertext(True),x.exponent))(r)
    cip,exp = np.expand_dims(a[0],axis=0),np.expand_dims(a[1],axis=0)
    a = np.concatenate([cip,exp],axis=0)
    # a = np.tensordot(a,b,axes=axes)
    return a


def _tensordot(a, b, axes, is_sparse):
    x = max([a, b], key=lambda x: x.__array_priority__)
    # tensordot = tensordot_lookup.dispatch(type(x))
    tensordot = dot_
    # print(tensordot.__code__)
    x = tensordot(a, b, axes=axes)
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
def tensordot(lhs, rhs, axes=2):
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
    )
    if concatenate:
        return intermediate
    else:
        return intermediate.sum(axis=left_axes)


@derived_from(np, ua_args=["out"])
def dot(a, b):
    return tensordot(a, b, axes=((a.ndim - 1,), (b.ndim - 2,)))

#
# data = toTensor(np.random.random_sample((2,3,4))).encrypt()
# data2 = toTensor(np.random.random_sample((4,2)))
# data = dot(data,data2)
#
# print(data.compute().shape)
