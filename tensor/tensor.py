import math
import sys
from nppaillier.function import \
    get_random_lt_n, \
    get_mantissa, \
    powmod, \
    mulmod, invert, l_function, crt_func
from task.config import pub, qub
from nppaillier.dafunction import raw_encrypt, raw_add
import dask.array as da
import logging
import numpy as np
from dask.array import Array, reduction
from dask.utils import derived_from
from tensor.op import sum_chunk, tensordot


class Tensor(Array):
    BASE = 16
    LOG2_BASE = math.log(BASE, 2)
    FLOAT_MANTISSA_BITS = sys.float_info.mant_dig
    n = pub.n
    nsquare = n ** 2
    max_int = n // 3 - 1

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

    def __str__(self):
        r = super(Tensor, self).__str__()
        return f"type is:{type(self).__name__}\nis encrypted :{self.is_encrypt}\narray information: {r}"

    def init_public_key_informathion(self):
        self.nsquare = self.n ** 2
        self.max_int = self.n // 3

    def reshape(self, *shape, merge_chunks=True, limit=None):
        super(Tensor, self).reshape(*shape, merge_chunks=True, limit=None)

    def encrypt(self):
        self.init_public_key_informathion()
        if len(self.shape) == 0:
            self = array_to_tensor(self.reshape((1, 1)))
        encoding, exponent = self.encode(self.n)
        ciphertext = raw_encrypt(self.n, self.nsquare, self.max_int, encoding, 1)
        ciphertext = self.obfuscate(ciphertext)
        r = da.stack([ciphertext, exponent], axis=0)
        r = r.rechunk((2,) + self.chunksize)
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
        encoding = ((self * array_to_tensor(da.power(self.BASE, -exponent))).astype(int)) % n
        return encoding, array_to_tensor(exponent)

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
        if not self.is_encrypt:
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
        """Add an int, float, `EncryptedNumber` or `EncodedNumber`."""
        if self.is_encrypt and (isinstance(other, float) or isinstance(other, int)):
            other = array_to_tensor(da.full(self.shape[1:], fill_value=other))
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
                logging.warning("\nYou try to use sum(axis=0) on the ciphertext aaaa\n"
                                "it won't work because axis 0 is used to put cip and exp,"
                                " and we will return the original value to you")
            obj = sums(self, axis=axis, keepdims=True)
            obj = array_to_tensor(obj, is_encrypt=True)
            return obj
        else:
            return array_to_tensor(super(Tensor, self).sum(axis=axis))

    def dot(self, other):
        return dot(self, other)




def array_to_tensor(array, is_encrypt=False):
    tensor = Tensor(array.dask, name=array.name, chunks=array.chunks, dtype=array.dtype, meta=array._meta,
                    shape=array.shape, is_encrypt=is_encrypt)
    return tensor


def sums(array, axis=None, keepdims=True):
    array = array.rechunk(chunks=(2,)+array.chunks[1:])
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


@derived_from(np, ua_args=["out"])
def dot(a, b) -> Tensor:
    if not a.is_encrypt and not b.is_encrypt:
        return array_to_tensor(da.dot(a, b))

    if a.is_encrypt and b.is_encrypt:
        raise "0.0"

    return array_to_tensor(tensordot(a, b, axes=((a.ndim - 1,), (b.ndim - 2,)), right=b.is_encrypt), is_encrypt=True)


def toTensor(x,
             com=None,
             dname=None,
             chunks="auto",
             name=None,
             lock=False,
             asarray=None,
             fancy=True,
             getitem=None,
             meta=None,
             inline_array=False,
             ) -> Tensor:
    if x is None:
        assert com and dname, print("while x is None, you should send com and dname")
        result = array_to_tensor(da.empty(shape=(0, 0))).from_other(com, dname=dname)
        result = array_to_tensor(da.asanyarray(result['data']), is_encrypt=result['is_encrypt'])
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
