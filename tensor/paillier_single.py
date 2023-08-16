from dask.array.core import Array
import numpy as np
import math
import sys
import random
from paillier_func import powmod, mulmod, invert, getprimeover


def generate_paillier_keypair(n_length=1024):
    """Return a new :class:`PaillierPublicKey` and :class:`PaillierPrivateKey`.

    Add the private key to *private_keyring* if given.

    Args:
      n_length: key size in bits.

    Returns:
      tuple: The generated :class:`PaillierPublicKey` and
      :class:`PaillierPrivateKey`
    """
    p = q = n = None
    n_len = 0
    while n_len != n_length:
        p = getprimeover(n_length // 2)
        q = p
        while q == p:
            q = getprimeover(n_length // 2)
        n = p * q
        n_len = n.bit_length()

    public_key = PaillierPublicKey(n)
    private_key = PaillierPrivateKey(n, p, q)

    return public_key, private_key


class PaillierPublicKey(object):
    """Contains a public key and associated encryption methods.

    Args:

      n (int): the modulus of the public key - see paaaa's paper.

    Attributes:
      g (int): part of the public key - see paaaa's paper.
      n (int): part of the public key - see paaaa's paper.
      nsquare (int): :attr:`n` ** 2, stored for frequent use.
      max_int (int): Maximum int that may safely be stored. This can be
        increased, if you are happy to redefine "safely" and lower the
        chance of detecting an integer overflow.
    """

    def __init__(self, n):
        self.g = n + 1
        self.n = n
        self.nsquare = n * n
        self.max_int = n // 3 - 1

    def __repr__(self):
        publicKeyHash = hex(hash(self))[2:]
        return "<PaillierPublicKey {}>".format(publicKeyHash[:10])

    def __eq__(self, other):
        return self.n == other

    def __hash__(self):
        return hash(self.n)


class PaillierPrivateKey(object):
    def __init__(self, n, p, q):
        if not p * q == n:
            raise ValueError('given public key does not match the given p and q.')
        if p == q:
            # check that p and q are different, otherwise we can't compute p^-1 mod q
            raise ValueError('p and q have to be different')
        self.n = n
        if q < p:  # ensure that p < q.
            self.p = q
            self.q = p
        else:
            self.p = p
            self.q = q

    def __repr__(self):
        pub_repr = repr(self.n)
        return "<PaillierPrivateKey for {}>".format(pub_repr)

    def __eq__(self, other):
        return self.p == other.p and self.q == other.q

    def __hash__(self):
        return hash((self.p, self.q))


class EncryptedNumber(object):
    def __init__(self, n, ciphertext, exponent):
        self.n = n
        self.nsquare = n * n
        self.max_int = n // 3 - 1
        self.__ciphertext = ciphertext
        self.exponent = exponent
        self.__is_obfuscated = False

    def __add__(self, other):
        """Add an int, float, `EncryptedNumber` or `EncodedNumber`."""
        if isinstance(other, EncryptedNumber):
            return self._add_encrypted(other)
        elif isinstance(other, EncodedNumber):
            return self._add_encoded(other)
        else:
            return self._add_scalar(other)

    def __radd__(self, other):
        """Called when Python evaluates `34 + <EncryptedNumber>`
        Required for builtin `sum` to work.
        """
        return self.__add__(other)

    def __mul__(self, other):
        """Multiply by an int, float, or EncodedNumber."""
        if isinstance(other, EncryptedNumber):
            raise NotImplementedError('Good luck with that...')

        if isinstance(other, EncodedNumber):
            encoding = other
        else:
            encoding = EncodedNumber.encode(self.n, other)
        product = self._raw_mul(encoding.encoding)
        exponent = self.exponent + encoding.exponent
        return EncryptedNumber(self.n, product, exponent)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self + (other * -1)

    def __rsub__(self, other):
        return other + (self * -1)

    def __truediv__(self, scalar):
        return self.__mul__(1 / scalar)

    def ciphertext(self, be_secure=True):
        if be_secure and not self.__is_obfuscated:
            self.obfuscate()
        return self.__ciphertext

    def decrease_exponent_to(self, new_exp):
        if np.any(new_exp > self.exponent):
            raise ValueError('New exponent %i should be more negative than '
                             'old exponent %i' % (new_exp, self.exponent))
        # new_base = np.power(EncodedNumber.BASE, self.exponent - new_exp)
        new_base = np.frompyfunc(lambda x: pow(EncodedNumber.BASE, x), 1, 1)(self.exponent - new_exp)
        multiplied = self * new_base
        multiplied.exponent = new_exp

        return multiplied

    def get_random_lt_n(self):
        """Return a cryptographically random number less than :attr:`n`"""
        return random.SystemRandom().randrange(1, self.n)

    def obfuscate(self):
        r = self.get_random_lt_n()
        r_pow_n = powmod(r, self.n, self.nsquare)
        self.__ciphertext = np.frompyfunc(mulmod, 3, 1)(self.__ciphertext, r_pow_n, self.nsquare)
        self.__is_obfuscated = True

    def _add_encrypted(self, other):
        """Returns E(a + b) given E(a) and E(b).

        Args:
          other (EncryptedNumber): an `EncryptedNumber` to add to self.

        Returns:
          EncryptedNumber: E(a + b), calculated by taking the product
            of E(a) and E(b) modulo :attr:`~PaillierPublicKey.n` ** 2.

        Raises:
          ValueError: if numbers were encrypted against different keys.
        """
        if self.n != other.n:
            raise ValueError("Attempted to add numbers encrypted against "
                             "different public keys!")

        # In order to add two numbers, their exponents must match.
        a, b = self, other
        if isinstance(a.exponent, Array):
            mins = np.minimum(a.exponent, b.exponent)

            if np.any(a.exponent != mins):
                a = self.decrease_exponent_to(mins)

            if np.any(b.exponent != mins):
                b = b.decrease_exponent_to(mins)

        else:
            if a.exponent > b.exponent:
                a = self.decrease_exponent_to(b.exponent)
            elif a.exponent < b.exponent:
                b = b.decrease_exponent_to(a.exponent)

        sum_ciphertext = a._raw_add(a.ciphertext(False), b.ciphertext(False))
        return EncryptedNumber(a.n, sum_ciphertext, a.exponent)

    def _add_scalar(self, scalar):
        encoded = EncodedNumber.encode(self.n, scalar,
                                       max_exponent=self.exponent)
        return self._add_encoded(encoded)

    def _add_encoded(self, encoded):
        """Returns E(a + b), given self=E(a) and b.

        Args:
          encoded (EncodedNumber): an :class:`EncodedNumber` to be added
            to `self`.

        Returns:
          EncryptedNumber: E(a + b), calculated by encrypting b and
            taking the product of E(a) and E(b) modulo
            :attr:`~PaillierPublicKey.n` ** 2.

        Raises:
          ValueError: if scalar is out of range or precision.
        """
        if self.n != encoded.n:
            raise ValueError("Attempted to add numbers encoded against "
                             "different public keys!")

        # In order to add two numbers, their exponents must match.
        a, b = self, encoded

        if isinstance(a.exponent, Array):
            mins = np.minimum(a.exponent, b.exponent)
            if np.any(a.exponent != mins):
                a = self.decrease_exponent_to(mins)

            if np.any(b.exponent != mins):
                b = b.decrease_exponent_to(mins)
        else:
            if a.exponent > b.exponent:
                a = self.decrease_exponent_to(b.exponent)
            elif a.exponent < b.exponent:
                b = b.decrease_exponent_to(a.exponent)
        # Don't bother to salt/obfuscate in a basic operation, do it
        # just before leaving the computer.
        encrypted_scalar = PaillierPublicKey(a.n).raw_encrypt(b.encoding, 1)
        sum_ciphertext = a._raw_add(a.ciphertext(False), encrypted_scalar)
        return EncryptedNumber(a.n, sum_ciphertext, a.exponent)

    def _raw_add(self, e_a, e_b):
        """Returns the integer E(a + b) given ints E(a) and E(b).

        N.B. this returns an int, not an `EncryptedNumber`, and ignores
        :attr:`ciphertext`

        Args:
          e_a (int): E(a), first term
          e_b (int): E(b), second term

        Returns:
          int: E(a + b), calculated by taking the product of E(a) and
            E(b) modulo :attr:`~PaillierPublicKey.n` ** 2.
        """
        return np.frompyfunc(lambda x, y: mulmod(x, y, self.nsquare), 2, 1)(e_a, e_b)

    def _raw_mul(self, plaintext):
        """Returns the integer E(a * plaintext), where E(a) = ciphertext

        Args:
          plaintext (int): number by which to multiply the
            `EncryptedNumber`. *plaintext* is typically an encoding.
            0 <= *plaintext* < :attr:`~PaillierPublicKey.n`

        Returns:
          int: Encryption of the product of `self` and the scalar
            encoded in *plaintext*.

        Raises:
          TypeError: if *plaintext* is not an int.
          ValueError: if *plaintext* is not between 0 and
            :attr:`PaillierPublicKey.n`.
        """
        if not (isinstance(plaintext, int) or isinstance(plaintext, np.ndarray)):
            raise TypeError('Expected ciphertext to be int, not %s' %
                            type(plaintext))

        if np.any(plaintext < 0) or np.any(plaintext >= self.n):
            raise ValueError('Scalar out of bounds: %i' % plaintext)
        if np.any(self.n - self.max_int <= plaintext):
            # Very large plaintext, play a sneaky trick using inverses
            neg_c = np.frompyfunc(invert, 2, 1)(self.ciphertext(False), self.nsquare)
            neg_scalar = self.n - plaintext
            return np.frompyfunc(powmod, 3, 1)(neg_c, neg_scalar, self.nsquare)
        else:
            return np.frompyfunc(powmod, 3, 1)(self.ciphertext(False), plaintext, self.nsquare)

    def toArray(self):
        return np.frompyfunc(EncryptedNumber, 3, 1)(self.n, self.__ciphertext, self.exponent)


class EncodedNumber(object):
    BASE = 16
    LOG2_BASE = math.log(BASE, 2)
    FLOAT_MANTISSA_BITS = sys.float_info.mant_dig

    def __init__(self, n, encoding: Array, exponent: Array):
        self.n = n
        self.max_int = n // 3 - 1
        self.encoding = encoding
        self.exponent = exponent

    @classmethod
    def encode(cls, n, scalar, precision=None, max_exponent=None):
        # Calculate the maximum exponent for desired precision
        if isinstance(scalar, int) or isinstance(scalar, float):
            scalar = np.array(scalar)

        if np.issubdtype(scalar.dtype, np.int16) or np.issubdtype(scalar.dtype, np.int32) \
                or np.issubdtype(scalar.dtype, np.int64):
            prec_exponent = np.zeros(scalar.shape, dtype=np.int32)

        elif np.issubdtype(scalar.dtype, np.float16) \
                or np.issubdtype(scalar.dtype, np.float32) or np.issubdtype(scalar.dtype, np.float64):
            bin_flt_exponent = np.frexp(scalar)[1]
            bin_lsb_exponent = bin_flt_exponent - cls.FLOAT_MANTISSA_BITS
            prec_exponent = np.floor(bin_lsb_exponent / cls.LOG2_BASE).astype(np.int32)

        elif np.issubdtype(scalar.dtype, object):
            prec_exponent = np.zeros(scalar.shape, dtype=np.int32)
        else:
            raise TypeError("Don't know the precision of type %s."
                            % type(scalar))

        if max_exponent is None:
            exponent = prec_exponent
        else:
            exponent = np.minimum(max_exponent, prec_exponent).min()

        def power(x, scalar, n):
            return int(scalar * pow(cls.BASE, x)) % n

        int_rep = np.frompyfunc(lambda x, y, z: power(x, y, z), 3, 1)(-exponent, scalar, n)
        return cls(n, int_rep, exponent)

    def get_mantissa(self, x, y, z):
        if x >= self.n:
            # Should be mod n
            raise ValueError('Attempted to decode corrupted number')
        elif x <= self.max_int:
            # Positive
            mantissa = x
        elif x >= self.n - self.max_int:
            # Negative
            mantissa = x - self.n
        else:
            raise OverflowError('Overflow detected in decrypted number')
        return mantissa * pow(y, int(z))

    def decode(self):
        """Decode plaintext and return the result.

        Returns:
          an int or float: the decoded number. N.B. if the number
            returned is an integer, it will not be of type float.

        Raises:
          OverflowError: if overflow is detected in the decrypted number.
        """
        mantissa = np.frompyfunc(self.get_mantissa, 3, 1)(self.encoding, self.BASE, self.exponent)
        return mantissa

    def decrease_exponent_to(self, new_exp):
        if np.any(new_exp > self.exponent):
            raise ValueError('New exponent should be more negative than'
                             'old exponent ')
        factor = np.power(self.BASE, self.exponent - new_exp)
        new_enc = self.encoding * factor % self.n
        return self.__class__(self.n, new_enc, new_exp)
