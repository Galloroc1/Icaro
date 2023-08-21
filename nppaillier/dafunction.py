from nppaillier.function import  get_nude,powmod,mulmod,get_random_lt_n
import dask.array as da


def raw_encrypt(n, nsquare, max_int, encoding, r_value=None):
    nude_ciphertext = da.frompyfunc(get_nude, 4, 1)(n, nsquare, max_int, encoding)
    r = r_value or get_random_lt_n()
    obfuscator = powmod(r, n, nsquare)
    return da.frompyfunc(mulmod, 3, 1)(nude_ciphertext, obfuscator, nsquare)


def raw_add(e_a, e_b, nsquare):
    return da.frompyfunc(lambda x, y: mulmod(x, y, nsquare), 2, 1)(e_a, e_b)