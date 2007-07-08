import scipy.special

maxnu = 10
n_zeros = 20

zeros = []
for n in range(0,maxnu+1):
    zeros.append(list(scipy.special.jn_zeros(n, n_zeros)))

outf = open("bessel_zeros.py", "w").write("bessel_zeros = %s" % zeros)
