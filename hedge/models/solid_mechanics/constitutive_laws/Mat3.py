# -*- coding: utf8 -*-
"""Utility code for 3x3 matrix operations"""

from __future__ import division

__copyright__ = "Copyright (C) 2013 Andreas Klockner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from pytools import memoize_method
def copyMat2ToMat3(m2):
    m3 = [1,0,0,0,1,0,0,0,1]
    m3[0] = m2[0]
    m3[1] = m2[1]
    m3[3] = m2[2]
    m3[4] = m2[3]
    return m3

def copyMat3(A):
    B = [0]*9
    try:
        for i in range(9):
            B[i] = A[i]
    except IndexError:
        for i in range(9):
            B[i] = A[i%3][i/3]
    return B

def det(A):
    return A[0]*(A[4]*A[8]-A[5]*A[7]) \
          -A[1]*(A[3]*A[8]-A[5]*A[6]) \
          +A[2]*(A[3]*A[7]-A[4]*A[6])

def inv(A, detA=None):
    if detA is not None:
        detA = det(A)
    
    detinv = 1.0/detA

    Ainv = [0]*9

    Ainv[0] = detinv*( A[4]*A[8]-A[5]*A[7])
    Ainv[1] = detinv*(-A[1]*A[8]+A[2]*A[7])
    Ainv[2] = detinv*( A[1]*A[5]-A[2]*A[4])
    Ainv[3] = detinv*(-A[3]*A[8]+A[5]*A[6])
    Ainv[4] = detinv*( A[0]*A[8]-A[2]*A[6])
    Ainv[5] = detinv*(-A[0]*A[5]+A[2]*A[3])
    Ainv[6] = detinv*( A[3]*A[7]-A[4]*A[6])
    Ainv[7] = detinv*(-A[0]*A[7]+A[1]*A[6])
    Ainv[8] = detinv*( A[0]*A[4]-A[1]*A[3])

    return Ainv

def trace(A):
    return A[0] + A[4] + A[8]

def scalarMult(c, A):
    B = [0]*9
    for i in range(9):
        B[i] = c * A[i]
    return B

def add(A, B):
    C = [0]*9
    for i in range(9):
        C[i] = A[i] + B[i]
    return C

def scaleMat(c):
    A = [1,0,0, \
         0,1,0, \
         0,0,1]
    for i in range(9):
        A[i] = c*A[i]
    return A

# A and B not transposed, returns AB
def mulss(A, B):
    C = [0]*9
    C[0] = A[0]*B[0] + A[1]*B[3] + A[2]*B[6]
    C[1] = A[0]*B[1] + A[1]*B[4] + A[2]*B[7]
    C[2] = A[0]*B[2] + A[1]*B[5] + A[2]*B[8]
    C[3] = A[3]*B[0] + A[4]*B[3] + A[5]*B[6]
    C[4] = A[3]*B[1] + A[4]*B[4] + A[5]*B[7]
    C[5] = A[3]*B[2] + A[4]*B[5] + A[5]*B[8]
    C[6] = A[6]*B[0] + A[7]*B[3] + A[8]*B[6]
    C[7] = A[6]*B[1] + A[7]*B[4] + A[8]*B[7]
    C[8] = A[6]*B[2] + A[7]*B[5] + A[8]*B[8]
    return C

# A is transposed, B is not, returns A'B
def mults(A, B):
    C = [0]*9
    C[0] = A[0]*B[0] + A[3]*B[3] + A[6]*B[6]
    C[1] = A[0]*B[1] + A[3]*B[4] + A[6]*B[7]
    C[2] = A[0]*B[2] + A[3]*B[5] + A[6]*B[8]
    C[3] = A[1]*B[0] + A[4]*B[3] + A[7]*B[6]
    C[4] = A[1]*B[1] + A[4]*B[4] + A[7]*B[7]
    C[5] = A[1]*B[2] + A[4]*B[5] + A[7]*B[8]
    C[6] = A[2]*B[0] + A[5]*B[3] + A[8]*B[6]
    C[7] = A[2]*B[1] + A[5]*B[4] + A[8]*B[7]
    C[8] = A[2]*B[2] + A[5]*B[5] + A[8]*B[8]
    return C

def printMat(A):
    for i in range(3):
        str = ""
        for j in range(3):
            str += "{0} ".format(A[3*i+j])
        print(str)

