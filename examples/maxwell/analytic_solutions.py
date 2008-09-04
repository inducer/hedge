# Hedge - the Hybrid'n'Easy DG Environment
# Copyright (C) 2007 Andreas Kloeckner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.




from __future__ import division
from hedge.tools import \
        cyl_bessel_j, \
        cyl_bessel_j_prime
from math import sqrt, pi, sin, cos, atan2, acos
import numpy
import numpy.linalg as la
import cmath




# solution adapters -----------------------------------------------------------
class RealPartAdapter:
    def __init__(self, adaptee):
        self.adaptee = adaptee

    @property
    def shape(self):
        return self.adaptee.shape

    def __call__(self, x, el):
        return [xi.real for xi in self.adaptee(x, el)]

class SplitComplexAdapter:
    def __init__(self, adaptee):
        self.adaptee = adaptee

    @property
    def shape(self):
        (n,) = self.adaptee.shape
        return (n*2,)

    def __call__(self, x, el):
        ad_x = self.adaptee(x, el)
        return [xi.real for xi in ad_x] + [xi.imag for xi in ad_x]




class CylindricalFieldAdapter:
    """Turns a field returned as (r, phi, z) components into
    (x,y,z) components.
    """
    def __init__(self, adaptee):
        self.adaptee = adaptee

    @property
    def shape(self):
        return self.adaptee.shape

    def __call__(self, x, el):
        xy = x[:2]
        r = la.norm(xy)
        phi = atan2(x[1], x[0])

        prev_result = self.adaptee(x, el)
        result = []
        i = 0
        while i < len(prev_result):
            fr, fphi, fz = prev_result[i:i+3]
            result.extend([
                    cos(phi)*fr - sin(phi)*fphi, # ex
                    sin(phi)*fr + cos(phi)*fphi, # ey
                    fz,
                    ])
            i += 3

        return result




class SphericalFieldAdapter:
    """Turns the adaptee's field C{(Er, Etheta, Ephi)} components into
    C{(Ex,Ey,Ez)} components.

    Conventions are those of
    http://mathworld.wolfram.com/SphericalCoordinates.html
    """
    def __init__(self, adaptee):
        self.adaptee = adaptee

    @property
    def shape(self):
        return self.adaptee.shape

    @staticmethod
    def r_theta_phi_from_xyz(x):
        r = la.norm(x)
        return r, atan2(x[1], x[0]), acos(x[2]/r)

    def __call__(self, x, el):
        r, theta, phi = self.r_theta_phi_from_xyz(x)

        er = numpy.array([
            cos(theta)*sin(phi),
            sin(theta)*sin(phi),
            cos(phi)])
        etheta = numpy.array([
            -sin(theta),
            cos(theta),
            0])
        ephi = numpy.array([
            cos(theta)*cos(phi),
            sin(theta)*cos(phi),
            -sin(phi)])

        adaptee_result = self.adaptee(x, el)
        result = numpy.empty_like(adaptee_result)
        i = 0
        while i < len(adaptee_result):
            fr, ftheta, fphi = adaptee_result[i:i+3]
            result[i:i+3] = fr*er + ftheta*etheta + fphi*ephi
            i += 3

        return result




# actual solutions ------------------------------------------------------------
class CylindricalCavityMode:
    """A cylindrical TM cavity mode.

    Taken from:
    J.D. Jackson, Classical Electrodynamics, Wiley.
    3rd edition, 2001.
    ch 8.7, p. 368f.
    """
    
    def __init__(self,  m, n, p, radius, height, epsilon, mu):
        try:
            from bessel_zeros import bessel_zeros
        except ImportError:
            print "*** You need to generate the bessel root data file."
            print "*** Execute generate-bessel-zeros.py at the command line."
            raise

        assert m >= 0 and m == int(m)
        assert n >= 1 and n == int(n)
        assert p >= 0 and p == int(p)
        self.m = m
        self.n = n
        self.p = p
        self.phi_sign = 1

        R = self.radius = radius
        d = self.height = height

        self.epsilon = epsilon
        self.mu = mu

        self.t = 0

        x_mn = bessel_zeros[m][n-1]

        self.omega = 1 / sqrt(mu*epsilon) * sqrt(
                x_mn**2 / R**2
                + p**2 * pi**2 / d**2)

        self.gamma_mn = x_mn/R

    def set_time(self, t):
        self.t = t

    shape = (6,)

    def __call__(self, x, el):
        # coordinates -----------------------------------------------------
        xy = x[:2]
        r = sqrt(xy*xy)
        phi = atan2(x[1], x[0])
        z = x[2]

        # copy instance variables for easier access -----------------------
        m = self.m
        p = self.p
        gamma = self.gamma_mn
        phi_sign = self.phi_sign
        omega = self.omega
        d = self.height
        epsilon = self.epsilon

        # common subexpressions -------------------------------------------
        tdep = cmath.exp(-1j * omega * self.t)
        phi_factor = cmath.exp(phi_sign * 1j * m * phi)

        # psi and derivatives ---------------------------------------------
        psi = cyl_bessel_j(m, gamma * r) * phi_factor
        psi_dr = gamma*cyl_bessel_j_prime(m, gamma*r) * phi_factor
        psi_dphi = (cyl_bessel_j(m, gamma * r) 
                * 1/r * phi_sign*1j*m * phi_factor)

        # field components in polar coordinates ---------------------------
        ez   = tdep * cos(p * pi * z / d) * psi

        e_transverse_factor = (tdep
                * (-p*pi/(d*gamma**2))
                * sin(p * pi * z / d))

        er   = e_transverse_factor * psi_dr
        ephi = e_transverse_factor * psi_dphi

        hz   = 0j

        # z x grad psi = z x (psi_x, psi_y)   = (-psi_y,   psi_x)
        # z x grad psi = z x (psi_r, psi_phi) = (-psi_phi, psi_r)
        h_transverse_factor = (tdep
                * 1j*epsilon*omega/gamma**2
                * cos(p * pi * z / d))

        hr   = h_transverse_factor * (-psi_dphi)
        hphi = h_transverse_factor * psi_dr

        return [er, ephi, ez, hr, hphi, hz]




class RectangularWaveguideMode:
    """A rectangular TM cavity mode."""
    
    def __init__(self, epsilon, mu, mode_indices, 
            dimensions=(1,1,1), coefficients=(1,0,0),
            forward_coeff=1, backward_coeff=0):
        for n in mode_indices:
            assert n >= 0 and n == int(n)
        self.mode_indices = mode_indices
        self.dimensions = dimensions
        self.coefficients = coefficients
        self.forward_coeff = forward_coeff
        self.backward_coeff = backward_coeff

        self.epsilon = epsilon
        self.mu = mu

        self.t = 0

        self.factors = [n*pi/a for n,  a in zip(self.mode_indices, self.dimensions)]

        c = 1/sqrt(mu*epsilon)
        self.k = sqrt(sum(f**2 for f in self.factors))
        self.omega = self.k*c

    def set_time(self, t):
        self.t = t

    def __call__(self, discr):
        f,g,h = self.factors
        omega = self.omega
        k = self.k

        x = discr.nodes[:,0]
        y = discr.nodes[:,1]
        z = discr.nodes[:,2]

        sx = numpy.sin(f*x)
        sy = numpy.sin(g*y)
        cx = numpy.cos(f*x)
        cy = numpy.cos(g*y)

        zdep_add = numpy.exp(1j*h*z)+numpy.exp(-1j*h*z)
        zdep_sub = numpy.exp(1j*h*z)-numpy.exp(-1j*h*z)

        tdep = numpy.exp(-1j * omega * self.t)

        C = 1j/(f**2+g**2)

        result = numpy.zeros((6,len(x)), dtype=zdep_add.dtype)

        result[0] = C*f*h*cx*sy*zdep_sub*tdep # ex
        result[1] = C*g*h*sx*cy*zdep_sub*tdep # ey
        result[2] = sx*sy*zdep_add*tdep # ez
        result[3] = -C*g*self.epsilon*omega*sx*cy*zdep_add*tdep # hx
        result[4] = C*f*self.epsilon*omega*cx*sy*zdep_add*tdep

        return result




class RectangularCavityMode(RectangularWaveguideMode):
    """A rectangular TM cavity mode."""
    
    def __init__(self, *args, **kwargs):
        if "scale" in kwargs:
            kwargs["forward_coeff"] = scale
            kwargs["backward_coeff"] = scale
        else:
            kwargs["forward_coeff"] = 1
            kwargs["backward_coeff"] = 1
        RectangularWaveguideMode.__init__(self, *args, **kwargs)





# dipole solution -------------------------------------------------------------
class DipoleFarField:
    """Dipole EM field.

    See U{http://piki.tiker.net/wiki/Dipole}.
    """
    def __init__(self, q, d, omega, epsilon, mu):
        self.q = q
        self.d = d

        self.epsilon = epsilon
        self.mu = mu
        self.c = c = 1/sqrt(mu*epsilon)

        self.omega = omega

        freq = omega/(2*pi)
        self.wavelength = c/freq

    def far_field_mask(self, x, el):
        if la.norm(x) > 0.5*self.wavelength:
            return 1
        else:
            return 0

    def set_time(self, t):
        self.t = t

    shape = (6,)

    def __call__(self, x, el):
        # coordinates ---------------------------------------------------------
        r, theta, phi = SphericalFieldAdapter.r_theta_phi_from_xyz(x)

        # copy instance variables for easier access ---------------------------
        q = self.q
        d = self.d

        epsilon = self.epsilon
        mu = self.mu
        c = self.c

        omega = self.omega
        t = self.t

        # field components ----------------------------------------------------
        tdep = omega*t-omega*r/c
        e_r  = (2*cos(phi)*q*d) / (4*pi*epsilon) * (
                1/r**3 * sin(tdep)
                +omega/(c*r**2) * cos(tdep))
        e_phi = (sin(phi)*q*d) / (4*pi*epsilon) * (
                (1/r**3 - omega**2/(c**2*r)*sin(tdep))
                + omega/(c*r**2)*cos(tdep))
        h_theta = (omega*sin(phi)*q*d)/(4*pi) * (
                - omega/(c*r)*sin(tdep)
                + 1/r**2*cos(tdep))

        return [e_r, 0, e_phi, 0, h_theta, 0]

    def source_modulation(self, t):
        j0 = self.q*self.d*self.omega/self.epsilon
        return j0*cos(self.omega*t)




# analytic solution tools -----------------------------------------------------
def check_time_harmonic_solution(discr, mode, c_sol):
    from hedge.discretization import bind_nabla, bind_mass_matrix
    from hedge.visualization import SiloVisualizer
    from hedge.silo import SiloFile
    from hedge.tools import dot, cross
    from hedge.silo import DB_VARTYPE_VECTOR

    def curl(field):
        return cross(nabla, field)

    vis = SiloVisualizer(discr)

    nabla = bind_nabla(discr)
    mass = bind_mass_matrix(discr)

    def rel_l2_error(err, base):
        def l2_norm(field):
            return sqrt(dot(field, mass*field))

        base_l2 = l2_norm(base)
        err_l2 = l2_norm(err)
        if base_l2 == 0:
            if err_l2 == 0:
                return 0.
            else:
                return float("inf")
        else:
            return err_l2/base_l2

    dt = 0.1

    for step in range(10):
        t = step*dt
        mode.set_time(t)
        fields = discr.interpolate_volume_function(c_sol)

        er = fields[0:3]
        hr = fields[3:6]
        ei = fields[6:9]
        hi = fields[9:12]

        silo = SiloFile("em-complex-%04d.silo" % step)
        vis.add_to_silo(silo,
                vectors=[
                    ("curl_er", curl(er)), 
                    ("om_hi", -mode.mu*mode.omega*hi), 
                    ("curl_hr", curl(hr)), 
                    ("om_ei", mode.epsilon*mode.omega*hi), 
                    ],
                expressions=[
                ("diff_er", "curl_er-om_hi", DB_VARTYPE_VECTOR),
                ("diff_hr", "curl_hr-om_ei", DB_VARTYPE_VECTOR),
                ],
                write_coarse_mesh=True,
                time=t, step=step
                )

        er_res = curl(er) + mode.mu     *mode.omega*hi
        ei_res = curl(ei) - mode.mu     *mode.omega*hr
        hr_res = curl(hr) - mode.epsilon*mode.omega*ei
        hi_res = curl(hi) + mode.epsilon*mode.omega*er

        print "time=%f, rel l2 residual in Re[E]=%g\tIm[E]=%g\tRe[H]=%g\tIm[H]=%g" % (
                t,
                rel_l2_error(er_res, er),
                rel_l2_error(ei_res, ei),
                rel_l2_error(hr_res, hr),
                rel_l2_error(hi_res, hi),
                )

