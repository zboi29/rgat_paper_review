#!/usr/bin/env python3
"""
Symbolic Validation: SI Lemma S2 — Small-angle Distance Expansion

Validates that for small generators u, v with norm <= ε:
    d_geo(exp(u), exp(v))^2 = 4||u - v||^2 + O(ε^4)

This is the local flatness statement used throughout the paper: at sufficiently
small angles, quaternion geometry agrees with the flat quadratic form up to
fourth-order error.

Reference: si_rgat_paper.tex, Lemma S2
"""
import sympy as sp
from sympy import Symbol, symbols, exp, Rational, simplify, cos, sin, acos, sqrt, series, Matrix

def test_small_angle_expansion():
    print("=" * 65)
    print("  SI LEMMA S2: SMALL-ANGLE DISTANCE EXPANSION")
    print("=" * 65)

    # Introduce a scalar t so the small-angle regime becomes a Taylor expansion
    # around t = 0 in the Lie algebra coordinates.
    t = Symbol('t', real=True) # Scaling parameter (represents epsilon)
    
    # Generator coordinates in the Lie algebra.
    u1, u2, u3 = symbols('u1 u2 u3', real=True)
    v1, v2, v3 = symbols('v1 v2 v3', real=True)
    
    # Leading Euclidean term predicted by the lemma.
    euclidean_dist_sq = (u1 - v1)**2 + (u2 - v2)**2 + (u3 - v3)**2
    
    print("\n[1/3] Defining Rotor Exponentials...")

    # Quaternion multiplication helper kept for completeness of the rotor model.
    def quat_mul(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return (
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        )

    # Exponential map for a pure-vector quaternion.
    def quat_exp(vx, vy, vz, scale):
        # Apply the small-angle scale inside the exponential.
        vx, vy, vz = vx*scale, vy*scale, vz*scale
        theta_sq = vx**2 + vy**2 + vz**2
        theta = sqrt(theta_sq)
        
        w = cos(theta)
        s = sin(theta) / theta # sinc(theta)
        
        # SymPy handles the theta -> 0 limit when the series is expanded.
        return (w, s*vx, s*vy, s*vz)

    # Rotor-valued curves q(t) = exp(tu), k(t) = exp(tv).
    q = quat_exp(u1, u2, u3, t)
    k = quat_exp(v1, v2, v3, t)

    print("   q(t) = exp(t*u)")
    print("   k(t) = exp(t*v)")

    print("\n[2/3] Computing Geodesic Distance Expansion...")
    
    # The geodesic distance depends on the quaternion inner product.
    inner = sum(a*b for a, b in zip(q, k))
    
    # Near t = 0 the inner product stays close to 1, so the absolute value does
    # not affect the local expansion.
    s_series = series(inner, t, 0, 5).removeO()
    print(f"   <q, k> approx = {s_series}")
    
    # Expand the squared distance directly. Squaring avoids the square-root
    # singularity in arccos(...) at x = 1.
    d_geo_sq_expr = 4 * acos(s_series)**2
    d_geo_expansion = series(d_geo_sq_expr, t, 0, 5).removeO()
    
    print(f"   d_geo^2 expansion = {d_geo_expansion}")

    print("\n[3/3] Verifying S2 Identity...")
    
    # Flat reference model after scaling by t.
    target_euclidean = 4 * t**2 * euclidean_dist_sq
    
    diff = simplify(d_geo_expansion - target_euclidean)
    print(f"   Difference (d_geo^2 - 4||u-v||^2) = {diff}")
    
    # An O(t^4) remainder means every coefficient below degree 4 vanishes.
    coeffs = [diff.coeff(t, i) for i in range(4)]
    
    if all(simplify(c) == 0 for c in coeffs):
        print("   SUCCESS: Leading terms match. Error is O(t^4).")
        print("   d_geo(q, k)^2 = 4||u-v||^2 + O(ε^4) confirmed.")
    else:
        print("   FAILURE: Lower order terms found in difference.")
        print(f"   Coeffs [0-3]: {coeffs}")
        assert False, "Expansion mismatch"

    print("\n" + "=" * 65)
    print("  LEMMA S2 VALIDATION COMPLETE")
    print("=" * 65)

if __name__ == "__main__":
    test_small_angle_expansion()
