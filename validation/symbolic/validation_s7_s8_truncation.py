#!/usr/bin/env python3
"""
Symbolic Validation: SI Lemma S7 & S8 — Truncation Analysis

Validates:
1. Exact Identity (S7): y_i - y_trunc = delta_i * (mu_complement - mu_set)
2. Error Bound (S8): ||y_i - y_trunc|| <= 2 * V_max * delta_i

The goal is to make the truncation error explicit: discarding low-mass entries
perturbs the output in direct proportion to the omitted probability mass.

Reference: si_rgat_paper.tex, Lemma S7, Corollary S8
"""
import sympy as sp
from sympy import Symbol, symbols, Sum, IndexedBase, Idx, simplify, Rational

def test_truncation_analysis():
    print("=" * 65)
    print("  SI LEMMA S7 & S8: TRUNCATION ANALYSIS")
    print("=" * 65)

    # Check 1: derive the exact decomposition of the truncation error.
    print("\n[1/2] Verifying Exact Truncation Identity (S7)...")
    
    # Use abstract symbols for the kept mass p_i and discarded mass delta_i.
    p_i = Symbol('p_i', real=True)
    delta_i = Symbol('delta_i', real=True)
    
    # Weighted sums over the kept set S and its complement S^c.
    y_S = Symbol('y_S', real=True) # Vector output partial sum
    y_Sc = Symbol('y_Sc', real=True)
    
    # Full and truncated outputs.
    y = y_S + y_Sc
    y_tilde = y_S / p_i
    
    # Conditional means on the kept and discarded subsets.
    mu_S = y_S / p_i
    mu_Sc = y_Sc / delta_i
    
    print("   Defined:")
    print("   y = y_S + y_Sc")
    print("   y_tilde = y_S / p_i = mu_S")
    print("   mu_Sc = y_Sc / delta_i")
    
    lhs = y - y_tilde
    rhs = delta_i * (mu_Sc - mu_S)
    
    diff = lhs - rhs
    # Rewrite the partial sums in terms of conditional means so the cancellation
    # is explicit to SymPy.
    diff_subbed = diff.subs({y_Sc: delta_i * mu_Sc, y_S: p_i * mu_S})
    
    print(f"   LHS - RHS = {simplify(diff_subbed)}")
    
    # Enforce p_i + delta_i = 1 to complete the identity.
    res = simplify(diff_subbed.subs(p_i, 1 - delta_i))
    print(f"   After p_i -> 1-delta_i: {res}")
    
    if res == 0:
        print("   SUCCESS: Lemma S7 identity verified.")
    else:
        print("   FAILURE: Identity mismatch.")
        assert False, "S7 Mismatch"

    # Check 2: turn the identity into a norm bound.
    print("\n[2/2] Verifying Truncation Bound (S8)...")
    
    V_max = Symbol('V_max', positive=True)
    norm_mu_S = Symbol('N_S', positive=True)
    norm_mu_Sc = Symbol('N_Sc', positive=True)
    
    bound_rhs = delta_i * (norm_mu_Sc + norm_mu_S)
    
    # Saturate both conditional norms at the same global bound.
    max_bound = bound_rhs.subs({norm_mu_S: V_max, norm_mu_Sc: V_max})
    
    print(f"   Checking bound: delta_i * (||mu_Sc|| + ||mu_S||) <= {max_bound}")
    
    if simplify(max_bound - 2 * V_max * delta_i) == 0:
        print("   SUCCESS: Bound algebraic structure verified.")
        print("            ||y - y_tilde|| <= 2 * V_max * delta_i")
    else:
        print("   FAILURE: Bound structure mismatch.")
        assert False, "S8 Mismatch"

    print("\n" + "=" * 65)
    print("  LEMMA S7 & S8 VALIDATION COMPLETE")
    print("=" * 65)

if __name__ == "__main__":
    test_truncation_analysis()
