/*
 * QEMU float support
 *
 * The code in this source file is derived from release 2a of the SoftFloat
 * IEC/IEEE Floating-point Arithmetic Package. Those parts of the code (and
 * some later contributions) are provided under that license, as detailed below.
 * It has subsequently been modified by contributors to the QEMU Project,
 * so some portions are provided under:
 *  the SoftFloat-2a license
 *  the BSD license
 *  GPL-v2-or-later
 *
 * Any future contributions to this file after December 1st 2014 will be
 * taken to be licensed under the Softfloat-2a license unless specifically
 * indicated otherwise.
 */

#include "softfloat-parts-arith.c.inc"

/*
 * Canonicalize the FloatParts structure.  Determine the class,
 * unbias the exponent, and normalize the fraction.
 */
static void partsN(canonicalize)(FloatPartsN *p, float_status *status,
                                 const FloatFmt *fmt)
{
    /*
     * It's target-dependent how to handle the case of exponent 0
     * and Integer bit set. Intel calls these "pseudodenormals",
     * and treats them as if the integer bit was 0, and never
     * produces them on output. This is the default behaviour for QEMU.
     * For m68k, the integer bit is considered validly part of the
     * input value when the exponent is 0, and may be 0 or 1,
     * giving extra range. They may also be generated as outputs.
     * (The m68k manual actually calls these values part of the
     * normalized number range, not the denormalized number range,
     * but that distinction is not important for us, because
     * m68k doesn't care about the input_denormal_used status flag.)
     * floatx80_pseudo_denormal_valid selects the m68k behaviour,
     * which changes both how we canonicalize such a value and
     * how we uncanonicalize results.
     */
    bool has_pseudo_denormals = fmt->has_explicit_bit &&
        (get_floatx80_behaviour(status) & floatx80_pseudo_denormal_valid);

    if (unlikely(p->exp == 0)) {
        if (likely(fracN(eqz)(p))) {
            p->cls = float_class_zero;
        } else if (get_flush_inputs_to_zero(status)) {
            float_raise(float_flag_input_denormal_flushed, status);
            p->cls = float_class_zero;
            fracN(clear)(p);
        } else {
            int shift = fracN(normalize)(p);
            p->cls = float_class_denormal;
            p->exp = fmt->frac_shift - fmt->exp_bias
                   - shift + !has_pseudo_denormals;
        }
        return;
    }
    if (unlikely(p->exp == fmt->exp_max)) {
        switch (fmt->exp_max_kind) {
        case float_expmax_ieee:
            if (likely(fracN(eqz)(p))) {
                p->cls = float_class_inf;
            } else {
                fracN(shl)(p, fmt->frac_shift);
                p->cls = (parts_is_snan_frac(p->frac_hi, status)
                          ? float_class_snan : float_class_qnan);
            }
            return;
        case float_expmax_normal:
            break;
        case float_expmax_e4m3:
            if (p->frac_hi == 0b111) {
                fracN(shl)(p, fmt->frac_shift);
                p->cls = (get_float_e4m3_nan_is_snan(status)
                          ? float_class_snan : float_class_qnan);
                return;
            }
            /* otherwise normal */
            break;
        default:
            g_assert_not_reached();
        }
    }

    p->cls = float_class_normal;
    p->exp -= fmt->exp_bias;
    fracN(shl)(p, fmt->frac_shift);
    p->frac_hi |= DECOMPOSED_IMPLICIT_BIT;
}

/*
 * Round and uncanonicalize a floating-point number by parts. There
 * are FRAC_SHIFT bits that may require rounding at the bottom of the
 * fraction; these bits will be removed. The exponent will be biased
 * by EXP_BIAS and must be bounded by [EXP_MAX-1, 0].
 *
 * For formats with has_saturation, saturate selects either max normal or
 * the format's non-saturating overflow value.
 */

/* Helper for uncanon_normal and uncanon, for FP8 E4M3. */
static void partsN(uncanon_e4m3_overflow)(FloatPartsN *p, float_status *s,
                                          const FloatFmt *fmt, bool saturate)
{
    assert(N == 64);
    p->exp = fmt->exp_max;
    if (saturate) {
        p->frac_hi = E4M3_NORMAL_FRAC_MAX;
    } else {
        /*
         * The class isn't actually used after this point in uncanon,
         * but for clarity while debugging, don't leave it set to normal.
         */
        p->cls = float_class_qnan;
        p->frac_hi = E4M3_NAN_FRAC;
    }
}

static void partsN(uncanon_normal)(FloatPartsN *p, float_status *s,
                                   const FloatFmt *fmt, bool saturate)
{
    const int exp_max = fmt->exp_max;
    const int frac_shift = fmt->frac_shift;
    const uint64_t round_mask = fmt->round_mask;
    const uint64_t frac_lsb = round_mask + 1;
    const uint64_t frac_lsbm1 = round_mask ^ (round_mask >> 1);
    const uint64_t roundeven_mask = round_mask | frac_lsb;
    uint64_t inc;
    bool overflow_norm = saturate;
    int exp;
    FloatExceptionFlags flags = 0;

    switch (get_float_rounding_mode(s)) {
    case float_round_nearest_even_max:
        overflow_norm = true;
        /* fall through */
    case float_round_nearest_even:
        if (N > 64 && frac_lsb == 0) {
            inc = ((p->frac_hi & 1) || (p->frac_lo & round_mask) != frac_lsbm1
                   ? frac_lsbm1 : 0);
        } else {
            inc = ((p->frac_lo & roundeven_mask) != frac_lsbm1
                   ? frac_lsbm1 : 0);
        }
        break;
    case float_round_ties_away:
        inc = frac_lsbm1;
        break;
    case float_round_to_zero:
        overflow_norm = true;
        inc = 0;
        break;
    case float_round_up:
        inc = p->sign ? 0 : round_mask;
        overflow_norm |= p->sign;
        break;
    case float_round_down:
        inc = p->sign ? round_mask : 0;
        overflow_norm |= !p->sign;
        break;
    case float_round_to_odd:
        overflow_norm = true;
        /* fall through */
    case float_round_to_odd_inf:
        if (N > 64 && frac_lsb == 0) {
            inc = p->frac_hi & 1 ? 0 : round_mask;
        } else {
            inc = p->frac_lo & frac_lsb ? 0 : round_mask;
        }
        break;
    default:
        g_assert_not_reached();
    }

    if (fmt->has_saturation) {
        overflow_norm = saturate;
    }

    exp = p->exp + fmt->exp_bias;
    if (likely(exp > 0)) {
        if (p->frac_lo & round_mask) {
            flags |= float_flag_inexact;
            if (fracN(addi)(p, p, inc)) {
                fracN(shr)(p, 1);
                p->frac_hi |= DECOMPOSED_IMPLICIT_BIT;
                exp++;
            }
            p->frac_lo &= ~round_mask;
        }

        if (unlikely(exp >= exp_max)) {
            switch (fmt->exp_max_kind) {
            case float_expmax_ieee:
                flags |= float_flag_overflow;
                if (get_float_rebias_overflow(s)) {
                    exp -= fmt->exp_re_bias;
                } else if (overflow_norm) {
                    flags |= float_flag_inexact;
                    exp = exp_max - 1;
                    fracN(allones)(p);
                    p->frac_lo &= ~round_mask;
                } else {
                    flags |= float_flag_inexact;
                    p->cls = float_class_inf;
                    exp = exp_max;
                    fracN(clear)(p);
                }
                break;

            case float_expmax_normal:
                if (unlikely(exp > exp_max)) {
                    /* Overflow.  Return the maximum normal.  */
                    flags = (fmt->overflow_raises_invalid
                             ? float_flag_invalid
                             : float_flag_overflow | float_flag_inexact);
                    exp = exp_max;
                    fracN(allones)(p);
                    p->frac_lo &= ~round_mask;
                }
                break;

            case float_expmax_e4m3:
                if (exp > exp_max || p->frac_hi > E4M3_NORMAL_FRAC_MAX) {
                    partsN(uncanon_e4m3_overflow)(p, s, fmt, overflow_norm);
                    exp = p->exp;
                    flags |= (float_flag_overflow | float_flag_inexact);
                }
                break;

            default:
                g_assert_not_reached();
            }
        }
        fracN(shr)(p, frac_shift);
    } else if (unlikely(get_float_rebias_underflow(s))) {
        flags |= float_flag_underflow;
        exp += fmt->exp_re_bias;
        if (p->frac_lo & round_mask) {
            flags |= float_flag_inexact;
            if (fracN(addi)(p, p, inc)) {
                fracN(shr)(p, 1);
                p->frac_hi |= DECOMPOSED_IMPLICIT_BIT;
                exp++;
            }
            p->frac_lo &= ~round_mask;
        }
        fracN(shr)(p, frac_shift);
    } else if (get_flush_to_zero(s) && get_ftz_before_rounding(s)) {
        flags |= float_flag_output_denormal_flushed;
        p->cls = float_class_zero;
        exp = 0;
        fracN(clear)(p);
    } else {
        bool is_tiny = get_tininess_before_rounding(s) || exp < 0;
        bool has_pseudo_denormals = fmt->has_explicit_bit &&
            (get_floatx80_behaviour(s) & floatx80_pseudo_denormal_valid);

        if (!is_tiny) {
            FloatPartsN discard;
            is_tiny = !fracN(addi)(&discard, p, inc);
        }

        fracN(shrjam)(p, !has_pseudo_denormals - exp);

        if (p->frac_lo & round_mask) {
            /* Need to recompute round-to-even/round-to-odd. */
            switch (get_float_rounding_mode(s)) {
            case float_round_nearest_even:
            case float_round_nearest_even_max:
                if (N > 64 && frac_lsb == 0) {
                    inc = ((p->frac_hi & 1) ||
                           (p->frac_lo & round_mask) != frac_lsbm1
                           ? frac_lsbm1 : 0);
                } else {
                    inc = ((p->frac_lo & roundeven_mask) != frac_lsbm1
                           ? frac_lsbm1 : 0);
                }
                break;
            case float_round_to_odd:
            case float_round_to_odd_inf:
                if (N > 64 && frac_lsb == 0) {
                    inc = p->frac_hi & 1 ? 0 : round_mask;
                } else {
                    inc = p->frac_lo & frac_lsb ? 0 : round_mask;
                }
                break;
            default:
                break;
            }
            flags |= float_flag_inexact;
            fracN(addi)(p, p, inc);
            p->frac_lo &= ~round_mask;
        }

        exp = (p->frac_hi & DECOMPOSED_IMPLICIT_BIT) && !has_pseudo_denormals;
        fracN(shr)(p, frac_shift);

        if (is_tiny) {
            if (get_flush_to_zero(s)) {
                assert(!get_ftz_before_rounding(s));
                flags |= float_flag_output_denormal_flushed;
                p->cls = float_class_zero;
                exp = 0;
                fracN(clear)(p);
            } else if (flags & float_flag_inexact) {
                flags |= float_flag_underflow;
            }
            if (exp == 0 && fracN(eqz)(p)) {
                p->cls = float_class_zero;
            }
        }
    }
    p->exp = exp;
    float_raise(flags, s);
}

static void partsN(uncanon)(FloatPartsN *p, float_status *s,
                            const FloatFmt *fmt, bool saturate)
{
    if (likely(is_anynorm(p->cls))) {
        partsN(uncanon_normal)(p, s, fmt, saturate);
    } else {
        switch (p->cls) {
        case float_class_zero:
            p->exp = 0;
            fracN(clear)(p);
            return;
        case float_class_inf:
            switch (fmt->exp_max_kind) {
            case float_expmax_ieee:
                p->exp = fmt->exp_max;
                fracN(clear)(p);
                break;
            case float_expmax_e4m3:
                partsN(uncanon_e4m3_overflow)(p, s, fmt, saturate);
                fracN(shr)(p, fmt->frac_shift);
                break;
            case float_expmax_normal:
            default:
                g_assert_not_reached();
            }
            return;
        case float_class_qnan:
        case float_class_snan:
            p->exp = fmt->exp_max;
            switch (fmt->exp_max_kind) {
            case float_expmax_e4m3:
                /*
                 * There is only one NaN encoding for E4M3, and with a
                 * conversion from another format, the input NaN fraction
                 * may not apply.
                 */
                assert(N == 64);
                p->frac_hi = E4M3_NAN_FRAC;
                /* fall through */
            case float_expmax_ieee:
                fracN(shr)(p, fmt->frac_shift);
                break;
            case float_expmax_normal:
            default:
                g_assert_not_reached();
            }
            return;
        default:
            break;
        }
        g_assert_not_reached();
    }
}

/*
 * Returns the result of dividing the floating-point value `a' by the
 * corresponding value `b'. The operation is performed according to
 * the IEC/IEEE Standard for Binary Floating-Point Arithmetic.
 */
FloatPartsN partsN(div)(const FloatPartsN *a, const FloatPartsN *b,
                        float_status *s)
{
    int ab_mask = float_cmask(a->cls) | float_cmask(b->cls);
    FloatPartsN r = *a;

    r.sign ^= b->sign;
    r.exp -= b->exp;

    if (likely(cmask_is_only_normals(ab_mask))) {
        record_denormals_used(ab_mask, s);
        r.exp -= fracN(div)(&r, b);
        return r;
    }

    /* 0/0 or Inf/Inf => NaN */
    if (unlikely(ab_mask == float_cmask_zero)) {
        float_raise(float_flag_invalid | float_flag_invalid_zdz, s);
        return partsN(default_nan)(s);
    }
    if (unlikely(ab_mask == float_cmask_inf)) {
        float_raise(float_flag_invalid | float_flag_invalid_idi, s);
        return partsN(default_nan)(s);
    }

    /* All the NaN cases */
    if (unlikely(ab_mask & float_cmask_anynan)) {
        return partsN(pick_nan)(a, b, s);
    }

    if (b->cls != float_class_zero) {
        record_denormals_used(ab_mask, s);
    }

    /* Inf / X */
    if (r.cls == float_class_inf) {
        return r;
    }

    /* 0 / X */
    if (r.cls == float_class_zero) {
        return r;
    }

    /* X / Inf */
    if (b->cls == float_class_inf) {
        r.cls = float_class_zero;
        return r;
    }

    /* X / 0 => Inf */
    assert(b->cls == float_class_zero);
    float_raise(float_flag_divbyzero, s);
    r.cls = float_class_inf;
    return r;
}

/*
 * Floating point remainder, per IEC/IEEE, or modulus.
 */
static FloatPartsN *partsN(modrem)(FloatPartsN *a, FloatPartsN *b,
                                   uint64_t *mod_quot, float_status *s)
{
    int ab_mask = float_cmask(a->cls) | float_cmask(b->cls);

    if (likely(cmask_is_only_normals(ab_mask))) {
        record_denormals_used(ab_mask, s);
        fracN(modrem)(a, b, mod_quot);
        return a;
    }

    if (mod_quot) {
        *mod_quot = 0;
    }

    /* All the NaN cases */
    if (unlikely(ab_mask & float_cmask_anynan)) {
        *a = partsN(pick_nan)(a, b, s);
        return a;
    }

    /* Inf % N; N % 0 */
    if (a->cls == float_class_inf || b->cls == float_class_zero) {
        float_raise(float_flag_invalid, s);
        *a = partsN(default_nan)(s);
        return a;
    }

    record_denormals_used(ab_mask, s);

    /* N % Inf; 0 % N */
    g_assert(b->cls == float_class_inf || a->cls == float_class_zero);
    return a;
}

/*
 * Square Root
 *
 * The base algorithm is lifted from
 * https://git.musl-libc.org/cgit/musl/tree/src/math/sqrtf.c
 * https://git.musl-libc.org/cgit/musl/tree/src/math/sqrt.c
 * https://git.musl-libc.org/cgit/musl/tree/src/math/sqrtl.c
 * and is thus MIT licenced.
 */
static void partsN(sqrt)(FloatPartsN *a, float_status *status,
                         const FloatFmt *fmt)
{
    const uint32_t three32 = 3u << 30;
    const uint64_t three64 = 3ull << 62;
    uint32_t d32, m32, r32, s32, u32;            /* 32-bit computation */
    uint64_t d64, m64, r64, s64, u64;            /* 64-bit computation */
    uint64_t dh, dl, rh, rl, sh, sl, uh, ul;     /* 128-bit computation */
    uint64_t d0h, d0l, d1h, d1l, d2h, d2l;
    uint64_t discard;
    bool exp_odd;
    size_t index;

    if (unlikely(a->cls != float_class_normal)) {
        switch (a->cls) {
        case float_class_denormal:
            if (!a->sign) {
                /* -ve denormal will be InvalidOperation */
                float_raise(float_flag_input_denormal_used, status);
            }
            break;
        case float_class_snan:
        case float_class_qnan:
            *a = partsN(return_nan)(a, status);
            return;
        case float_class_zero:
            return;
        case float_class_inf:
            if (unlikely(a->sign)) {
                goto d_nan;
            }
            return;
        default:
            g_assert_not_reached();
        }
    }

    if (unlikely(a->sign)) {
        goto d_nan;
    }

    /*
     * Argument reduction.
     * x = 4^e frac; with integer e, and frac in [1, 4)
     * m = frac fixed point at bit 62, since we're in base 4.
     * If base-2 exponent is odd, exchange that for multiply by 2,
     * which results in no shift.
     */
    exp_odd = a->exp & 1;
    index = extract64(a->frac_hi, 57, 6) | (!exp_odd << 6);
    if (!exp_odd) {
        fracN(shr)(a, 1);
    }

    /*
     * Approximate r ~= 1/sqrt(m) and s ~= sqrt(m) when m in [1, 4).
     *
     * Initial estimate:
     * 7-bit lookup table (1-bit exponent and 6-bit significand).
     *
     * The relative error (e = r0*sqrt(m)-1) of a linear estimate
     * (r0 = a*m + b) is |e| < 0.085955 ~ 0x1.6p-4 at best;
     * a table lookup is faster and needs one less iteration.
     * The 7-bit table gives |e| < 0x1.fdp-9.
     *
     * A Newton-Raphson iteration for r is
     *   s = m*r
     *   d = s*r
     *   u = 3 - d
     *   r = r*u/2
     *
     * Fixed point representations:
     *   m, s, d, u, three are all 2.30; r is 0.32
     */
    m64 = a->frac_hi;
    m32 = m64 >> 32;

    r32 = rsqrt_tab[index] << 16;
    /* |r*sqrt(m) - 1| < 0x1.FDp-9 */

    s32 = ((uint64_t)m32 * r32) >> 32;
    d32 = ((uint64_t)s32 * r32) >> 32;
    u32 = three32 - d32;

    if (N == 64) {
        /* float64 or smaller */

        r32 = ((uint64_t)r32 * u32) >> 31;
        /* |r*sqrt(m) - 1| < 0x1.7Bp-16 */

        s32 = ((uint64_t)m32 * r32) >> 32;
        d32 = ((uint64_t)s32 * r32) >> 32;
        u32 = three32 - d32;

        if (fmt->frac_size <= 23) {
            /* float32 or smaller */

            s32 = ((uint64_t)s32 * u32) >> 32;  /* 3.29 */
            s32 = (s32 - 1) >> 6;               /* 9.23 */
            /* s < sqrt(m) < s + 0x1.08p-23 */

            /* compute nearest rounded result to 2.23 bits */
            uint32_t d0 = (m32 << 16) - s32 * s32;
            uint32_t d1 = s32 - d0;
            uint32_t d2 = d1 + s32 + 1;
            s32 += d1 >> 31;
            a->frac_hi = (uint64_t)s32 << (64 - 25);

            /* increment or decrement for inexact */
            if (d2 != 0) {
                a->frac_hi += ((int32_t)(d1 ^ d2) < 0 ? -1 : 1);
            }
            goto done;
        }

        /* float64 */

        r64 = (uint64_t)r32 * u32 * 2;
        /* |r*sqrt(m) - 1| < 0x1.37-p29; convert to 64-bit arithmetic */
        mul64To128(m64, r64, &s64, &discard);
        mul64To128(s64, r64, &d64, &discard);
        u64 = three64 - d64;

        mul64To128(s64, u64, &s64, &discard);  /* 3.61 */
        s64 = (s64 - 2) >> 9;                  /* 12.52 */

        /* Compute nearest rounded result */
        uint64_t d0 = (m64 << 42) - s64 * s64;
        uint64_t d1 = s64 - d0;
        uint64_t d2 = d1 + s64 + 1;
        s64 += d1 >> 63;
        a->frac_hi = s64 << (64 - 54);

        /* increment or decrement for inexact */
        if (d2 != 0) {
            a->frac_hi += ((int64_t)(d1 ^ d2) < 0 ? -1 : 1);
        }
        goto done;
    }

    r64 = (uint64_t)r32 * u32 * 2;
    /* |r*sqrt(m) - 1| < 0x1.7Bp-16; convert to 64-bit arithmetic */

    mul64To128(m64, r64, &s64, &discard);
    mul64To128(s64, r64, &d64, &discard);
    u64 = three64 - d64;
    mul64To128(u64, r64, &r64, &discard);
    r64 <<= 1;
    /* |r*sqrt(m) - 1| < 0x1.a5p-31 */

    mul64To128(m64, r64, &s64, &discard);
    mul64To128(s64, r64, &d64, &discard);
    u64 = three64 - d64;
    mul64To128(u64, r64, &rh, &rl);
    add128(rh, rl, rh, rl, &rh, &rl);
    /* |r*sqrt(m) - 1| < 0x1.c001p-59; change to 128-bit arithmetic */

    mul128To256(a->frac_hi, a->frac_lo, rh, rl, &sh, &sl, &discard, &discard);
    mul128To256(sh, sl, rh, rl, &dh, &dl, &discard, &discard);
    sub128(three64, 0, dh, dl, &uh, &ul);
    mul128To256(uh, ul, sh, sl, &sh, &sl, &discard, &discard);  /* 3.125 */
    /* -0x1p-116 < s - sqrt(m) < 0x3.8001p-125 */

    sub128(sh, sl, 0, 4, &sh, &sl);
    shift128Right(sh, sl, 13, &sh, &sl);  /* 16.112 */
    /* s < sqrt(m) < s + 1ulp */

    /* Compute nearest rounded result */
    mul64To128(sl, sl, &d0h, &d0l);
    d0h += 2 * sh * sl;
    sub128(a->frac_lo << 34, 0, d0h, d0l, &d0h, &d0l);
    sub128(sh, sl, d0h, d0l, &d1h, &d1l);
    add128(sh, sl, 0, 1, &d2h, &d2l);
    add128(d2h, d2l, d1h, d1l, &d2h, &d2l);
    add128(sh, sl, 0, d1h >> 63, &sh, &sl);
    shift128Left(sh, sl, 128 - 114, &sh, &sl);

    /* increment or decrement for inexact */
    if (d2h | d2l) {
        if ((int64_t)(d1h ^ d2h) < 0) {
            sub128(sh, sl, 0, 1, &sh, &sl);
        } else {
            add128(sh, sl, 0, 1, &sh, &sl);
        }
    }
    a->frac_lo = sl;
    a->frac_hi = sh;

 done:
    /* Convert back from base 4 to base 2. */
    a->exp >>= 1;
    if (!(a->frac_hi & DECOMPOSED_IMPLICIT_BIT)) {
        fracN(add)(a, a, a);
    } else {
        a->exp += 1;
    }
    return;

 d_nan:
    float_raise(float_flag_invalid | float_flag_invalid_sqrt, status);
    *a = partsN(default_nan)(status);
}

/*
 * Rounds the floating-point value `a' to an integer, and returns the
 * result as a floating-point value. The operation is performed
 * according to the IEC/IEEE Standard for Binary Floating-Point
 * Arithmetic.
 *
 * partsN(round_to_int_normal) is an internal helper function for
 * normal numbers only, returning true for inexact but not directly
 * raising float_flag_inexact.
 */
static bool partsN(round_to_int_normal)(FloatPartsN *a, FloatRoundMode rmode,
                                        int scale, int frac_size)
{
    uint64_t frac_lsb, frac_lsbm1, rnd_even_mask, rnd_mask, inc;
    int shift_adj;

    a->exp = exp_scalbn(a->exp, scale);

    if (a->exp < 0) {
        bool one;

        /* All fractional */
        switch (rmode) {
        case float_round_nearest_even:
        case float_round_nearest_even_max:
            one = false;
            if (a->exp == -1) {
                FloatPartsN tmp;
                /* Shift left one, discarding DECOMPOSED_IMPLICIT_BIT */
                fracN(add)(&tmp, a, a);
                /* Anything remaining means frac > 0.5. */
                one = !fracN(eqz)(&tmp);
            }
            break;
        case float_round_ties_away:
            one = a->exp == -1;
            break;
        case float_round_to_zero:
            one = false;
            break;
        case float_round_up:
            one = !a->sign;
            break;
        case float_round_down:
            one = a->sign;
            break;
        case float_round_to_odd:
        case float_round_to_odd_inf:
            one = true;
            break;
        default:
            g_assert_not_reached();
        }

        fracN(clear)(a);
        a->exp = 0;
        if (one) {
            a->frac_hi = DECOMPOSED_IMPLICIT_BIT;
        } else {
            a->cls = float_class_zero;
        }
        return true;
    }

    if (N > 64 && a->exp < N - 64) {
        /*
         * Rounding is not in the low word -- shift lsb to bit 2,
         * which leaves room for sticky and rounding bit.
         */
        shift_adj = (N - 1) - (a->exp + 2);
        fracN(shrjam)(a, shift_adj);
        frac_lsb = 1 << 2;
    } else {
        /*
         * Rounding is in the low word -- compute the lsb offset for rounding
         * and for clamping to the target precision, then map it to an offset
         * within frac_lo.
         */
        shift_adj = 0;
        frac_lsb = DECOMPOSED_IMPLICIT_BIT >> (MIN(a->exp, frac_size) & 63);
    }

    frac_lsbm1 = frac_lsb >> 1;
    rnd_mask = frac_lsb - 1;
    rnd_even_mask = rnd_mask | frac_lsb;

    if (!(a->frac_lo & rnd_mask)) {
        /* Fractional bits already clear, undo the shift above. */
        fracN(shl)(a, shift_adj);
        return false;
    }

    switch (rmode) {
    case float_round_nearest_even:
    case float_round_nearest_even_max:
        inc = ((a->frac_lo & rnd_even_mask) != frac_lsbm1 ? frac_lsbm1 : 0);
        break;
    case float_round_ties_away:
        inc = frac_lsbm1;
        break;
    case float_round_to_zero:
        inc = 0;
        break;
    case float_round_up:
        inc = a->sign ? 0 : rnd_mask;
        break;
    case float_round_down:
        inc = a->sign ? rnd_mask : 0;
        break;
    case float_round_to_odd:
    case float_round_to_odd_inf:
        inc = a->frac_lo & frac_lsb ? 0 : rnd_mask;
        break;
    default:
        g_assert_not_reached();
    }

    if (shift_adj == 0) {
        if (fracN(addi)(a, a, inc)) {
            fracN(shr)(a, 1);
            a->frac_hi |= DECOMPOSED_IMPLICIT_BIT;
            a->exp++;
        }
        a->frac_lo &= ~rnd_mask;
    } else {
        fracN(addi)(a, a, inc);
        a->frac_lo &= ~rnd_mask;
        /* Be careful shifting back, not to overflow */
        fracN(shl)(a, shift_adj - 1);
        if (a->frac_hi & DECOMPOSED_IMPLICIT_BIT) {
            a->exp++;
        } else {
            fracN(add)(a, a, a);
        }
    }
    return true;
}

FloatPartsN partsN(round_to_int)(const FloatPartsN *a,
                                 FloatRoundMode rmode,
                                 int scale, float_status *s,
                                 const FloatFmt *fmt)
{
    switch (a->cls) {
    case float_class_qnan:
    case float_class_snan:
        return partsN(return_nan)(a, s);
    case float_class_zero:
    case float_class_inf:
        return *a;
    case float_class_normal:
    case float_class_denormal:
        {
            FloatPartsN r = *a;
            if (partsN(round_to_int_normal)(&r, rmode, scale, fmt->frac_size)) {
                float_raise(float_flag_inexact, s);
            }
            return r;
        }
    default:
        g_assert_not_reached();
    }
}

/*
 * Returns the result of converting the floating-point value `a' to
 * the two's complement integer format. The conversion is performed
 * according to the IEC/IEEE Standard for Binary Floating-Point
 * Arithmetic---which means in particular that the conversion is
 * rounded according to the current rounding mode. If `a' is a NaN,
 * the largest positive integer is returned. Otherwise, if the
 * conversion overflows, the largest integer with the same sign as `a'
 * is returned.
 */
static int64_t partsN(float_to_sint)(FloatPartsN *p, FloatRoundMode rmode,
                                     int scale, int64_t min, int64_t max,
                                     float_status *s)
{
    FloatExceptionFlags flags = 0;
    uint64_t r;

    switch (p->cls) {
    case float_class_snan:
        flags |= float_flag_invalid_snan;
        /* fall through */
    case float_class_qnan:
        flags |= float_flag_invalid;
        r = max;
        break;

    case float_class_inf:
        flags = float_flag_invalid | float_flag_invalid_cvti;
        r = p->sign ? min : max;
        break;

    case float_class_zero:
        return 0;

    case float_class_normal:
    case float_class_denormal:
        /* TODO: N - 2 is frac_size for rounding; could use input fmt. */
        if (partsN(round_to_int_normal)(p, rmode, scale, N - 2)) {
            flags = float_flag_inexact;
        }

        if (p->exp <= DECOMPOSED_BINARY_POINT) {
            r = p->frac_hi >> (DECOMPOSED_BINARY_POINT - p->exp);
        } else {
            r = UINT64_MAX;
        }
        if (p->sign) {
            if (r <= -(uint64_t)min) {
                r = -r;
            } else {
                flags = float_flag_invalid | float_flag_invalid_cvti;
                r = min;
            }
        } else if (r > max) {
            flags = float_flag_invalid | float_flag_invalid_cvti;
            r = max;
        }
        break;

    default:
        g_assert_not_reached();
    }

    float_raise(flags, s);
    return r;
}

/*
 *  Returns the result of converting the floating-point value `a' to
 *  the unsigned integer format. The conversion is performed according
 *  to the IEC/IEEE Standard for Binary Floating-Point
 *  Arithmetic---which means in particular that the conversion is
 *  rounded according to the current rounding mode. If `a' is a NaN,
 *  the largest unsigned integer is returned. Otherwise, if the
 *  conversion overflows, the largest unsigned integer is returned. If
 *  the 'a' is negative, the result is rounded and zero is returned;
 *  values that do not round to zero will raise the inexact exception
 *  flag.
 */
static uint64_t partsN(float_to_uint)(FloatPartsN *p, FloatRoundMode rmode,
                                      int scale, uint64_t max, float_status *s)
{
    FloatExceptionFlags flags = 0;
    uint64_t r;

    switch (p->cls) {
    case float_class_snan:
        flags |= float_flag_invalid_snan;
        /* fall through */
    case float_class_qnan:
        flags |= float_flag_invalid;
        r = max;
        break;

    case float_class_inf:
        flags = float_flag_invalid | float_flag_invalid_cvti;
        r = p->sign ? 0 : max;
        break;

    case float_class_zero:
        return 0;

    case float_class_normal:
    case float_class_denormal:
        /* TODO: N - 2 is frac_size for rounding; could use input fmt. */
        if (partsN(round_to_int_normal)(p, rmode, scale, N - 2)) {
            flags = float_flag_inexact;
            if (p->cls == float_class_zero) {
                r = 0;
                break;
            }
        }

        if (p->sign) {
            flags = float_flag_invalid | float_flag_invalid_cvti;
            r = 0;
        } else if (p->exp > DECOMPOSED_BINARY_POINT) {
            flags = float_flag_invalid | float_flag_invalid_cvti;
            r = max;
        } else {
            r = p->frac_hi >> (DECOMPOSED_BINARY_POINT - p->exp);
            if (r > max) {
                flags = float_flag_invalid | float_flag_invalid_cvti;
                r = max;
            }
        }
        break;

    default:
        g_assert_not_reached();
    }

    float_raise(flags, s);
    return r;
}

/*
 * Integer to float conversions
 *
 * Returns the result of converting the two's complement integer `a'
 * to the floating-point format. The conversion is performed according
 * to the IEC/IEEE Standard for Binary Floating-Point Arithmetic.
 */
static void partsN(sint_to_float)(FloatPartsN *p, int64_t a,
                                  int scale, float_status *s)
{
    uint64_t f = a;
    int shift;

    memset(p, 0, sizeof(*p));

    if (a == 0) {
        p->cls = float_class_zero;
        return;
    }

    p->cls = float_class_normal;
    if (a < 0) {
        f = -f;
        p->sign = true;
    }
    shift = clz64(f);
    scale = MIN(MAX(scale, -0x10000), 0x10000);

    p->exp = DECOMPOSED_BINARY_POINT - shift + scale;
    p->frac_hi = f << shift;
}

/*
 * Unsigned Integer to float conversions
 *
 * Returns the result of converting the unsigned integer `a' to the
 * floating-point format. The conversion is performed according to the
 * IEC/IEEE Standard for Binary Floating-Point Arithmetic.
 */
static void partsN(uint_to_float)(FloatPartsN *p, uint64_t a,
                                  int scale, float_status *status)
{
    memset(p, 0, sizeof(*p));

    if (a == 0) {
        p->cls = float_class_zero;
    } else {
        int shift = clz64(a);
        scale = MIN(MAX(scale, -0x10000), 0x10000);
        p->cls = float_class_normal;
        p->exp = DECOMPOSED_BINARY_POINT - shift + scale;
        p->frac_hi = a << shift;
    }
}

/* Keep all integer bits until normalization, then jam for a narrower part. */
static void partsN(uint128_to_float)(FloatPartsN *p, Int128 a)
{
    FloatParts128 magnitude = {
        .frac_hi = int128_gethi(a),
        .frac_lo = int128_getlo(a),
    };
    int shift = frac128_normalize(&magnitude);

    memset(p, 0, sizeof(*p));
    if (shift == 128) {
        p->cls = float_class_zero;
        return;
    }
    p->cls = float_class_normal;
    p->exp = 127 - shift;
#if N == 64
    frac64_truncjam(p, &magnitude);
#else
    p->frac_hi = magnitude.frac_hi;
    p->frac_lo = magnitude.frac_lo;
#endif
}

static void partsN(sint128_to_float)(FloatPartsN *p, Int128 a)
{
    bool sign = !int128_nonneg(a);

    if (sign) {
        FloatParts128 magnitude = {
            .frac_hi = int128_gethi(a),
            .frac_lo = int128_getlo(a),
        };

        /* Unsigned fraction arithmetic also handles INT128_MIN. */
        frac128_neg(&magnitude);
        a = int128_make128(magnitude.frac_lo, magnitude.frac_hi);
    }
    partsN(uint128_to_float)(p, a);
    p->sign = sign;
}

/*
 * Float min/max.
 */
static FloatPartsN *partsN(minmax)(FloatPartsN *a, FloatPartsN *b,
                                   float_status *s, int flags)
{
    int ab_mask = float_cmask(a->cls) | float_cmask(b->cls);
    int a_exp, b_exp, cmp;

    if (unlikely(ab_mask & float_cmask_anynan)) {
        /*
         * For minNum/maxNum (IEEE 754-2008)
         * or minimumNumber/maximumNumber (IEEE 754-2019),
         * if one operand is a QNaN, and the other
         * operand is numerical, then return numerical argument.
         */
        if ((flags & (float_minmax_isnum | float_minmax_isnumber))
            && !(ab_mask & float_cmask_snan)
            && (ab_mask & ~float_cmask_qnan)) {
            record_denormals_used(ab_mask, s);
            return is_nan(a->cls) ? b : a;
        }

        /*
         * In IEEE 754-2019, minNum, maxNum, minNumMag and maxNumMag
         * are removed and replaced with minimum, minimumNumber, maximum
         * and maximumNumber.
         * minimumNumber/maximumNumber behavior for SNaN is changed to:
         *   If both operands are NaNs, a QNaN is returned.
         *   If either operand is a SNaN,
         *   an invalid operation exception is signaled,
         *   but unless both operands are NaNs,
         *   the SNaN is otherwise ignored and not converted to a QNaN.
         */
        if ((flags & float_minmax_isnumber)
            && (ab_mask & float_cmask_snan)
            && (ab_mask & ~float_cmask_anynan)) {
            float_raise(float_flag_invalid, s);
            return is_nan(a->cls) ? b : a;
        }

        *a = partsN(pick_nan)(a, b, s);
        return a;
    }

    record_denormals_used(ab_mask, s);

    a_exp = a->exp;
    b_exp = b->exp;

    if (unlikely(!cmask_is_only_normals(ab_mask))) {
        switch (a->cls) {
        case float_class_normal:
        case float_class_denormal:
            break;
        case float_class_inf:
            a_exp = INT16_MAX;
            break;
        case float_class_zero:
            a_exp = INT16_MIN;
            break;
        default:
            g_assert_not_reached();
        }
        switch (b->cls) {
        case float_class_normal:
        case float_class_denormal:
            break;
        case float_class_inf:
            b_exp = INT16_MAX;
            break;
        case float_class_zero:
            b_exp = INT16_MIN;
            break;
        default:
            g_assert_not_reached();
        }
    }

    /* Compare magnitudes. */
    cmp = a_exp - b_exp;
    if (cmp == 0) {
        cmp = fracN(cmp)(a, b);
    }

    /*
     * Take the sign into account.
     * For ismag, only do this if the magnitudes are equal.
     */
    if (!(flags & float_minmax_ismag) || cmp == 0) {
        if (a->sign != b->sign) {
            /* For differing signs, the negative operand is less. */
            cmp = a->sign ? -1 : 1;
        } else if (a->sign) {
            /* For two negative operands, invert the magnitude comparison. */
            cmp = -cmp;
        }
    }

    if (flags & float_minmax_ismin) {
        cmp = -cmp;
    }
    return cmp < 0 ? b : a;
}

/*
 * Floating point compare
 */
FloatRelation partsN(compare)(const FloatPartsN *a, const FloatPartsN *b,
                              float_status *s, bool is_quiet)
{
    int ab_mask = float_cmask(a->cls) | float_cmask(b->cls);

    if (likely(cmask_is_only_normals(ab_mask))) {
        FloatRelation cmp;

        record_denormals_used(ab_mask, s);

        if (a->sign != b->sign) {
            goto a_sign;
        }
        if (a->exp == b->exp) {
            cmp = fracN(cmp)(a, b);
        } else if (a->exp < b->exp) {
            cmp = float_relation_less;
        } else {
            cmp = float_relation_greater;
        }
        if (a->sign) {
            cmp = -cmp;
        }
        return cmp;
    }

    if (unlikely(ab_mask & float_cmask_anynan)) {
        if (ab_mask & float_cmask_snan) {
            float_raise(float_flag_invalid | float_flag_invalid_snan, s);
        } else if (!is_quiet) {
            float_raise(float_flag_invalid, s);
        }
        return float_relation_unordered;
    }

    record_denormals_used(ab_mask, s);

    if (ab_mask & float_cmask_zero) {
        if (ab_mask == float_cmask_zero) {
            return float_relation_equal;
        } else if (a->cls == float_class_zero) {
            goto b_sign;
        } else {
            goto a_sign;
        }
    }

    if (ab_mask == float_cmask_inf) {
        if (a->sign == b->sign) {
            return float_relation_equal;
        }
    } else if (b->cls == float_class_inf) {
        goto b_sign;
    } else {
        g_assert(a->cls == float_class_inf);
    }

 a_sign:
    return a->sign ? float_relation_less : float_relation_greater;
 b_sign:
    return b->sign ? float_relation_greater : float_relation_less;
}

/*
 * Multiply A by 2 raised to the power N.
 */
FloatPartsN partsN(scalbn)(const FloatPartsN *a, int n, float_status *s)
{
    switch (a->cls) {
    case float_class_snan:
    case float_class_qnan:
        return partsN(return_nan)(a, s);
    case float_class_zero:
    case float_class_inf:
        return *a;
    case float_class_denormal:
        float_raise(float_flag_input_denormal_used, s);
        /* fall through */
    case float_class_normal:
        {
            FloatPartsN r = *a;
            r.exp = exp_scalbn(r.exp, n);
            return r;
        }
    default:
        g_assert_not_reached();
    }
}
