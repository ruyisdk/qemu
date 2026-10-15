/*
 * QEMU floating-point arithmetic and NaN handling
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

FloatPartsN partsN(return_nan)(const FloatPartsN *a, float_status *s)
{
    switch (a->cls) {
    case float_class_snan:
        float_raise(float_flag_invalid | float_flag_invalid_snan, s);
        if (get_default_nan_mode(s)) {
            return partsN(default_nan)(s);
        } else {
            return partsN(silence_nan)(a, s);
        }
        break;
    case float_class_qnan:
        if (get_default_nan_mode(s)) {
            return partsN(default_nan)(s);
        }
        break;
    default:
        g_assert_not_reached();
    }
    return *a;
}

FloatPartsN partsN(pick_nan)(const FloatPartsN *a, const FloatPartsN *b,
                             float_status *s)
{
    bool have_snan = false;
    const FloatPartsN *ret;
    int cmp;

    if (is_snan(a->cls) || is_snan(b->cls)) {
        float_raise(float_flag_invalid | float_flag_invalid_snan, s);
        have_snan = true;
    }

    if (get_default_nan_mode(s)) {
        return partsN(default_nan)(s);
    }

    switch (get_float_2nan_prop_rule(s)) {
    case float_2nan_prop_s_ab:
        if (have_snan) {
            ret = is_snan(a->cls) ? a : b;
            break;
        }
        /* fall through */
    case float_2nan_prop_ab:
        ret = is_nan(a->cls) ? a : b;
        break;
    case float_2nan_prop_s_ba:
        if (have_snan) {
            ret = is_snan(b->cls) ? b : a;
            break;
        }
        /* fall through */
    case float_2nan_prop_ba:
        ret = is_nan(b->cls) ? b : a;
        break;
    case float_2nan_prop_x87:
        /*
         * This implements x87 NaN propagation rules:
         * SNaN + QNaN => return the QNaN
         * two SNaNs => return the one with the larger significand, silenced
         * two QNaNs => return the one with the larger significand
         * SNaN and a non-NaN => return the SNaN, silenced
         * QNaN and a non-NaN => return the QNaN
         *
         * If we get down to comparing significands and they are the same,
         * return the NaN with the positive sign bit (if any).
         */
        if (is_snan(a->cls)) {
            if (!is_snan(b->cls)) {
                ret = is_qnan(b->cls) ? b : a;
                break;
            }
        } else if (is_qnan(a->cls)) {
            if (is_snan(b->cls) || !is_qnan(b->cls)) {
                ret = a;
                break;
            }
        } else {
            ret = b;
            break;
        }
        cmp = fracN(cmp)(a, b);
        if (cmp == 0) {
            cmp = a->sign < b->sign;
        }
        ret = cmp > 0 ? a : b;
        break;
    default:
        g_assert_not_reached();
    }

    if (is_snan(ret->cls)) {
        return partsN(silence_nan)(ret, s);
    }
    return *ret;
}

static FloatPartsN partsN(pick_nan_muladd)(const FloatPartsN *a,
                                           const FloatPartsN *b,
                                           const FloatPartsN *c,
                                           float_status *s,
                                           int ab_mask, int abc_mask)
{
    bool infzero = (ab_mask == float_cmask_infzero);
    bool have_snan = (abc_mask & float_cmask_snan);
    FloatInfZeroNaNRule izn_rule = get_float_infzeronan_rule(s);
    const FloatPartsN *ret;

    if (unlikely(have_snan)) {
        float_raise(float_flag_invalid | float_flag_invalid_snan, s);
    }

    if (infzero && !(izn_rule & float_infzeronan_suppress_invalid)) {
        /* This is (0 * inf) + NaN or (inf * 0) + NaN */
        float_raise(float_flag_invalid | float_flag_invalid_imz, s);
    }

    if (get_default_nan_mode(s)) {
        /*
         * We guarantee not to require the target to tell us how to
         * pick a NaN if we're always returning the default NaN.
         * But if we're not in default-NaN mode then the target must
         * specify.
         */
        goto default_nan;
    } else if (infzero) {
        /*
         * Inf * 0 + NaN -- some implementations return the
         * default NaN here, and some return the input NaN.
         */
        switch (izn_rule & ~float_infzeronan_suppress_invalid) {
        case float_infzeronan_dnan_never:
            break;
        case float_infzeronan_dnan_always:
            goto default_nan;
        case float_infzeronan_dnan_if_qnan:
            if (is_qnan(c->cls)) {
                goto default_nan;
            }
            break;
        default:
            g_assert_not_reached();
        }
        ret = c;
    } else {
        const FloatPartsN *val[R_3NAN_1ST_MASK + 1] = { a, b, c };
        Float3NaNPropRule rule = get_float_3nan_prop_rule(s);

        assert(rule != float_3nan_prop_none);
        if (have_snan && (rule & R_3NAN_SNAN_MASK)) {
            /* We have at least one SNaN input and should prefer it */
            do {
                ret = val[rule & R_3NAN_1ST_MASK];
                rule >>= R_3NAN_1ST_LENGTH;
            } while (!is_snan(ret->cls));
        } else {
            do {
                ret = val[rule & R_3NAN_1ST_MASK];
                rule >>= R_3NAN_1ST_LENGTH;
            } while (!is_nan(ret->cls));
        }
    }

    if (is_snan(ret->cls)) {
        return partsN(silence_nan)(ret, s);
    }
    return *ret;

 default_nan:
    return partsN(default_nan)(s);
}

/*
 * Returns the result of adding or subtracting the values of the
 * floating-point values `a' and `b'. The operation is performed
 * according to the IEC/IEEE Standard for Binary Floating-Point
 * Arithmetic.
 */
FloatPartsN partsN(addsub)(const FloatPartsN *a_orig,
                           const FloatPartsN *b_orig,
                           float_status *s, bool subtract)
{
    int ab_mask = float_cmask(a_orig->cls) | float_cmask(b_orig->cls);

    if (unlikely(ab_mask & float_cmask_anynan)) {
        return partsN(pick_nan)(a_orig, b_orig, s);
    }

    /*
     * For addition and subtraction, we will consume an
     * input denormal unless the other input is a NaN.
     */
    record_denormals_used(ab_mask, s);

    FloatPartsN a = *a_orig;
    FloatPartsN b = *b_orig;

    b.sign ^= subtract;

    if (a.sign != b.sign) {
        /* Subtraction */
        if (likely(cmask_is_only_normals(ab_mask))) {
            if (partsN(sub_normal)(&a, &b)) {
                return a;
            }
            /* Subtract was exact, fall through to set sign. */
            ab_mask = float_cmask_zero;
        }

        if (ab_mask == float_cmask_zero) {
            /* 0 - 0 */
            a.sign = get_float_rounding_mode(s) == float_round_down;
            return a;
        }

        if (ab_mask & float_cmask_inf) {
            if (a.cls != float_class_inf) {
                /* N - Inf */
                return b;
            }
            if (b.cls != float_class_inf) {
                /* Inf - N */
                return a;
            }
            /* Inf - Inf */
            float_raise(float_flag_invalid | float_flag_invalid_isi, s);
            return partsN(default_nan)(s);
        }
    } else {
        /* Addition */
        if (likely(cmask_is_only_normals(ab_mask))) {
            partsN(add_normal)(&a, &b);
            return a;
        }

        if (ab_mask == float_cmask_zero) {
            /* 0 + 0 */
            return a;
        }

        if (ab_mask & float_cmask_inf) {
            /* N + Inf or Inf + N */
            a.cls = float_class_inf;
            return a;
        }
    }

    /* 0 +/- N or N +/- 0 */
    assert((ab_mask & float_cmask_zero) && (ab_mask & float_cmask_anynorm));
    return b.cls == float_class_zero ? a : b;
}

/*
 * Returns the result of multiplying the floating-point values `a' and
 * `b'. The operation is performed according to the IEC/IEEE Standard
 * for Binary Floating-Point Arithmetic.
 */
FloatPartsN partsN(mul)(const FloatPartsN *a, const FloatPartsN *b,
                        float_status *s)
{
    int ab_mask = float_cmask(a->cls) | float_cmask(b->cls);
    bool sign = a->sign ^ b->sign;

    if (likely(cmask_is_only_normals(ab_mask))) {
        FloatPartsW tmp;
        FloatPartsN r = {
            .cls = float_class_normal,
            .sign = sign,
            .exp = a->exp + b->exp + 1,
        };

        record_denormals_used(ab_mask, s);

        fracN(mulw)(&tmp, a, b);
        fracN(truncjam)(&r, &tmp);

        if (!(r.frac_hi & DECOMPOSED_IMPLICIT_BIT)) {
            fracN(add)(&r, &r, &r);
            r.exp -= 1;
        }

        return r;
    }

    /* Inf * Zero == NaN */
    if (unlikely(ab_mask == float_cmask_infzero)) {
        float_raise(float_flag_invalid | float_flag_invalid_imz, s);
        return partsN(default_nan)(s);
    }

    if (unlikely(ab_mask & float_cmask_anynan)) {
        return partsN(pick_nan)(a, b, s);
    }

    /* Multiply by 0 or Inf */
    record_denormals_used(ab_mask, s);

    if (ab_mask & float_cmask_inf) {
        return (FloatPartsN){ .cls = float_class_inf, .sign = sign };
    }

    g_assert(ab_mask & float_cmask_zero);
    return (FloatPartsN){ .cls = float_class_zero, .sign = sign };
}

/*
 * Returns the result of multiplying the floating-point values `a' and
 * `b' then adding 'c', with no intermediate rounding step after the
 * multiplication. The operation is performed according to the
 * IEC/IEEE Standard for Binary Floating-Point Arithmetic 754-2008.
 * The flags argument allows the caller to select negation of the addend
 * or the intermediate product. (The difference between this and having
 * the caller do a separate negation is that negating externally will
 * flip the sign bit on NaNs.) Note that float_muladd_negate_result
 * is not applied here, and should be handled separately after rounding
 * chooses the final sign of 0.0.
 *
 * The product and aligned addend use W-bit fractions before the result
 * is narrowed to N bits with a sticky bit for final rounding.
 */
FloatPartsN partsN(muladd)(const FloatPartsN *a, const FloatPartsN *b,
                           const FloatPartsN *c, int flags, float_status *s)
{
    int ab_mask = float_cmask(a->cls) | float_cmask(b->cls);
    int c_mask = float_cmask(c->cls);
    int abc_mask = ab_mask | c_mask;
    bool c_sign = c->sign ^ !!(flags & float_muladd_negate_c);
    bool p_sign = a->sign ^ b->sign ^ !!(flags & float_muladd_negate_product);

    /*
     * The "likely" case is A and B normal, so that the product is normal,
     * and C normal or zero so that the result is normal.
     */
    int likely_mask = ab_mask | (c_mask & ~float_cmask_zero);
    if (likely(cmask_is_only_normals(likely_mask))) {
        record_denormals_used(abc_mask, s);

        /* Perform the multiplication step. */
        FloatPartsW p_widen = { .sign = p_sign, .exp = a->exp + b->exp + 1 };
        fracN(mulw)(&p_widen, a, b);
        if (!(p_widen.frac_hi & DECOMPOSED_IMPLICIT_BIT)) {
            fracW(add)(&p_widen, &p_widen, &p_widen);
            p_widen.exp -= 1;
        }

        /* Perform the addition step. */
        if (!(c_mask & float_cmask_zero)) {
            /* Zero-extend C to less significant bits. */
            FloatPartsW c_widen = { .sign = c_sign, .exp = c->exp };
            fracN(widen)(&c_widen, c);

            if (p_sign == c_sign) {
                partsW(add_normal)(&p_widen, &c_widen);
            } else if (!partsW(sub_normal)(&p_widen, &c_widen)) {
                goto return_sub_zero;
            }
        }

        /* Narrow with sticky bit, for proper rounding later. */
        FloatPartsN r = {
            .sign = p_widen.sign,
            .exp = p_widen.exp,
            .cls = float_class_normal,
        };
        fracN(truncjam)(&r, &p_widen);
        return r;
    }

    /*
     * It is implementation-defined whether the cases of (0,inf,qnan)
     * and (inf,0,qnan) raise InvalidOperation or not (and what QNaN
     * they return if they do), so we have to hand this information
     * off to the target-specific pick-a-NaN routine.
     */
    if (unlikely(abc_mask & float_cmask_anynan)) {
        return partsN(pick_nan_muladd)(a, b, c, s, ab_mask, abc_mask);
    }

    if (unlikely(ab_mask == float_cmask_infzero)) {
        /* Inf * Zero == NaN */
        float_raise(float_flag_invalid | float_flag_invalid_imz, s);
        goto d_nan;
    }

    if (unlikely(ab_mask & float_cmask_inf)) {
        if ((c_mask & float_cmask_inf) && p_sign != c_sign) {
            /* Inf - Inf == NaN */
            float_raise(float_flag_invalid | float_flag_invalid_isi, s);
            goto d_nan;
        }
        /* Inf + C == Inf */
        record_denormals_used(abc_mask, s);
        return (FloatPartsN){ .sign = p_sign, .cls = float_class_inf };
    }
    record_denormals_used(abc_mask, s);

    /* Only remaining cases are zero product or inf addend. */
    assert((ab_mask & float_cmask_zero) | (c_mask & float_cmask_inf));

    /*
     * P + Inf == Inf, or
     * 0 + C == C,
     * except for 0 - 0, which needs special rounding,
     * except for when we want to suppress this addition step.
     */
    if (!(c_mask & float_cmask_zero)
        || p_sign == c_sign
        || (flags & float_muladd_suppress_add_product_zero)) {
        FloatPartsN r = *c;
        r.sign = c_sign;
        return r;
    }

 return_sub_zero:
    /* 0 - 0 == -0 for round_down, +0 otherwise. */
    return (FloatPartsN){
        .sign = get_float_rounding_mode(s) == float_round_down,
        .cls = float_class_zero
    };

 d_nan:
    return partsN(default_nan)(s);
}
