/*
 * RISC-V Attached Matrix Extension (AME) floating-point execution
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/host-utils.h"
#include "fpu/softfloat.h"
#include "fpu/softfloat-helpers.h"
#include "fpu/softfloat-parts.h"
#include "target/riscv/ame/fp.h"
#include "target/riscv/ame/int.h"

static bool ame_fp_rounding_mode(AmeFPRoundMode mode,
                                 FloatRoundMode *rounding)
{
    switch (mode) {
    case AME_FP_RNE:
        *rounding = float_round_nearest_even;
        return true;
    case AME_FP_RTZ:
        *rounding = float_round_to_zero;
        return true;
    case AME_FP_RDN:
        *rounding = float_round_down;
        return true;
    case AME_FP_RUP:
        *rounding = float_round_up;
        return true;
    case AME_FP_RMM:
        *rounding = float_round_ties_away;
        return true;
    case AME_FP_RNO:
        *rounding = float_round_to_odd_inf;
        return true;
    default:
        return false;
    }
}

static bool ame_fp_dtype_valid(const AmeDTypeDesc *dtype)
{
    const AmeFPFormatDesc *format;

    if (dtype == NULL || (dtype->class != AME_DTYPE_CLASS_IEEE &&
                          dtype->class != AME_DTYPE_CLASS_OCP)) {
        return false;
    }

    format = riscv_ame_fp_format_desc(dtype->fp.format);
    return format && dtype->class == format->class &&
           dtype->bits == format->bits;
}

bool riscv_ame_fp_supports(AmeOperation op, const AmeDTypeDesc *dest,
                           const AmeDTypeDesc *a, const AmeDTypeDesc *b)
{
    bool di = riscv_ame_dtype_is_integer(dest);
    bool ai = riscv_ame_dtype_is_integer(a);
    bool bi = riscv_ame_dtype_is_integer(b);
    bool df = ame_fp_dtype_valid(dest);
    bool af = ame_fp_dtype_valid(a);
    bool bf = ame_fp_dtype_valid(b);
    bool dn = di || df;
    bool an = ai || af;
    bool bn = bi || bf;

    if (!dn || !an) {
        return false;
    }
    switch (op) {
    case AME_OP_CONV:
        return b == NULL && (df || af);
    case AME_OP_CMPGE:
    case AME_OP_CMPLT:
        return di && bn;
    case AME_OP_ABS:
        return b == NULL;
    case AME_OP_SIN:
    case AME_OP_COS:
    case AME_OP_SQRT:
    case AME_OP_REC:
    case AME_OP_RSQRT:
    case AME_OP_EXP2:
    case AME_OP_LOG2:
    case AME_OP_TANH:
    case AME_OP_FRINTM:
    case AME_OP_FRINTN:
    case AME_OP_FRINTP:
    case AME_OP_FRINTZ:
        return df && af && b == NULL;
    case AME_OP_LDEXP:
    case AME_OP_RDEXP:
    case AME_OP_LDEXPACC:
    case AME_OP_RDEXPACC:
        return bi;
    case AME_OP_ABSDIFF:
    case AME_OP_ADD:
    case AME_OP_SUB:
    case AME_OP_HDIFF:
    case AME_OP_MEAN:
    case AME_OP_MUL:
    case AME_OP_NMUL:
    case AME_OP_MULACC:
    case AME_OP_MULNACC:
    case AME_OP_MULADD:
    case AME_OP_MULSUB:
    case AME_OP_MIN:
    case AME_OP_MAX:
        return bn;
    case AME_OP_LOG2SUB:
    case AME_OP_SUBLOG2:
        return df && af && bn;
    default:
        return false;
    }
}

void riscv_ame_fp_status_init(float_status *status)
{
    g_assert(status != NULL);
    memset(status, 0, sizeof(*status));
    set_float_detect_tininess(float_tininess_after_rounding, status);
    set_default_nan_mode(true, status);
    set_float_default_nan_pattern(0x40, status);
    set_snan_rule(float_snan_bit_is_zero, status);
    set_float_e4m3_nan_is_snan(false, status);
}

bool riscv_ame_fp_status_prepare(float_status *status,
                                 const AmeDTypeDesc *dest_dtype)
{
    FloatRoundMode rounding;

    if (status == NULL) {
        return false;
    }
    if (riscv_ame_dtype_is_integer(dest_dtype)) {
        if ((unsigned)dest_dtype->integer.rounding >= AME_INT_ROUND_MAX) {
            return false;
        }
        /* Integer results use the destination's integer rounding rules. */
        riscv_ame_fp_status_init(status);
        return true;
    }
    if (!ame_fp_dtype_valid(dest_dtype) ||
        !ame_fp_rounding_mode(dest_dtype->fp.rounding, &rounding)) {
        return false;
    }

    riscv_ame_fp_status_init(status);
    set_float_rounding_mode(rounding, status);
    return true;
}

bool riscv_ame_fp_status_set_rounding(float_status *status,
                                      const AmeDTypeDesc *dest_dtype)
{
    FloatRoundMode rounding;

    if (status == NULL || !ame_fp_dtype_valid(dest_dtype) ||
        !ame_fp_rounding_mode(dest_dtype->fp.rounding, &rounding)) {
        return false;
    }

    set_float_rounding_mode(rounding, status);
    return true;
}

uint8_t riscv_ame_fp_status_flags(const float_status *status)
{
    FloatExceptionFlags flags;
    uint8_t result = 0;

    g_assert(status != NULL);
    flags = get_float_exception_flags(status);

    if (flags & float_flag_invalid) {
        result |= AME_FFLAG_NV;
    }
    if (flags & float_flag_divbyzero) {
        result |= AME_FFLAG_DZ;
    }
    if (flags & float_flag_overflow) {
        result |= AME_FFLAG_OF;
    }
    if (flags & float_flag_underflow) {
        result |= AME_FFLAG_UF;
    }
    if (flags & float_flag_inexact) {
        result |= AME_FFLAG_NX;
    }
    return result;
}

static FloatParts64 ame_fp_unpack(const AmeDTypeDesc *dtype, uint64_t value,
                                  float_status *status)
{
    switch (dtype->fp.format) {
    case AME_FP_FORMAT_FP16:
        return float16_unpack_canonical(value, status);
    case AME_FP_FORMAT_BF16:
        return bfloat16_unpack_canonical(value, status);
    case AME_FP_FORMAT_FP32:
        return float32_unpack_canonical(make_float32(value), status);
    case AME_FP_FORMAT_FP64:
        return float64_unpack_canonical(make_float64(value), status);
    case AME_FP_FORMAT_OCP_E4M3:
        return float8_e4m3_unpack_canonical(value, status);
    case AME_FP_FORMAT_OCP_E5M2:
        /* OCP FP8 does not distinguish quiet and signaling NaNs. */
        if ((value & UINT8_C(0x7c)) == UINT8_C(0x7c) &&
            (value & UINT8_C(0x03)) != 0) {
            value = (value & UINT8_C(0x80)) | UINT8_C(0x7e);
        }
        return float8_e5m2_unpack_canonical(value, status);
    default:
        g_assert_not_reached();
    }
}

static uint64_t ame_fp_pack(const AmeDTypeDesc *dtype, FloatParts64 parts,
                            float_status *status)
{
    bool infinity = parts.cls == float_class_inf;

    if (parts.cls >= float_class_qnan) {
        parts = parts64_return_nan(&parts, status);
    }
    switch (dtype->fp.format) {
    case AME_FP_FORMAT_FP16:
        return float16_val(float16_round_pack_canonical(&parts, status));
    case AME_FP_FORMAT_BF16:
        return bfloat16_round_pack_canonical(&parts, status);
    case AME_FP_FORMAT_FP32:
        return float32_val(float32_round_pack_canonical(&parts, status));
    case AME_FP_FORMAT_FP64:
        return float64_val(float64_round_pack_canonical(&parts, status));
    case AME_FP_FORMAT_OCP_E4M3:
        if (infinity) {
            float_raise(float_flag_overflow | float_flag_inexact, status);
        }
        return float8_e4m3_round_pack_canonical(&parts, status, true);
    case AME_FP_FORMAT_OCP_E5M2:
        return float8_e5m2_round_pack_canonical(&parts, status, false);
    default:
        g_assert_not_reached();
    }
}

static bool ame_fp_is_nan(const AmeDTypeDesc *dtype, AmeValue value)
{
    switch (dtype->fp.format) {
    case AME_FP_FORMAT_FP16:
        return float16_is_any_nan((float16)value.lo);
    case AME_FP_FORMAT_BF16:
        return bfloat16_is_any_nan((bfloat16)value.lo);
    case AME_FP_FORMAT_FP32:
        return float32_is_any_nan(make_float32(value.lo));
    case AME_FP_FORMAT_FP64:
        return float64_is_any_nan(make_float64(value.lo));
    case AME_FP_FORMAT_OCP_E4M3:
        return (value.lo & UINT8_C(0x7f)) == UINT8_C(0x7f);
    case AME_FP_FORMAT_OCP_E5M2:
        return (value.lo & UINT8_C(0x7c)) == UINT8_C(0x7c) &&
               (value.lo & UINT8_C(0x03)) != 0;
    default:
        g_assert_not_reached();
    }
}

static bool ame_fp_is_snan(const AmeDTypeDesc *dtype, AmeValue value,
                           float_status *status)
{
    switch (dtype->fp.format) {
    case AME_FP_FORMAT_FP16:
        return float16_is_signaling_nan((float16)value.lo, status);
    case AME_FP_FORMAT_BF16:
        return bfloat16_is_signaling_nan((bfloat16)value.lo, status);
    case AME_FP_FORMAT_FP32:
        return float32_is_signaling_nan(make_float32(value.lo), status);
    case AME_FP_FORMAT_FP64:
        return float64_is_signaling_nan(make_float64(value.lo), status);
    case AME_FP_FORMAT_OCP_E4M3:
    case AME_FP_FORMAT_OCP_E5M2:
        return false;
    default:
        g_assert_not_reached();
    }
}

static AmeValue ame_fp_default_nan(const AmeDTypeDesc *dtype,
                                   float_status *status)
{
    AmeValue result = { 0 };

    switch (dtype->fp.format) {
    case AME_FP_FORMAT_FP16:
        result.lo = float16_val(float16_default_nan(status));
        break;
    case AME_FP_FORMAT_BF16:
        result.lo = bfloat16_default_nan(status);
        break;
    case AME_FP_FORMAT_FP32:
        result.lo = float32_val(float32_default_nan(status));
        break;
    case AME_FP_FORMAT_FP64:
        result.lo = float64_val(float64_default_nan(status));
        break;
    case AME_FP_FORMAT_OCP_E4M3:
        result.lo = UINT8_C(0x7f);
        break;
    case AME_FP_FORMAT_OCP_E5M2:
        result.lo = UINT8_C(0x7e);
        break;
    default:
        g_assert_not_reached();
    }
    return result;
}

static float64 ame_fp_to_float64(const AmeDTypeDesc *dtype, AmeValue value,
                                 float_status *status)
{
    switch (dtype->fp.format) {
    case AME_FP_FORMAT_FP16:
        return float16_to_float64((float16)value.lo, true, status);
    case AME_FP_FORMAT_BF16:
        return bfloat16_to_float64((bfloat16)value.lo, status);
    case AME_FP_FORMAT_FP32:
        return float32_to_float64(make_float32(value.lo), status);
    case AME_FP_FORMAT_FP64:
        return make_float64(value.lo);
    case AME_FP_FORMAT_OCP_E4M3:
    case AME_FP_FORMAT_OCP_E5M2:
    {
        FloatParts64 parts = ame_fp_unpack(dtype, value.lo, status);

        if (parts.cls >= float_class_qnan) {
            parts = parts64_return_nan(&parts, status);
        }
        return float64_round_pack_canonical(&parts, status);
    }
    default:
        g_assert_not_reached();
    }
}

static AmeValue ame_fp_from_float64(const AmeDTypeDesc *dtype, float64 value,
                                    float_status *status)
{
    AmeValue result = { 0 };

    switch (dtype->fp.format) {
    case AME_FP_FORMAT_FP16:
        result.lo = float16_val(float64_to_float16(value, true, status));
        break;
    case AME_FP_FORMAT_BF16:
        result.lo = float64_to_bfloat16(value, status);
        break;
    case AME_FP_FORMAT_FP32:
        result.lo = float32_val(float64_to_float32(value, status));
        break;
    case AME_FP_FORMAT_FP64:
        result.lo = float64_val(value);
        break;
    case AME_FP_FORMAT_OCP_E4M3:
    case AME_FP_FORMAT_OCP_E5M2:
    {
        FloatParts64 parts = float64_unpack_canonical(value, status);

        result.lo = ame_fp_pack(dtype, parts, status);
        break;
    }
    default:
        g_assert_not_reached();
    }
    return result;
}

static int64_t ame_sign_extend64(uint64_t value, unsigned bits)
{
    g_assert(bits > 0 && bits <= 64);
    if (bits == 64) {
        return value;
    }
    return (int64_t)(value << (64 - bits)) >> (64 - bits);
}

static Int128 ame_integer_value(const AmeDTypeDesc *src_dtype, AmeValue source)
{
    uint64_t hi = source.hi;
    uint64_t lo = source.lo;

    g_assert(src_dtype->bits <= 128);
    if (src_dtype->bits < 64) {
        uint64_t mask = MAKE_64BIT_MASK(0, src_dtype->bits);

        lo &= mask;
        if (src_dtype->integer.is_signed &&
            (lo & BIT_ULL(src_dtype->bits - 1))) {
            lo |= ~mask;
            hi = UINT64_MAX;
        } else {
            hi = 0;
        }
    } else if (src_dtype->bits == 64) {
        hi = src_dtype->integer.is_signed && (int64_t)lo < 0 ? UINT64_MAX : 0;
    }
    return int128_make128(lo, hi);
}

static AmeValue ame_int_to_fp(const AmeDTypeDesc *dest_dtype,
                              const AmeDTypeDesc *src_dtype, AmeValue source,
                              float_status *status)
{
    Int128 value = ame_integer_value(src_dtype, source);
    AmeValue result = { 0 };

    switch (dest_dtype->fp.format) {
    case AME_FP_FORMAT_FP16:
        result.lo = float16_val(src_dtype->integer.is_signed ?
            int128_to_float16(value, status) :
            uint128_to_float16(value, status));
        break;
    case AME_FP_FORMAT_BF16:
        result.lo = src_dtype->integer.is_signed ?
            int128_to_bfloat16(value, status) :
            uint128_to_bfloat16(value, status);
        break;
    case AME_FP_FORMAT_FP32:
        result.lo = float32_val(src_dtype->integer.is_signed ?
            int128_to_float32(value, status) :
            uint128_to_float32(value, status));
        break;
    case AME_FP_FORMAT_FP64:
        result.lo = float64_val(src_dtype->integer.is_signed ?
            int128_to_float64(value, status) :
            uint128_to_float64(value, status));
        break;
    case AME_FP_FORMAT_OCP_E4M3:
        result.lo = src_dtype->integer.is_signed ?
            int128_to_float8_e4m3(value, true, status) :
            uint128_to_float8_e4m3(value, true, status);
        break;
    case AME_FP_FORMAT_OCP_E5M2:
        result.lo = src_dtype->integer.is_signed ?
            int128_to_float8_e5m2(value, false, status) :
            uint128_to_float8_e5m2(value, false, status);
        break;
    default:
        g_assert_not_reached();
    }
    return result;
}

/*
 * Round to the destination's integer mode before checking its range. Large
 * integral inputs only need their low AME_VALUE_BITS and an out-of-range
 * magnitude: shifting by at most AME_VALUE_BITS preserves both. This also
 * provides a clamping sentinel for NaN and infinity without host FP arithmetic.
 */
static AmeExecResult ame_fp_parts_to_int(const AmeDTypeDesc *dest_dtype,
                                         FloatParts64 parts,
                                         float_status *status)
{
    AmeIntProduct significand = { .negative = parts.sign };
    AmeIntAccumulator acc = { 0 };
    AmeDTypeDesc clamped = *dest_dtype;
    AmeExecResult result;
    const int binary_point = AME_VALUE_WORD_BITS - 1;
    bool nonfinite = parts.cls >= float_class_inf;
    bool inexact = false;

    if (nonfinite) {
        significand.word[AME_VALUE_WORDS] = 1;
        /* NaN, of either sign, returns the positive destination maximum. */
        significand.negative = parts.cls == float_class_inf && parts.sign;
    } else if (parts.cls != float_class_zero) {
        if (parts.exp >= binary_point) {
            unsigned shift = MIN(parts.exp - binary_point, AME_VALUE_BITS);
            unsigned word = shift / AME_VALUE_WORD_BITS;
            unsigned bits = shift % AME_VALUE_WORD_BITS;

            significand.word[word] = parts.frac << bits;
            if (bits) {
                significand.word[word + 1] =
                    parts.frac >> (AME_VALUE_WORD_BITS - bits);
            }
        } else {
            significand.word[0] = parts.frac;
            inexact = parts.exp < 0 ||
                      (parts.frac & MAKE_64BIT_MASK(0,
                                                   binary_point - parts.exp));
        }
    }
    riscv_ame_int_acc_add_product(&acc, &significand);
    if (!nonfinite && parts.cls != float_class_zero &&
        parts.exp < binary_point) {
        riscv_ame_int_acc_round_shift(&acc, dest_dtype,
                                      binary_point - parts.exp);
    }

    /* Check the rounded value even when the requested result must wrap. */
    clamped.integer.saturating = true;
    riscv_ame_int_acc_finish(&acc, &clamped, &result);
    if (nonfinite) {
        result.saturated = false;
        float_raise(float_flag_invalid, status);
    } else if (result.saturated) {
        if (!dest_dtype->integer.saturating) {
            riscv_ame_int_acc_finish(&acc, dest_dtype, &result);
        }
        float_raise(float_flag_invalid, status);
    } else if (inexact) {
        float_raise(float_flag_inexact, status);
    }
    return result;
}

static FloatParts128 ame_number_unpack(const AmeDTypeDesc *dtype,
                                       AmeValue value, float_status *status)
{
    if (dtype->class == AME_DTYPE_CLASS_INTEGER) {
        Int128 integer = ame_integer_value(dtype, value);

        return dtype->integer.is_signed ? int128_unpack_canonical(integer) :
                                          uint128_unpack_canonical(integer);
    } else {
        FloatParts64 p = ame_fp_unpack(dtype, value.lo, status);

        return (FloatParts128) {
            .cls = p.cls, .sign = p.sign, .exp = p.exp, .frac_hi = p.frac,
        };
    }
}

static FloatParts64 ame_number_narrow(FloatParts128 value)
{
    return (FloatParts64) {
        .cls = value.cls, .sign = value.sign, .exp = value.exp,
        .frac = value.frac_hi | (value.frac_lo != 0),
    };
}

static AmeExecResult ame_number_finish(const AmeDTypeDesc *dest,
                                        FloatParts128 value,
                                        float_status *status)
{
    FloatParts64 narrow = ame_number_narrow(value);

    if (dest->class == AME_DTYPE_CLASS_INTEGER) {
        /* Finite integer results use the exact binary sum, not this path. */
        g_assert(value.cls >= float_class_inf);
        return ame_fp_parts_to_int(dest, narrow, status);
    }
    return (AmeExecResult) { .value.lo = ame_fp_pack(dest, narrow, status) };
}

static AmeIntProduct ame_number_term(const FloatParts128 *parts)
{
    return (AmeIntProduct) {
        .word = { parts->frac_lo, parts->frac_hi }, .negative = parts->sign,
    };
}

static AmeIntProduct ame_number_product(const FloatParts128 *left,
                                         const FloatParts128 *right,
                                         bool negate)
{
    const AmeDTypeDesc magnitude = {
        .class = AME_DTYPE_CLASS_INTEGER, .bits = AME_VALUE_BITS,
    };
    AmeValue a = { .lo = left->frac_lo, .hi = left->frac_hi };
    AmeValue b = { .lo = right->frac_lo, .hi = right->frac_hi };
    AmeIntProduct product;

    riscv_ame_int_product_init(&product, &magnitude, &a, &magnitude, &b,
                               left->sign ^ right->sign ^ negate);
    return product;
}

static void ame_number_integer_flags(AmeExecResult *result,
                                      float_status *status)
{
    if (result->fflags & AME_FFLAG_NV) {
        float_raise(float_flag_invalid, status);
    } else if (result->fflags & AME_FFLAG_NX) {
        float_raise(float_flag_inexact, status);
    }
    result->fflags = riscv_ame_fp_status_flags(status);
}

static void ame_exec_number_binary(AmeOperation op, const AmeDTypeDesc *dest,
                                    const AmeDTypeDesc *a, const AmeValue *av,
                                    const AmeDTypeDesc *b, const AmeValue *bv,
                                    AmeExecResult *result, float_status *status)
{
    FloatParts128 left, right, value;
    const int binary_point = AME_VALUE_BITS - 1;
    bool subtract = op == AME_OP_SUB || op == AME_OP_HDIFF;
    bool absolute = op == AME_OP_ABSDIFF;
    bool half = op == AME_OP_MEAN || op == AME_OP_HDIFF;
    bool multiply = op == AME_OP_MUL || op == AME_OP_NMUL;
    bool wide = a->bits == AME_VALUE_BITS || b->bits == AME_VALUE_BITS;

    left = ame_number_unpack(a, *av, status);
    right = ame_number_unpack(b, *bv, status);
    if (op == AME_OP_CMPGE || op == AME_OP_CMPLT) {
        FloatRelation relation = parts128_compare(&left, &right, status, true);
        bool predicate = relation != float_relation_unordered &&
            (op == AME_OP_CMPGE ? relation >= float_relation_equal :
                                 relation < float_relation_equal);

        *result = (AmeExecResult) { 0 };
        if (predicate) {
            result->value.lo = dest->bits >= AME_VALUE_WORD_BITS ? UINT64_MAX :
                               MAKE_64BIT_MASK(0, dest->bits);
            result->value.hi = dest->bits == AME_VALUE_BITS ? UINT64_MAX : 0;
        }
        result->fflags = riscv_ame_fp_status_flags(status);
        return;
    }
    if (op == AME_OP_MIN || op == AME_OP_MAX) {
        if (left.cls >= float_class_qnan || right.cls >= float_class_qnan) {
            value = parts128_pick_nan(&left, &right, status);
        } else {
            FloatRelation relation = parts128_compare(&left, &right,
                                                       status, true);

            value = ((op == AME_OP_MIN && relation == float_relation_greater) ||
                     (op == AME_OP_MAX && relation == float_relation_less)) ?
                    right : left;
            if (left.cls == float_class_zero && right.cls == float_class_zero) {
                value.sign = op == AME_OP_MIN ? left.sign || right.sign :
                                               left.sign && right.sign;
            }
        }
        if (dest->class == AME_DTYPE_CLASS_INTEGER &&
            value.cls < float_class_inf) {
            AmeIntProduct term = ame_number_term(&value);

            riscv_ame_int_scaled_sum(&term, value.exp - binary_point, NULL, 0,
                                      false, dest, result);
            ame_number_integer_flags(result, status);
        } else {
            *result = ame_number_finish(dest, value, status);
            result->fflags = riscv_ame_fp_status_flags(status);
        }
        return;
    }
    if (dest->class == AME_DTYPE_CLASS_INTEGER && left.cls < float_class_inf &&
        right.cls < float_class_inf) {
        AmeIntProduct first = ame_number_term(&left);
        AmeIntProduct second = ame_number_term(&right);
        int first_exp = left.exp - binary_point;
        int second_exp = right.exp - binary_point;

        if (multiply) {
            first = ame_number_product(&left, &right, op == AME_OP_NMUL);
            first_exp += second_exp;
            memset(&second, 0, sizeof(second));
        } else if (subtract) {
            first.negative = !first.negative;
        } else if (absolute) {
            second.negative = !second.negative;
        }
        riscv_ame_int_scaled_sum(&first, first_exp - half, &second,
                                  second_exp - half, absolute, dest, result);
        ame_number_integer_flags(result, status);
        return;
    }
    if (multiply) {
        left.sign ^= op == AME_OP_NMUL;
        value = parts128_mul(&left, &right, status);
    } else {
        const FloatParts128 *first = subtract ? &right : &left;
        const FloatParts128 *second = subtract ? &left : &right;

        if (wide) {
            FloatParts256 first256 = parts128_to_parts256(first);
            FloatParts256 second256 = parts128_to_parts256(second);
            FloatParts256 result256 = parts256_addsub(
                &first256, &second256, status, subtract || absolute);

            value = parts256_to_parts128(&result256);
        } else {
            value = parts128_addsub(first, second, status,
                                    subtract || absolute);
        }
        if (absolute) {
            value.sign = false;
        }
    }
    if (half) {
        value = parts128_scalbn(&value, -1, status);
    }
    *result = ame_number_finish(dest, value, status);
    result->fflags = riscv_ame_fp_status_flags(status);
}

static void ame_exec_number_ternary(AmeOperation op, const AmeDTypeDesc *dest,
                                     const AmeDTypeDesc *a, const AmeValue *av,
                                     const AmeDTypeDesc *b, const AmeValue *bv,
                                     const AmeValue *current,
                                     AmeExecResult *result,
                                     float_status *status)
{
    FloatParts128 left = ame_number_unpack(a, *av, status);
    FloatParts128 right = ame_number_unpack(b, *bv, status);
    FloatParts128 addend = ame_number_unpack(dest, *current, status);
    bool negate = op == AME_OP_MULNACC || op == AME_OP_MULSUB;
    FloatParts128 value;
    const int binary_point = AME_VALUE_BITS - 1;

    if (op == AME_OP_MULADD || op == AME_OP_MULSUB) {
        value = addend;
        addend = left;
        left = value;
    }
    if (dest->class == AME_DTYPE_CLASS_INTEGER && left.cls < float_class_inf &&
        right.cls < float_class_inf && addend.cls < float_class_inf) {
        AmeIntProduct product = ame_number_product(&left, &right, negate);
        AmeIntProduct term = ame_number_term(&addend);

        riscv_ame_int_scaled_sum(
            &product, left.exp + right.exp - 2 * binary_point,
            &term, addend.exp - binary_point, false, dest, result);
        ame_number_integer_flags(result, status);
        return;
    }
    if (a->bits == AME_VALUE_BITS || b->bits == AME_VALUE_BITS) {
        FloatParts256 left256 = parts128_to_parts256(&left);
        FloatParts256 right256 = parts128_to_parts256(&right);
        FloatParts256 addend256 = parts128_to_parts256(&addend);
        FloatParts256 result256 = parts256_muladd(
            &left256, &right256, &addend256,
            negate ? float_muladd_negate_product : 0, status);

        value = parts256_to_parts128(&result256);
    } else {
        value = parts128_muladd(&left, &right, &addend,
                                negate ? float_muladd_negate_product : 0,
                                status);
    }
    *result = ame_number_finish(dest, value, status);
    result->fflags = riscv_ame_fp_status_flags(status);
}

static int ame_fp_scale(const AmeDTypeDesc *dtype, AmeValue exponent,
                         bool negate);

static void ame_exec_number_scale(AmeOperation op, const AmeDTypeDesc *dest,
                                   const AmeDTypeDesc *data,
                                   const AmeValue *data_value,
                                   const AmeDTypeDesc *exponent,
                                   const AmeValue *exponent_value,
                                   const AmeValue *current,
                                   AmeExecResult *result, float_status *status)
{
    FloatParts128 value = ame_number_unpack(data, *data_value, status);
    FloatParts128 addend = current ? ame_number_unpack(dest, *current, status) :
                          (FloatParts128) { .cls = float_class_zero };
    int scale = ame_fp_scale(exponent, *exponent_value,
                             op == AME_OP_RDEXP || op == AME_OP_RDEXPACC);
    const int binary_point = AME_VALUE_BITS - 1;

    if (dest->class == AME_DTYPE_CLASS_INTEGER && value.cls < float_class_inf &&
        addend.cls < float_class_inf) {
        AmeIntProduct term = ame_number_term(&value);
        AmeIntProduct previous = ame_number_term(&addend);

        riscv_ame_int_scaled_sum(
            &term, sadd32_saturate(value.exp - binary_point, scale),
            &previous, addend.exp - binary_point, false, dest, result);
        ame_number_integer_flags(result, status);
        return;
    }
    value = parts128_scalbn(&value, scale, status);
    if (current) {
        if (data->bits == AME_VALUE_BITS) {
            FloatParts256 value256 = parts128_to_parts256(&value);
            FloatParts256 addend256 = parts128_to_parts256(&addend);
            FloatParts256 result256 = parts256_addsub(
                &value256, &addend256, status, false);

            value = parts256_to_parts128(&result256);
        } else {
            value = parts128_addsub(&value, &addend, status, false);
        }
    }
    *result = ame_number_finish(dest, value, status);
    result->fflags = riscv_ame_fp_status_flags(status);
}

void riscv_ame_exec_fp_convert(const AmeDTypeDesc *dest_dtype,
                               const AmeDTypeDesc *src_dtype,
                               const AmeValue *source, AmeExecResult *result,
                               float_status *status)
{
    AmeExecResult executed = { 0 };

    if (dest_dtype->class == AME_DTYPE_CLASS_INTEGER) {
        executed = ame_fp_parts_to_int(
            dest_dtype, ame_fp_unpack(src_dtype, source->lo, status), status);
    } else if (src_dtype->class == AME_DTYPE_CLASS_INTEGER) {
        executed.value = ame_int_to_fp(dest_dtype, src_dtype, *source,
                                       status);
    } else {
        if (ame_fp_is_nan(src_dtype, *source)) {
            if (ame_fp_is_snan(src_dtype, *source, status)) {
                float_raise(float_flag_invalid, status);
            }
            executed.value = ame_fp_default_nan(dest_dtype, status);
        } else {
            FloatParts64 parts = ame_fp_unpack(src_dtype, source->lo,
                                               status);

            executed.value.lo = ame_fp_pack(dest_dtype, parts, status);
        }
    }

    executed.fflags = riscv_ame_fp_status_flags(status);
    *result = executed;
}

static uint64_t ame_fp_abs(const AmeDTypeDesc *dtype, uint64_t value)
{
    return value & MAKE_64BIT_MASK(0, dtype->bits - 1);
}

static uint64_t ame_fp_negate(const AmeDTypeDesc *dtype, uint64_t value)
{
    return value ^ BIT_ULL(dtype->bits - 1);
}

static const FloatFmt *ame_fp_format_params(const AmeDTypeDesc *dtype)
{
    switch (dtype->fp.format) {
    case AME_FP_FORMAT_FP16:
        return &float16_params;
    case AME_FP_FORMAT_BF16:
        return &bfloat16_params;
    case AME_FP_FORMAT_FP32:
        return &float32_params;
    case AME_FP_FORMAT_FP64:
        return &float64_params;
    case AME_FP_FORMAT_OCP_E4M3:
        return &float8_e4m3_params;
    case AME_FP_FORMAT_OCP_E5M2:
        return &float8_e5m2_params;
    default:
        g_assert_not_reached();
    }
}

static uint64_t ame_fp_one(const AmeDTypeDesc *dtype)
{
    switch (dtype->fp.format) {
    case AME_FP_FORMAT_FP16:
        return UINT16_C(0x3c00);
    case AME_FP_FORMAT_BF16:
        return UINT16_C(0x3f80);
    case AME_FP_FORMAT_FP32:
        return UINT32_C(0x3f800000);
    case AME_FP_FORMAT_FP64:
        return UINT64_C(0x3ff0000000000000);
    case AME_FP_FORMAT_OCP_E4M3:
        return UINT8_C(0x38);
    case AME_FP_FORMAT_OCP_E5M2:
        return UINT8_C(0x3c);
    default:
        g_assert_not_reached();
    }
}

static uint64_t ame_fp_exponent_mask(const AmeDTypeDesc *dtype)
{
    switch (dtype->fp.format) {
    case AME_FP_FORMAT_FP16:
        return UINT16_C(0x7c00);
    case AME_FP_FORMAT_BF16:
        return UINT16_C(0x7f80);
    case AME_FP_FORMAT_FP32:
        return UINT32_C(0x7f800000);
    case AME_FP_FORMAT_FP64:
        return UINT64_C(0x7ff0000000000000);
    case AME_FP_FORMAT_OCP_E4M3:
        return UINT8_C(0x78);
    case AME_FP_FORMAT_OCP_E5M2:
        return UINT8_C(0x7c);
    default:
        g_assert_not_reached();
    }
}

static bool ame_fp_is_finite_raw(const AmeDTypeDesc *dtype, uint64_t value)
{
    AmeValue input = { .lo = value };

    if (ame_fp_is_nan(dtype, input)) {
        return false;
    }
    return !riscv_ame_fp_format_desc(dtype->fp.format)->has_inf ||
           (value & ame_fp_abs(dtype, UINT64_MAX)) !=
               ame_fp_exponent_mask(dtype);
}

static float128 ame_fp_to_float128(const AmeDTypeDesc *dtype, uint64_t source,
                                   float_status *status)
{
    AmeValue input = { .lo = source };
    float64 value = ame_fp_to_float64(dtype, input, status);

    return float64_to_float128(value, status);
}

static uint64_t ame_fp_from_float128(const AmeDTypeDesc *dtype,
                                     float128 source, float_status *status)
{
    if (dtype->fp.format == AME_FP_FORMAT_FP64) {
        return float64_val(float128_to_float64(source, status));
    }
    if (dtype->fp.format == AME_FP_FORMAT_FP32) {
        return float32_val(float128_to_float32(source, status));
    }
    if (dtype->fp.format == AME_FP_FORMAT_BF16) {
        return float128_to_bfloat16(source, status);
    }
    if (dtype->fp.format == AME_FP_FORMAT_FP16) {
        return float16_val(float128_to_float16(source, status));
    }
    if (dtype->fp.format == AME_FP_FORMAT_OCP_E4M3) {
        uint64_t result = float128_to_float8_e4m3(source, true, status);

        if ((source.high & UINT64_C(0x7fff000000000000)) ==
            UINT64_C(0x7fff000000000000) &&
            (source.high & UINT64_C(0x0000ffffffffffff)) == 0 &&
            source.low == 0) {
            float_raise(float_flag_overflow | float_flag_inexact, status);
        }
        return result;
    }
    g_assert(dtype->fp.format == AME_FP_FORMAT_OCP_E5M2);
    return float128_to_float8_e5m2(source, false, status);
}

static float128 ame_fp_log2_float128(float64 source, float_status *status)
{
    const float128 one = make_float128(UINT64_C(0x3fff000000000000), 0);
    const float128 two = make_float128(UINT64_C(0x4000000000000000), 0);
    const float128 invln2 = make_float128(UINT64_C(0x3fff71547652b82f),
                                          UINT64_C(0xe1777d0ffda0d23a));
    uint64_t raw = float64_val(source);
    uint64_t fraction = raw & UINT64_C(0x000fffffffffffff);
    uint64_t exponent_field = extract64(raw, 52, 11);
    FloatRoundMode saved_rounding;
    float128 input, mantissa, z, square, term, sum, result;
    int exponent, divisor;

    if (exponent_field == 0x7ff) {
        if (fraction) {
            return float64_to_float128(float64_add(source, source, status),
                                       status);
        }
        if (!(raw >> 63)) {
            return make_float128(UINT64_C(0x7fff000000000000), 0);
        }
        float_raise(float_flag_invalid, status);
        return make_float128(UINT64_C(0x7fff800000000000), 0);
    }
    if ((raw << 1) == 0) {
        float_raise(float_flag_divbyzero, status);
        return make_float128(UINT64_C(0xffff000000000000), 0);
    }
    if (raw >> 63) {
        float_raise(float_flag_invalid, status);
        return make_float128(UINT64_C(0x7fff800000000000), 0);
    }
    if (raw == UINT64_C(0x3ff0000000000000)) {
        return float128_zero;
    }

    if (exponent_field == 0) {
        int shift = clz64(fraction) - 11;

        exponent = -1022 - shift;
    } else {
        exponent = exponent_field - 1023;
    }
    input = float64_to_float128(source, status);
    mantissa = float128_scalbn(input, -exponent, status);

    saved_rounding = get_float_rounding_mode(status);
    set_float_rounding_mode(float_round_nearest_even, status);
    z = float128_div(float128_sub(mantissa, one, status),
                     float128_add(mantissa, one, status), status);
    square = float128_mul(z, z, status);
    term = sum = z;
    for (divisor = 3; divisor <= 81; divisor += 2) {
        term = float128_mul(term, square, status);
        sum = float128_add(sum,
                          float128_div(term,
                              int32_to_float128(divisor, status), status),
                          status);
    }
    result = float128_mul(float128_mul(two, sum, status), invln2, status);
    result = float128_add(int32_to_float128(exponent, status), result,
                          status);
    set_float_rounding_mode(saved_rounding, status);
    return result;
}

static uint64_t ame_fp_log2_raw(const AmeDTypeDesc *dest_dtype,
                                const AmeDTypeDesc *source_dtype,
                                uint64_t source, float_status *status)
{
    AmeValue input = { .lo = source };
    float64 value = ame_fp_to_float64(source_dtype, input, status);

    return ame_fp_from_float128(dest_dtype,
                               ame_fp_log2_float128(value, status), status);
}

static float128 ame_fp_exp2_finite_float128(float128 input,
                                            float_status *status)
{
    const float128 one = make_float128(UINT64_C(0x3fff000000000000), 0);
    const float128 ln2 = make_float128(UINT64_C(0x3ffe62e42fefa39e),
                                       UINT64_C(0xf35793c7673007e6));
    FloatRoundMode saved_rounding;
    float128 integral, remainder, term, sum, y;
    int32_t exponent;
    int i;

    saved_rounding = get_float_rounding_mode(status);
    set_float_rounding_mode(float_round_nearest_even, status);
    integral = float128_round_to_int(input, status);
    exponent = float128_to_int32(integral, status);
    remainder = float128_sub(input, integral, status);

    if ((remainder.high << 1) == 0 && remainder.low == 0) {
        set_float_rounding_mode(saved_rounding, status);
        return float128_scalbn(one, exponent, status);
    }
    y = float128_mul(remainder, ln2, status);
    term = sum = one;
    for (i = 1; i <= 32; i++) {
        term = float128_mul(term, y, status);
        term = float128_div(term, int32_to_float128(i, status), status);
        sum = float128_add(sum, term, status);
    }
    sum = float128_scalbn(sum, exponent, status);
    set_float_rounding_mode(saved_rounding, status);
    return sum;
}

static float128 ame_fp_exp2_float128(float64 source, float_status *status)
{
    const float64 positive_limit = make_float64(UINT64_C(0x4090000000000000));
    const float64 negative_limit = make_float64(UINT64_C(0xc090cc0000000000));
    const float128 one = make_float128(UINT64_C(0x3fff000000000000), 0);
    uint64_t raw = float64_val(source);

    if ((raw & UINT64_C(0x7ff0000000000000)) ==
        UINT64_C(0x7ff0000000000000)) {
        if (raw & UINT64_C(0x000fffffffffffff)) {
            return float64_to_float128(float64_add(source, source, status),
                                       status);
        }
        return raw >> 63 ? float128_zero :
               make_float128(UINT64_C(0x7fff000000000000), 0);
    }
    if ((raw << 1) == 0) {
        return one;
    }
    if (float64_compare_quiet(source, positive_limit, status) >=
        float_relation_equal) {
        return float128_scalbn(one, 2000, status);
    }
    if (float64_compare_quiet(source, negative_limit, status) <
        float_relation_equal) {
        return float128_scalbn(one, -2000, status);
    }

    return ame_fp_exp2_finite_float128(float64_to_float128(source, status),
                                       status);
}

static uint64_t ame_fp_exp2_raw(const AmeDTypeDesc *dest_dtype,
                                const AmeDTypeDesc *source_dtype,
                                uint64_t source,
                                float_status *status)
{
    AmeValue input = { .lo = source };
    float64 value = ame_fp_to_float64(source_dtype, input, status);
    float_status work_status = *status;
    float128 result;

    /*
     * Tiny Taylor terms do not make a result near one underflow. The
     * kernel bounds finite results to 2^[-2000,2000], normal in binary128
     * and wider than every supported D; only D's packing raises OF/UF.
     */
    set_float_exception_flags(0, &work_status);
    result = ame_fp_exp2_float128(value, &work_status);
    float_raise(get_float_exception_flags(&work_status) &
                (float_flag_invalid | float_flag_inexact), status);
    return ame_fp_from_float128(dest_dtype, result, status);
}

static float128 ame_fp_tanh_float128(float64 source, float_status *status)
{
    const float128 one = make_float128(UINT64_C(0x3fff000000000000), 0);
    const float128 log2e = make_float128(UINT64_C(0x3fff71547652b82f),
                                         UINT64_C(0xe1777d0ffda0d23a));
    uint64_t raw = float64_val(source);
    bool negative = raw >> 63;
    uint64_t magnitude_raw = raw & UINT64_C(0x7fffffffffffffff);
    FloatRoundMode saved_rounding;
    float128 input, square, polynomial, result;

    if ((magnitude_raw & UINT64_C(0x7ff0000000000000)) ==
        UINT64_C(0x7ff0000000000000)) {
        if (raw & UINT64_C(0x000fffffffffffff)) {
            return float64_to_float128(float64_add(source, source, status),
                                       status);
        }
        return negative ?
               make_float128(UINT64_C(0xbfff000000000000), 0) : one;
    }
    if ((raw << 1) == 0) {
        return float64_to_float128(source, status);
    }
    float_raise(float_flag_inexact, status);
    if (magnitude_raw >= UINT64_C(0x4034000000000000)) {
        return negative ?
               make_float128(UINT64_C(0xbfff000000000000), 0) : one;
    }

    saved_rounding = get_float_rounding_mode(status);
    set_float_rounding_mode(float_round_nearest_even, status);
    input = float64_to_float128(source, status);
    if (magnitude_raw < UINT64_C(0x3eb0000000000000)) {
        square = float128_mul(input, input, status);
        polynomial = float128_div(int32_to_float128(-17, status),
                                  int32_to_float128(315, status), status);
        polynomial = float128_add(
            float128_div(int32_to_float128(2, status),
                         int32_to_float128(15, status), status),
            float128_mul(square, polynomial, status), status);
        polynomial = float128_add(
            float128_div(int32_to_float128(-1, status),
                         int32_to_float128(3, status), status),
            float128_mul(square, polynomial, status), status);
        result = float128_add(
            input,
            float128_mul(float128_mul(input, square, status), polynomial,
                         status),
            status);
        set_float_rounding_mode(saved_rounding, status);
        return result;
    }

    input.high &= ~BIT_ULL(63);
    result = float128_mul(input, float128_scalbn(log2e, 1, status), status);
    result = ame_fp_exp2_finite_float128(result, status);
    result = float128_div(float128_sub(result, one, status),
                          float128_add(result, one, status), status);
    if (negative) {
        result.high ^= BIT_ULL(63);
    }
    set_float_rounding_mode(saved_rounding, status);
    return result;
}

static uint64_t ame_fp_tanh_raw(const AmeDTypeDesc *dest_dtype,
                                const AmeDTypeDesc *source_dtype,
                                uint64_t source,
                                float_status *status)
{
    AmeValue input = { .lo = source };
    FloatExceptionFlags accrued = get_float_exception_flags(status);
    float64 value;
    uint64_t result;

    /* Tininess concerns this element, not inexactness from an earlier one. */
    set_float_exception_flags(0, status);
    value = ame_fp_to_float64(source_dtype, input, status);
    result = ame_fp_from_float128(dest_dtype,
                                 ame_fp_tanh_float128(value, status), status);
    if (get_float_exception_flags(status) & float_flag_inexact) {
        if ((result & ame_fp_exponent_mask(dest_dtype)) == 0) {
            float_raise(float_flag_underflow, status);
        }
    }
    float_raise(accrued, status);
    return result;
}

static float64 ame_fp_sincos_float64(AmeOperation op, float64 source,
                                     float_status *status)
{
    static const uint64_t sin_coefficient_bits[] = {
        UINT64_C(0xbfc5555555555549),
        UINT64_C(0x3f8111111110f8a6),
        UINT64_C(0xbf2a01a019c161d5),
        UINT64_C(0x3ec71de357b1fe7d),
        UINT64_C(0xbe5ae5e68a2b9ceb),
        UINT64_C(0x3de5d93a5acfd57c),
    };
    static const uint64_t cos_coefficient_bits[] = {
        UINT64_C(0x3fa555555555554c),
        UINT64_C(0xbf56c16c16c15177),
        UINT64_C(0x3efa01a019cb1590),
        UINT64_C(0xbe927e4f809c52ad),
        UINT64_C(0x3e21ee9ebdb4b1c4),
        UINT64_C(0xbda8fae9be8838d4),
    };
    const float64 invpio2 = make_float64(UINT64_C(0x3fe45f306dc9c883));
    const float64 pio2_hi = make_float64(UINT64_C(0x3ff921fb54400000));
    const float64 pio2_lo = make_float64(UINT64_C(0x3dd0b4611a626331));
    const float64 half = make_float64(UINT64_C(0x3fe0000000000000));
    uint64_t raw = float64_val(source);
    FloatRoundMode saved_rounding;
    float64 multiple, r, square, sin_result, cos_result, polynomial;
    int32_t quadrant;
    size_t i;

    if ((raw & UINT64_C(0x7ff0000000000000)) ==
        UINT64_C(0x7ff0000000000000)) {
        if (raw & UINT64_C(0x000fffffffffffff)) {
            return float64_add(source, source, status);
        }
        return float64_sub(source, source, status);
    }
    if ((raw << 1) == 0) {
        return op == AME_OP_SIN ? source : float64_one;
    }

    saved_rounding = get_float_rounding_mode(status);
    set_float_rounding_mode(float_round_nearest_even, status);
    multiple = float64_round_to_int(float64_mul(source, invpio2, status),
                                    status);
    quadrant = float64_to_int32(multiple, status);
    r = float64_sub(source, float64_mul(multiple, pio2_hi, status), status);
    r = float64_sub(r, float64_mul(multiple, pio2_lo, status), status);
    set_float_rounding_mode(saved_rounding, status);

    square = float64_mul(r, r, status);
    i = ARRAY_SIZE(sin_coefficient_bits) - 1;
    polynomial = make_float64(sin_coefficient_bits[i]);
    while (i-- > 0) {
        polynomial = float64_add(
            make_float64(sin_coefficient_bits[i]),
            float64_mul(square, polynomial, status), status);
    }
    sin_result = float64_add(
        r, float64_mul(float64_mul(r, square, status), polynomial, status),
        status);

    i = ARRAY_SIZE(cos_coefficient_bits) - 1;
    polynomial = make_float64(cos_coefficient_bits[i]);
    while (i-- > 0) {
        polynomial = float64_add(
            make_float64(cos_coefficient_bits[i]),
            float64_mul(square, polynomial, status), status);
    }
    cos_result = float64_add(
        float64_sub(float64_one, float64_mul(half, square, status), status),
        float64_mul(float64_mul(square, square, status), polynomial, status),
        status);

    float_raise(float_flag_inexact, status);
    switch (quadrant & 3) {
    case 0:
        return op == AME_OP_SIN ? sin_result : cos_result;
    case 1:
        return op == AME_OP_SIN ? cos_result :
               make_float64(float64_val(sin_result) ^ BIT_ULL(63));
    case 2:
        return op == AME_OP_SIN ?
               make_float64(float64_val(sin_result) ^ BIT_ULL(63)) :
               make_float64(float64_val(cos_result) ^ BIT_ULL(63));
    case 3:
        return op == AME_OP_SIN ?
               make_float64(float64_val(cos_result) ^ BIT_ULL(63)) :
               sin_result;
    default:
        g_assert_not_reached();
    }
}

/* Return x - n * pi/2 and n modulo four, without converting n to an integer. */
static float128 ame_fp_reduce_pio2(float64 source, unsigned *quadrant,
                                   float_status *status)
{
    /*
     * floor(2/pi * 2^1280), least-significant limb first. Even for the
     * largest binary64 input, truncating this table changes x * 2/pi by
     * less than 2^-256. For non-small binary64 inputs the distance from
     * an integer or half-integer exceeds 2^-81, leaving ample precision
     * for the binary128 residual as well as unambiguous quadrant bits.
     */
    static const uint64_t two_over_pi[] = {
        UINT64_C(0xf0cfbc209af4361d), UINT64_C(0x56033046fc7b6bab),
        UINT64_C(0x6bfb5fb11f8d5d08), UINT64_C(0x3d0739f78a5292ea),
        UINT64_C(0x7527bac7ebe5f17b), UINT64_C(0x4f463f669e5fea2d),
        UINT64_C(0x6d367ecf27cb09b7), UINT64_C(0xef2f118b5a0a6d1f),
        UINT64_C(0x1ff897ffde05980f), UINT64_C(0x9c845f8bbdf9283b),
        UINT64_C(0x3991d639835339f4), UINT64_C(0xe99c7026b45f7e41),
        UINT64_C(0xe88235f52ebb4484), UINT64_C(0xfe1deb1cb129a73e),
        UINT64_C(0x06492eea09d1921c), UINT64_C(0xb7246e3a424dd2e0),
        UINT64_C(0xfe5163abdebbc561), UINT64_C(0xdb6295993c439041),
        UINT64_C(0xfc2757d1f534ddc0), UINT64_C(0xa2f9836e4e441529),
    };
    const float128 pio2 = make_float128(UINT64_C(0x3fff921fb54442d1),
                                        UINT64_C(0x8469898cc51701b8));
    uint64_t raw = float64_val(source);
    uint64_t magnitude = raw & ~BIT_ULL(63);
    uint64_t product[ARRAY_SIZE(two_over_pi) + 1];
    uint64_t significand, carry = 0;
    FloatParts128 fraction = { .cls = float_class_normal };
    unsigned point, word, offset;
    bool complement;
    int top;

    if (magnitude <= UINT64_C(0x3fe921fb54442d18)) {
        *quadrant = 0;
        return float64_to_float128(source, status);
    }

    /* x = significand * 2^(exponent - 52), with a 53-bit significand. */
    significand = (magnitude & MAKE_64BIT_MASK(0, 52)) | BIT_ULL(52);
    point = ARRAY_SIZE(two_over_pi) * 64 + 52 -
            ((int)extract64(magnitude, 52, 11) - 1023);
    /* |x| > pi/4 bounds point to [309, 1333], within the 21-limb product. */
    for (unsigned i = 0; i < ARRAY_SIZE(two_over_pi); i++) {
        uint64_t high, low;

        mulu64(&low, &high, significand, two_over_pi[i]);
        product[i] = low + carry;
        carry = high + (product[i] < low);
    }
    product[ARRAY_SIZE(two_over_pi)] = carry;
    *quadrant = ((product[point / 64] >> (point % 64)) & 1) |
                (((product[(point + 1) / 64] >> ((point + 1) % 64)) & 1)
                 << 1);
    complement = (product[(point - 1) / 64] >> ((point - 1) % 64)) & 1;
    *quadrant = (*quadrant + complement) & 3;

    /*
     * Keep the signed fraction in [-1/2, 1/2]. Subtract from one in the
     * integer domain: rounding the fraction first would lose small
     * residuals near integer multiples of pi/2.
     */
    word = point / 64;
    offset = point % 64;
    if (complement) {
        carry = 1;
        for (unsigned i = 0; i <= word; i++) {
            product[i] = ~product[i] + carry;
            carry &= product[i] == 0;
        }
    }
    product[word] = offset ? product[word] & MAKE_64BIT_MASK(0, offset) : 0;
    /* The table is odd; the product has at most 52 trailing zero bits. */
    top = word;
    while (top > 0 && product[top] == 0) {
        top--;
    }
    g_assert(product[top] != 0);
    top = top * 64 + 63 - clz64(product[top]);
    fraction.exp = top - (int)point;
    fraction.sign = complement ^ (raw >> 63);
    for (unsigned i = 0; i < 128 && (int)i <= top; i++) {
        unsigned bit = top - i;
        uint64_t value = (product[bit / 64] >> (bit % 64)) & 1;

        if (i < 64) {
            fraction.frac_hi |= value << (63 - i);
        } else {
            fraction.frac_lo |= value << (127 - i);
        }
    }
    for (int i = 0; i < top - 127; i += 64) {
        unsigned bits = MIN(64, top - 127 - i);

        if (product[i / 64] & MAKE_64BIT_MASK(0, bits)) {
            fraction.frac_lo |= 1;
            break;
        }
    }
    if (raw >> 63) {
        *quadrant = -*quadrant & 3;
    }
    return float128_mul(float128_round_pack_canonical(&fraction, status),
                         pio2, status);
}

static float128 ame_fp_sincos_float128(AmeOperation op, float64 source,
                                       float_status *status)
{
    const float128 one = make_float128(UINT64_C(0x3fff000000000000), 0);
    uint64_t raw = float64_val(source);
    unsigned quadrant;
    float128 r, square, term, sum;
    bool cosine;

    if ((raw & UINT64_C(0x7ff0000000000000)) ==
        UINT64_C(0x7ff0000000000000)) {
        return float64_to_float128(float64_sub(source, source, status),
                                    status);
    }
    if ((raw << 1) == 0) {
        return op == AME_OP_SIN ? float64_to_float128(source, status) : one;
    }
    if ((raw & ~BIT_ULL(63)) < UINT64_C(0x3e10000000000000)) {
        /*
         * Below 2^-30, sin(x) is just inside x and cos(x) just below one
         * at every supported precision. Preserve that side of the
         * rounding boundary; the exact input must not mask inexactness.
         */
        sum = op == AME_OP_SIN ? float64_to_float128(source, status) : one;
        if (sum.low-- == 0) {
            sum.high--;
        }
        float_raise(float_flag_inexact, status);
        return sum;
    }
    r = ame_fp_reduce_pio2(source, &quadrant, status);
    cosine = (op == AME_OP_COS) ^ (quadrant & 1);
    square = float128_mul(r, r, status);
    square.high ^= BIT_ULL(63);
    term = sum = cosine ? one : r;
    /* On [-pi/4, pi/4], the first omitted term is below 2^-181. */
    for (unsigned i = 1; i <= 20; i++) {
        unsigned factor = (2 * i) * (cosine ? 2 * i - 1 : 2 * i + 1);

        term = float128_div(float128_mul(term, square, status),
                            int32_to_float128(factor, status), status);
        sum = float128_add(sum, term, status);
    }
    if (((quadrant + (op == AME_OP_COS)) & 2) != 0) {
        sum.high ^= BIT_ULL(63);
    }
    float_raise(float_flag_inexact, status);
    return sum;
}

static uint64_t ame_fp_sincos_raw(AmeOperation op,
                                  const AmeDTypeDesc *dest_dtype,
                                  const AmeDTypeDesc *source_dtype,
                                  uint64_t source,
                                  float_status *status)
{
    AmeValue input = { .lo = source };
    float64 value = ame_fp_to_float64(source_dtype, input, status);
    uint64_t result;

    if (dest_dtype->fp.format == source_dtype->fp.format &&
        (dest_dtype->fp.format == AME_FP_FORMAT_FP16 ||
         dest_dtype->class == AME_DTYPE_CLASS_OCP)) {
        value = ame_fp_sincos_float64(op, value, status);
        result = ame_fp_from_float64(dest_dtype, value, status).lo;
    } else {
        float_status work_status = *status;
        float128 wide;

        set_float_rounding_mode(float_round_nearest_even, &work_status);
        set_float_exception_flags(0, &work_status);
        wide = ame_fp_sincos_float128(op, value, &work_status);
        float_raise(get_float_exception_flags(&work_status) &
                    (float_flag_invalid | float_flag_inexact), status);
        result = ame_fp_from_float128(dest_dtype, wide, status);
    }
    if (ame_fp_abs(source_dtype, source) != 0 &&
        ame_fp_is_finite_raw(source_dtype, source) &&
        (result & ame_fp_exponent_mask(dest_dtype)) == 0) {
        float_raise(float_flag_underflow, status);
    }
    return result;
}

static uint64_t ame_fp_rsqrt_raw(const AmeDTypeDesc *dest_dtype,
                                 const AmeDTypeDesc *source_dtype,
                                 uint64_t source,
                                 float_status *status)
{
    const float128 one = make_float128(UINT64_C(0x3fff000000000000), 0);
    FloatRoundMode saved_rounding = get_float_rounding_mode(status);
    float128 value = ame_fp_to_float128(source_dtype, source, status);

    set_float_rounding_mode(float_round_nearest_even, status);
    value = float128_div(one, float128_sqrt(value, status), status);
    set_float_rounding_mode(saved_rounding, status);
    return ame_fp_from_float128(dest_dtype, value, status);
}

static uint64_t ame_fp_parts_unary_raw(AmeOperation op,
                                       const AmeDTypeDesc *dtype,
                                       uint64_t source,
                                       float_status *status)
{
    FloatParts64 input;
    FloatParts64 result;

    if (op == AME_OP_SQRT) {
        if (dtype->fp.format == AME_FP_FORMAT_OCP_E4M3) {
            return float8_e4m3_sqrt(source, status);
        }

        g_assert(dtype->fp.format == AME_FP_FORMAT_OCP_E5M2);
        if ((source & UINT8_C(0x7c)) == UINT8_C(0x7c) &&
            (source & UINT8_C(0x03)) != 0) {
            source = (source & UINT8_C(0x80)) | UINT8_C(0x7e);
        }
        return float8_e5m2_sqrt(source, status);
    }

    input = ame_fp_unpack(dtype, source, status);
    g_assert(op == AME_OP_REC);
    FloatParts64 one = ame_fp_unpack(dtype, ame_fp_one(dtype), status);

    result = parts64_div(&one, &input, status);
    return ame_fp_pack(dtype, result, status);
}

static uint64_t ame_fp_unary_raw(AmeOperation op,
                                 const AmeDTypeDesc *dest_dtype,
                                 const AmeDTypeDesc *source_dtype,
                                 uint64_t source,
                                 float_status *status)
{
    FloatRoundMode saved_rounding = get_float_rounding_mode(status);
    uint64_t value;

    if (op == AME_OP_FRINTM || op == AME_OP_FRINTN ||
        op == AME_OP_FRINTP || op == AME_OP_FRINTZ) {
        FloatRoundMode rounding =
            op == AME_OP_FRINTM ? float_round_down :
            op == AME_OP_FRINTN ? float_round_nearest_even :
            op == AME_OP_FRINTP ? float_round_up : float_round_to_zero;
        FloatExceptionFlags old_nx =
            get_float_exception_flags(status) & float_flag_inexact;
        FloatParts64 input = ame_fp_unpack(source_dtype, source, status);
        FloatParts64 result;

        set_float_rounding_mode(rounding, status);
        result = parts64_round_to_int(&input, rounding, 0, status,
                                      ame_fp_format_params(source_dtype));
        /* Only representing the integral result in D may add NX. */
        set_float_exception_flags(
            (get_float_exception_flags(status) & ~float_flag_inexact) | old_nx,
            status);
        value = ame_fp_pack(dest_dtype, result, status);
        set_float_rounding_mode(saved_rounding, status);
        return value;
    }

    if (op == AME_OP_LOG2) {
        return ame_fp_log2_raw(dest_dtype, source_dtype, source, status);
    }
    if (op == AME_OP_EXP2) {
        return ame_fp_exp2_raw(dest_dtype, source_dtype, source, status);
    }
    if (op == AME_OP_TANH) {
        return ame_fp_tanh_raw(dest_dtype, source_dtype, source, status);
    }
    if (op == AME_OP_SIN || op == AME_OP_COS) {
        return ame_fp_sincos_raw(op, dest_dtype, source_dtype, source, status);
    }
    if (op == AME_OP_RSQRT) {
        return ame_fp_rsqrt_raw(dest_dtype, source_dtype, source, status);
    }

    if (dest_dtype->fp.format != source_dtype->fp.format) {
        FloatParts64 input, result;

        if (op == AME_OP_SQRT) {
            float128 wide = ame_fp_to_float128(source_dtype, source, status);

            /*
             * Round-to-odd retains the information needed by the final
             * rounding, unlike a rounded-to-nearest intermediate. There
             * are 113 significant bits here and at most 53 in D; sqrt of
             * every finite nonzero source is normal in binary128. This
             * is an internal jamming step, independent of the guest RNO.
             */
            set_float_rounding_mode(float_round_to_odd, status);
            wide = float128_sqrt(wide, status);
            set_float_rounding_mode(saved_rounding, status);
            return ame_fp_from_float128(dest_dtype, wide, status);
        }

        input = ame_fp_unpack(source_dtype, source, status);
        g_assert(op == AME_OP_REC);
        FloatParts64 one = float64_unpack_canonical(float64_one, status);

        result = parts64_div(&one, &input, status);
        value = ame_fp_pack(dest_dtype, result, status);
        set_float_rounding_mode(saved_rounding, status);
        return value;
    }

    if (dest_dtype->class == AME_DTYPE_CLASS_OCP) {
        return ame_fp_parts_unary_raw(op, dest_dtype, source, status);
    }

    switch (dest_dtype->fp.format) {
    case AME_FP_FORMAT_FP16:
        switch (op) {
        case AME_OP_SQRT:
            value = float16_val(float16_sqrt(source, status));
            break;
        case AME_OP_REC:
            value = float16_val(float16_div(float16_one, source, status));
            break;
        default:
            g_assert_not_reached();
        }
        break;
    case AME_FP_FORMAT_BF16:
        switch (op) {
        case AME_OP_SQRT:
            value = bfloat16_sqrt(source, status);
            break;
        case AME_OP_REC:
            value = bfloat16_div(bfloat16_one, source, status);
            break;
        default:
            g_assert_not_reached();
        }
        break;
    case AME_FP_FORMAT_FP32:
        switch (op) {
        case AME_OP_SQRT:
            value = float32_val(float32_sqrt(make_float32(source), status));
            break;
        case AME_OP_REC:
            value = float32_val(float32_div(float32_one,
                                            make_float32(source), status));
            break;
        default:
            g_assert_not_reached();
        }
        break;
    case AME_FP_FORMAT_FP64:
        switch (op) {
        case AME_OP_SQRT:
            value = float64_val(float64_sqrt(make_float64(source), status));
            break;
        case AME_OP_REC:
            value = float64_val(float64_div(float64_one,
                                            make_float64(source), status));
            break;
        default:
            g_assert_not_reached();
        }
        break;
    default:
        g_assert_not_reached();
    }

    set_float_rounding_mode(saved_rounding, status);
    return value;
}

void riscv_ame_exec_fp_unary(AmeOperation op,
                             const AmeDTypeDesc *dest_dtype,
                             const AmeDTypeDesc *src_dtype,
                             const AmeValue *source, AmeExecResult *result,
                             float_status *status)
{
    AmeExecResult executed = { 0 };

    if (op == AME_OP_ABS) {
        FloatParts128 parts = ame_number_unpack(src_dtype, *source, status);

        parts.sign = false;
        if (dest_dtype->class == AME_DTYPE_CLASS_INTEGER &&
            parts.cls < float_class_inf) {
            AmeIntProduct term = ame_number_term(&parts);

            riscv_ame_int_scaled_sum(
                &term, parts.exp - (int)(AME_VALUE_BITS - 1), NULL, 0,
                false, dest_dtype, &executed);
            ame_number_integer_flags(&executed, status);
        } else {
            executed = ame_number_finish(dest_dtype, parts, status);
        }
    } else {
        executed.value.lo = ame_fp_unary_raw(op, dest_dtype, src_dtype,
                                             source->lo,
                                             status);
    }
    executed.fflags = riscv_ame_fp_status_flags(status);
    *result = executed;
}

static FloatRelation ame_fp_compare_raw(const AmeDTypeDesc *dtype,
                                        uint64_t src1, uint64_t src2,
                                        float_status *status);

static uint64_t ame_fp_half_raw(AmeOperation op,
                                const AmeDTypeDesc *dtype,
                                uint64_t src1, uint64_t src2,
                                float_status *status)
{
    int flags = op == AME_OP_HDIFF ? float_muladd_negate_product : 0;

    switch (dtype->fp.format) {
    case AME_FP_FORMAT_FP16:
        return float16_val(float16_muladd_scalbn(
            src1, float16_one, src2, -1, flags, status));
    case AME_FP_FORMAT_BF16:
    {
        FloatParts64 factor = bfloat16_unpack_canonical(src1, status);
        FloatParts64 one = bfloat16_unpack_canonical(bfloat16_one, status);
        FloatParts64 addend = bfloat16_unpack_canonical(src2, status);
        FloatParts64 value = parts64_muladd(&factor, &one, &addend,
                                            flags, status);

        value = parts64_scalbn(&value, -1, status);
        return bfloat16_round_pack_canonical(&value, status);
    }
    case AME_FP_FORMAT_FP32:
        return float32_val(float32_muladd_scalbn(
            make_float32(src1), float32_one, make_float32(src2),
            -1, flags, status));
    case AME_FP_FORMAT_FP64:
        return float64_val(float64_muladd_scalbn(
            make_float64(src1), float64_one, make_float64(src2),
            -1, flags, status));
    default:
        g_assert_not_reached();
    }
}

static uint64_t ame_fp_parts_binary_raw(AmeOperation op,
                                        const AmeDTypeDesc *dest_dtype,
                                        const AmeDTypeDesc *src1_dtype,
                                        uint64_t src1,
                                        const AmeDTypeDesc *src2_dtype,
                                        uint64_t src2,
                                        float_status *status)
{
    FloatParts64 left = ame_fp_unpack(src1_dtype, src1, status);
    FloatParts64 right = ame_fp_unpack(src2_dtype, src2, status);
    FloatParts64 value;

    switch (op) {
    case AME_OP_ADD:
    case AME_OP_MEAN:
        value = parts64_addsub(&left, &right, status, false);
        break;
    case AME_OP_SUB:
    case AME_OP_HDIFF:
        value = parts64_addsub(&right, &left, status, true);
        break;
    case AME_OP_ABSDIFF:
        value = parts64_addsub(&left, &right, status, true);
        value.sign = false;
        break;
    case AME_OP_MIN:
    case AME_OP_MAX:
        if (left.cls >= float_class_qnan ||
            right.cls >= float_class_qnan) {
            if (left.cls == float_class_snan ||
                right.cls == float_class_snan) {
                float_raise(float_flag_invalid, status);
            }
            value = parts64_default_nan(status);
        } else {
            FloatRelation relation = parts64_compare(
                &left, &right, status, true);

            if (relation == float_relation_equal &&
                left.cls == float_class_zero &&
                right.cls == float_class_zero) {
                value = left;
                value.sign = op == AME_OP_MIN ?
                             left.sign || right.sign :
                             left.sign && right.sign;
            } else if ((op == AME_OP_MIN &&
                        relation == float_relation_greater) ||
                       (op == AME_OP_MAX &&
                        relation == float_relation_less)) {
                value = right;
            } else {
                value = left;
            }
        }
        break;
    case AME_OP_MUL:
    case AME_OP_NMUL:
        if (op == AME_OP_NMUL) {
            left.sign = !left.sign;
        }
        value = parts64_mul(&left, &right, status);
        break;
    default:
        g_assert_not_reached();
    }
    if (op == AME_OP_HDIFF || op == AME_OP_MEAN) {
        value = parts64_scalbn(&value, -1, status);
    }
    return ame_fp_pack(dest_dtype, value, status);
}

static uint64_t ame_fp_binary_raw(AmeOperation op,
                                  const AmeDTypeDesc *dtype,
                                  uint64_t src1, uint64_t src2,
                                  float_status *status)
{
    uint64_t factor1 = op == AME_OP_NMUL ?
                       ame_fp_negate(dtype, src1) : src1;
    uint64_t value;

    if (op == AME_OP_LOG2SUB || op == AME_OP_SUBLOG2) {
        uint64_t logarithm = ame_fp_log2_raw(
            dtype, dtype, ame_fp_abs(dtype, src1), status);

        if (op == AME_OP_LOG2SUB) {
            return ame_fp_binary_raw(AME_OP_SUB, dtype, src2,
                                     logarithm, status);
        }
        return ame_fp_binary_raw(AME_OP_SUB, dtype, logarithm,
                                 src2, status);
    }

    if (dtype->class == AME_DTYPE_CLASS_OCP) {
        return ame_fp_parts_binary_raw(op, dtype, dtype, src1, dtype, src2,
                                       status);
    }

    if (op == AME_OP_HDIFF || op == AME_OP_MEAN) {
        return ame_fp_half_raw(op, dtype, src1, src2, status);
    }

    if (op == AME_OP_ABSDIFF &&
        ame_fp_compare_raw(dtype, src1, src2, status) ==
            float_relation_less) {
        uint64_t tmp = src1;

        src1 = src2;
        src2 = tmp;
    }

    switch (dtype->fp.format) {
    case AME_FP_FORMAT_FP16:
        if (op == AME_OP_ADD) {
            value = float16_val(float16_add(src1, src2, status));
        } else if (op == AME_OP_SUB) {
            value = float16_val(float16_sub(src2, src1, status));
        } else if (op == AME_OP_ABSDIFF) {
            value = float16_val(float16_sub(src1, src2, status));
        } else if (op == AME_OP_MIN || op == AME_OP_MAX) {
            value = float16_val(float16_minmax(src1, src2, status,
                op == AME_OP_MIN ? float_minmax_ismin : 0));
        } else {
            value = float16_val(float16_mul(factor1, src2, status));
        }
        break;
    case AME_FP_FORMAT_BF16:
        if (op == AME_OP_ADD) {
            value = bfloat16_add(src1, src2, status);
        } else if (op == AME_OP_SUB) {
            value = bfloat16_sub(src2, src1, status);
        } else if (op == AME_OP_ABSDIFF) {
            value = bfloat16_sub(src1, src2, status);
        } else if (op == AME_OP_MIN || op == AME_OP_MAX) {
            value = bfloat16_minmax(src1, src2, status,
                op == AME_OP_MIN ? float_minmax_ismin : 0);
        } else {
            value = bfloat16_mul(factor1, src2, status);
        }
        break;
    case AME_FP_FORMAT_FP32:
        if (op == AME_OP_ADD) {
            value = float32_val(float32_add(make_float32(src1),
                                            make_float32(src2), status));
        } else if (op == AME_OP_SUB) {
            value = float32_val(float32_sub(make_float32(src2),
                                            make_float32(src1), status));
        } else if (op == AME_OP_ABSDIFF) {
            value = float32_val(float32_sub(make_float32(src1),
                                            make_float32(src2), status));
        } else if (op == AME_OP_MIN || op == AME_OP_MAX) {
            value = float32_val(float32_minmax(make_float32(src1),
                make_float32(src2), status,
                op == AME_OP_MIN ? float_minmax_ismin : 0));
        } else {
            value = float32_val(float32_mul(make_float32(factor1),
                                            make_float32(src2), status));
        }
        break;
    case AME_FP_FORMAT_FP64:
        if (op == AME_OP_ADD) {
            value = float64_val(float64_add(make_float64(src1),
                                            make_float64(src2), status));
        } else if (op == AME_OP_SUB) {
            value = float64_val(float64_sub(make_float64(src2),
                                            make_float64(src1), status));
        } else if (op == AME_OP_ABSDIFF) {
            value = float64_val(float64_sub(make_float64(src1),
                                            make_float64(src2), status));
        } else if (op == AME_OP_MIN || op == AME_OP_MAX) {
            value = float64_val(float64_minmax(make_float64(src1),
                make_float64(src2), status,
                op == AME_OP_MIN ? float_minmax_ismin : 0));
        } else {
            value = float64_val(float64_mul(make_float64(factor1),
                                            make_float64(src2), status));
        }
        break;
    default:
        g_assert_not_reached();
    }

    if (op == AME_OP_ABSDIFF) {
        value = ame_fp_abs(dtype, value);
    }
    return value;
}

static int ame_fp_scale(const AmeDTypeDesc *dtype, AmeValue exponent,
                        bool negate)
{
    int64_t value;

    g_assert(dtype->class == AME_DTYPE_CLASS_INTEGER);
    if (dtype->bits == 128) {
        if (dtype->integer.is_signed && (int64_t)exponent.hi < 0) {
            if (exponent.hi == UINT64_MAX &&
                exponent.lo >= UINT64_MAX - INT_MAX) {
                value = (int64_t)exponent.lo;
            } else {
                return negate ? INT_MAX : INT_MIN;
            }
        } else {
            if (exponent.hi != 0 ||
                exponent.lo > (uint64_t)INT_MAX + negate) {
                return negate ? INT_MIN : INT_MAX;
            }
            value = exponent.lo;
        }
    } else {
        g_assert(dtype->bits <= 64);
        if (dtype->integer.is_signed) {
            value = ame_sign_extend64(exponent.lo, dtype->bits);
        } else if (exponent.lo > (uint64_t)INT_MAX + negate) {
            return negate ? INT_MIN : INT_MAX;
        } else {
            value = dtype->bits == 64 ? exponent.lo :
                    exponent.lo & MAKE_64BIT_MASK(0, dtype->bits);
        }
    }

    if (negate) {
        if (value <= -(int64_t)INT_MAX) {
            return INT_MAX;
        }
        if (value >= (int64_t)INT_MAX + 1) {
            return INT_MIN;
        }
        value = -value;
    }
    return value > INT_MAX ? INT_MAX : value < INT_MIN ? INT_MIN : value;
}

static uint64_t ame_fp_scalbn_raw(const AmeDTypeDesc *dtype, uint64_t source,
                                  int scale, float_status *status)
{
    if (dtype->class == AME_DTYPE_CLASS_OCP) {
        FloatParts64 parts = ame_fp_unpack(dtype, source, status);

        parts = parts64_scalbn(&parts, scale, status);
        return ame_fp_pack(dtype, parts, status);
    }

    switch (dtype->fp.format) {
    case AME_FP_FORMAT_FP16:
        return float16_val(float16_scalbn(source, scale, status));
    case AME_FP_FORMAT_BF16:
        return bfloat16_scalbn(source, scale, status);
    case AME_FP_FORMAT_FP32:
        return float32_val(float32_scalbn(make_float32(source), scale,
                                          status));
    case AME_FP_FORMAT_FP64:
        return float64_val(float64_scalbn(make_float64(source), scale,
                                          status));
    default:
        g_assert_not_reached();
    }
}

static FloatRelation ame_fp_compare_raw(const AmeDTypeDesc *dtype,
                                        uint64_t src1, uint64_t src2,
                                        float_status *status)
{
    switch (dtype->fp.format) {
    case AME_FP_FORMAT_FP16:
        return float16_compare_quiet(src1, src2, status);
    case AME_FP_FORMAT_BF16:
        return bfloat16_compare_quiet(src1, src2, status);
    case AME_FP_FORMAT_FP32:
        return float32_compare_quiet(make_float32(src1), make_float32(src2),
                                     status);
    case AME_FP_FORMAT_FP64:
        return float64_compare_quiet(make_float64(src1), make_float64(src2),
                                     status);
    case AME_FP_FORMAT_OCP_E4M3:
    case AME_FP_FORMAT_OCP_E5M2:
    {
        FloatParts64 left = ame_fp_unpack(dtype, src1, status);
        FloatParts64 right = ame_fp_unpack(dtype, src2, status);

        return parts64_compare(&left, &right, status, true);
    }
    default:
        g_assert_not_reached();
    }
}

void riscv_ame_fp_compare_zero(const AmeDTypeDesc *dtype,
                               const AmeValue *value,
                               bool *ordered, bool *negative,
                               float_status *status)
{
    FloatRelation relation;

    relation = ame_fp_compare_raw(dtype, value->lo, 0, status);
    *ordered = relation != float_relation_unordered;
    *negative = relation == float_relation_less;
}

void riscv_ame_exec_fp_binary(AmeOperation op,
                              const AmeDTypeDesc *dest_dtype,
                              const AmeDTypeDesc *src1_dtype,
                              const AmeValue *src1,
                              const AmeDTypeDesc *src2_dtype,
                              const AmeValue *src2, AmeExecResult *result,
                              float_status *status)
{
    AmeExecResult executed = { 0 };
    bool di = dest_dtype->class == AME_DTYPE_CLASS_INTEGER;
    bool ai = src1_dtype->class == AME_DTYPE_CLASS_INTEGER;
    bool bi = src2_dtype->class == AME_DTYPE_CLASS_INTEGER;

    switch (op) {
    case AME_OP_CMPGE:
    case AME_OP_CMPLT:
    {
        FloatRelation relation;
        float64 left, right;
        bool predicate;

        if (ai || bi) {
            ame_exec_number_binary(op, dest_dtype, src1_dtype, src1,
                                   src2_dtype, src2, result, status);
            return;
        }
        left = ame_fp_to_float64(src1_dtype, *src1, status);
        right = ame_fp_to_float64(src2_dtype, *src2, status);
        relation = float64_compare_quiet(left, right, status);
        predicate = relation != float_relation_unordered &&
            (op == AME_OP_CMPGE ? relation >= float_relation_equal :
                                    relation < float_relation_equal);
        if (predicate) {
            executed.value.lo = dest_dtype->bits >= AME_VALUE_WORD_BITS ?
                                UINT64_MAX :
                                MAKE_64BIT_MASK(0, dest_dtype->bits);
            executed.value.hi = dest_dtype->bits == AME_VALUE_BITS ?
                                UINT64_MAX : 0;
        }
        break;
    }
    case AME_OP_LDEXP:
    case AME_OP_RDEXP:
    {
        int scale;

        if (di || ai) {
            ame_exec_number_scale(op, dest_dtype, src1_dtype, src1,
                                  src2_dtype, src2, NULL, result, status);
            return;
        }
        scale = ame_fp_scale(src2_dtype, *src2, op == AME_OP_RDEXP);
        if (dest_dtype->fp.format == src1_dtype->fp.format) {
            executed.value.lo = ame_fp_scalbn_raw(dest_dtype, src1->lo,
                                                  scale, status);
        } else {
            FloatParts64 parts = ame_fp_unpack(src1_dtype, src1->lo, status);

            parts = parts64_scalbn(&parts, scale, status);
            executed.value.lo = ame_fp_pack(dest_dtype, parts, status);
        }
        break;
    }
    case AME_OP_LOG2SUB:
    case AME_OP_SUBLOG2:
        if (!bi && dest_dtype->fp.format == src1_dtype->fp.format &&
            dest_dtype->fp.format == src2_dtype->fp.format) {
            executed.value.lo = ame_fp_binary_raw(op, dest_dtype, src1->lo,
                                                  src2->lo, status);
        } else {
            AmeExecResult bias;
            uint64_t logarithm = ame_fp_log2_raw(
                dest_dtype, src1_dtype,
                ame_fp_abs(src1_dtype, src1->lo), status);

            riscv_ame_exec_fp_convert(dest_dtype, src2_dtype, src2,
                                      &bias, status);
            executed.value.lo = op == AME_OP_LOG2SUB ?
                ame_fp_binary_raw(AME_OP_SUB, dest_dtype, bias.value.lo,
                                  logarithm, status) :
                ame_fp_binary_raw(AME_OP_SUB, dest_dtype, logarithm,
                                  bias.value.lo, status);
        }
        break;
    case AME_OP_ABSDIFF:
    case AME_OP_ADD:
    case AME_OP_HDIFF:
    case AME_OP_MAX:
    case AME_OP_MEAN:
    case AME_OP_MIN:
    case AME_OP_MUL:
    case AME_OP_NMUL:
    case AME_OP_SUB:
        if (di || ai || bi) {
            ame_exec_number_binary(op, dest_dtype, src1_dtype, src1,
                                   src2_dtype, src2, result, status);
            return;
        }
        if (dest_dtype->fp.format == src1_dtype->fp.format &&
            dest_dtype->fp.format == src2_dtype->fp.format) {
            executed.value.lo = ame_fp_binary_raw(op, dest_dtype, src1->lo,
                                                  src2->lo, status);
        } else {
            executed.value.lo = ame_fp_parts_binary_raw(
                op, dest_dtype, src1_dtype, src1->lo, src2_dtype, src2->lo,
                status);
        }
        break;
    default:
        g_assert_not_reached();
    }
    executed.fflags = riscv_ame_fp_status_flags(status);
    *result = executed;
}

static uint64_t ame_fp_muladd_raw(const AmeDTypeDesc *dest_dtype,
                                 const AmeDTypeDesc *factor1_dtype,
                                 uint64_t factor1,
                                 const AmeDTypeDesc *factor2_dtype,
                                 uint64_t factor2,
                                 const AmeDTypeDesc *addend_dtype,
                                 uint64_t addend, int flags,
                                 float_status *status)
{
    FloatParts64 left, right, current, value;
    bool same_format = dest_dtype->fp.format == factor1_dtype->fp.format &&
                       dest_dtype->fp.format == factor2_dtype->fp.format &&
                       dest_dtype->fp.format == addend_dtype->fp.format;

    if (same_format && dest_dtype->class != AME_DTYPE_CLASS_OCP) {
        switch (dest_dtype->fp.format) {
        case AME_FP_FORMAT_FP16:
            return float16_val(float16_muladd(
                factor1, factor2, addend, flags, status));
        case AME_FP_FORMAT_BF16:
            return bfloat16_muladd(factor1, factor2, addend, flags, status);
        case AME_FP_FORMAT_FP32:
            return float32_val(float32_muladd(
                make_float32(factor1), make_float32(factor2),
                make_float32(addend), flags, status));
        case AME_FP_FORMAT_FP64:
            return float64_val(float64_muladd(
                make_float64(factor1), make_float64(factor2),
                make_float64(addend), flags, status));
        default:
            g_assert_not_reached();
        }
    }

    left = ame_fp_unpack(factor1_dtype, factor1, status);
    right = ame_fp_unpack(factor2_dtype, factor2, status);
    current = ame_fp_unpack(addend_dtype, addend, status);
    value = parts64_muladd(&left, &right, &current, flags, status);

    return ame_fp_pack(dest_dtype, value, status);
}

static uint64_t ame_fp_mixed_scale_acc_raw(const AmeDTypeDesc *dest_dtype,
                                           const AmeDTypeDesc *data_dtype,
                                           uint64_t data, uint64_t current,
                                           int scale, float_status *status)
{
    FloatParts64 scaled = ame_fp_unpack(data_dtype, data, status);
    FloatParts64 addend = ame_fp_unpack(dest_dtype, current, status);
    FloatParts64 sum;

    scaled = parts64_scalbn(&scaled, scale, status);
    sum = parts64_addsub(&addend, &scaled, status, false);
    return ame_fp_pack(dest_dtype, sum, status);
}

void riscv_ame_exec_fp_scale_acc(AmeOperation op,
                                 const AmeDTypeDesc *dest_dtype,
                                 const AmeDTypeDesc *data_dtype,
                                 const AmeValue *data,
                                 const AmeDTypeDesc *exponent_dtype,
                                 const AmeValue *exponent,
                                 const AmeValue *current,
                                 AmeExecResult *result,
                                 float_status *status)
{
    AmeExecResult executed = { 0 };
    int scale;

    if (dest_dtype->class == AME_DTYPE_CLASS_INTEGER ||
        data_dtype->class == AME_DTYPE_CLASS_INTEGER) {
        ame_exec_number_scale(op, dest_dtype, data_dtype, data,
                              exponent_dtype, exponent, current,
                              result, status);
        return;
    }

    scale = ame_fp_scale(exponent_dtype, *exponent,
                         op == AME_OP_RDEXPACC);
    executed = (AmeExecResult) {
        .value.lo = ame_fp_mixed_scale_acc_raw(
            dest_dtype, data_dtype, data->lo, current->lo, scale, status),
    };
    executed.fflags = riscv_ame_fp_status_flags(status);
    *result = executed;
}

void riscv_ame_exec_fp_ternary(AmeOperation op,
                               const AmeDTypeDesc *dest_dtype,
                               const AmeDTypeDesc *src1_dtype,
                               const AmeValue *src1,
                               const AmeDTypeDesc *src2_dtype,
                               const AmeValue *src2,
                               const AmeValue *current,
                               AmeExecResult *result,
                               float_status *status)
{
    AmeExecResult executed;
    const AmeDTypeDesc *factor1_dtype, *factor2_dtype, *addend_dtype;
    uint64_t factor1, factor2, addend;
    int flags = 0;

    if (dest_dtype->class == AME_DTYPE_CLASS_INTEGER ||
        src1_dtype->class == AME_DTYPE_CLASS_INTEGER ||
        src2_dtype->class == AME_DTYPE_CLASS_INTEGER) {
        ame_exec_number_ternary(op, dest_dtype, src1_dtype, src1,
                                src2_dtype, src2, current, result, status);
        return;
    }

    switch (op) {
    case AME_OP_MULACC:
        factor1_dtype = src1_dtype;
        factor1 = src1->lo;
        factor2_dtype = src2_dtype;
        factor2 = src2->lo;
        addend_dtype = dest_dtype;
        addend = current->lo;
        break;
    case AME_OP_MULNACC:
        factor1_dtype = src1_dtype;
        factor1 = src1->lo;
        factor2_dtype = src2_dtype;
        factor2 = src2->lo;
        addend_dtype = dest_dtype;
        addend = current->lo;
        flags = float_muladd_negate_product;
        break;
    case AME_OP_MULADD:
        factor1_dtype = src2_dtype;
        factor1 = src2->lo;
        factor2_dtype = dest_dtype;
        factor2 = current->lo;
        addend_dtype = src1_dtype;
        addend = src1->lo;
        break;
    case AME_OP_MULSUB:
        factor1_dtype = src2_dtype;
        factor1 = src2->lo;
        factor2_dtype = dest_dtype;
        factor2 = current->lo;
        addend_dtype = src1_dtype;
        addend = src1->lo;
        flags = float_muladd_negate_product;
        break;
    default:
        g_assert_not_reached();
    }

    executed = (AmeExecResult) {
        .value.lo = ame_fp_muladd_raw(
            dest_dtype, factor1_dtype, factor1, factor2_dtype, factor2,
            addend_dtype, addend, flags, status),
    };
    executed.fflags = riscv_ame_fp_status_flags(status);
    *result = executed;
}
