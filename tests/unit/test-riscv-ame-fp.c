/*
 * RISC-V Attached Matrix Extension IEEE floating-point tests
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "fpu/softfloat.h"
#include "target/riscv/ame/fp.h"

static const AmeDTypeDesc *dtype(uint32_t encoding)
{
    const AmeDTypeDesc *desc = riscv_ame_dtype_by_encoding(encoding);

    g_assert_nonnull(desc);
    return desc;
}

static const AmeDTypeDesc *fp_dtype(AmeFPFormat format, AmeFPRoundMode round)
{
    static const uint32_t base[] = {
        [AME_FP_FORMAT_FP16] = UINT32_C(0x14300110),
        [AME_FP_FORMAT_BF16] = UINT32_C(0x20300110),
        [AME_FP_FORMAT_FP32] = UINT32_C(0x20300120),
        [AME_FP_FORMAT_FP64] = UINT32_C(0x2c300140),
        [AME_FP_FORMAT_OCP_E4M3] = UINT32_C(0x10100108),
        [AME_FP_FORMAT_OCP_E5M2] = UINT32_C(0x14300108),
    };

    g_assert_cmpuint(format, >=, AME_FP_FORMAT_FP16);
    g_assert_cmpuint(format, <=, AME_FP_FORMAT_OCP_E5M2);
    return dtype(base[format] | ((uint32_t)round << 22));
}

static void exec_fp_ternary_same_dtype(AmeOperation op,
                                       const AmeDTypeDesc *dtype,
                                       const AmeValue *src1,
                                       const AmeValue *src2,
                                       const AmeValue *current,
                                       AmeExecResult *result,
                                       float_status *status)
{
    riscv_ame_exec_fp_ternary(op, dtype, dtype, src1, dtype,
                src2, current, result, status);
}

typedef enum OcpRefClass {
    OCP_REF_ZERO,
    OCP_REF_FINITE,
    OCP_REF_INFINITY,
    OCP_REF_NAN,
} OcpRefClass;

typedef struct OcpRefValue {
    OcpRefClass cls;
    bool sign;
    uint64_t significand;
    int exponent;
} OcpRefValue;

typedef struct OcpRefResult {
    uint8_t value;
    uint8_t fflags;
} OcpRefResult;

static void ocp_ref_params(AmeFPFormat format, unsigned *exp_bits,
                           unsigned *frac_bits, int *bias,
                           uint8_t *max_finite)
{
    if (format == AME_FP_FORMAT_OCP_E4M3) {
        *exp_bits = 4;
        *frac_bits = 3;
        *bias = 7;
        *max_finite = UINT8_C(0x7e);
    } else {
        g_assert_cmpuint(format, ==, AME_FP_FORMAT_OCP_E5M2);
        *exp_bits = 5;
        *frac_bits = 2;
        *bias = 15;
        *max_finite = UINT8_C(0x7b);
    }
}

static OcpRefValue ocp_ref_decode(AmeFPFormat format, uint8_t raw)
{
    unsigned exp_bits, frac_bits;
    int bias;
    uint8_t max_finite;
    uint8_t fraction, exponent;
    OcpRefValue result = { .sign = raw >> 7 };

    ocp_ref_params(format, &exp_bits, &frac_bits, &bias, &max_finite);
    fraction = raw & MAKE_64BIT_MASK(0, frac_bits);
    exponent = (raw >> frac_bits) & MAKE_64BIT_MASK(0, exp_bits);
    if (format == AME_FP_FORMAT_OCP_E4M3 &&
        exponent == MAKE_64BIT_MASK(0, exp_bits) &&
        fraction == MAKE_64BIT_MASK(0, frac_bits)) {
        result.cls = OCP_REF_NAN;
    } else if (format == AME_FP_FORMAT_OCP_E5M2 &&
               exponent == MAKE_64BIT_MASK(0, exp_bits)) {
        result.cls = fraction == 0 ? OCP_REF_INFINITY : OCP_REF_NAN;
    } else if (exponent == 0 && fraction == 0) {
        result.cls = OCP_REF_ZERO;
    } else {
        result.cls = OCP_REF_FINITE;
        result.significand = exponent == 0 ? fraction :
                             BIT_ULL(frac_bits) | fraction;
        result.exponent = (exponent == 0 ? 1 : exponent) - bias -
                          frac_bits;
    }
    return result;
}

static int ocp_ref_msb(uint64_t value)
{
    g_assert_cmpuint(value, !=, 0);
    return 63 - clz64(value);
}

static bool ocp_ref_round_increment(AmeFPRoundMode mode, bool sign,
                                    uint64_t rounded, uint64_t remainder,
                                    int shift)
{
    bool above_halfway = false;
    bool exactly_halfway = false;

    g_assert_cmpint(shift, >, 0);
    if (shift < 64) {
        uint64_t halfway = BIT_ULL(shift - 1);

        above_halfway = remainder > halfway;
        exactly_halfway = remainder == halfway;
    } else if (shift == 64) {
        above_halfway = remainder > BIT_ULL(63);
        exactly_halfway = remainder == BIT_ULL(63);
    }

    switch (mode) {
    case AME_FP_RNE:
        return above_halfway || (exactly_halfway && (rounded & 1));
    case AME_FP_RTZ:
        return false;
    case AME_FP_RDN:
        return sign && remainder != 0;
    case AME_FP_RUP:
        return !sign && remainder != 0;
    case AME_FP_RMM:
        return above_halfway || exactly_halfway;
    default:
        g_assert_not_reached();
    }
}

static bool ocp_ref_tiny_after_rounding(AmeFPFormat format,
                                        AmeFPRoundMode mode, bool sign,
                                        uint64_t significand, int exponent)
{
    unsigned exp_bits, frac_bits;
    int bias, result_exp, shift;
    uint8_t max_finite;
    uint64_t rounded, remainder = 0;

    ocp_ref_params(format, &exp_bits, &frac_bits, &bias, &max_finite);
    result_exp = ocp_ref_msb(significand) + exponent - frac_bits;
    shift = result_exp - exponent;
    if (shift > 0) {
        rounded = significand >> shift;
        remainder = significand & MAKE_64BIT_MASK(0, shift);
        rounded += ocp_ref_round_increment(mode, sign, rounded, remainder,
                                           shift);
    } else {
        rounded = significand << -shift;
    }
    if (rounded >= BIT_ULL(frac_bits + 1)) {
        result_exp++;
    }

    return result_exp + (int)frac_bits < 1 - bias;
}

static OcpRefResult ocp_ref_round(AmeFPFormat format, AmeFPRoundMode mode,
                                  bool sign, uint64_t significand,
                                  int exponent)
{
    unsigned exp_bits, frac_bits;
    int bias, result_exp, shift;
    uint8_t max_finite;
    uint64_t rounded, remainder = 0;
    bool increment = false;
    bool is_tiny;
    OcpRefResult result = { 0 };

    ocp_ref_params(format, &exp_bits, &frac_bits, &bias, &max_finite);
    if (significand == 0) {
        result.value = sign ? UINT8_C(0x80) : 0;
        return result;
    }

    is_tiny = ocp_ref_tiny_after_rounding(format, mode, sign, significand,
                                          exponent);
    result_exp = MAX(1 - bias - (int)frac_bits,
                     ocp_ref_msb(significand) + exponent -
                     (int)frac_bits);
    shift = result_exp - exponent;
    if (shift > 0) {
        if (shift >= 64) {
            rounded = 0;
            remainder = significand;
        } else {
            rounded = significand >> shift;
            remainder = significand & MAKE_64BIT_MASK(0, shift);
        }
        increment = ocp_ref_round_increment(mode, sign, rounded, remainder,
                                            shift);
    } else {
        rounded = significand << -shift;
    }
    rounded += increment;
    while (rounded >= BIT_ULL(frac_bits + 1)) {
        rounded >>= 1;
        result_exp++;
    }

    if (rounded < BIT_ULL(frac_bits)) {
        result.value = rounded;
    } else {
        int encoded_exp = result_exp + frac_bits + bias;

        if (encoded_exp >= (1 << exp_bits)) {
            result.value = UINT8_MAX;
        } else {
            result.value = (encoded_exp << frac_bits) |
                           (rounded - BIT_ULL(frac_bits));
        }
    }
    if (result.value > max_finite) {
        result.fflags = AME_FFLAG_OF | AME_FFLAG_NX;
        result.value = format == AME_FP_FORMAT_OCP_E5M2 ?
                       UINT8_C(0x7c) : max_finite;
    } else if (remainder != 0) {
        result.fflags = AME_FFLAG_NX;
        if (is_tiny) {
            result.fflags |= AME_FFLAG_UF;
        }
    }
    result.value |= sign ? UINT8_C(0x80) : 0;
    return result;
}

static OcpRefResult ocp_ref_binary(AmeOperation op, AmeFPFormat format,
                                   AmeFPRoundMode mode, uint8_t raw1,
                                   uint8_t raw2, bool *comparable)
{
    OcpRefValue left = ocp_ref_decode(format, raw1);
    OcpRefValue right = ocp_ref_decode(format, raw2);
    int exponent;
    int64_t coefficient, right_coefficient;

    *comparable = left.cls <= OCP_REF_FINITE &&
                  right.cls <= OCP_REF_FINITE;
    if (!*comparable) {
        return (OcpRefResult) { 0 };
    }

    if (op == AME_OP_MUL) {
        if (left.cls == OCP_REF_ZERO || right.cls == OCP_REF_ZERO) {
            return (OcpRefResult) {
                .value = (left.sign ^ right.sign) ? UINT8_C(0x80) : 0,
            };
        }
        return ocp_ref_round(format, mode, left.sign ^ right.sign,
                             left.significand * right.significand,
                             left.exponent + right.exponent);
    }

    if (left.cls == OCP_REF_ZERO && right.cls == OCP_REF_ZERO) {
        *comparable = false;
        return (OcpRefResult) { 0 };
    }
    exponent = MIN(left.cls == OCP_REF_ZERO ? right.exponent : left.exponent,
                   right.cls == OCP_REF_ZERO ? left.exponent :
                                               right.exponent);
    coefficient = left.cls == OCP_REF_ZERO ? 0 :
                  left.significand << (left.exponent - exponent);
    coefficient = left.sign ? -coefficient : coefficient;
    right_coefficient = right.cls == OCP_REF_ZERO ? 0 :
                        right.significand <<
                            (right.exponent - exponent);
    right_coefficient = right.sign ? -right_coefficient : right_coefficient;
    if (op == AME_OP_ADD || op == AME_OP_MEAN) {
        coefficient += right_coefficient;
    } else {
        g_assert_true(op == AME_OP_SUB || op == AME_OP_HDIFF);
        coefficient = right_coefficient - coefficient;
    }
    if (coefficient == 0) {
        *comparable = false;
        return (OcpRefResult) { 0 };
    }
    return ocp_ref_round(format, mode, coefficient < 0,
                         coefficient < 0 ? -(uint64_t)coefficient :
                                           coefficient,
                         exponent - (op == AME_OP_HDIFF ||
                                     op == AME_OP_MEAN));
}

static int64_t ocp_ref_coefficient(const OcpRefValue *value, int exponent)
{
    unsigned shift;
    int64_t coefficient;

    if (value->cls == OCP_REF_ZERO) {
        return 0;
    }
    g_assert_cmpint(value->cls, ==, OCP_REF_FINITE);
    shift = value->exponent - exponent;
    g_assert_cmpuint(shift, <, 63);
    coefficient = value->significand << shift;
    return value->sign ? -coefficient : coefficient;
}

static OcpRefResult ocp_ref_fused(AmeOperation op, AmeFPFormat format,
                                  AmeFPRoundMode mode, uint8_t raw1,
                                  uint8_t raw2, uint8_t raw_current,
                                  bool *comparable)
{
    OcpRefValue src1 = ocp_ref_decode(format, raw1);
    OcpRefValue src2 = ocp_ref_decode(format, raw2);
    OcpRefValue current = ocp_ref_decode(format, raw_current);
    const OcpRefValue *factor1 = &src1;
    const OcpRefValue *factor2 = &src2;
    const OcpRefValue *addend = &current;
    int product_exponent, exponent;
    int64_t product, coefficient;
    bool negate_product = false;

    *comparable = src1.cls <= OCP_REF_FINITE &&
                  src2.cls <= OCP_REF_FINITE &&
                  current.cls <= OCP_REF_FINITE;
    if (!*comparable) {
        return (OcpRefResult) { 0 };
    }

    switch (op) {
    case AME_OP_MULACC:
        break;
    case AME_OP_MULNACC:
        negate_product = true;
        break;
    case AME_OP_MULADD:
        factor1 = &src2;
        factor2 = &current;
        addend = &src1;
        break;
    case AME_OP_MULSUB:
        factor1 = &src2;
        factor2 = &current;
        addend = &src1;
        negate_product = true;
        break;
    default:
        g_assert_not_reached();
    }

    if (factor1->cls == OCP_REF_ZERO || factor2->cls == OCP_REF_ZERO) {
        product_exponent = addend->exponent;
        product = 0;
    } else {
        product_exponent = factor1->exponent + factor2->exponent;
        product = factor1->significand * factor2->significand;
        if (factor1->sign ^ factor2->sign ^ negate_product) {
            product = -product;
        }
    }
    if (addend->cls == OCP_REF_ZERO) {
        exponent = product_exponent;
    } else {
        exponent = MIN(product_exponent, addend->exponent);
    }
    g_assert_cmpuint(product_exponent - exponent, <, 63);
    product *= INT64_C(1) << (product_exponent - exponent);
    coefficient = product + ocp_ref_coefficient(addend, exponent);
    if (coefficient == 0) {
        *comparable = false;
        return (OcpRefResult) { 0 };
    }
    return ocp_ref_round(format, mode, coefficient < 0,
                         coefficient < 0 ? -(uint64_t)coefficient :
                                           coefficient,
                         exponent);
}

static void test_ieee_status(void)
{
    static const int softfloat_rounding[] = {
        [AME_FP_RNE] = float_round_nearest_even,
        [AME_FP_RTZ] = float_round_to_zero,
        [AME_FP_RDN] = float_round_down,
        [AME_FP_RUP] = float_round_up,
        [AME_FP_RMM] = float_round_ties_away,
        [AME_FP_RNO] = float_round_to_odd_inf,
    };
    float_status status;
    const AmeDTypeDesc *fp32;
    size_t i;

    for (i = 0; i < ARRAY_SIZE(softfloat_rounding); i++) {
        fp32 = fp_dtype(AME_FP_FORMAT_FP32, i);
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
        g_assert_cmpuint(get_float_rounding_mode(&status), ==,
                         softfloat_rounding[i]);
        g_assert_false(status.tininess_before_rounding);
        g_assert_true(status.default_nan_mode);
        g_assert_cmpuint(status.default_nan_pattern, ==, 0x40);
    }

    set_float_exception_flags(float_flag_invalid | float_flag_divbyzero |
                              float_flag_overflow | float_flag_underflow |
                              float_flag_inexact, &status);
    g_assert_cmphex(riscv_ame_fp_status_flags(&status), ==,
                    AME_FFLAG_NV | AME_FFLAG_DZ | AME_FFLAG_OF |
                    AME_FFLAG_UF | AME_FFLAG_NX);
    g_assert_true(riscv_ame_fp_status_set_rounding(
        &status, fp_dtype(AME_FP_FORMAT_FP32, AME_FP_RDN)));
    g_assert_cmpuint(get_float_rounding_mode(&status), ==, float_round_down);
    g_assert_cmphex(riscv_ame_fp_status_flags(&status), ==,
                    AME_FFLAGS_MASK);
    g_assert_true(riscv_ame_fp_status_prepare(&status,
        dtype(UINT32_C(0x40000020))));
}

static void test_ieee_conversion(void)
{
    static const struct {
        AmeFPFormat format;
        uint64_t one;
        uint64_t quiet_nan;
        uint64_t signaling_nan;
        uint64_t default_nan;
    } formats[] = {
        { AME_FP_FORMAT_FP16, 0x3c00, 0x7e55, 0x7d00, 0x7e00 },
        { AME_FP_FORMAT_BF16, 0x3f80, 0x7fc5, 0x7fa0, 0x7fc0 },
        { AME_FP_FORMAT_FP32, 0x3f800000, 0x7fc12345, 0x7fa00000,
          0x7fc00000 },
        { AME_FP_FORMAT_FP64, UINT64_C(0x3ff0000000000000),
          UINT64_C(0x7ff8123456789abc), UINT64_C(0x7ff4000000000000),
          UINT64_C(0x7ff8000000000000) },
    };
    static const uint32_t halfway_expected[] = {
        [AME_FP_RNE] = UINT32_C(0x3f800000),
        [AME_FP_RTZ] = UINT32_C(0x3f800000),
        [AME_FP_RDN] = UINT32_C(0x3f800000),
        [AME_FP_RUP] = UINT32_C(0x3f800001),
        [AME_FP_RMM] = UINT32_C(0x3f800001),
        [AME_FP_RNO] = UINT32_C(0x3f800001),
    };
    const AmeDTypeDesc *fp16 = fp_dtype(AME_FP_FORMAT_FP16, AME_FP_RNE);
    const AmeDTypeDesc *fp32 = fp_dtype(AME_FP_FORMAT_FP32, AME_FP_RNE);
    const AmeDTypeDesc *fp64 = fp_dtype(AME_FP_FORMAT_FP64, AME_FP_RNE);
    const AmeDTypeDesc *int64 = dtype(UINT32_C(0x40000040));
    const AmeDTypeDesc *uint64 = dtype(UINT32_C(0x00000040));
    const AmeDTypeDesc *int128 = dtype(UINT32_C(0x40000080));
    float_status status;
    AmeExecResult result;
    AmeValue source;
    size_t i, j;

    for (i = 0; i < ARRAY_SIZE(formats); i++) {
        const AmeDTypeDesc *src = fp_dtype(formats[i].format, AME_FP_RNE);

        for (j = 0; j < ARRAY_SIZE(formats); j++) {
            const AmeDTypeDesc *dest = fp_dtype(formats[j].format,
                                                AME_FP_RNE);

            source.lo = formats[i].one;
            g_assert_true(riscv_ame_fp_status_prepare(&status, dest));
            riscv_ame_exec_fp_convert(dest, src, &source,
                                      &result, &status);
            g_assert_cmphex(result.value.lo, ==, formats[j].one);
            g_assert_cmphex(result.fflags, ==, 0);

            source.lo = formats[i].quiet_nan;
            g_assert_true(riscv_ame_fp_status_prepare(&status, dest));
            riscv_ame_exec_fp_convert(dest, src, &source,
                                      &result, &status);
            g_assert_cmphex(result.value.lo, ==, formats[j].default_nan);
            g_assert_cmphex(result.fflags, ==, 0);

            source.lo = formats[i].signaling_nan;
            g_assert_true(riscv_ame_fp_status_prepare(&status, dest));
            riscv_ame_exec_fp_convert(dest, src, &source,
                                      &result, &status);
            g_assert_cmphex(result.value.lo, ==, formats[j].default_nan);
            g_assert_cmphex(result.fflags, ==, AME_FFLAG_NV);
        }

        source.lo = 1;
        g_assert_true(riscv_ame_fp_status_prepare(&status, src));
        riscv_ame_exec_fp_convert(src, int64, &source,
                                 &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].one);
        g_assert_cmphex(result.fflags, ==, 0);
    }

    source.lo = UINT64_C(16777217);
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    riscv_ame_exec_fp_convert(fp32, int64, &source, &result,
                             &status);
    g_assert_cmphex(result.value.lo, ==, UINT32_C(0x4b800000));
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);

    source.lo = UINT64_MAX;
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp64));
    riscv_ame_exec_fp_convert(fp64, uint64, &source, &result,
                             &status);
    g_assert_cmphex(result.value.lo, ==, UINT64_C(0x43f0000000000000));
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);

    source.lo = UINT64_C(0x3ff0000010000000);
    for (i = 0; i < ARRAY_SIZE(halfway_expected); i++) {
        const AmeDTypeDesc *dest = fp_dtype(AME_FP_FORMAT_FP32, i);

        g_assert_true(riscv_ame_fp_status_prepare(&status, dest));
        riscv_ame_exec_fp_convert(dest, fp64, &source, &result,
                                 &status);
        g_assert_cmphex(result.value.lo, ==, halfway_expected[i]);
        g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);
    }

    source.lo = UINT64_C(0x3ff0000008000000);
    g_assert_true(riscv_ame_fp_status_prepare(&status,
        fp_dtype(AME_FP_FORMAT_FP32, AME_FP_RNO)));
    riscv_ame_exec_fp_convert(
        fp_dtype(AME_FP_FORMAT_FP32, AME_FP_RNO), fp64, &source, &result,
        &status);
    g_assert_cmphex(result.value.lo, ==, UINT32_C(0x3f800001));
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);

    source.lo = UINT16_C(0x7e55);
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    riscv_ame_exec_fp_convert(fp32, fp16, &source, &result,
                             &status);
    g_assert_cmphex(result.value.lo, ==, UINT32_C(0x7fc00000));
    g_assert_cmphex(result.fflags, ==, 0);

    source.lo = UINT16_C(0x7d00);
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    riscv_ame_exec_fp_convert(fp32, fp16, &source, &result,
                             &status);
    g_assert_cmphex(result.value.lo, ==, UINT32_C(0x7fc00000));
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NV);

    source.lo = UINT16_C(0x7c00);
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    riscv_ame_exec_fp_convert(fp32, fp16, &source, &result,
                             &status);
    g_assert_cmphex(result.value.lo, ==, UINT32_C(0x7f800000));
    g_assert_cmphex(result.fflags, ==, 0);

    source.lo = UINT16_C(0x0001);
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    riscv_ame_exec_fp_convert(fp32, fp16, &source, &result,
                             &status);
    g_assert_cmphex(result.value.lo, ==, UINT32_C(0x33800000));
    g_assert_cmphex(result.fflags, ==, 0);

    source = (AmeValue) { .lo = 1 };
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    riscv_ame_exec_fp_convert(fp32, int128, &source, &result,
                             &status);
    g_assert_cmphex(result.value.lo, ==, UINT32_C(0x3f800000));
    g_assert_cmphex(result.fflags, ==, 0);
    riscv_ame_exec_fp_convert(int64, fp32, &source, &result,
                             &status);
    g_assert_cmphex(result.value.lo, ==, 0);
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);
}

/* Decode the storage bits independently of SoftFloat's canonical parts. */
static OcpRefValue fp_int_ref_decode(AmeFPFormat format, uint64_t raw)
{
    static const struct {
        unsigned exponent_bits;
        unsigned fraction_bits;
        int bias;
    } ieee[] = {
        [AME_FP_FORMAT_FP16] = { 5, 10, 15 },
        [AME_FP_FORMAT_BF16] = { 8, 7, 127 },
        [AME_FP_FORMAT_FP32] = { 8, 23, 127 },
        [AME_FP_FORMAT_FP64] = { 11, 52, 1023 },
    };
    unsigned e, f;
    uint64_t exponent, fraction;
    OcpRefValue result = { 0 };

    if (format >= AME_FP_FORMAT_OCP_E4M3) {
        return ocp_ref_decode(format, raw);
    }
    e = ieee[format].exponent_bits;
    f = ieee[format].fraction_bits;
    result.sign = extract64(raw, e + f, 1);
    exponent = extract64(raw, f, e);
    fraction = extract64(raw, 0, f);
    if (exponent == MAKE_64BIT_MASK(0, e)) {
        result.cls = fraction ? OCP_REF_NAN : OCP_REF_INFINITY;
    } else if (exponent == 0 && fraction == 0) {
        result.cls = OCP_REF_ZERO;
    } else {
        result.cls = OCP_REF_FINITE;
        result.significand = fraction | (exponent ? BIT_ULL(f) : 0);
        result.exponent = (int)(exponent ? exponent : 1) -
                          ieee[format].bias - f;
    }
    return result;
}

static Int128 fp_int_ref_mask(unsigned bits)
{
    return bits == 128 ? int128_not(int128_zero()) :
           int128_sub(int128_lshift(int128_one(), bits), int128_one());
}

/* Round the unsigned magnitude, then apply sign, range and result policy. */
static AmeExecResult fp_int_ref(const AmeDTypeDesc *dest,
                                AmeFPFormat format, uint64_t raw)
{
    OcpRefValue value = fp_int_ref_decode(format, raw);
    Int128 magnitude = int128_zero(), rounded, limit;
    Int128 maximum = fp_int_ref_mask(dest->bits - dest->integer.is_signed);
    Int128 minimum = dest->integer.is_signed ?
        int128_lshift(int128_one(), dest->bits - 1) : int128_zero();
    bool negative = value.sign && value.cls != OCP_REF_NAN;
    bool nonfinite = value.cls == OCP_REF_NAN ||
                     value.cls == OCP_REF_INFINITY;
    bool overflow = false, inexact = false, increment = false;
    bool above_half = false, tie = false;
    AmeExecResult result = { 0 };

    if (value.cls == OCP_REF_FINITE) {
        if (value.exponent >= 0) {
            overflow = ocp_ref_msb(value.significand) + value.exponent >= 128;
            if (value.exponent < 128) {
                magnitude = int128_lshift(int128_make64(value.significand),
                                          value.exponent);
            }
        } else {
            unsigned shift = -value.exponent;
            uint64_t remainder = value.significand;

            if (shift < 64) {
                magnitude = int128_make64(value.significand >> shift);
                remainder &= MAKE_64BIT_MASK(0, shift);
            }
            inexact = remainder != 0;
            if (shift <= 64) {
                above_half = remainder > BIT_ULL(shift - 1);
                tie = remainder == BIT_ULL(shift - 1);
            }
            switch (dest->integer.rounding) {
            case AME_INT_RNU:
                increment = above_half || (tie && !negative);
                break;
            case AME_INT_RNE:
                increment = above_half ||
                            (tie && (int128_getlo(magnitude) & 1));
                break;
            case AME_INT_RDN:
                increment = negative && inexact;
                break;
            case AME_INT_ROD:
                increment = inexact && !(int128_getlo(magnitude) & 1);
                break;
            default:
                g_assert_not_reached();
            }
            magnitude = int128_add(magnitude, int128_make64(increment));
        }
    }
    rounded = negative ? int128_neg(magnitude) : magnitude;
    limit = negative ? minimum : maximum;
    overflow |= int128_ult(limit, magnitude);
    if (nonfinite || (overflow && dest->integer.saturating)) {
        rounded = negative ? int128_neg(minimum) : maximum;
        result.saturated = !nonfinite;
    }
    rounded = int128_and(rounded, fp_int_ref_mask(dest->bits));
    result.value.lo = int128_getlo(rounded);
    result.value.hi = int128_gethi(rounded);
    result.fflags = nonfinite || overflow ? AME_FFLAG_NV :
                    inexact ? AME_FFLAG_NX : 0;
    return result;
}

static void check_fp_int(const AmeDTypeDesc *dest, const AmeDTypeDesc *src,
                         uint64_t raw)
{
    AmeExecResult expected = fp_int_ref(dest, src->fp.format, raw), actual;
    AmeValue value = { .lo = raw };
    float_status status;

    g_assert_true(riscv_ame_fp_status_prepare(&status, dest));
    riscv_ame_exec_fp_convert(dest, src, &value, &actual,
                             &status);
    if (actual.value.lo != expected.value.lo ||
        actual.value.hi != expected.value.hi ||
        actual.fflags != expected.fflags ||
        actual.saturated != expected.saturated) {
        g_error("FP-to-int: D=0x%x A=0x%x raw=0x%" PRIx64
                " got=0x%" PRIx64 ":0x%" PRIx64 " flags=0x%x sat=%d"
                " expected=0x%" PRIx64 ":0x%" PRIx64 " flags=0x%x sat=%d",
                dest->encoding, src->encoding, raw,
                actual.value.hi, actual.value.lo, actual.fflags,
                actual.saturated, expected.value.hi, expected.value.lo,
                expected.fflags, expected.saturated);
    }
}

static void test_fp_int_exhaustive(void)
{
    static const AmeFPFormat formats[] = {
        AME_FP_FORMAT_FP16, AME_FP_FORMAT_BF16,
        AME_FP_FORMAT_OCP_E4M3, AME_FP_FORMAT_OCP_E5M2,
    };

    for (unsigned f = 0; f < G_N_ELEMENTS(formats); f++) {
        const AmeDTypeDesc *src = fp_dtype(formats[f], AME_FP_RNE);

        for (unsigned d = 0; d < AME_DTYPE_COUNT; d++) {
            const AmeDTypeDesc *dest = &riscv_ame_dtypes[d];

            if (dest->class != AME_DTYPE_CLASS_INTEGER) {
                continue;
            }
            for (unsigned raw = 0; raw < BIT(src->bits); raw++) {
                check_fp_int(dest, src, raw);
            }
        }
    }
}

static void test_fp_int_boundaries(void)
{
    static const uint64_t fp64[] = {
        0, 1, 0x3fdfffffffffffff, 0x3fe0000000000000, 0x3fe0000000000001,
        0x3ff8000000000000, 0x4004000000000000, 0x400c000000000000,
        0x401fffffffffffff, 0x4020000000000000, 0x405fdfffffffffff,
        0x405fe00000000000, 0x405fe00000000001, 0x405fffffffffffff,
        0x4060000000000000, 0x406ff00000000000,
        0x41dfffffffffffff, 0x41e0000000000000, 0x41efffffffffffff,
        0x41f0000000000000, 0x43dfffffffffffff, 0x43e0000000000000,
        0x43efffffffffffff, 0x43f0000000000000, 0x43f0000000000001,
        0x47dfffffffffffff, 0x47e0000000000000, 0x47efffffffffffff,
        0x47f0000000000000, 0x47f0000000000001, 0x4be0000000000001,
        0x4bf0000000000001, 0x7fefffffffffffff, 0x7ff0000000000000,
        0x7ff0000000000001, 0x7ff8000000000000, 0x7fffffffffffffff,
    };
    static const uint32_t fp32[] = {
        0, 1, 0x3effffff, 0x3f000000, 0x3f000001, 0x3fc00000,
        0x40200000, 0x42ff0000, 0x43000000, 0x437f8000,
        0x4effffff, 0x4f000000, 0x4f7fffff, 0x4f800000,
        0x5effffff, 0x5f000000, 0x5f7fffff, 0x5f800000,
        0x7effffff, 0x7f000000, 0x7f7fffff, 0x7f800000,
        0x7f800001, 0x7fc00000, 0x7fffffff,
    };

    for (unsigned d = 0; d < AME_DTYPE_COUNT; d++) {
        const AmeDTypeDesc *dest = &riscv_ame_dtypes[d];
        uint64_t random = UINT64_C(0x19f5a072db63e841);

        if (dest->class != AME_DTYPE_CLASS_INTEGER) {
            continue;
        }
        for (unsigned round = 0; round < AME_FP_ROUND_MAX; round++) {
            for (unsigned sign = 0; sign < 2; sign++) {
                for (unsigned i = 0; i < G_N_ELEMENTS(fp64); i++) {
                    check_fp_int(dest, fp_dtype(AME_FP_FORMAT_FP64, round),
                                  fp64[i] | ((uint64_t)sign << 63));
                }
                for (unsigned i = 0; i < G_N_ELEMENTS(fp32); i++) {
                    check_fp_int(dest, fp_dtype(AME_FP_FORMAT_FP32, round),
                                  fp32[i] | ((uint64_t)sign << 31));
                }
            }
        }
        for (unsigned i = 0; i < 4096; i++) {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            check_fp_int(dest, fp_dtype(AME_FP_FORMAT_FP64, AME_FP_RNE),
                          random);
            check_fp_int(dest, fp_dtype(AME_FP_FORMAT_FP32, AME_FP_RNE),
                          (uint32_t)random);
        }
    }
}

static void test_fp_int_accrued_flags(void)
{
    const AmeDTypeDesc *dest = dtype(0x40000008);
    const AmeDTypeDesc *src = fp_dtype(AME_FP_FORMAT_FP32, AME_FP_RUP);
    AmeExecResult result;
    float_status status;

    for (unsigned invalid_first = 0; invalid_first < 2; invalid_first++) {
        g_assert_true(riscv_ame_fp_status_prepare(&status, dest));
        for (unsigned i = 0; i < 2; i++) {
            bool invalid = i == invalid_first;
            AmeValue value = { .lo = invalid ? 0x7f800000 : 0x3fc00000 };

            riscv_ame_exec_fp_convert(dest, src, &value,
                                     &result, &status);
            g_assert_cmphex(result.fflags, ==, i == 1 ?
                            AME_FFLAG_NV | AME_FFLAG_NX :
                            invalid ? AME_FFLAG_NV : AME_FFLAG_NX);
        }
    }
}

static void test_int128_fp_conversion(void)
{
    static const struct {
        AmeFPFormat format;
        uint64_t tie_lo;
        uint64_t tie_hi;
        uint64_t even;
        uint64_t odd;
    } formats[] = {
        { AME_FP_FORMAT_OCP_E4M3, 17, 0, 0x58, 0x59 },
        { AME_FP_FORMAT_OCP_E5M2, 9, 0, 0x48, 0x49 },
        { AME_FP_FORMAT_FP16, 2049, 0, 0x6800, 0x6801 },
        { AME_FP_FORMAT_BF16, 257, 0, 0x4380, 0x4381 },
        { AME_FP_FORMAT_FP32, 16777217, 0, 0x4b800000, 0x4b800001 },
        { AME_FP_FORMAT_FP64, UINT64_C(9007199254740993), 0,
          UINT64_C(0x4340000000000000),
          UINT64_C(0x4340000000000001) },
    };
    static const bool rounds_up[] = {
        [AME_FP_RNE] = false,
        [AME_FP_RTZ] = false,
        [AME_FP_RDN] = false,
        [AME_FP_RUP] = true,
        [AME_FP_RMM] = true,
        [AME_FP_RNO] = true,
    };
    const AmeDTypeDesc *uint128 = dtype(UINT32_C(0x00000080));
    const AmeDTypeDesc *int128 = dtype(UINT32_C(0x40000080));
    AmeExecResult result;
    float_status status;

    for (size_t i = 0; i < ARRAY_SIZE(formats); i++) {
        for (size_t round = AME_FP_RNE; round <= AME_FP_RNO; round++) {
            const AmeDTypeDesc *dest;
            AmeValue source = {
                .lo = formats[i].tie_lo,
                .hi = formats[i].tie_hi,
            };

            if (formats[i].format >= AME_FP_FORMAT_OCP_E4M3 &&
                round == AME_FP_RNO) {
                continue;
            }
            dest = fp_dtype(formats[i].format, round);
            g_assert_true(riscv_ame_fp_status_prepare(&status, dest));
            riscv_ame_exec_fp_convert(
                dest, uint128, &source, &result, &status);
            g_test_message("format: %u, round: %zu", formats[i].format,
                           round);
            g_assert_cmphex(result.value.lo, ==,
                            rounds_up[round] ? formats[i].odd :
                                               formats[i].even);
            g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);
        }
    }

    g_assert_true(riscv_ame_fp_status_prepare(
        &status, fp_dtype(AME_FP_FORMAT_FP64, AME_FP_RNE)));
    riscv_ame_exec_fp_convert(
        fp_dtype(AME_FP_FORMAT_FP64, AME_FP_RNE), int128,
        &(AmeValue) { .hi = UINT64_C(0x8000000000000000) },
        &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT64_C(0xc7e0000000000000));
    g_assert_cmphex(result.fflags, ==, 0);

    g_assert_true(riscv_ame_fp_status_prepare(
        &status, fp_dtype(AME_FP_FORMAT_FP64, AME_FP_RNE)));
    riscv_ame_exec_fp_convert(
        fp_dtype(AME_FP_FORMAT_FP64, AME_FP_RNE), uint128,
        &(AmeValue) { .lo = 1, .hi = UINT64_C(0x8000000000000400) },
        &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT64_C(0x47e0000000000001));
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);

    {
        static const struct {
            AmeFPFormat format;
            uint64_t expected;
            uint8_t flags;
        } overflow[] = {
            { AME_FP_FORMAT_OCP_E4M3, 0x7e,
              AME_FFLAG_OF | AME_FFLAG_NX },
            { AME_FP_FORMAT_OCP_E5M2, 0x7c,
              AME_FFLAG_OF | AME_FFLAG_NX },
            { AME_FP_FORMAT_FP16, 0x7c00,
              AME_FFLAG_OF | AME_FFLAG_NX },
            { AME_FP_FORMAT_BF16, 0x7f80,
              AME_FFLAG_OF | AME_FFLAG_NX },
            { AME_FP_FORMAT_FP32, UINT32_C(0x7f800000),
              AME_FFLAG_OF | AME_FFLAG_NX },
            { AME_FP_FORMAT_FP64, UINT64_C(0x47f0000000000000),
              AME_FFLAG_NX },
        };
        AmeValue maximum = { .lo = UINT64_MAX, .hi = UINT64_MAX };

        for (size_t i = 0; i < ARRAY_SIZE(overflow); i++) {
            const AmeDTypeDesc *dest = fp_dtype(overflow[i].format,
                                                AME_FP_RNE);

            g_assert_true(riscv_ame_fp_status_prepare(&status, dest));
            riscv_ame_exec_fp_convert(
                dest, uint128, &maximum, &result, &status);
            g_assert_cmphex(result.value.lo, ==, overflow[i].expected);
            g_assert_cmphex(result.fflags, ==, overflow[i].flags);
        }
    }
}

static void test_ieee_unary(void)
{
    static const struct {
        AmeFPFormat format;
        uint64_t negative_zero;
        uint64_t negative_one;
        uint64_t infinity;
        uint64_t negative_infinity;
        uint64_t subnormal;
        uint64_t one;
        uint64_t one_point_five;
        uint64_t negative_one_point_five;
        uint64_t two;
        uint64_t quiet_nan;
        uint64_t signaling_nan;
        uint64_t default_nan;
    } formats[] = {
        { AME_FP_FORMAT_FP16, 0x8000, 0xbc00, 0x7c00, 0xfc00, 0x0001,
          0x3c00, 0x3e00, 0xbe00, 0x4000, 0x7e55, 0x7d00, 0x7e00 },
        { AME_FP_FORMAT_BF16, 0x8000, 0xbf80, 0x7f80, 0xff80, 0x0001,
          0x3f80, 0x3fc0, 0xbfc0, 0x4000, 0x7fc5, 0x7fa0, 0x7fc0 },
        { AME_FP_FORMAT_FP32, UINT32_C(0x80000000),
          UINT32_C(0xbf800000), UINT32_C(0x7f800000),
          UINT32_C(0xff800000), UINT32_C(0x00000001),
          UINT32_C(0x3f800000), UINT32_C(0x3fc00000),
          UINT32_C(0xbfc00000),
          UINT32_C(0x40000000),
          UINT32_C(0x7fc12345), UINT32_C(0x7fa00000),
          UINT32_C(0x7fc00000) },
        { AME_FP_FORMAT_FP64, UINT64_C(0x8000000000000000),
          UINT64_C(0xbff0000000000000),
          UINT64_C(0x7ff0000000000000),
          UINT64_C(0xfff0000000000000),
          UINT64_C(0x0000000000000001),
          UINT64_C(0x3ff0000000000000),
          UINT64_C(0x3ff8000000000000),
          UINT64_C(0xbff8000000000000),
          UINT64_C(0x4000000000000000),
          UINT64_C(0x7ff8123456789abc), UINT64_C(0x7ff4000000000000),
          UINT64_C(0x7ff8000000000000) },
    };
    float_status status;
    AmeExecResult result;
    AmeValue source;
    size_t i;

    for (i = 0; i < ARRAY_SIZE(formats); i++) {
        const AmeDTypeDesc *fp = fp_dtype(formats[i].format, AME_FP_RNE);

        source.lo = formats[i].negative_zero;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_unary(AME_OP_ABS, fp, fp,
                               &source, &result, &status);
        g_assert_cmphex(result.value.lo, ==, 0);
        g_assert_cmphex(result.fflags, ==, 0);

        source.lo = formats[i].infinity;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_unary(AME_OP_ABS, fp, fp,
                               &source, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].infinity);
        g_assert_cmphex(result.fflags, ==, 0);

        source.lo = formats[i].subnormal;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_unary(AME_OP_ABS, fp, fp,
                               &source, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].subnormal);
        g_assert_cmphex(result.fflags, ==, 0);

        source.lo = formats[i].quiet_nan;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_unary(AME_OP_ABS, fp, fp,
                               &source, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].default_nan);
        g_assert_cmphex(result.fflags, ==, 0);

        source.lo = formats[i].signaling_nan;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_unary(AME_OP_ABS, fp, fp,
                               &source, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].default_nan);
        g_assert_cmphex(result.fflags, ==, AME_FFLAG_NV);

        source.lo = formats[i].negative_one;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_unary(AME_OP_SQRT, fp, fp,
                               &source, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].default_nan);
        g_assert_cmphex(result.fflags, ==, AME_FFLAG_NV);

        source.lo = 0;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_unary(AME_OP_REC, fp, fp,
                               &source, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].infinity);
        g_assert_cmphex(result.fflags, ==, AME_FFLAG_DZ);

        source.lo = formats[i].negative_zero;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_unary(AME_OP_REC, fp, fp,
                               &source, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].negative_infinity);
        g_assert_cmphex(result.fflags, ==, AME_FFLAG_DZ);
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_unary(AME_OP_SQRT, fp, fp,
                               &source, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].negative_zero);
        g_assert_cmphex(result.fflags, ==, 0);

        source.lo = formats[i].infinity;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_unary(AME_OP_REC, fp, fp,
                               &source, &result, &status);
        g_assert_cmphex(result.value.lo, ==, 0);
        g_assert_cmphex(result.fflags, ==, 0);
        source.lo = formats[i].negative_infinity;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_unary(AME_OP_REC, fp, fp,
                               &source, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].negative_zero);
        g_assert_cmphex(result.fflags, ==, 0);

        source.lo = formats[i].one_point_five;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_unary(AME_OP_FRINTN, fp, fp,
                               &source, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].two);
        g_assert_cmphex(result.fflags, ==, 0);
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_unary(AME_OP_FRINTM, fp, fp,
                               &source, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].one);
        g_assert_cmphex(result.fflags, ==, 0);
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_unary(AME_OP_FRINTP, fp, fp,
                               &source, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].two);
        g_assert_cmphex(result.fflags, ==, 0);
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_unary(AME_OP_FRINTZ, fp, fp,
                               &source, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].one);
        g_assert_cmphex(result.fflags, ==, 0);

        source.lo = formats[i].negative_one_point_five;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_unary(AME_OP_FRINTN, fp, fp,
                               &source, &result, &status);
        g_assert_cmphex(result.value.lo, ==,
                        formats[i].two | formats[i].negative_zero);
        g_assert_cmphex(result.fflags, ==, 0);
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_unary(AME_OP_FRINTM, fp, fp,
                               &source, &result, &status);
        g_assert_cmphex(result.value.lo, ==,
                        formats[i].two | formats[i].negative_zero);
        g_assert_cmphex(result.fflags, ==, 0);
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_unary(AME_OP_FRINTP, fp, fp,
                               &source, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].negative_one);
        g_assert_cmphex(result.fflags, ==, 0);
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_unary(AME_OP_FRINTZ, fp, fp,
                               &source, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].negative_one);
        g_assert_cmphex(result.fflags, ==, 0);
    }
}

static void assert_nonlinear_unary(AmeOperation op,
                                   const AmeDTypeDesc *dtype,
                                   uint64_t source, uint64_t expected,
                                   uint8_t expected_flags)
{
    AmeValue input = { .lo = source };
    AmeExecResult result;
    float_status status;

    g_assert_true(riscv_ame_fp_status_prepare(&status, dtype));
    riscv_ame_exec_fp_unary(op, dtype, dtype, &input,
                           &result, &status);
    g_assert_cmphex(result.value.lo, ==, expected);
    g_assert_cmphex(result.fflags, ==, expected_flags);
}

static void test_ieee_nonlinear_special(void)
{
    static const struct {
        AmeFPFormat format;
        uint64_t negative_zero;
        uint64_t negative_one;
        uint64_t infinity;
        uint64_t negative_infinity;
        uint64_t subnormal;
        uint64_t one;
        uint64_t two;
        uint64_t quiet_nan;
        uint64_t signaling_nan;
        uint64_t default_nan;
        uint64_t tanh_one;
    } formats[] = {
        { AME_FP_FORMAT_FP16, 0x8000, 0xbc00, 0x7c00, 0xfc00, 0x0001,
          0x3c00, 0x4000, 0x7e55, 0x7d00, 0x7e00, 0x3a18 },
        { AME_FP_FORMAT_BF16, 0x8000, 0xbf80, 0x7f80, 0xff80, 0x0001,
          0x3f80, 0x4000, 0x7fc5, 0x7fa0, 0x7fc0, 0x3f43 },
        { AME_FP_FORMAT_FP32, UINT32_C(0x80000000),
          UINT32_C(0xbf800000), UINT32_C(0x7f800000),
          UINT32_C(0xff800000), UINT32_C(0x00000001),
          UINT32_C(0x3f800000), UINT32_C(0x40000000),
          UINT32_C(0x7fc12345), UINT32_C(0x7fa00000),
          UINT32_C(0x7fc00000), UINT32_C(0x3f42f7d6) },
        { AME_FP_FORMAT_FP64, UINT64_C(0x8000000000000000),
          UINT64_C(0xbff0000000000000),
          UINT64_C(0x7ff0000000000000),
          UINT64_C(0xfff0000000000000),
          UINT64_C(0x0000000000000001),
          UINT64_C(0x3ff0000000000000),
          UINT64_C(0x4000000000000000),
          UINT64_C(0x7ff8123456789abc),
          UINT64_C(0x7ff4000000000000),
          UINT64_C(0x7ff8000000000000),
          UINT64_C(0x3fe85efab514f394) },
    };
    size_t i;

    for (i = 0; i < ARRAY_SIZE(formats); i++) {
        const AmeDTypeDesc *fp = fp_dtype(formats[i].format, AME_FP_RNE);

        assert_nonlinear_unary(AME_OP_EXP2, fp, 0, formats[i].one, 0);
        assert_nonlinear_unary(AME_OP_EXP2, fp, formats[i].one,
                               formats[i].two, 0);
        assert_nonlinear_unary(AME_OP_EXP2, fp, formats[i].infinity,
                               formats[i].infinity, 0);
        assert_nonlinear_unary(AME_OP_EXP2, fp,
                               formats[i].negative_infinity, 0, 0);
        assert_nonlinear_unary(AME_OP_EXP2, fp, formats[i].quiet_nan,
                               formats[i].default_nan, 0);
        assert_nonlinear_unary(AME_OP_EXP2, fp, formats[i].signaling_nan,
                               formats[i].default_nan, AME_FFLAG_NV);

        assert_nonlinear_unary(AME_OP_LOG2, fp, 0,
                               formats[i].negative_infinity, AME_FFLAG_DZ);
        assert_nonlinear_unary(AME_OP_LOG2, fp, formats[i].negative_one,
                               formats[i].default_nan, AME_FFLAG_NV);
        assert_nonlinear_unary(AME_OP_LOG2, fp, formats[i].one, 0, 0);
        assert_nonlinear_unary(AME_OP_LOG2, fp, formats[i].two,
                               formats[i].one, 0);
        assert_nonlinear_unary(AME_OP_LOG2, fp, formats[i].infinity,
                               formats[i].infinity, 0);

        assert_nonlinear_unary(AME_OP_RSQRT, fp, 0,
                               formats[i].infinity, AME_FFLAG_DZ);
        assert_nonlinear_unary(AME_OP_RSQRT, fp, formats[i].negative_zero,
                               formats[i].negative_infinity, AME_FFLAG_DZ);
        assert_nonlinear_unary(AME_OP_RSQRT, fp, formats[i].negative_one,
                               formats[i].default_nan, AME_FFLAG_NV);
        assert_nonlinear_unary(AME_OP_RSQRT, fp, formats[i].one,
                               formats[i].one, 0);
        assert_nonlinear_unary(AME_OP_RSQRT, fp, formats[i].infinity, 0, 0);

        assert_nonlinear_unary(AME_OP_TANH, fp, 0, 0, 0);
        assert_nonlinear_unary(AME_OP_TANH, fp, formats[i].negative_zero,
                               formats[i].negative_zero, 0);
        assert_nonlinear_unary(AME_OP_TANH, fp, formats[i].infinity,
                               formats[i].one, 0);
        assert_nonlinear_unary(AME_OP_TANH, fp,
                               formats[i].negative_infinity,
                               formats[i].negative_one, 0);
        assert_nonlinear_unary(AME_OP_TANH, fp, formats[i].one,
                               formats[i].tanh_one, AME_FFLAG_NX);
        assert_nonlinear_unary(AME_OP_TANH, fp, formats[i].subnormal,
                               formats[i].subnormal,
                               AME_FFLAG_UF | AME_FFLAG_NX);
    }

    assert_nonlinear_unary(
        AME_OP_LOG2, fp_dtype(AME_FP_FORMAT_FP64, AME_FP_RNE),
        UINT64_C(0x3fefffffffffffff), UINT64_C(0xbca71547652b82fe),
        AME_FFLAG_NX);
    assert_nonlinear_unary(
        AME_OP_LOG2, fp_dtype(AME_FP_FORMAT_FP64, AME_FP_RNE),
        UINT64_C(0x3ff0000000000001), UINT64_C(0x3cb71547652b82fd),
        AME_FFLAG_NX);
}

static uint64_t nonlinear_signature_update(uint64_t signature,
                                           const AmeExecResult *result)
{
    uint32_t encoded = result->value.lo | (uint32_t)result->fflags << 16;
    size_t i;

    for (i = 0; i < sizeof(encoded); i++) {
        signature ^= extract32(encoded, i * 8, 8);
        signature *= UINT64_C(0x100000001b3);
    }
    return signature;
}

static void test_tanh_accrued_flags(void)
{
    static const struct {
        AmeFPFormat format;
        uint64_t one;
    } formats[] = {
        { AME_FP_FORMAT_FP16, UINT16_C(0x3c00) },
        { AME_FP_FORMAT_BF16, UINT16_C(0x3f80) },
        { AME_FP_FORMAT_FP32, UINT32_C(0x3f800000) },
        { AME_FP_FORMAT_FP64, UINT64_C(0x3ff0000000000000) },
        { AME_FP_FORMAT_OCP_E4M3, UINT8_C(0x38) },
        { AME_FP_FORMAT_OCP_E5M2, UINT8_C(0x3c) },
    };
    static const FloatExceptionFlags prior_flags[] = {
        0, float_flag_inexact, float_flag_invalid, float_flag_underflow,
        float_flag_divbyzero | float_flag_overflow | float_flag_inexact,
    };

    for (size_t f = 0; f < ARRAY_SIZE(formats); f++) {
        for (AmeFPRoundMode mode = AME_FP_RNE; mode <= AME_FP_RNO; mode++) {
            const AmeDTypeDesc *fp = fp_dtype(formats[f].format, mode);
            uint64_t sign = BIT_ULL(fp->bits - 1);

            if (fp->class == AME_DTYPE_CLASS_OCP && mode == AME_FP_RNO) {
                continue;
            }
            for (size_t p = 0; p < ARRAY_SIZE(prior_flags); p++) {
                float_status status;
                AmeExecResult result;
                AmeValue input;
                uint8_t prior;

                g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
                float_raise(prior_flags[p], &status);
                prior = riscv_ame_fp_status_flags(&status);
                for (unsigned negative = 0; negative < 2; negative++) {
                    input = (AmeValue) { .lo = negative ? sign : 0 };
                    riscv_ame_exec_fp_unary(
                        AME_OP_TANH, fp, fp, &input, &result, &status);
                    g_assert_cmphex(result.value.lo, ==, input.lo);
                    g_assert_cmphex(result.fflags, ==, prior);
                }

                input = (AmeValue) { .lo = formats[f].one };
                riscv_ame_exec_fp_unary(
                    AME_OP_TANH, fp, fp, &input, &result, &status);
                g_assert_cmphex(result.fflags, ==, prior | AME_FFLAG_NX);
                input.lo = 0;
                riscv_ame_exec_fp_unary(
                    AME_OP_TANH, fp, fp, &input, &result, &status);
                g_assert_cmphex(result.value.lo, ==, 0);
                g_assert_cmphex(result.fflags, ==, prior | AME_FFLAG_NX);

                /* A nonzero subnormal is genuinely tiny and inexact. */
                input.lo = 1;
                riscv_ame_exec_fp_unary(
                    AME_OP_TANH, fp, fp, &input, &result, &status);
                g_assert_cmphex(result.fflags, ==,
                                prior | AME_FFLAG_UF | AME_FFLAG_NX);
            }
        }
    }
}

static void test_ieee_nonlinear_exhaustive(void)
{
    static const uint64_t expected[2][6][4] = {
        {
            { UINT64_C(0x437671f4f94a416a),
              UINT64_C(0x25d24e43dbfe23e7),
              UINT64_C(0xbd0ed9f011c7edc3),
              UINT64_C(0x8585a4c45fe5e865) },
            { UINT64_C(0xb934a19f193b51e0),
              UINT64_C(0xa0c9fdcbdde4e0a3),
              UINT64_C(0x53541680bdc70eca),
              UINT64_C(0x7504468ae1df1b9d) },
            { UINT64_C(0xb934a19f193b51e0),
              UINT64_C(0xd613fd7c7c73b6e0),
              UINT64_C(0x53541680bdc70eca),
              UINT64_C(0x0c83fa87de99516e) },
            { UINT64_C(0xf238713619d27972),
              UINT64_C(0x7cc8e31fce96bd90),
              UINT64_C(0x543ae38a164ec535),
              UINT64_C(0xbe902c1a450249f2) },
            { UINT64_C(0x437671f4f94a416a),
              UINT64_C(0x25d24e43dbfe23e7),
              UINT64_C(0x87e57f21d0d858ae),
              UINT64_C(0x8585a4c45fe5e865) },
            { UINT64_C(0xd404df9d8462e9b0),
              UINT64_C(0x3a7cd3daf4c8bda2),
              UINT64_C(0xee78c822e255ba7e),
              UINT64_C(0x66ce673bb5d8d4bd) },
        },
        {
            { UINT64_C(0xf8906f3674bdaff6),
              UINT64_C(0x9a26edf0763c4114),
              UINT64_C(0x5c8c2656e4928967),
              UINT64_C(0xe0837477f163c5d5) },
            { UINT64_C(0xd2a976dc0e2f1559),
              UINT64_C(0x585c7343bd026893),
              UINT64_C(0xfbaa31e9789374f7),
              UINT64_C(0x6bec323d33b20205) },
            { UINT64_C(0xd2a976dc0e2f1559),
              UINT64_C(0x4a34091ff8fcda1f),
              UINT64_C(0xfbaa31e9789374f7),
              UINT64_C(0x17739e1900267089) },
            { UINT64_C(0x70817a2ef904cd50),
              UINT64_C(0xaa2509110d93d59f),
              UINT64_C(0x5c232beb77721b80),
              UINT64_C(0x4b5e89c738f7eb45) },
            { UINT64_C(0xf8906f3674bdaff6),
              UINT64_C(0x9a26edf0763c4114),
              UINT64_C(0x47171e33ae2bb046),
              UINT64_C(0xe0837477f163c5d5) },
            { UINT64_C(0xe861bd6cfa431a28),
              UINT64_C(0x4e41e503302b047a),
              UINT64_C(0xb573deaaf0f7dfde),
              UINT64_C(0x44a094ef8398bef5) },
        },
    };
    static const AmeFPFormat formats[] = {
        AME_FP_FORMAT_FP16,
        AME_FP_FORMAT_BF16,
    };
    static const AmeOperation operations[] = {
        AME_OP_RSQRT,
        AME_OP_LOG2,
        AME_OP_EXP2,
        AME_OP_TANH,
    };
    size_t format, rounding, operation;

    for (format = 0; format < ARRAY_SIZE(formats); format++) {
        for (rounding = AME_FP_RNE; rounding <= AME_FP_RNO; rounding++) {
            const AmeDTypeDesc *fp = fp_dtype(formats[format], rounding);

            for (operation = 0; operation < ARRAY_SIZE(operations);
                 operation++) {
                uint64_t signature = UINT64_C(0xcbf29ce484222325);
                uint32_t raw;

                for (raw = 0; raw <= UINT16_MAX; raw++) {
                    AmeValue source = { .lo = raw };
                    AmeExecResult result;
                    float_status status;

                    g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
                    riscv_ame_exec_fp_unary(
                        operations[operation], fp, fp, &source, &result,
                        &status);
                    signature = nonlinear_signature_update(signature,
                                                            &result);
                }
                g_assert_cmphex(signature, ==,
                                expected[format][rounding][operation]);
            }
        }
    }
}

static void test_ieee_sincos(void)
{
    const AmeDTypeDesc *fp16 = fp_dtype(AME_FP_FORMAT_FP16, AME_FP_RNE);
    const AmeDTypeDesc *bf16 = fp_dtype(AME_FP_FORMAT_BF16, AME_FP_RNE);
    float_status status;
    AmeExecResult result;
    AmeValue source;

    source.lo = UINT16_C(0x0000);
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp16));
    riscv_ame_exec_fp_unary(AME_OP_SIN, fp16, fp16,
                           &source, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT16_C(0x0000));
    g_assert_cmphex(result.fflags, ==, 0);
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp16));
    riscv_ame_exec_fp_unary(AME_OP_COS, fp16, fp16,
                           &source, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT16_C(0x3c00));
    g_assert_cmphex(result.fflags, ==, 0);

    source.lo = UINT16_C(0x8000);
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp16));
    riscv_ame_exec_fp_unary(AME_OP_SIN, fp16, fp16,
                           &source, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT16_C(0x8000));
    g_assert_cmphex(result.fflags, ==, 0);
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp16));
    riscv_ame_exec_fp_unary(AME_OP_COS, fp16, fp16,
                           &source, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT16_C(0x3c00));
    g_assert_cmphex(result.fflags, ==, 0);

    source.lo = UINT16_C(0x3c00);
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp16));
    riscv_ame_exec_fp_unary(AME_OP_SIN, fp16, fp16,
                           &source, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT16_C(0x3abb));
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp16));
    riscv_ame_exec_fp_unary(AME_OP_COS, fp16, fp16,
                           &source, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT16_C(0x3853));
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);

    source.lo = UINT16_C(0x0001);
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp16));
    riscv_ame_exec_fp_unary(AME_OP_SIN, fp16, fp16,
                           &source, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT16_C(0x0001));
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_UF | AME_FFLAG_NX);

    source.lo = UINT16_C(0x7c00);
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp16));
    riscv_ame_exec_fp_unary(AME_OP_SIN, fp16, fp16,
                           &source, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT16_C(0x7e00));
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NV);
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp16));
    riscv_ame_exec_fp_unary(AME_OP_COS, fp16, fp16,
                           &source, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT16_C(0x7e00));
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NV);

    source.lo = UINT16_C(0x7e55);
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp16));
    riscv_ame_exec_fp_unary(AME_OP_SIN, fp16, fp16,
                           &source, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT16_C(0x7e00));
    g_assert_cmphex(result.fflags, ==, 0);
    source.lo = UINT16_C(0x7d00);
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp16));
    riscv_ame_exec_fp_unary(AME_OP_COS, fp16, fp16,
                           &source, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT16_C(0x7e00));
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NV);

    source.lo = 0;
    g_assert_true(riscv_ame_fp_status_prepare(&status, bf16));
    riscv_ame_exec_fp_unary(AME_OP_SIN, bf16, bf16,
                           &source, &result, &status);
    g_assert_cmphex(result.value.lo, ==, 0);
    g_assert_cmphex(result.fflags, ==, 0);
}

static void test_ieee_sincos_exhaustive(void)
{
    static const uint64_t expected[6][2] = {
        { UINT64_C(0x425b56923a1075bd),
          UINT64_C(0xd6feac23c3d04179) },
        { UINT64_C(0x0a84bc9790c036e5),
          UINT64_C(0x27bf296cfaa42f9d) },
        { UINT64_C(0x432fe96cf09d0a35),
          UINT64_C(0x51087c13db09ede9) },
        { UINT64_C(0xbdb94351aa29fa21),
          UINT64_C(0x5a17571e93a6a55d) },
        { UINT64_C(0x425b56923a1075bd),
          UINT64_C(0xd6feac23c3d04179) },
        { UINT64_C(0x95dbea1daf19f04d),
          UINT64_C(0xae3626b05b4aadbd) },
    };
    static const AmeOperation operations[] = {
        AME_OP_SIN,
        AME_OP_COS,
    };
    size_t rounding, operation;

    for (rounding = AME_FP_RNE; rounding <= AME_FP_RNO; rounding++) {
        const AmeDTypeDesc *fp = fp_dtype(AME_FP_FORMAT_FP16, rounding);

        for (operation = 0; operation < ARRAY_SIZE(operations); operation++) {
            uint64_t signature = UINT64_C(0xcbf29ce484222325);
            uint32_t raw;

            for (raw = 0; raw <= UINT16_MAX; raw++) {
                AmeValue source = { .lo = raw };
                AmeExecResult result;
                float_status status;

                g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
                riscv_ame_exec_fp_unary(
                    operations[operation], fp, fp, &source, &result,
                    &status);
                signature = nonlinear_signature_update(signature, &result);
            }
            g_assert_cmphex(signature, ==,
                            expected[rounding][operation]);
        }
    }
}

static void test_ieee_log2_composite(void)
{
    static const struct {
        AmeFPFormat format;
        uint64_t negative_two;
        uint64_t two;
        uint64_t three;
    } formats[] = {
        { AME_FP_FORMAT_FP16, 0xc000, 0x4000, 0x4200 },
        { AME_FP_FORMAT_BF16, 0xc000, 0x4000, 0x4040 },
        { AME_FP_FORMAT_FP32, UINT32_C(0xc0000000),
          UINT32_C(0x40000000), UINT32_C(0x40400000) },
        { AME_FP_FORMAT_FP64, UINT64_C(0xc000000000000000),
          UINT64_C(0x4000000000000000),
          UINT64_C(0x4008000000000000) },
    };
    size_t i;

    for (i = 0; i < ARRAY_SIZE(formats); i++) {
        const AmeDTypeDesc *fp = fp_dtype(formats[i].format, AME_FP_RNE);
        AmeValue left = { .lo = formats[i].negative_two };
        AmeValue right = { .lo = formats[i].three };
        AmeExecResult result;
        float_status status;

        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_binary(
            AME_OP_LOG2SUB, fp, fp, &left, fp, &right, &result,
            &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].negative_two);
        g_assert_cmphex(result.fflags, ==, 0);

        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_binary(
            AME_OP_SUBLOG2, fp, fp, &left, fp, &right, &result,
            &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].two);
        g_assert_cmphex(result.fflags, ==, 0);
    }
}

static void test_ieee_binary(void)
{
    static const uint32_t halfway_expected[] = {
        [AME_FP_RNE] = UINT32_C(0x3f800000),
        [AME_FP_RTZ] = UINT32_C(0x3f800000),
        [AME_FP_RDN] = UINT32_C(0x3f800000),
        [AME_FP_RUP] = UINT32_C(0x3f800001),
        [AME_FP_RMM] = UINT32_C(0x3f800001),
        [AME_FP_RNO] = UINT32_C(0x3f800001),
    };
    static const struct {
        AmeFPFormat format;
        uint64_t one;
        uint64_t two;
        uint64_t three;
        uint64_t half;
        uint64_t negative_zero;
        uint64_t infinity;
        uint64_t negative_infinity;
        uint64_t maximum;
        uint64_t subnormal;
        uint64_t quiet_nan;
        uint64_t signaling_nan;
        uint64_t default_nan;
    } formats[] = {
        { AME_FP_FORMAT_FP16, 0x3c00, 0x4000, 0x4200, 0x3800, 0x8000,
          0x7c00, 0xfc00, 0x7bff, 0x0001, 0x7e55, 0x7d00, 0x7e00 },
        { AME_FP_FORMAT_BF16, 0x3f80, 0x4000, 0x4040, 0x3f00, 0x8000,
          0x7f80, 0xff80, 0x7f7f, 0x0001, 0x7fc5, 0x7fa0, 0x7fc0 },
        { AME_FP_FORMAT_FP32, 0x3f800000, 0x40000000, 0x40400000,
          0x3f000000, 0x80000000, 0x7f800000, 0xff800000, 0x7f7fffff,
          0x00000001, 0x7fc12345, 0x7fa00000, 0x7fc00000 },
        { AME_FP_FORMAT_FP64, UINT64_C(0x3ff0000000000000),
          UINT64_C(0x4000000000000000), UINT64_C(0x4008000000000000),
          UINT64_C(0x3fe0000000000000), UINT64_C(0x8000000000000000),
          UINT64_C(0x7ff0000000000000), UINT64_C(0xfff0000000000000),
          UINT64_C(0x7fefffffffffffff), UINT64_C(0x0000000000000001),
          UINT64_C(0x7ff8123456789abc), UINT64_C(0x7ff4000000000000),
          UINT64_C(0x7ff8000000000000) },
    };
    const AmeDTypeDesc *fp32 = fp_dtype(AME_FP_FORMAT_FP32, AME_FP_RNE);
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    const AmeDTypeDesc *int64 = dtype(UINT32_C(0x40000040));
    const AmeDTypeDesc *int128 = dtype(UINT32_C(0x40000080));
    float_status status;
    AmeExecResult result;
    AmeValue left, right;
    size_t i;

    for (i = 0; i < ARRAY_SIZE(formats); i++) {
        const AmeDTypeDesc *fp = fp_dtype(formats[i].format, AME_FP_RNE);

        left.lo = formats[i].one;
        right.lo = formats[i].two;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_binary(AME_OP_ADD, fp, fp, &left,
                                fp, &right, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].three);
        g_assert_cmphex(result.fflags, ==, 0);

        left.lo = formats[i].maximum;
        right = left;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_binary(AME_OP_ADD, fp, fp, &left,
                                fp, &right, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].infinity);
        g_assert_cmphex(result.fflags, ==, AME_FFLAG_OF | AME_FFLAG_NX);

        left.lo = formats[i].subnormal;
        right.lo = formats[i].half;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_binary(AME_OP_MUL, fp, fp, &left,
                                fp, &right, &result, &status);
        g_assert_cmphex(result.value.lo, ==, 0);
        g_assert_cmphex(result.fflags, ==, AME_FFLAG_UF | AME_FFLAG_NX);

        left.lo = formats[i].infinity;
        right.lo = formats[i].negative_infinity;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_binary(AME_OP_ADD, fp, fp, &left,
                                fp, &right, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].default_nan);
        g_assert_cmphex(result.fflags, ==, AME_FFLAG_NV);

        left.lo = 0;
        right.lo = formats[i].infinity;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_binary(AME_OP_MUL, fp, fp, &left,
                                fp, &right, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].default_nan);
        g_assert_cmphex(result.fflags, ==, AME_FFLAG_NV);

        left.lo = formats[i].negative_zero;
        right.lo = 0;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_binary(AME_OP_MIN, fp, fp, &left,
                                fp, &right, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].negative_zero);
        g_assert_cmphex(result.fflags, ==, 0);
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_binary(AME_OP_MAX, fp, fp, &left,
                                fp, &right, &result, &status);
        g_assert_cmphex(result.value.lo, ==, 0);
        g_assert_cmphex(result.fflags, ==, 0);

        left.lo = formats[i].quiet_nan;
        right.lo = formats[i].one;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_binary(AME_OP_MIN, fp, fp, &left,
                                fp, &right, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].default_nan);
        g_assert_cmphex(result.fflags, ==, 0);
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_binary(AME_OP_CMPLT, int8, fp,
                                &left, fp, &right, &result,
                                &status);
        g_assert_cmphex(result.value.lo, ==, 0);
        g_assert_cmphex(result.fflags, ==, 0);

        left.lo = formats[i].signaling_nan;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_binary(AME_OP_MAX, fp, fp, &left,
                                fp, &right, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].default_nan);
        g_assert_cmphex(result.fflags, ==, AME_FFLAG_NV);
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_binary(AME_OP_CMPGE, int8, fp,
                                &left, fp, &right, &result,
                                &status);
        g_assert_cmphex(result.value.lo, ==, 0);
        g_assert_cmphex(result.fflags, ==, AME_FFLAG_NV);

        left.lo = formats[i].maximum;
        right = (AmeValue) { .lo = 1 };
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_binary(AME_OP_LDEXP, fp, fp,
                                &left, int8, &right, &result,
                                &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].infinity);
        g_assert_cmphex(result.fflags, ==, AME_FFLAG_OF | AME_FFLAG_NX);

        left.lo = formats[i].subnormal;
        right = (AmeValue) { .lo = UINT64_MAX };
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_binary(AME_OP_LDEXP, fp, fp,
                                &left, int8, &right, &result,
                                &status);
        g_assert_cmphex(result.value.lo, ==, 0);
        g_assert_cmphex(result.fflags, ==, AME_FFLAG_UF | AME_FFLAG_NX);
    }

    left.lo = UINT32_C(0x3f800000);
    right.lo = UINT32_C(0x33800000); /* Half an FP32 ulp at 1.0. */
    for (i = 0; i < ARRAY_SIZE(halfway_expected); i++) {
        const AmeDTypeDesc *dest = fp_dtype(AME_FP_FORMAT_FP32, i);

        g_assert_true(riscv_ame_fp_status_prepare(&status, dest));
        riscv_ame_exec_fp_binary(AME_OP_ADD, dest, fp32,
                                &left, fp32, &right, &result,
                                &status);
        g_assert_cmphex(result.value.lo, ==, halfway_expected[i]);
        g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);
    }

    left.lo = UINT32_C(0x7f7fffff);
    right = left;
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    riscv_ame_exec_fp_binary(AME_OP_ADD, fp32, fp32, &left,
                            fp32, &right, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT32_C(0x7f800000));
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_OF | AME_FFLAG_NX);

    left.lo = UINT32_C(0x00000001);
    right.lo = UINT32_C(0x3f000000);
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    riscv_ame_exec_fp_binary(AME_OP_MUL, fp32, fp32, &left,
                            fp32, &right, &result, &status);
    g_assert_cmphex(result.value.lo, ==, 0);
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_UF | AME_FFLAG_NX);

    left.lo = UINT32_C(0x33000000); /* 2^-25 */
    right.lo = UINT32_C(0x3f800000); /* 1 */
    g_assert_true(riscv_ame_fp_status_prepare(
        &status, fp_dtype(AME_FP_FORMAT_FP32, AME_FP_RDN)));
    riscv_ame_exec_fp_binary(
        AME_OP_ABSDIFF, fp_dtype(AME_FP_FORMAT_FP32, AME_FP_RDN), fp32,
        &left, fp32, &right, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT32_C(0x3f7fffff));
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);
    g_assert_true(riscv_ame_fp_status_prepare(
        &status, fp_dtype(AME_FP_FORMAT_FP32, AME_FP_RUP)));
    riscv_ame_exec_fp_binary(
        AME_OP_ABSDIFF, fp_dtype(AME_FP_FORMAT_FP32, AME_FP_RUP), fp32,
        &left, fp32, &right, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT32_C(0x3f800000));
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);

    left.lo = UINT32_C(0x3f800001);
    right = left;
    g_assert_true(riscv_ame_fp_status_prepare(
        &status, fp_dtype(AME_FP_FORMAT_FP32, AME_FP_RDN)));
    riscv_ame_exec_fp_binary(
        AME_OP_NMUL, fp_dtype(AME_FP_FORMAT_FP32, AME_FP_RDN), fp32,
        &left, fp32, &right, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT32_C(0xbf800003));
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);
    g_assert_true(riscv_ame_fp_status_prepare(
        &status, fp_dtype(AME_FP_FORMAT_FP32, AME_FP_RUP)));
    riscv_ame_exec_fp_binary(
        AME_OP_NMUL, fp_dtype(AME_FP_FORMAT_FP32, AME_FP_RUP), fp32,
        &left, fp32, &right, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT32_C(0xbf800002));
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);

    left.lo = UINT32_C(0x7f800000);
    right.lo = UINT32_C(0xff800000);
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    riscv_ame_exec_fp_binary(AME_OP_ADD, fp32, fp32, &left,
                            fp32, &right, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT32_C(0x7fc00000));
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NV);

    left.lo = UINT32_C(0x3f800000);  /* A = 1 */
    right.lo = UINT32_C(0x40400000); /* B = 3 */
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    riscv_ame_exec_fp_binary(AME_OP_SUB, fp32, fp32, &left,
                            fp32, &right, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT32_C(0x40000000));
    g_assert_cmphex(result.fflags, ==, 0);

    left.lo = UINT32_C(0x80000000);
    right.lo = 0;
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    riscv_ame_exec_fp_binary(AME_OP_MIN, fp32, fp32, &left,
                            fp32, &right, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT32_C(0x80000000));
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    riscv_ame_exec_fp_binary(AME_OP_MAX, fp32, fp32, &left,
                            fp32, &right, &result, &status);
    g_assert_cmphex(result.value.lo, ==, 0);

    left.lo = UINT32_C(0x7fc12345);
    right.lo = UINT32_C(0x3f800000);
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    riscv_ame_exec_fp_binary(AME_OP_MIN, fp32, fp32, &left,
                            fp32, &right, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT32_C(0x7fc00000));
    g_assert_cmphex(result.fflags, ==, 0);

    left.lo = UINT32_C(0x7f812345);
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    riscv_ame_exec_fp_binary(AME_OP_MAX, fp32, fp32, &left,
                            fp32, &right, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT32_C(0x7fc00000));
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NV);

    left.lo = UINT32_C(0x7fc12345);
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    riscv_ame_exec_fp_binary(AME_OP_CMPLT, int8, fp32, &left,
                            fp32, &right, &result, &status);
    g_assert_cmphex(result.value.lo, ==, 0);
    g_assert_cmphex(result.fflags, ==, 0);
    left.lo = UINT32_C(0x7f812345);
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    riscv_ame_exec_fp_binary(AME_OP_CMPGE, int8, fp32, &left,
                            fp32, &right, &result, &status);
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NV);

    left.lo = UINT32_C(0x3f800000);
    right.lo = UINT64_MAX;
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    riscv_ame_exec_fp_binary(AME_OP_LDEXP, fp32, fp32, &left,
                            int8, &right, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT32_C(0x3f000000));

    /* Scaling clamps only after applying the mrdexp exponent negation. */
    left.lo = UINT32_C(0x3f800000);
    right = (AmeValue) { .lo = (uint64_t)INT_MAX + 1 };
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    riscv_ame_exec_fp_binary(AME_OP_RDEXP, fp32, fp32,
                            &left, int64, &right, &result,
                            &status);
    g_assert_cmphex(result.value.lo, ==, 0);
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_UF | AME_FFLAG_NX);

    right = (AmeValue) { .lo = 0, .hi = 1 };
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    riscv_ame_exec_fp_binary(AME_OP_RDEXP, fp32, fp32,
                            &left, int128, &right, &result,
                            &status);
    g_assert_cmphex(result.value.lo, ==, 0);
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_UF | AME_FFLAG_NX);

    right = (AmeValue) { .lo = 0, .hi = UINT64_MAX };
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    riscv_ame_exec_fp_binary(AME_OP_RDEXP, fp32, fp32,
                            &left, int128, &right, &result,
                            &status);
    g_assert_cmphex(result.value.lo, ==, UINT32_C(0x7f800000));
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_OF | AME_FFLAG_NX);

    right = (AmeValue) { .lo = 1, .hi = 0 };
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    riscv_ame_exec_fp_binary(AME_OP_LDEXP, fp32, fp32, &left,
                            int128, &right, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT32_C(0x40000000));
    right = (AmeValue) { .lo = UINT64_MAX, .hi = UINT64_MAX };
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    riscv_ame_exec_fp_binary(AME_OP_LDEXP, fp32, fp32, &left,
                            int128, &right, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT32_C(0x3f000000));

    left.lo = UINT32_C(0x7f7fffff);
    right = (AmeValue) { .lo = 1 };
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    riscv_ame_exec_fp_binary(AME_OP_LDEXP, fp32, fp32, &left,
                            int8, &right, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT32_C(0x7f800000));
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_OF | AME_FFLAG_NX);

    left.lo = UINT32_C(0x00000001);
    right = (AmeValue) { .lo = UINT64_MAX };
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    riscv_ame_exec_fp_binary(AME_OP_LDEXP, fp32, fp32, &left,
                            int8, &right, &result, &status);
    g_assert_cmphex(result.value.lo, ==, 0);
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_UF | AME_FFLAG_NX);

    left.lo = UINT32_C(0x80000000);
    right = (AmeValue) { .lo = 100 };
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    riscv_ame_exec_fp_binary(AME_OP_LDEXP, fp32, fp32, &left,
                            int8, &right, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT32_C(0x80000000));
    g_assert_cmphex(result.fflags, ==, 0);
}

static void test_ieee_fused(void)
{
    static const struct {
        AmeFPFormat format;
        uint64_t one;
        uint64_t two;
        uint64_t three;
        uint64_t ten;
        uint64_t twenty_three;
        uint64_t negative_seventeen;
        uint64_t sixteen;
        uint64_t four;
        uint64_t infinity;
        uint64_t default_nan;
        uint64_t half_ulp;
        uint64_t next_one;
    } formats[] = {
        { AME_FP_FORMAT_FP16, 0x3c00, 0x4000, 0x4200, 0x4900, 0x4dc0,
          0xcc40, 0x4c00, 0x4400, 0x7c00, 0x7e00, 0x1000, 0x3c01 },
        { AME_FP_FORMAT_BF16, 0x3f80, 0x4000, 0x4040, 0x4120, 0x41b8,
          0xc188, 0x4180, 0x4080, 0x7f80, 0x7fc0, 0x3b80, 0x3f81 },
        { AME_FP_FORMAT_FP32, 0x3f800000, 0x40000000, 0x40400000,
          0x41200000, 0x41b80000, 0xc1880000, 0x41800000, 0x40800000,
          0x7f800000, 0x7fc00000, 0x33800000, 0x3f800001 },
        { AME_FP_FORMAT_FP64, UINT64_C(0x3ff0000000000000),
          UINT64_C(0x4000000000000000), UINT64_C(0x4008000000000000),
          UINT64_C(0x4024000000000000), UINT64_C(0x4037000000000000),
          UINT64_C(0xc031000000000000), UINT64_C(0x4030000000000000),
          UINT64_C(0x4010000000000000), UINT64_C(0x7ff0000000000000),
          UINT64_C(0x7ff8000000000000), UINT64_C(0x3ca0000000000000),
          UINT64_C(0x3ff0000000000001) },
    };
    static const bool rounds_up[] = {
        [AME_FP_RNE] = false,
        [AME_FP_RTZ] = false,
        [AME_FP_RDN] = false,
        [AME_FP_RUP] = true,
        [AME_FP_RMM] = true,
        [AME_FP_RNO] = true,
    };
    const AmeDTypeDesc *fp32 = fp_dtype(AME_FP_FORMAT_FP32, AME_FP_RNE);
    float_status status;
    AmeExecResult result;
    AmeValue left, right, current;
    size_t i, round;

    for (i = 0; i < ARRAY_SIZE(formats); i++) {
        const AmeDTypeDesc *fp = fp_dtype(formats[i].format, AME_FP_RNE);

        left.lo = formats[i].ten;
        right.lo = formats[i].two;
        current.lo = formats[i].three;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        exec_fp_ternary_same_dtype(AME_OP_MULACC, fp, &left,
                                 &right, &current, &result,
                                 &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].twenty_three);
        g_assert_cmphex(result.fflags, ==, 0);
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        exec_fp_ternary_same_dtype(AME_OP_MULNACC, fp, &left,
                                 &right, &current, &result,
                                 &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].negative_seventeen);
        g_assert_cmphex(result.fflags, ==, 0);
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        exec_fp_ternary_same_dtype(AME_OP_MULADD, fp, &left,
                                 &right, &current, &result,
                                 &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].sixteen);
        g_assert_cmphex(result.fflags, ==, 0);
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        exec_fp_ternary_same_dtype(AME_OP_MULSUB, fp, &left,
                                 &right, &current, &result,
                                 &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].four);
        g_assert_cmphex(result.fflags, ==, 0);

        left.lo = 0;
        right.lo = formats[i].infinity;
        current.lo = 0;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        exec_fp_ternary_same_dtype(AME_OP_MULACC, fp, &left,
                                 &right, &current, &result,
                                 &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].default_nan);
        g_assert_cmphex(result.fflags, ==, AME_FFLAG_NV);

        left.lo = formats[i].one;
        right.lo = formats[i].half_ulp;
        current.lo = formats[i].one;
        for (round = 0; round < ARRAY_SIZE(rounds_up); round++) {
            const AmeDTypeDesc *dest = fp_dtype(formats[i].format, round);

            g_assert_true(riscv_ame_fp_status_prepare(&status, dest));
            exec_fp_ternary_same_dtype(
                AME_OP_MULACC, dest, &left, &right, &current, &result,
                &status);
            g_assert_cmphex(result.value.lo, ==,
                            rounds_up[round] ? formats[i].next_one :
                            formats[i].one);
            g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);
        }
    }

    /* This result distinguishes one fused operation from rounded mul+add. */
    left.lo = UINT32_C(0x3f800001);
    right.lo = UINT32_C(0x3f7fffff);
    current.lo = UINT32_C(0xbf800000);
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    exec_fp_ternary_same_dtype(AME_OP_MULACC, fp32, &left,
                             &right, &current, &result,
                             &status);
    g_assert_cmphex(result.value.lo, ==, UINT32_C(0x337ffffe));
    g_assert_cmphex(result.fflags, ==, 0);

    left.lo = UINT32_C(0x3f800000);
    right.lo = UINT32_C(0x7f800000);
    current.lo = UINT32_C(0xff800000);
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    exec_fp_ternary_same_dtype(AME_OP_MULACC, fp32, &left,
                             &right, &current, &result,
                             &status);
    g_assert_cmphex(result.value.lo, ==, UINT32_C(0x7fc00000));
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NV);
}

static void test_ieee_scale_acc(void)
{
    static const struct {
        AmeFPFormat format;
        uint64_t three;
        uint64_t seven;
        uint64_t ten;
        uint64_t sixteen;
        uint64_t twenty_three;
        uint64_t maximum;
        uint64_t negative_maximum;
        uint64_t infinity;
        uint64_t negative_infinity;
        uint64_t default_nan;
        uint64_t minimum_subnormal;
        uint64_t twice_minimum_subnormal;
    } formats[] = {
        { AME_FP_FORMAT_FP16, 0x4200, 0x4700, 0x4900, 0x4c00, 0x4dc0,
          0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00, 0x0001, 0x0002 },
        { AME_FP_FORMAT_BF16, 0x4040, 0x40e0, 0x4120, 0x4180, 0x41b8,
          0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc0, 0x0001, 0x0002 },
        { AME_FP_FORMAT_FP32, UINT32_C(0x40400000),
          UINT32_C(0x40e00000), UINT32_C(0x41200000),
          UINT32_C(0x41800000), UINT32_C(0x41b80000),
          UINT32_C(0x7f7fffff), UINT32_C(0xff7fffff),
          UINT32_C(0x7f800000), UINT32_C(0xff800000),
          UINT32_C(0x7fc00000), UINT32_C(0x00000001),
          UINT32_C(0x00000002) },
        { AME_FP_FORMAT_FP64, UINT64_C(0x4008000000000000),
          UINT64_C(0x401c000000000000), UINT64_C(0x4024000000000000),
          UINT64_C(0x4030000000000000), UINT64_C(0x4037000000000000),
          UINT64_C(0x7fefffffffffffff), UINT64_C(0xffefffffffffffff),
          UINT64_C(0x7ff0000000000000), UINT64_C(0xfff0000000000000),
          UINT64_C(0x7ff8000000000000), UINT64_C(0x0000000000000001),
          UINT64_C(0x0000000000000002) },
    };
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    const AmeDTypeDesc *bf16 = fp_dtype(AME_FP_FORMAT_BF16, AME_FP_RNE);
    float_status status;
    AmeExecResult result;
    AmeValue data, exponent, current;
    size_t i;

    for (i = 0; i < ARRAY_SIZE(formats); i++) {
        const AmeDTypeDesc *fp = fp_dtype(formats[i].format, AME_FP_RNE);

        data.lo = formats[i].ten;
        exponent.lo = 1;
        current.lo = formats[i].three;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_scale_acc(
            AME_OP_LDEXPACC, fp, fp, &data, int8, &exponent, &current,
            &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].twenty_three);
        g_assert_cmphex(result.fflags, ==, 0);

        data.lo = formats[i].sixteen;
        exponent.lo = 2;
        current.lo = formats[i].three;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_scale_acc(
            AME_OP_RDEXPACC, fp, fp, &data, int8, &exponent, &current,
            &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].seven);
        g_assert_cmphex(result.fflags, ==, 0);

        /* Scaling the product must not overflow before cancellation. */
        data.lo = formats[i].maximum;
        exponent.lo = 1;
        current.lo = formats[i].negative_maximum;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_scale_acc(
            AME_OP_LDEXPACC, fp, fp, &data, int8, &exponent, &current,
            &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].maximum);
        g_assert_cmphex(result.fflags, ==, 0);

        /* The exact 1.5-minsubnormal sum is rounded only once. */
        data.lo = formats[i].minimum_subnormal;
        exponent.lo = UINT8_MAX;
        current.lo = formats[i].minimum_subnormal;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_scale_acc(
            AME_OP_LDEXPACC, fp, fp, &data, int8, &exponent, &current,
            &result, &status);
        g_assert_cmphex(result.value.lo, ==,
                        formats[i].twice_minimum_subnormal);
        g_assert_cmphex(result.fflags, ==,
                        AME_FFLAG_UF | AME_FFLAG_NX);

        data.lo = formats[i].infinity;
        exponent.lo = 0;
        current.lo = formats[i].negative_infinity;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_scale_acc(
            AME_OP_LDEXPACC, fp, fp, &data, int8, &exponent, &current,
            &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].default_nan);
        g_assert_cmphex(result.fflags, ==, AME_FFLAG_NV);
    }

    data = (AmeValue) { .lo = UINT16_C(0x3c00) };
    exponent = (AmeValue) { 0 };
    current = (AmeValue) { .lo = UINT16_C(0x3f80) };
    g_assert_true(riscv_ame_fp_status_prepare(&status, bf16));
    riscv_ame_exec_fp_scale_acc(
        AME_OP_LDEXPACC, bf16, fp_dtype(AME_FP_FORMAT_FP16, AME_FP_RNE),
        &data, int8, &exponent, &current, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT16_C(0x4000));
    g_assert_cmphex(result.fflags, ==, 0);
}

static void test_fp_mixed_operations(void)
{
    const AmeDTypeDesc *fp16 = fp_dtype(AME_FP_FORMAT_FP16, AME_FP_RNE);
    const AmeDTypeDesc *bf16 = fp_dtype(AME_FP_FORMAT_BF16, AME_FP_RNE);
    const AmeDTypeDesc *fp32 = fp_dtype(AME_FP_FORMAT_FP32, AME_FP_RNE);
    const AmeDTypeDesc *fp64 = fp_dtype(AME_FP_FORMAT_FP64, AME_FP_RNE);
    AmeValue fp16_one_point_five = { .lo = UINT16_C(0x3e00) };
    AmeValue bf16_two = { .lo = UINT16_C(0x4000) };
    AmeValue fp32_quarter = { .lo = UINT32_C(0x3e800000) };
    AmeExecResult result;
    float_status status;

    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    riscv_ame_exec_fp_binary(
        AME_OP_ADD, fp32, fp16, &fp16_one_point_five,
        bf16, &bf16_two, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT32_C(0x40600000));
    g_assert_cmphex(result.fflags, ==, 0);

    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    riscv_ame_exec_fp_ternary(
        AME_OP_MULACC, fp32, fp16, &fp16_one_point_five,
        bf16, &bf16_two, &fp32_quarter, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT32_C(0x40500000));
    g_assert_cmphex(result.fflags, ==, 0);

    g_assert_true(riscv_ame_fp_status_prepare(&status, fp64));
    riscv_ame_exec_fp_unary(
        AME_OP_ABS, fp64, fp16, &fp16_one_point_five,
        &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT64_C(0x3ff8000000000000));

    g_assert_true(riscv_ame_fp_status_prepare(&status, fp16));
    riscv_ame_exec_fp_binary(
        AME_OP_ADD, fp16, fp32, &fp32_quarter,
        fp16, &fp16_one_point_five, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT16_C(0x3f00));
    g_assert_cmphex(result.fflags, ==, 0);

    /* A source-format conversion would destroy this cancellation. */
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp16));
    riscv_ame_exec_fp_binary(
        AME_OP_ADD, fp16, fp32,
        &(AmeValue) { .lo = UINT32_C(0x3f801001) },
        fp16, &(AmeValue) { .lo = UINT16_C(0xbc00) },
        &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT16_C(0x1000));
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);

    /* The product and addend must remain fused across source formats. */
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp16));
    riscv_ame_exec_fp_ternary(
        AME_OP_MULACC, fp16, fp32,
        &(AmeValue) { .lo = UINT32_C(0x3f801000) },
        fp32, &(AmeValue) { .lo = UINT32_C(0x3f7fe000) },
        &(AmeValue) { .lo = UINT16_C(0xbc00) },
        &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT16_C(0x8004));
    g_assert_cmphex(result.fflags, ==, 0);

    for (size_t round = AME_FP_RNE; round <= AME_FP_RNO; round++) {
        static const bool rounds_up[] = {
            false, false, false, true, true, true,
        };
        const AmeDTypeDesc *dest = fp_dtype(AME_FP_FORMAT_FP16, round);

        g_assert_true(riscv_ame_fp_status_prepare(&status, dest));
        riscv_ame_exec_fp_binary(
            AME_OP_ADD, dest, fp32,
            &(AmeValue) { .lo = UINT32_C(0x3f801000) },
            bf16, &(AmeValue) { 0 }, &result, &status);
        g_assert_cmphex(result.value.lo, ==,
                        rounds_up[round] ? UINT16_C(0x3c01) :
                                           UINT16_C(0x3c00));
        g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);
    }

    g_assert_true(riscv_ame_fp_status_prepare(
        &status, fp_dtype(AME_FP_FORMAT_OCP_E4M3, AME_FP_RNE)));
    riscv_ame_exec_fp_binary(
        AME_OP_MEAN,
        fp_dtype(AME_FP_FORMAT_OCP_E4M3, AME_FP_RNE),
        fp_dtype(AME_FP_FORMAT_OCP_E5M2, AME_FP_RNE),
        &(AmeValue) { .lo = UINT8_C(0x3c) },
        fp_dtype(AME_FP_FORMAT_OCP_E4M3, AME_FP_RNE),
        &(AmeValue) { .lo = UINT8_C(0x40) }, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT8_C(0x3c));
    g_assert_cmphex(result.fflags, ==, 0);

    /* OCP same-format half operations use the canonical-parts path too. */
    g_assert_true(riscv_ame_fp_status_prepare(
        &status, fp_dtype(AME_FP_FORMAT_OCP_E4M3, AME_FP_RNE)));
    riscv_ame_exec_fp_binary(
        AME_OP_HDIFF,
        fp_dtype(AME_FP_FORMAT_OCP_E4M3, AME_FP_RNE),
        fp_dtype(AME_FP_FORMAT_OCP_E4M3, AME_FP_RNE),
        &(AmeValue) { .lo = UINT8_C(0x38) },
        fp_dtype(AME_FP_FORMAT_OCP_E4M3, AME_FP_RNE),
        &(AmeValue) { .lo = UINT8_C(0x40) }, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT8_C(0x30));
    g_assert_cmphex(result.fflags, ==, 0);

    g_assert_true(riscv_ame_fp_status_prepare(
        &status, fp_dtype(AME_FP_FORMAT_OCP_E4M3, AME_FP_RNE)));
    riscv_ame_exec_fp_binary(
        AME_OP_ADD,
        fp_dtype(AME_FP_FORMAT_OCP_E4M3, AME_FP_RNE), fp32,
        &(AmeValue) { .lo = UINT32_C(0x7fa00000) }, bf16,
        &(AmeValue) { .lo = UINT16_C(0x3f80) }, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT8_C(0x7f));
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NV);

    g_assert_true(riscv_ame_fp_status_prepare(
        &status, fp_dtype(AME_FP_FORMAT_OCP_E4M3, AME_FP_RNE)));
    riscv_ame_exec_fp_binary(
        AME_OP_ADD,
        fp_dtype(AME_FP_FORMAT_OCP_E4M3, AME_FP_RNE),
        fp_dtype(AME_FP_FORMAT_OCP_E5M2, AME_FP_RNE),
        &(AmeValue) { .lo = UINT8_C(0x7c) }, fp16,
        &(AmeValue) { 0 }, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT8_C(0x7e));
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_OF | AME_FFLAG_NX);

    g_assert_true(riscv_ame_fp_status_prepare(
        &status, fp_dtype(AME_FP_FORMAT_FP16, AME_FP_RDN)));
    riscv_ame_exec_fp_binary(
        AME_OP_ADD, fp_dtype(AME_FP_FORMAT_FP16, AME_FP_RDN),
        fp_dtype(AME_FP_FORMAT_OCP_E4M3, AME_FP_RNE),
        &(AmeValue) { 0 },
        fp_dtype(AME_FP_FORMAT_OCP_E5M2, AME_FP_RNE),
        &(AmeValue) { .lo = UINT8_C(0x80) }, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT16_C(0x8000));
    g_assert_cmphex(result.fflags, ==, 0);

    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    riscv_ame_exec_fp_binary(
        AME_OP_ADD, fp32,
        fp_dtype(AME_FP_FORMAT_OCP_E4M3, AME_FP_RNE),
        &(AmeValue) { .lo = UINT8_C(0x01) }, fp16,
        &(AmeValue) { 0 }, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT32_C(0x3b000000));
    g_assert_cmphex(result.fflags, ==, 0);
}

static void test_ieee_half_operations(void)
{
    static const struct {
        AmeFPFormat format;
        uint64_t maximum;
        uint64_t negative_maximum;
        uint64_t one;
        uint64_t next_one;
        uint64_t two;
        uint64_t three;
    } formats[] = {
        { AME_FP_FORMAT_FP16, UINT16_C(0x7bff), UINT16_C(0xfbff),
          UINT16_C(0x3c00), UINT16_C(0x3c01), UINT16_C(0x4000),
          UINT16_C(0x4200) },
        { AME_FP_FORMAT_BF16, UINT16_C(0x7f7f), UINT16_C(0xff7f),
          UINT16_C(0x3f80), UINT16_C(0x3f81), UINT16_C(0x4000),
          UINT16_C(0x4040) },
        { AME_FP_FORMAT_FP32, UINT32_C(0x7f7fffff),
          UINT32_C(0xff7fffff), UINT32_C(0x3f800000),
          UINT32_C(0x3f800001), UINT32_C(0x40000000),
          UINT32_C(0x40400000) },
        { AME_FP_FORMAT_FP64, UINT64_C(0x7fefffffffffffff),
          UINT64_C(0xffefffffffffffff),
          UINT64_C(0x3ff0000000000000),
          UINT64_C(0x3ff0000000000001),
          UINT64_C(0x4000000000000000),
          UINT64_C(0x4008000000000000) },
    };
    AmeExecResult result;
    float_status status;
    size_t i;

    for (i = 0; i < ARRAY_SIZE(formats); i++) {
        const AmeDTypeDesc *fp = fp_dtype(formats[i].format, AME_FP_RNE);
        AmeValue left = { .lo = formats[i].one };
        AmeValue right = { .lo = formats[i].three };

        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_binary(
            AME_OP_MEAN, fp, fp, &left, fp, &right, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].two);
        g_assert_cmphex(result.fflags, ==, 0);

        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_binary(
            AME_OP_HDIFF, fp, fp, &left, fp, &right, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].one);
        g_assert_cmphex(result.fflags, ==, 0);

        left.lo = formats[i].maximum;
        right.lo = formats[i].maximum;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_binary(
            AME_OP_MEAN, fp, fp, &left, fp, &right, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].maximum);
        g_assert_cmphex(result.fflags, ==, 0);

        left.lo = formats[i].negative_maximum;
        right.lo = formats[i].maximum;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp));
        riscv_ame_exec_fp_binary(
            AME_OP_HDIFF, fp, fp, &left, fp, &right, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].maximum);
        g_assert_cmphex(result.fflags, ==, 0);

        left.lo = formats[i].one;
        right.lo = formats[i].next_one;
        for (unsigned int round = AME_FP_RNE; round < AME_FP_ROUND_MAX;
             round++) {
            const AmeDTypeDesc *rounded = fp_dtype(formats[i].format,
                                                   round);
            bool round_up = round == AME_FP_RUP || round == AME_FP_RMM ||
                            round == AME_FP_RNO;

            g_assert_true(riscv_ame_fp_status_prepare(&status, rounded));
            riscv_ame_exec_fp_binary(
                AME_OP_MEAN, rounded, fp, &left, fp, &right,
                &result, &status);
            g_assert_cmphex(result.value.lo, ==,
                            round_up ? formats[i].next_one : formats[i].one);
            g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);
        }
    }
}

static void test_ieee_invalid_requests(void)
{
    const AmeDTypeDesc *fp16 = fp_dtype(AME_FP_FORMAT_FP16, AME_FP_RNE);
    const AmeDTypeDesc *fp32 = fp_dtype(AME_FP_FORMAT_FP32, AME_FP_RNE);
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    AmeDTypeDesc invalid_fp = *fp32;
    float_status status, before;

    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    set_float_exception_flags(float_flag_inexact, &status);
    before = status;

    g_assert_false(riscv_ame_fp_supports(AME_OP_COS, fp32, int8, NULL));
    g_assert_false(riscv_ame_fp_supports(AME_OP_LOG2SUB, int8, fp32, fp16));

    invalid_fp.bits = 24;
    g_assert_false(riscv_ame_fp_status_prepare(&status, &invalid_fp));
    g_assert_cmpmem(&status, sizeof(status), &before, sizeof(before));
    invalid_fp = *fp32;
    invalid_fp.fp.rounding = AME_FP_ROUND_MAX;
    g_assert_false(riscv_ame_fp_status_set_rounding(&status, &invalid_fp));
    g_assert_cmpmem(&status, sizeof(status), &before, sizeof(before));
    invalid_fp.bits = 24;
    g_assert_false(riscv_ame_fp_supports(AME_OP_CONV, int8, &invalid_fp, NULL));
}

static void test_ocp_conversion(void)
{
    const AmeDTypeDesc *fp32 = fp_dtype(AME_FP_FORMAT_FP32, AME_FP_RNE);
    const AmeDTypeDesc *int64 = dtype(UINT32_C(0x40000040));
    const AmeDTypeDesc *uint64 = dtype(UINT32_C(0x00000040));
    static const struct {
        uint64_t raw;
        bool is_signed;
    } integer_cases[] = {
        { INT64_MIN, true },
        { INT64_MAX, true },
        { UINT64_MAX, false },
    };
    float_status status;
    AmeExecResult widened, narrowed;
    AmeValue source;
    size_t format, round, raw;

    for (format = AME_FP_FORMAT_OCP_E4M3;
         format <= AME_FP_FORMAT_OCP_E5M2; format++) {
        const AmeDTypeDesc *src = fp_dtype(format, AME_FP_RNE);

        for (round = AME_FP_RNE; round <= AME_FP_RMM; round++) {
            const AmeDTypeDesc *dest = fp_dtype(format, round);

            for (raw = 0; raw <= UINT8_MAX; raw++) {
                OcpRefValue decoded = ocp_ref_decode(format, raw);
                uint8_t expected = decoded.cls == OCP_REF_NAN ?
                    (format == AME_FP_FORMAT_OCP_E4M3 ?
                     UINT8_C(0x7f) : UINT8_C(0x7e)) : raw;

                source = (AmeValue) { .lo = raw };
                g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
                riscv_ame_exec_fp_convert(
                    fp32, src, &source, &widened, &status);
                g_assert_cmphex(widened.fflags, ==, 0);
                g_assert_true(riscv_ame_fp_status_prepare(&status, dest));
                riscv_ame_exec_fp_convert(
                    dest, fp32, &widened.value, &narrowed, &status);
                g_assert_cmphex(narrowed.value.lo, ==, expected);
                g_assert_cmphex(narrowed.fflags, ==, 0);
            }
        }
    }

    source.lo = UINT32_C(0x7f800000);
    g_assert_true(riscv_ame_fp_status_prepare(
        &status, fp_dtype(AME_FP_FORMAT_OCP_E4M3, AME_FP_RNE)));
    riscv_ame_exec_fp_convert(
        fp_dtype(AME_FP_FORMAT_OCP_E4M3, AME_FP_RNE), fp32, &source,
        &narrowed, &status);
    g_assert_cmphex(narrowed.value.lo, ==, UINT8_C(0x7e));
    g_assert_cmphex(narrowed.fflags, ==, AME_FFLAG_OF | AME_FFLAG_NX);

    g_assert_true(riscv_ame_fp_status_prepare(
        &status, fp_dtype(AME_FP_FORMAT_OCP_E5M2, AME_FP_RNE)));
    riscv_ame_exec_fp_convert(
        fp_dtype(AME_FP_FORMAT_OCP_E5M2, AME_FP_RNE), fp32, &source,
        &narrowed, &status);
    g_assert_cmphex(narrowed.value.lo, ==, UINT8_C(0x7c));
    g_assert_cmphex(narrowed.fflags, ==, 0);

    source.lo = UINT32_C(0x44000000); /* 512, beyond E4M3 max 448. */
    g_assert_true(riscv_ame_fp_status_prepare(
        &status, fp_dtype(AME_FP_FORMAT_OCP_E4M3, AME_FP_RNE)));
    riscv_ame_exec_fp_convert(
        fp_dtype(AME_FP_FORMAT_OCP_E4M3, AME_FP_RNE), fp32, &source,
        &narrowed, &status);
    g_assert_cmphex(narrowed.value.lo, ==, UINT8_C(0x7e));
    g_assert_cmphex(narrowed.fflags, ==, AME_FFLAG_OF | AME_FFLAG_NX);

    source.lo = UINT32_C(0x3a800000); /* Half of E4M3 min subnormal. */
    g_assert_true(riscv_ame_fp_status_prepare(
        &status, fp_dtype(AME_FP_FORMAT_OCP_E4M3, AME_FP_RNE)));
    riscv_ame_exec_fp_convert(
        fp_dtype(AME_FP_FORMAT_OCP_E4M3, AME_FP_RNE), fp32, &source,
        &narrowed, &status);
    g_assert_cmphex(narrowed.value.lo, ==, 0);
    g_assert_cmphex(narrowed.fflags, ==, AME_FFLAG_UF | AME_FFLAG_NX);
    g_assert_true(riscv_ame_fp_status_prepare(
        &status, fp_dtype(AME_FP_FORMAT_OCP_E4M3, AME_FP_RMM)));
    riscv_ame_exec_fp_convert(
        fp_dtype(AME_FP_FORMAT_OCP_E4M3, AME_FP_RMM), fp32, &source,
        &narrowed, &status);
    g_assert_cmphex(narrowed.value.lo, ==, UINT8_C(0x01));
    g_assert_cmphex(narrowed.fflags, ==, AME_FFLAG_UF | AME_FFLAG_NX);

    for (format = AME_FP_FORMAT_OCP_E4M3;
         format <= AME_FP_FORMAT_OCP_E5M2; format++) {
        for (round = AME_FP_RNE; round <= AME_FP_RMM; round++) {
            const AmeDTypeDesc *dest = fp_dtype(format, round);
            size_t i;

            for (i = 0; i < ARRAY_SIZE(integer_cases); i++) {
                int64_t signed_value = integer_cases[i].raw;
                bool sign = integer_cases[i].is_signed && signed_value < 0;
                uint64_t magnitude = sign ? -(uint64_t)signed_value :
                                             integer_cases[i].raw;
                OcpRefResult expected = ocp_ref_round(
                    format, round, sign, magnitude, 0);
                const AmeDTypeDesc *source_dtype =
                    integer_cases[i].is_signed ? int64 : uint64;

                source.lo = integer_cases[i].raw;
                g_assert_true(riscv_ame_fp_status_prepare(&status, dest));
                riscv_ame_exec_fp_convert(
                    dest, source_dtype, &source, &narrowed, &status);
                g_assert_cmphex(narrowed.value.lo, ==, expected.value);
                g_assert_cmphex(narrowed.fflags, ==, expected.fflags);
            }
        }
    }
}

static void test_ocp_conversion_exhaustive(void)
{
    const AmeDTypeDesc *fp16 = fp_dtype(AME_FP_FORMAT_FP16, AME_FP_RNE);
    const AmeDTypeDesc *int16 = dtype(UINT32_C(0x40000010));
    const AmeDTypeDesc *uint16 = dtype(UINT32_C(0x00000010));
    const AmeDTypeDesc *integer_dtypes[] = { int16, uint16 };
    size_t format, round;

    for (format = AME_FP_FORMAT_OCP_E4M3;
         format <= AME_FP_FORMAT_OCP_E5M2; format++) {
        for (round = AME_FP_RNE; round <= AME_FP_RMM; round++) {
            const AmeDTypeDesc *fp8 = fp_dtype(format, round);
            uint32_t raw;

            for (raw = 0; raw <= UINT16_MAX; raw++) {
                unsigned exponent = extract32(raw, 10, 5);
                unsigned fraction = extract32(raw, 0, 10);
                bool sign = raw >> 15;
                OcpRefResult expected;
                AmeValue source = { .lo = raw };
                AmeExecResult actual;
                float_status status;

                if (exponent == 0x1f && fraction != 0) {
                    expected = (OcpRefResult) {
                        .value = format == AME_FP_FORMAT_OCP_E4M3 ?
                                 UINT8_C(0x7f) : UINT8_C(0x7e),
                        .fflags = (fraction & BIT(9)) ? 0 : AME_FFLAG_NV,
                    };
                } else if (exponent == 0x1f) {
                    expected = (OcpRefResult) {
                        .value = format == AME_FP_FORMAT_OCP_E4M3 ?
                                 UINT8_C(0x7e) : UINT8_C(0x7c),
                        .fflags = format == AME_FP_FORMAT_OCP_E4M3 ?
                                  AME_FFLAG_OF | AME_FFLAG_NX : 0,
                    };
                    expected.value |= sign ? UINT8_C(0x80) : 0;
                } else {
                    uint64_t significand = exponent == 0 ? fraction :
                                           BIT_ULL(10) | fraction;
                    int binary_exponent = (exponent == 0 ? 1 : exponent) -
                                          15 - 10;

                    expected = ocp_ref_round(format, round, sign,
                                             significand,
                                             binary_exponent);
                }
                g_assert_true(riscv_ame_fp_status_prepare(&status, fp8));
                riscv_ame_exec_fp_convert(
                    fp8, fp16, &source, &actual, &status);
                if (actual.value.lo != expected.value ||
                    actual.fflags != expected.fflags) {
                    g_test_message("format=%zu round=%zu fp16=0x%04x",
                                   format, round, raw);
                }
                g_assert_cmphex(actual.value.lo, ==, expected.value);
                g_assert_cmphex(actual.fflags, ==, expected.fflags);
            }

            for (raw = 0; raw <= UINT16_MAX; raw++) {
                size_t integer_index;

                for (integer_index = 0;
                     integer_index < ARRAY_SIZE(integer_dtypes);
                     integer_index++) {
                    const AmeDTypeDesc *integer_dtype =
                        integer_dtypes[integer_index];
                    int64_t signed_value = (int16_t)raw;
                    bool sign = integer_dtype->integer.is_signed &&
                                signed_value < 0;
                    uint64_t magnitude = integer_dtype->integer.is_signed ?
                        (sign ? -(uint64_t)signed_value : signed_value) : raw;
                    OcpRefResult expected = magnitude == 0 ?
                        (OcpRefResult) { 0 } :
                        ocp_ref_round(format, round, sign, magnitude, 0);
                    AmeValue source = { .lo = raw };
                    AmeExecResult actual;
                    float_status status;

                    g_assert_true(riscv_ame_fp_status_prepare(&status, fp8));
                    riscv_ame_exec_fp_convert(
                        fp8, integer_dtype, &source, &actual, &status);
                    g_assert_cmphex(actual.value.lo, ==, expected.value);
                    g_assert_cmphex(actual.fflags, ==, expected.fflags);
                }
            }
        }
    }
}

static void test_ocp_binary_exhaustive(void)
{
    static const AmeOperation operations[] = {
        AME_OP_ADD, AME_OP_SUB, AME_OP_MUL,
        AME_OP_HDIFF, AME_OP_MEAN,
    };
    size_t format, round, operation, raw1, raw2;

    for (format = AME_FP_FORMAT_OCP_E4M3;
         format <= AME_FP_FORMAT_OCP_E5M2; format++) {
        for (round = AME_FP_RNE; round <= AME_FP_RMM; round++) {
            const AmeDTypeDesc *fp8 = fp_dtype(format, round);

            for (operation = 0; operation < ARRAY_SIZE(operations);
                 operation++) {
                for (raw1 = 0; raw1 <= UINT8_MAX; raw1++) {
                    for (raw2 = 0; raw2 <= UINT8_MAX; raw2++) {
                        OcpRefResult expected;
                        AmeExecResult actual;
                        AmeValue left = { .lo = raw1 };
                        AmeValue right = { .lo = raw2 };
                        float_status status;
                        bool comparable;

                        expected = ocp_ref_binary(
                            operations[operation], format, round, raw1, raw2,
                            &comparable);
                        if (!comparable) {
                            continue;
                        }
                        g_assert_true(riscv_ame_fp_status_prepare(&status,
                                                                  fp8));
                        riscv_ame_exec_fp_binary(
                            operations[operation], fp8, fp8, &left, fp8,
                            &right, &actual, &status);
                        if (actual.value.lo != expected.value ||
                            actual.fflags != expected.fflags) {
                            g_test_message("format=%zu round=%zu op=%u "
                                           "left=0x%02zx right=0x%02zx",
                                           format, round,
                                           operations[operation], raw1,
                                           raw2);
                        }
                        g_assert_cmphex(actual.value.lo, ==, expected.value);
                        g_assert_cmphex(actual.fflags, ==, expected.fflags);
                    }
                }
            }
        }
    }
}

static void test_ocp_basic_operations(void)
{
    static const struct {
        AmeFPFormat format;
        uint8_t zero;
        uint8_t one;
        uint8_t one_point_five;
        uint8_t two;
        uint8_t four;
        uint8_t maximum;
        uint8_t infinity;
        uint8_t nan;
    } formats[] = {
        { AME_FP_FORMAT_OCP_E4M3, 0x00, 0x38, 0x3c, 0x40, 0x48,
          0x7e, 0x00, 0x7f },
        { AME_FP_FORMAT_OCP_E5M2, 0x00, 0x3c, 0x3e, 0x40, 0x44,
          0x7b, 0x7c, 0x7e },
    };
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    float_status status;
    AmeExecResult result;
    AmeValue source, left, right, exponent;
    size_t i;

    for (i = 0; i < ARRAY_SIZE(formats); i++) {
        const AmeDTypeDesc *fp8 = fp_dtype(formats[i].format, AME_FP_RNE);

        source.lo = formats[i].four;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp8));
        riscv_ame_exec_fp_unary(
            AME_OP_SQRT, fp8, fp8, &source, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].two);
        g_assert_cmphex(result.fflags, ==, 0);

        source.lo = formats[i].zero;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp8));
        riscv_ame_exec_fp_unary(
            AME_OP_REC, fp8, fp8, &source, &result, &status);
        g_assert_cmphex(result.value.lo, ==,
                        formats[i].format == AME_FP_FORMAT_OCP_E4M3 ?
                        formats[i].maximum : formats[i].infinity);
        g_assert_cmphex(result.fflags, ==,
                        formats[i].format == AME_FP_FORMAT_OCP_E4M3 ?
                        AME_FFLAG_DZ | AME_FFLAG_OF | AME_FFLAG_NX :
                        AME_FFLAG_DZ);

        source.lo = formats[i].one_point_five;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp8));
        riscv_ame_exec_fp_unary(
            AME_OP_FRINTN, fp8, fp8, &source, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].two);
        g_assert_cmphex(result.fflags, ==, 0);

        left.lo = formats[i].one;
        exponent = (AmeValue) { .lo = 1 };
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp8));
        riscv_ame_exec_fp_binary(
            AME_OP_LDEXP, fp8, fp8, &left, int8, &exponent, &result,
            &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].two);
        g_assert_cmphex(result.fflags, ==, 0);

        left.lo = formats[i].zero;
        right.lo = UINT8_C(0x80);
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp8));
        riscv_ame_exec_fp_binary(
            AME_OP_MIN, fp8, fp8, &left, fp8, &right, &result,
            &status);
        g_assert_cmphex(result.value.lo, ==, UINT8_C(0x80));
        g_assert_cmphex(result.fflags, ==, 0);
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp8));
        riscv_ame_exec_fp_binary(
            AME_OP_MAX, fp8, fp8, &left, fp8, &right, &result,
            &status);
        g_assert_cmphex(result.value.lo, ==, 0);
        g_assert_cmphex(result.fflags, ==, 0);

        source.lo = UINT8_C(0x80);
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp8));
        riscv_ame_exec_fp_unary(
            AME_OP_ABS, fp8, fp8, &source, &result, &status);
        g_assert_cmphex(result.value.lo, ==, 0);
        g_assert_cmphex(result.fflags, ==, 0);

        source.lo = formats[i].nan;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp8));
        riscv_ame_exec_fp_unary(
            AME_OP_SQRT, fp8, fp8, &source, &result, &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].nan);
        g_assert_cmphex(result.fflags, ==, 0);

        left.lo = formats[i].nan;
        right.lo = formats[i].one;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp8));
        riscv_ame_exec_fp_binary(
            AME_OP_MEAN, fp8, fp8, &left, fp8, &right, &result,
            &status);
        g_assert_cmphex(result.value.lo, ==, formats[i].nan);
        g_assert_cmphex(result.fflags, ==, 0);

        left.lo = formats[i].one;
        right.lo = formats[i].two;
        g_assert_true(riscv_ame_fp_status_prepare(&status, fp8));
        riscv_ame_exec_fp_binary(
            AME_OP_CMPLT, int8, fp8, &left, fp8, &right, &result,
            &status);
        g_assert_cmphex(result.value.lo, ==, UINT8_MAX);
        g_assert_cmphex(result.fflags, ==, 0);
    }

    {
        const AmeDTypeDesc *e5m2 = fp_dtype(AME_FP_FORMAT_OCP_E5M2,
                                            AME_FP_RNE);

        left.lo = UINT8_C(0x7c);
        right.lo = UINT8_C(0xfc);
        g_assert_true(riscv_ame_fp_status_prepare(&status, e5m2));
        riscv_ame_exec_fp_binary(
            AME_OP_ADD, e5m2, e5m2, &left, e5m2, &right, &result,
            &status);
        g_assert_cmphex(result.value.lo, ==, UINT8_C(0x7e));
        g_assert_cmphex(result.fflags, ==, AME_FFLAG_NV);

        left.lo = UINT8_C(0x7c);
        right.lo = UINT8_C(0xfc);
        g_assert_true(riscv_ame_fp_status_prepare(&status, e5m2));
        riscv_ame_exec_fp_binary(
            AME_OP_MEAN, e5m2, e5m2, &left, e5m2, &right, &result,
            &status);
        g_assert_cmphex(result.value.lo, ==, UINT8_C(0x7e));
        g_assert_cmphex(result.fflags, ==, AME_FFLAG_NV);

        left.lo = 0;
        right.lo = UINT8_C(0x7c);
        g_assert_true(riscv_ame_fp_status_prepare(&status, e5m2));
        riscv_ame_exec_fp_binary(
            AME_OP_MUL, e5m2, e5m2, &left, e5m2, &right, &result,
            &status);
        g_assert_cmphex(result.value.lo, ==, UINT8_C(0x7e));
        g_assert_cmphex(result.fflags, ==, AME_FFLAG_NV);

        source.lo = UINT8_C(0xbc);
        g_assert_true(riscv_ame_fp_status_prepare(&status, e5m2));
        riscv_ame_exec_fp_unary(
            AME_OP_SQRT, e5m2, e5m2, &source, &result, &status);
        g_assert_cmphex(result.value.lo, ==, UINT8_C(0x7e));
        g_assert_cmphex(result.fflags, ==, AME_FFLAG_NV);

        source.lo = UINT8_C(0x80);
        g_assert_true(riscv_ame_fp_status_prepare(&status, e5m2));
        riscv_ame_exec_fp_unary(
            AME_OP_REC, e5m2, e5m2, &source, &result, &status);
        g_assert_cmphex(result.value.lo, ==, UINT8_C(0xfc));
        g_assert_cmphex(result.fflags, ==, AME_FFLAG_DZ);

        source.lo = UINT8_C(0x7d);
        g_assert_true(riscv_ame_fp_status_prepare(&status, e5m2));
        riscv_ame_exec_fp_unary(
            AME_OP_SQRT, e5m2, e5m2, &source, &result, &status);
        g_assert_cmphex(result.value.lo, ==, UINT8_C(0x7e));
        g_assert_cmphex(result.fflags, ==, 0);
    }
}

static void test_ocp_fused_and_scale_acc(void)
{
    static const AmeOperation operations[] = {
        AME_OP_MULACC, AME_OP_MULADD,
        AME_OP_MULNACC, AME_OP_MULSUB,
    };
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    size_t format, round, operation, raw1, raw2;

    for (format = AME_FP_FORMAT_OCP_E4M3;
         format <= AME_FP_FORMAT_OCP_E5M2; format++) {
        for (round = AME_FP_RNE; round <= AME_FP_RMM; round++) {
            const AmeDTypeDesc *fp8 = fp_dtype(format, round);

            for (operation = 0; operation < ARRAY_SIZE(operations);
                 operation++) {
                for (raw1 = 0; raw1 <= UINT8_MAX; raw1++) {
                    for (raw2 = 0; raw2 <= UINT8_MAX; raw2++) {
                        uint8_t raw_current =
                            ((raw1 * 17 + raw2 * 31 + operation * 13) &
                             UINT8_C(0x6f)) |
                            ((raw1 ^ raw2) & UINT8_C(0x80));
                        OcpRefResult expected;
                        AmeValue src1 = { .lo = raw1 };
                        AmeValue src2 = { .lo = raw2 };
                        AmeValue current = { .lo = raw_current };
                        AmeExecResult actual;
                        float_status status;
                        bool comparable;

                        expected = ocp_ref_fused(
                            operations[operation], format, round,
                            raw1, raw2, raw_current, &comparable);
                        if (!comparable) {
                            continue;
                        }
                        g_assert_true(riscv_ame_fp_status_prepare(&status,
                                                                  fp8));
                        exec_fp_ternary_same_dtype(
                            operations[operation], fp8, &src1, &src2,
                            &current, &actual, &status);
                        if (actual.value.lo != expected.value ||
                            actual.fflags != expected.fflags) {
                            g_test_message("format=%zu round=%zu op=%u "
                                           "src1=0x%02zx src2=0x%02zx "
                                           "current=0x%02x",
                                           format, round,
                                           operations[operation], raw1,
                                           raw2, raw_current);
                        }
                        g_assert_cmphex(actual.value.lo, ==, expected.value);
                        g_assert_cmphex(actual.fflags, ==, expected.fflags);
                    }
                }
            }
        }
    }

    for (format = AME_FP_FORMAT_OCP_E4M3;
         format <= AME_FP_FORMAT_OCP_E5M2; format++) {
        uint8_t one = format == AME_FP_FORMAT_OCP_E4M3 ?
                      UINT8_C(0x38) : UINT8_C(0x3c);
        uint8_t four = format == AME_FP_FORMAT_OCP_E4M3 ?
                       UINT8_C(0x48) : UINT8_C(0x44);
        uint8_t maximum = format == AME_FP_FORMAT_OCP_E4M3 ?
                          UINT8_C(0x7e) : UINT8_C(0x7b);
        OcpRefValue minimum = ocp_ref_decode(format, UINT8_C(0x01));

        for (round = AME_FP_RNE; round <= AME_FP_RMM; round++) {
            const AmeDTypeDesc *fp8 = fp_dtype(format, round);
            OcpRefResult expected = ocp_ref_round(
                format, round, false, 3, minimum.exponent - 1);
            AmeValue data = { .lo = UINT8_C(0x01) };
            AmeValue exponent = { .lo = UINT8_MAX };
            AmeValue current = { .lo = UINT8_C(0x01) };
            AmeExecResult actual;
            float_status status;

            g_assert_true(riscv_ame_fp_status_prepare(&status, fp8));
            riscv_ame_exec_fp_scale_acc(
                AME_OP_LDEXPACC, fp8, fp8, &data, int8, &exponent,
                &current, &actual, &status);
            g_assert_cmphex(actual.value.lo, ==, expected.value);
            g_assert_cmphex(actual.fflags, ==, expected.fflags);

            data.lo = maximum;
            exponent.lo = 1;
            current.lo = maximum | UINT8_C(0x80);
            g_assert_true(riscv_ame_fp_status_prepare(&status, fp8));
            riscv_ame_exec_fp_scale_acc(
                AME_OP_LDEXPACC, fp8, fp8, &data, int8, &exponent,
                &current, &actual, &status);
            g_assert_cmphex(actual.value.lo, ==, maximum);
            g_assert_cmphex(actual.fflags, ==, 0);

            data.lo = four;
            exponent.lo = 2;
            current.lo = one;
            g_assert_true(riscv_ame_fp_status_prepare(&status, fp8));
            riscv_ame_exec_fp_scale_acc(
                AME_OP_RDEXPACC, fp8, fp8, &data, int8, &exponent,
                &current, &actual, &status);
            g_assert_cmphex(actual.value.lo, ==, UINT8_C(0x40));
            g_assert_cmphex(actual.fflags, ==, 0);
        }
    }

    {
        const AmeDTypeDesc *e4m3 = fp_dtype(AME_FP_FORMAT_OCP_E4M3,
                                            AME_FP_RNE);
        const AmeDTypeDesc *e5m2 = fp_dtype(AME_FP_FORMAT_OCP_E5M2,
                                            AME_FP_RNE);
        const AmeDTypeDesc *fp16 = fp_dtype(AME_FP_FORMAT_FP16,
                                            AME_FP_RNE);
        AmeValue zero = { 0 };
        AmeValue one_e4m3 = { .lo = UINT8_C(0x38) };
        AmeValue two_e5m2 = { .lo = UINT8_C(0x40) };
        AmeValue one_fp16 = { .lo = UINT16_C(0x3c00) };
        AmeValue maximum = { .lo = UINT8_C(0x7e) };
        AmeValue two = { .lo = UINT8_C(0x40) };
        AmeValue infinity = { .lo = UINT8_C(0x7c) };
        AmeExecResult result;
        float_status status;

        g_assert_true(riscv_ame_fp_status_prepare(&status, e4m3));
        exec_fp_ternary_same_dtype(
            AME_OP_MULACC, e4m3, &maximum, &two, &zero, &result,
            &status);
        g_assert_cmphex(result.value.lo, ==, UINT8_C(0x7e));
        g_assert_cmphex(result.fflags, ==, AME_FFLAG_OF | AME_FFLAG_NX);

        g_assert_true(riscv_ame_fp_status_prepare(&status, e5m2));
        exec_fp_ternary_same_dtype(
            AME_OP_MULACC, e5m2, &zero, &infinity, &zero, &result,
            &status);
        g_assert_cmphex(result.value.lo, ==, UINT8_C(0x7e));
        g_assert_cmphex(result.fflags, ==, AME_FFLAG_NV);

        g_assert_true(riscv_ame_fp_status_prepare(&status, fp16));
        riscv_ame_exec_fp_ternary(
            AME_OP_MULACC, fp16, e4m3, &one_e4m3,
            e5m2, &two_e5m2, &one_fp16, &result, &status);
        g_assert_cmphex(result.value.lo, ==, UINT16_C(0x4200));
        g_assert_cmphex(result.fflags, ==, 0);

        g_assert_true(riscv_ame_fp_status_prepare(&status, fp16));
        riscv_ame_exec_fp_scale_acc(
            AME_OP_LDEXPACC, fp16, e4m3, &one_e4m3, int8,
            &(AmeValue) { .lo = 1 }, &one_fp16, &result, &status);
        g_assert_cmphex(result.value.lo, ==, UINT16_C(0x4200));
        g_assert_cmphex(result.fflags, ==, 0);
    }
}

static void test_ocp_exp2_overflow(void)
{
    for (AmeFPRoundMode mode = AME_FP_RNE; mode <= AME_FP_RMM; mode++) {
        const AmeDTypeDesc *e5m2 = fp_dtype(AME_FP_FORMAT_OCP_E5M2, mode);

        /* 2^16 exceeds E5M2's range; 2^-16 is its minimum subnormal. */
        assert_nonlinear_unary(AME_OP_EXP2, e5m2, UINT8_C(0x4c),
                              UINT8_C(0x7c), AME_FFLAG_OF | AME_FFLAG_NX);
        assert_nonlinear_unary(AME_OP_EXP2, e5m2, UINT8_C(0xcc),
                              UINT8_C(0x01), 0);
    }
}

static void test_ocp_nonlinear_exhaustive(void)
{
    static const uint64_t expected[2][5][6] = {
        {
            { 0xcc223c0eb689dc5e, 0xae5407fd964ed4e6,
              0x37c55ecaf2c68b82, 0x4a1635040fed5825,
              0xb9c6bc367c9804c5, 0x8b0cc1966b92f485 },
            { 0xd40b63a190d81b5a, 0xd06091867bea20c8,
              0x11f2d771e178b8b9, 0x7375ebd2904f4745,
              0x3928244f5ff019e5, 0xe94ce51430d32445 },
            { 0xd40b63a190d81b5a, 0xcfc3925d75c2a294,
              0x11f2d771e178b8b9, 0x80193228196346da,
              0x6f420240b679b7f5, 0xfd7b030408266ca5 },
            { 0xf2cd5f5d9894949b, 0x93e32bf45e2def04,
              0xc5baa67035216568, 0x3a4199e105719c9a,
              0x5f564a252635cfb5, 0x45ae00e0339251c5 },
            { 0xcc223c0eb689dc5e, 0xae5407fd964ed4e6,
              0x16c0ed0a690751d3, 0x4a1635040fed5825,
              0xb9c6bc367c9804c5, 0x8b0cc1966b92f485 },
        },
        {
            { 0x41f0b6764199a627, 0x520504ea0e67d3c2,
              0xe544795a089b7af4, 0x7334e61f4a3c9e15,
              0x4325df6450bfb465, 0x59b1304fec885c55 },
            { 0x8a3c62dc36d1e926, 0x1c33546ae7a1c77e,
              0xd98affc7401bbd50, 0xd6184ab5cbfc54a5,
              0x05b149e8bb8a38c5, 0x0aede2a583fc9d65 },
            { 0x8a3c62dc36d1e926, 0xbc8f01230007e8a5,
              0xd98affc7401bbd50, 0x60b9740e03125609,
              0xcb711f507e158e18, 0xf0a99a869f611eb5 },
            { 0x7a7e8d6d494ebd23, 0x76d388eee457e6b8,
              0xb56f5f90e22e23bf, 0x24ead0bcd3764299,
              0x53182ad70ecac6f8, 0xeacd9cde489e1545 },
            { 0x41f0b6764199a627, 0xfc722cc5222235a2,
              0xe544795a089b7af4, 0x7334e61f4a3c9e15,
              0x4325df6450bfb465, 0x59b1304fec885c55 },
        },
    };
    static const AmeFPFormat formats[] = {
        AME_FP_FORMAT_OCP_E4M3,
        AME_FP_FORMAT_OCP_E5M2,
    };
    static const AmeOperation operations[] = {
        AME_OP_RSQRT, AME_OP_LOG2, AME_OP_EXP2,
        AME_OP_TANH, AME_OP_SIN, AME_OP_COS,
    };
    size_t format, rounding, operation;

    for (format = 0; format < ARRAY_SIZE(formats); format++) {
        for (rounding = AME_FP_RNE; rounding <= AME_FP_RMM; rounding++) {
            const AmeDTypeDesc *fp8 = fp_dtype(formats[format], rounding);

            for (operation = 0; operation < ARRAY_SIZE(operations);
                 operation++) {
                uint64_t signature = UINT64_C(0xcbf29ce484222325);
                uint32_t raw;

                for (raw = 0; raw <= UINT8_MAX; raw++) {
                    AmeValue source = { .lo = raw };
                    AmeExecResult result;
                    float_status status;

                    g_assert_true(riscv_ame_fp_status_prepare(&status,
                                                               fp8));
                    riscv_ame_exec_fp_unary(
                        operations[operation], fp8, fp8, &source, &result,
                        &status);
                    signature = nonlinear_signature_update(signature,
                                                            &result);
                }
                g_assert_cmphex(signature, ==,
                                expected[format][rounding][operation]);
            }
        }
    }
}

static void test_fp_mixed_log2_composite(void)
{
    const AmeDTypeDesc *e4m3 = fp_dtype(AME_FP_FORMAT_OCP_E4M3,
                                        AME_FP_RNE);
    const AmeDTypeDesc *e5m2 = fp_dtype(AME_FP_FORMAT_OCP_E5M2,
                                        AME_FP_RNE);
    const AmeDTypeDesc *fp16 = fp_dtype(AME_FP_FORMAT_FP16, AME_FP_RNE);
    const AmeDTypeDesc *bf16 = fp_dtype(AME_FP_FORMAT_BF16, AME_FP_RNE);
    const AmeDTypeDesc *fp32 = fp_dtype(AME_FP_FORMAT_FP32, AME_FP_RNE);
    AmeValue logarithm_source = { .lo = UINT8_C(0x40) };
    AmeValue bias = { .lo = UINT8_C(0x38) };
    AmeExecResult result = {
        .value = { .lo = UINT64_C(0x0123456789abcdef),
                   .hi = UINT64_C(0xfedcba9876543210) },
        .fflags = UINT8_C(0xa5),
        .saturated = true,
    };
    float_status status;

    g_assert_true(riscv_ame_fp_status_prepare(&status, e4m3));
    riscv_ame_exec_fp_binary(
        AME_OP_LOG2SUB, e4m3, e5m2, &logarithm_source, e5m2, &bias,
        &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT8_C(0x30));
    g_assert_cmphex(result.fflags, ==, 0);

    g_assert_true(riscv_ame_fp_status_prepare(&status, fp16));
    riscv_ame_exec_fp_binary(
        AME_OP_SUBLOG2, fp16, fp32,
        &(AmeValue) { .lo = UINT32_C(0x40000000) }, bf16,
        &(AmeValue) { .lo = UINT16_C(0x3f00) }, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT16_C(0xb800));
    g_assert_cmphex(result.fflags, ==, 0);

    /* log2(1.5) rounds to FP16 before the exactly equal bias is removed. */
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp16));
    riscv_ame_exec_fp_binary(
        AME_OP_LOG2SUB, fp16, fp32,
        &(AmeValue) { .lo = UINT32_C(0x3fc00000) }, fp32,
        &(AmeValue) { .lo = UINT32_C(0x3f15c000) }, &result, &status);
    g_assert_cmphex(result.value.lo, ==, 0);
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);

    /* Bias conversion is also a separately rounded, flag-producing step. */
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp16));
    riscv_ame_exec_fp_binary(
        AME_OP_LOG2SUB, fp16, fp32,
        &(AmeValue) { .lo = UINT32_C(0x40000000) }, fp32,
        &(AmeValue) { .lo = UINT32_C(0x3f801001) }, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT16_C(0x9400));
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);

    g_assert_false(riscv_ame_fp_supports(AME_OP_AND, e4m3, e4m3, e5m2));
}

static void test_integer_log2_bias(void)
{
    const AmeDTypeDesc *f32 = fp_dtype(AME_FP_FORMAT_FP32, AME_FP_RNE);
    const AmeDTypeDesc *f64 = fp_dtype(AME_FP_FORMAT_FP64, AME_FP_RNE);
    AmeValue eight = { .lo = 0x41000000 };
    AmeValue minus_two = { .lo = 0xfe };
    AmeExecResult result;
    float_status status;

    g_assert_true(riscv_ame_fp_supports(
        AME_OP_LOG2SUB, f64, f32, dtype(0x40000008)));
    g_assert_true(riscv_ame_fp_status_prepare(&status, f64));
    riscv_ame_exec_fp_binary(
        AME_OP_LOG2SUB, f64, f32, &eight, dtype(0x40000008), &minus_two,
        &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT64_C(0x4014000000000000));
    g_assert_cmphex(result.fflags, ==, 0);
    g_assert_true(riscv_ame_fp_status_prepare(&status, f64));
    riscv_ame_exec_fp_binary(
        AME_OP_SUBLOG2, f64, f32, &eight, dtype(0x40000008), &minus_two,
        &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT64_C(0xc014000000000000));
    g_assert_cmphex(result.fflags, ==, 0);

    /* The bias conversion remains a separately rounded step. */
    g_assert_true(riscv_ame_fp_status_prepare(&status, f64));
    riscv_ame_exec_fp_binary(
        AME_OP_LOG2SUB, f64, f32, &(AmeValue) { .lo = 0x3f800000 },
        dtype(0x00000080),
        &(AmeValue) { .lo = 1, .hi = UINT64_C(0x8000000000000000) },
        &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT64_C(0xc7e0000000000000));
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);
    g_assert_false(result.saturated);
}

static void test_integer_float_cancellation(void)
{
    const AmeDTypeDesc *f64 = fp_dtype(AME_FP_FORMAT_FP64, AME_FP_RNE);
    const AmeDTypeDesc *u128 = dtype(0x00000080);
    AmeValue large = { .lo = 1, .hi = UINT64_C(0x8000000000000000) };
    AmeValue negative = { .lo = UINT64_C(0xc7e0000000000000) };
    AmeValue one = { .lo = UINT64_C(0x3ff0000000000000) };
    AmeExecResult result;
    float_status status;

    g_assert_true(riscv_ame_fp_status_prepare(&status, f64));
    riscv_ame_exec_fp_binary(
        AME_OP_ADD, f64, u128, &large, f64, &negative, &result, &status);
    g_assert_cmphex(result.value.lo, ==, one.lo);
    g_assert_cmphex(result.fflags, ==, 0);
    g_assert_true(riscv_ame_fp_status_prepare(&status, f64));
    riscv_ame_exec_fp_binary(
        AME_OP_ADD, f64, f64, &negative, u128, &large, &result, &status);
    g_assert_cmphex(result.value.lo, ==, one.lo);
    g_assert_cmphex(result.fflags, ==, 0);
    g_assert_true(riscv_ame_fp_status_prepare(&status, f64));
    riscv_ame_exec_fp_ternary(
        AME_OP_MULACC, f64, u128, &large, f64, &one, &negative,
        &result, &status);
    g_assert_cmphex(result.value.lo, ==, one.lo);
    g_assert_cmphex(result.fflags, ==, 0);

    /* Integer destinations must retain a fraction below Int128's LSB. */
    for (unsigned round = 0; round < AME_INT_ROUND_MAX; round++) {
        const AmeDTypeDesc *d = dtype(0x00000080 | (round << 27));
        AmeValue half = { .lo = UINT64_C(0x3fe0000000000000) };
        uint64_t expected = round == AME_INT_RNU || round == AME_INT_RNE ?
                            2 : 1;

        g_assert_true(riscv_ame_fp_status_prepare(&status, d));
        riscv_ame_exec_fp_binary(
            AME_OP_ADD, d, u128, &large, f64, &half, &result, &status);
        g_assert_cmphex(result.value.lo, ==, expected);
        g_assert_cmphex(result.value.hi, ==, large.hi);
        g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);
        g_assert_false(result.saturated);
    }
}

static void test_mixed_unary(void)
{
    static const struct {
        AmeFPFormat format;
        uint64_t one, two, four, half, one_half;
    } formats[] = {
        { AME_FP_FORMAT_FP16, 0x3c00, 0x4000, 0x4400, 0x3800, 0x3e00 },
        { AME_FP_FORMAT_BF16, 0x3f80, 0x4000, 0x4080, 0x3f00, 0x3fc0 },
        { AME_FP_FORMAT_FP32, 0x3f800000, 0x40000000, 0x40800000,
          0x3f000000, 0x3fc00000 },
        { AME_FP_FORMAT_FP64, UINT64_C(0x3ff0000000000000),
          UINT64_C(0x4000000000000000), UINT64_C(0x4010000000000000),
          UINT64_C(0x3fe0000000000000), UINT64_C(0x3ff8000000000000) },
        { AME_FP_FORMAT_OCP_E4M3, 0x38, 0x40, 0x48, 0x30, 0x3c },
        { AME_FP_FORMAT_OCP_E5M2, 0x3c, 0x40, 0x44, 0x38, 0x3e },
    };

    for (unsigned s = 0; s < ARRAY_SIZE(formats); s++) {
        const AmeDTypeDesc *src = fp_dtype(formats[s].format, AME_FP_RTZ);

        for (unsigned d = 0; d < ARRAY_SIZE(formats); d++) {
            unsigned modes = d < 4 ? 6 : 5;

            for (unsigned rm = 0; rm < modes; rm++) {
                const AmeDTypeDesc *dest = fp_dtype(formats[d].format, rm);
                const struct {
                    AmeOperation op;
                    uint64_t input, output;
                    uint8_t flags;
                } cases[] = {
                    { AME_OP_SQRT, formats[s].four, formats[d].two },
                    { AME_OP_REC, formats[s].two, formats[d].half },
                    { AME_OP_RSQRT, formats[s].four, formats[d].half },
                    { AME_OP_LOG2, formats[s].two, formats[d].one },
                    { AME_OP_EXP2, formats[s].one, formats[d].two },
                    { AME_OP_TANH, 0, 0 },
                    { AME_OP_SIN, 0, 0 },
                    { AME_OP_COS, 0, formats[d].one },
                    { AME_OP_FRINTM, formats[s].one_half,
                      formats[d].one },
                    { AME_OP_FRINTN, formats[s].one_half,
                      formats[d].two },
                    { AME_OP_FRINTP, formats[s].one_half,
                      formats[d].two },
                    { AME_OP_FRINTZ, formats[s].one_half,
                      formats[d].one },
                };

                for (unsigned c = 0; c < ARRAY_SIZE(cases); c++) {
                    AmeValue input = { .lo = cases[c].input };
                    AmeExecResult result;
                    float_status status;
                    FloatRoundMode saved_rounding;

                    g_assert_true(riscv_ame_fp_status_prepare(&status, dest));
                    saved_rounding = get_float_rounding_mode(&status);
                    riscv_ame_exec_fp_unary(
                        cases[c].op, dest, src, &input, &result, &status);
                    g_assert_cmphex(result.value.lo, ==, cases[c].output);
                    g_assert_cmphex(result.fflags, ==, cases[c].flags);
                    g_assert_cmpint(get_float_rounding_mode(&status), ==,
                                    saved_rounding);
                }
            }
        }
    }
}

static void test_mixed_unary_rounding(void)
{
    static const struct {
        AmeOperation op;
        uint32_t destination;
        uint64_t source, expected;
        uint8_t flags;
    } cases[] = {
        /* These three inputs straddle the exact FP16 square-root midpoint. */
        { AME_OP_SQRT, 0x14300110, UINT64_C(0x3ff004003fffffff),
          0x3c00, AME_FFLAG_NX },
        { AME_OP_SQRT, 0x14300110, UINT64_C(0x3ff0040040000000),
          0x3c00, AME_FFLAG_NX },
        { AME_OP_SQRT, 0x14300110, UINT64_C(0x3ff0040040000001),
          0x3c01, AME_FFLAG_NX },
        { AME_OP_SQRT, 0x15300110, UINT64_C(0x3ff0040040000000),
          0x3c01, AME_FFLAG_NX },
        /* Source-to-destination conversion must not precede the operation. */
        { AME_OP_REC, 0x14300110, UINT64_C(0x40f0000000000000),
          0x0100, 0 }, /* 1 / 65536: input overflows FP16, result is exact. */
        { AME_OP_SQRT, 0x14300110, UINT64_C(0x40f0000000000000),
          0x5c00, 0 },
        { AME_OP_REC, 0x14300110, 1, 0x7c00,
          AME_FFLAG_OF | AME_FFLAG_NX },
        { AME_OP_REC, 0x14700110, 1, 0x7bff,
          AME_FFLAG_OF | AME_FFLAG_NX },
        { AME_OP_REC, 0x10100108, UINT64_C(0x8000000000000000),
          0xfe, AME_FFLAG_DZ | AME_FFLAG_OF | AME_FFLAG_NX },
        { AME_OP_SQRT, 0x14300110, UINT64_C(0x8000000000000000),
          0x8000, 0 },
        { AME_OP_SQRT, 0x14300110, UINT64_C(0xbff0000000000000),
          0x7e00, AME_FFLAG_NV },
        { AME_OP_SQRT, 0x14300110, UINT64_C(0x7ff0000000000001),
          0x7e00, AME_FFLAG_NV },
        { AME_OP_SQRT, 0x14300110, UINT64_C(0x7ff8000000000001),
          0x7e00, 0 },
        /* FRINT's direction applies even when packing an integral value. */
        { AME_OP_FRINTM, 0x14f00110, UINT64_C(0x40a0020000000000),
          0x6800, AME_FFLAG_NX }, /* 2049 -> 2048, despite D.RUP */
        { AME_OP_FRINTP, 0x14b00110, UINT64_C(0x40a0020000000000),
          0x6801, AME_FFLAG_NX }, /* 2049 -> 2050, despite D.RDN */
        { AME_OP_TANH, 0x14300110, 1, 0,
          AME_FFLAG_UF | AME_FFLAG_NX },
        { AME_OP_EXP2, 0x14300110, 1, 0x3c00, AME_FFLAG_NX },
        { AME_OP_EXP2, 0x14300110, UINT64_C(0x7fefffffffffffff),
          0x7c00, AME_FFLAG_OF | AME_FFLAG_NX },
        { AME_OP_EXP2, 0x14f00110, UINT64_C(0xffefffffffffffff),
          1, AME_FFLAG_UF | AME_FFLAG_NX },
        { AME_OP_SIN, 0x14f00110, 1, 1,
          AME_FFLAG_UF | AME_FFLAG_NX },
        { AME_OP_COS, 0x14700110, 1, 0x3bff, AME_FFLAG_NX },
    };
    const AmeDTypeDesc *src = fp_dtype(AME_FP_FORMAT_FP64, AME_FP_RUP);

    for (unsigned i = 0; i < ARRAY_SIZE(cases); i++) {
        const AmeDTypeDesc *dest = dtype(cases[i].destination);
        AmeValue source = { .lo = cases[i].source };
        AmeExecResult result;
        float_status status;
        FloatRoundMode saved_rounding;

        g_assert_true(riscv_ame_fp_status_prepare(&status, dest));
        saved_rounding = get_float_rounding_mode(&status);
        float_raise(float_flag_divbyzero, &status);
        riscv_ame_exec_fp_unary(cases[i].op, dest, src,
                               &source, &result, &status);
        g_test_message("mixed unary rounding case %u", i);
        g_assert_cmphex(result.value.lo, ==, cases[i].expected);
        g_assert_cmphex(result.fflags, ==, cases[i].flags | AME_FFLAG_DZ);
        g_assert_cmpint(get_float_rounding_mode(&status), ==, saved_rounding);
    }
}

static void test_mixed_unary_special(void)
{
    static const struct {
        AmeFPFormat format;
        uint64_t one, nan, infinity;
    } formats[] = {
        { AME_FP_FORMAT_FP16, 0x3c00, 0x7e00, 0x7c00 },
        { AME_FP_FORMAT_BF16, 0x3f80, 0x7fc0, 0x7f80 },
        { AME_FP_FORMAT_FP32, 0x3f800000, 0x7fc00000, 0x7f800000 },
        { AME_FP_FORMAT_FP64, UINT64_C(0x3ff0000000000000),
          UINT64_C(0x7ff8000000000000), UINT64_C(0x7ff0000000000000) },
        { AME_FP_FORMAT_OCP_E4M3, 0x38, 0x7f, 0x7e },
        { AME_FP_FORMAT_OCP_E5M2, 0x3c, 0x7e, 0x7c },
    };
    static const AmeOperation ops[] = {
        AME_OP_REC, AME_OP_FRINTM, AME_OP_FRINTN,
        AME_OP_FRINTP, AME_OP_FRINTZ, AME_OP_LOG2,
        AME_OP_EXP2, AME_OP_TANH, AME_OP_RSQRT,
        AME_OP_SQRT, AME_OP_SIN, AME_OP_COS,
    };

    for (unsigned s = 0; s < ARRAY_SIZE(formats); s++) {
        const AmeDTypeDesc *src = fp_dtype(formats[s].format, AME_FP_RNE);

        for (unsigned d = 0; d < ARRAY_SIZE(formats); d++) {
            for (unsigned rm = 0; rm < (d < 4 ? 6 : 5); rm++) {
                const AmeDTypeDesc *dest = fp_dtype(formats[d].format, rm);

                for (unsigned o = 0; o < ARRAY_SIZE(ops); o++) {
                    AmeValue input = { .lo = formats[s].nan };
                    AmeExecResult result;
                    float_status status;
                    uint64_t expected = BIT_ULL(dest->bits - 1);
                    uint8_t flags = AME_FFLAG_OF;

                    g_assert_true(riscv_ame_fp_status_prepare(&status, dest));
                    float_raise(float_flag_overflow, &status);
                    riscv_ame_exec_fp_unary(
                        ops[o], dest, src, &input, &result, &status);
                    g_assert_cmphex(result.value.lo, ==, formats[d].nan);
                    g_assert_cmphex(result.fflags, ==, flags);

                    input.lo = BIT_ULL(src->bits - 1);
                    if (ops[o] == AME_OP_REC || ops[o] == AME_OP_RSQRT ||
                        ops[o] == AME_OP_LOG2) {
                        expected |= formats[d].infinity;
                        flags |= AME_FFLAG_DZ;
                        if (dest->fp.format == AME_FP_FORMAT_OCP_E4M3) {
                            flags |= AME_FFLAG_NX;
                        }
                    } else if (ops[o] == AME_OP_EXP2 ||
                               ops[o] == AME_OP_COS) {
                        expected = formats[d].one;
                    }
                    riscv_ame_exec_fp_unary(
                        ops[o], dest, src, &input, &result, &status);
                    g_assert_cmphex(result.value.lo, ==, expected);
                    g_assert_cmphex(result.fflags, ==, flags);
                }
            }
        }
    }
}

static void test_sincos_full_range(void)
{
    /* MPFR 4096-bit directed bounds agree on these binary64 RNE results. */
    static const struct {
        uint64_t input, sine, cosine;
    } cases[] = {
        { UINT64_C(0x432fffffffffffff), UINT64_C(0x3fefffccfceaf107),
          UINT64_C(0xbf7c91a4321f73c8) },
        { UINT64_C(0x4330000000000000), UINT64_C(0x3febf996908bb506),
          UINT64_C(0xbfdf1300d681503f) },
        { UINT64_C(0x4330000000000001), UINT64_C(0x3fb053c35068e10d),
          UINT64_C(0xbfefef526a39c993) },
        { UINT64_C(0x7f30000000000000), UINT64_C(0xbfefa983f18b760a),
          UINT64_C(0x3fc28cde21c192df) },
        { UINT64_C(0x7fefffffffffffff), UINT64_C(0x3f7452fc98b34e97),
          UINT64_C(0xbfefffe62ecfab75) },
        /* Ng's difficult near-multiple of pi/2. */
        { UINT64_C(0x7506ac5b262ca1ff), UINT64_C(0x3ff0000000000000),
          UINT64_C(0xbc214ae72e6ba22f) },
    };
    const AmeDTypeDesc *fp64 = fp_dtype(AME_FP_FORMAT_FP64, AME_FP_RNE);

    for (unsigned i = 0; i < ARRAY_SIZE(cases); i++) {
        for (unsigned negative = 0; negative < 2; negative++) {
            AmeValue input = { .lo = cases[i].input |
                                    (uint64_t)negative << 63 };
            AmeExecResult result;
            float_status status;

            g_assert_true(riscv_ame_fp_status_prepare(&status, fp64));
            riscv_ame_exec_fp_unary(
                AME_OP_SIN, fp64, fp64, &input, &result, &status);
            g_assert_cmphex(result.value.lo, ==,
                            cases[i].sine ^ (uint64_t)negative << 63);
            g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);
            g_assert_true(riscv_ame_fp_status_prepare(&status, fp64));
            riscv_ame_exec_fp_unary(
                AME_OP_COS, fp64, fp64, &input, &result, &status);
            g_assert_cmphex(result.value.lo, ==, cases[i].cosine);
            g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);
        }
    }
}

static void test_sincos_tiny_rounding(void)
{
    static const uint64_t centers[] = {
        UINT64_C(0x3c40000000000000), /* 2^-59 */
        UINT64_C(0x3c60000000000000), /* 2^-57 */
        UINT64_C(0x3e10000000000000), /* 2^-30: tiny-path boundary */
    };
    const AmeDTypeDesc *source = fp_dtype(AME_FP_FORMAT_FP64, AME_FP_RNE);

    for (unsigned i = 0; i < ARRAY_SIZE(centers); i++) {
        for (int delta = -1; delta <= 1; delta++) {
            for (unsigned negative = 0; negative < 2; negative++) {
                uint64_t raw = (centers[i] + delta) |
                               (uint64_t)negative << 63;
                AmeValue input = { .lo = raw };

                for (unsigned rm = AME_FP_RNE; rm <= AME_FP_RNO; rm++) {
                    const AmeDTypeDesc *dest = fp_dtype(AME_FP_FORMAT_FP64,
                                                       rm);
                    AmeExecResult result;
                    float_status status;
                    uint64_t expected = raw;

                    if (rm == AME_FP_RTZ ||
                        (rm == AME_FP_RDN && !negative) ||
                        (rm == AME_FP_RUP && negative)) {
                        expected--;
                    } else if (rm == AME_FP_RNO) {
                        expected = (expected - 1) | 1;
                    }
                    g_assert_true(riscv_ame_fp_status_prepare(&status, dest));
                    riscv_ame_exec_fp_unary(
                        AME_OP_SIN, dest, source, &input, &result, &status);
                    g_assert_cmphex(result.value.lo, ==, expected);
                    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);

                    expected = UINT64_C(0x3ff0000000000000) -
                               (rm == AME_FP_RTZ || rm == AME_FP_RDN ||
                                rm == AME_FP_RNO);
                    g_assert_true(riscv_ame_fp_status_prepare(&status, dest));
                    riscv_ame_exec_fp_unary(
                        AME_OP_COS, dest, source, &input, &result, &status);
                    g_assert_cmphex(result.value.lo, ==, expected);
                    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);
                }
            }
        }
    }
}

/* Small exact rational values, independent of the AME arithmetic helpers. */
static AmeExecResult numeric_reference(const AmeDTypeDesc *d, int64_t numerator,
                                       unsigned fractional_bits)
{
    AmeExecResult r = { 0 };
    int64_t divisor = INT64_C(1) << fractional_bits;

    if (d->class == AME_DTYPE_CLASS_INTEGER) {
        int64_t rounded = numerator / divisor;
        int64_t remainder = numerator % divisor;
        int64_t minimum = !d->integer.is_signed ? 0 : d->bits >= 64 ?
                          INT64_MIN : -(INT64_C(1) << (d->bits - 1));
        int64_t maximum = d->bits >= 64 ? INT64_MAX :
                          (INT64_C(1) << (d->bits - d->integer.is_signed)) - 1;
        bool invalid;

        if (remainder < 0) {
            rounded--;
            remainder += divisor;
        }
        switch (d->integer.rounding) {
        case AME_INT_RNU:
            rounded += 2 * remainder >= divisor;
            break;
        case AME_INT_RNE:
            rounded += 2 * remainder > divisor ||
                       (2 * remainder == divisor && (rounded & 1));
            break;
        case AME_INT_RDN:
            break;
        case AME_INT_ROD:
            rounded |= remainder != 0;
            break;
        default:
            g_assert_not_reached();
        }
        invalid = rounded < minimum || rounded > maximum;
        if (invalid && d->integer.saturating) {
            rounded = CLAMP(rounded, minimum, maximum);
            r.saturated = true;
        }
        r.value.lo = rounded;
        if (d->bits < 64) {
            r.value.lo &= MAKE_64BIT_MASK(0, d->bits);
        }
        r.value.hi = d->bits == 128 && rounded < 0 ? UINT64_MAX : 0;
        r.fflags = invalid ? AME_FFLAG_NV : remainder ? AME_FFLAG_NX : 0;
    } else if (d->class == AME_DTYPE_CLASS_OCP) {
        OcpRefResult ocp = ocp_ref_round(d->fp.format, d->fp.rounding,
                                         numerator < 0, llabs(numerator),
                                         -(int)fractional_bits);

        r.value.lo = ocp.value;
        r.fflags = ocp.fflags;
    } else {
        float_status status;

        g_assert_true(riscv_ame_fp_status_prepare(&status, d));
        switch (d->fp.format) {
        case AME_FP_FORMAT_FP16:
            r.value.lo = int64_to_float16_scalbn(numerator,
                                                 -(int)fractional_bits,
                                                 &status);
            break;
        case AME_FP_FORMAT_BF16:
            r.value.lo = int64_to_bfloat16_scalbn(numerator,
                                                  -(int)fractional_bits,
                                                  &status);
            break;
        case AME_FP_FORMAT_FP32:
            r.value.lo = float32_val(int64_to_float32_scalbn(
                numerator, -(int)fractional_bits, &status));
            break;
        case AME_FP_FORMAT_FP64:
            r.value.lo = float64_val(int64_to_float64_scalbn(
                numerator, -(int)fractional_bits, &status));
            break;
        default:
            g_assert_not_reached();
        }
        r.fflags = riscv_ame_fp_status_flags(&status);
    }
    return r;
}

static void assert_numeric_result(const AmeDTypeDesc *d, AmeExecResult actual,
                                   AmeExecResult expected, int64_t exact)
{
    /* Rational arithmetic has no signed zero; check its sign separately. */
    if (!exact && d->class != AME_DTYPE_CLASS_INTEGER) {
        actual.value.lo &= ~BIT_ULL(d->bits - 1);
    }
    g_assert_cmphex(actual.value.lo, ==, expected.value.lo);
    g_assert_cmphex(actual.value.hi, ==, expected.value.hi);
    g_assert_cmphex(actual.fflags, ==, expected.fflags);
    g_assert_cmpint(actual.saturated, ==, expected.saturated);
}

static void test_mixed_numeric_small(void)
{
    static const AmeOperation ops[] = {
        AME_OP_ADD, AME_OP_SUB, AME_OP_ABSDIFF, AME_OP_HDIFF, AME_OP_MEAN,
        AME_OP_MUL, AME_OP_NMUL, AME_OP_MULACC, AME_OP_MULNACC,
        AME_OP_MULADD, AME_OP_MULSUB,
    };
    static const int integers[] = { -7, -1, 0, 1, 7 };
    static const int twice_floats[] = { -12, -3, -1, 0, 1, 3, 12 };
    static const uint32_t integer_types[] = {
        0x40000004, 0x40000080, 0x00000008, 0x00000080,
    };
    uint64_t checked = 0;

    for (unsigned di = 0; di < AME_DTYPE_COUNT; di++) {
        const AmeDTypeDesc *d = &riscv_ame_dtypes[di];
        AmeValue current;

        if (d->class == AME_DTYPE_CLASS_OCP && d->fp.rounding == AME_FP_RNO) {
            continue;
        }
        current = numeric_reference(d, 3, 0).value;
        for (unsigned f = AME_FP_FORMAT_FP16; f <= AME_FP_FORMAT_OCP_E5M2;
             f++) {
            const AmeDTypeDesc *fp = fp_dtype(f, AME_FP_RNE);

            for (unsigned ii = 0; ii < G_N_ELEMENTS(integer_types); ii++) {
                const AmeDTypeDesc *integer = dtype(integer_types[ii]);

                for (unsigned ai = 0; ai < G_N_ELEMENTS(integers); ai++) {
                    int av = integers[ai];
                    AmeValue iv;

                    if (!integer->integer.is_signed && av < 0) {
                        continue;
                    }
                    iv = numeric_reference(integer, av, 0).value;
                    for (unsigned bi = 0; bi < G_N_ELEMENTS(twice_floats);
                         bi++) {
                        int bv = twice_floats[bi];
                        AmeValue fv = numeric_reference(fp, bv, 1).value;

                        for (unsigned order = 0; order < 2; order++) {
                            const AmeDTypeDesc *a = order ? fp : integer;
                            const AmeDTypeDesc *b = order ? integer : fp;
                            AmeValue left = order ? fv : iv;
                            AmeValue right = order ? iv : fv;
                            int64_t x = order ? bv : 2 * av;
                            int64_t y = order ? 2 * av : bv;
                            const int64_t exact[] = {
                                2 * (x + y), 2 * (y - x), 2 * llabs(x - y),
                                y - x, x + y, x * y, -x * y,
                                x * y + 12, 12 - x * y,
                                6 * y + 2 * x, 2 * x - 6 * y,
                            };

                            for (unsigned op = 0; op < G_N_ELEMENTS(ops);
                                 op++) {
                                AmeExecResult actual;
                                AmeExecResult expected = numeric_reference(
                                    d, exact[op], 2);
                                float_status status;

                                g_assert_true(riscv_ame_fp_status_prepare(
                                    &status, d));
                                if (op < 7) {
                                    riscv_ame_exec_fp_binary(
                                        ops[op], d, a, &left, b, &right,
                                        &actual, &status);
                                } else {
                                    riscv_ame_exec_fp_ternary(
                                        ops[op], d, a, &left, b, &right,
                                        &current, &actual, &status);
                                }
                                assert_numeric_result(d, actual, expected,
                                                       exact[op]);
                                checked++;
                            }
                        }
                    }
                }
            }
        }
    }
    g_test_message("mixed numeric rational cases: %" PRIu64, checked);
}

static void test_mixed_numeric_special(void)
{
    const AmeDTypeDesc *fp64 = fp_dtype(AME_FP_FORMAT_FP64, AME_FP_RNE);
    const AmeDTypeDesc *u128 = dtype(0x00000080);
    const AmeDTypeDesc *i8sat = dtype(0x60000008);
    const AmeValue large = { .lo = 1, .hi = BIT_ULL(63) };
    const AmeValue power = { .lo = UINT64_C(0x47e0000000000000) };
    const AmeValue zero = { 0 };
    const AmeValue minus_one = { .lo = UINT64_C(0xbff0000000000000) };
    const uint64_t invalid[] = {
        UINT64_C(0x7ff8000000000000), UINT64_C(0xfff0000000000001),
        UINT64_C(0x7ff0000000000000), UINT64_C(0xfff0000000000000),
    };
    AmeExecResult result;
    float_status status;

    /* Comparisons retain the Int128 low bit and return a full-width mask. */
    g_assert_true(riscv_ame_fp_status_prepare(&status, u128));
    riscv_ame_exec_fp_binary(
        AME_OP_CMPGE, u128, u128, &large, fp64, &power, &result, &status);
    g_assert_cmphex(result.value.lo, ==, UINT64_MAX);
    g_assert_cmphex(result.value.hi, ==, UINT64_MAX);
    g_assert_cmphex(result.fflags, ==, 0);
    riscv_ame_exec_fp_binary(
        AME_OP_CMPGE, u128, fp64, &power, u128, &large, &result, &status);
    g_assert_cmphex(result.value.lo, ==, 0);
    g_assert_cmphex(result.value.hi, ==, 0);

    for (unsigned i = 0; i < G_N_ELEMENTS(invalid); i++) {
        AmeValue input = { .lo = invalid[i] };

        g_assert_true(riscv_ame_fp_status_prepare(&status, i8sat));
        riscv_ame_exec_fp_binary(
            AME_OP_ADD, i8sat, i8sat, &zero, fp64, &input, &result, &status);
        g_assert_cmphex(result.value.lo, ==, i == 3 ? 128 : 127);
        g_assert_cmphex(result.fflags, ==, AME_FFLAG_NV);
        g_assert_false(result.saturated);
    }
    /* Zero times infinity produces NaN before conversion to the integer. */
    g_assert_true(riscv_ame_fp_status_prepare(&status, i8sat));
    riscv_ame_exec_fp_binary(
        AME_OP_MUL, i8sat, i8sat, &zero, fp64,
        &(AmeValue) { .lo = invalid[2] }, &result, &status);
    g_assert_cmphex(result.value.lo, ==, 127);
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NV);
    g_assert_false(result.saturated);

    g_assert_true(riscv_ame_fp_status_prepare(&status, fp64));
    riscv_ame_exec_fp_binary(
        AME_OP_MUL, fp64, u128, &zero, fp64, &minus_one, &result, &status);
    g_assert_cmphex(result.value.lo, ==, BIT_ULL(63));
    g_assert_cmphex(result.fflags, ==, 0);

    g_assert_true(riscv_ame_fp_status_prepare(&status, fp64));
    riscv_ame_exec_fp_unary(
        AME_OP_ABS, fp64, dtype(0x40000080),
        &(AmeValue) { .hi = BIT_ULL(63) }, &result, &status);
    g_assert_cmphex(result.value.lo, ==, power.lo);
    g_assert_cmphex(result.fflags, ==, 0);
    g_assert_true(riscv_ame_fp_status_prepare(&status, i8sat));
    riscv_ame_exec_fp_unary(
        AME_OP_ABS, i8sat, fp64,
        &(AmeValue) { .lo = UINT64_C(0xbff8000000000000) }, &result, &status);
    g_assert_cmphex(result.value.lo, ==, 2);
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);
    g_assert_false(result.saturated);

    /* A floating source also permits an integer exponent-scaled result. */
    g_assert_true(riscv_ame_fp_status_prepare(&status, u128));
    riscv_ame_exec_fp_binary(
        AME_OP_RDEXP, u128, fp64,
        &(AmeValue) { .lo = UINT64_C(0x4008000000000000) },
        u128, &(AmeValue) { .lo = 1 }, &result, &status);
    g_assert_cmphex(result.value.lo, ==, 2);
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);
    g_assert_true(riscv_ame_fp_status_prepare(&status, u128));
    riscv_ame_exec_fp_scale_acc(
        AME_OP_LDEXPACC, u128, fp64, &minus_one, i8sat,
        &(AmeValue) { .lo = 127 }, &large, &result, &status);
    g_assert_cmphex(result.value.lo, ==, 1);
    g_assert_cmphex(result.value.hi, ==, 0);
    g_assert_cmphex(result.fflags, ==, 0);

    /* Scaling extremes must not overflow the host exponent calculation. */
    static const struct {
        AmeOperation op;
        uint32_t dest, exponent;
        uint64_t source, expected;
        uint8_t flags;
        bool saturated;
    } cases[] = {
        { AME_OP_LDEXP, 0x00000080, 0x7fffffff,
          UINT64_C(0x7fefffffffffffff), 0, AME_FFLAG_NV, false },
        { AME_OP_LDEXP, 0x00000080, 0x80000000,
          1, 0, AME_FFLAG_NX, false },
        { AME_OP_RDEXP, 0x00000080, 0x80000000,
          UINT64_C(0x7fefffffffffffff), 0, AME_FFLAG_NV, false },
        { AME_OP_RDEXP, 0x00000080, 0x7fffffff,
          1, 0, AME_FFLAG_NX, false },
        { AME_OP_LDEXPACC, 0x00000080, 0x7fffffff,
          UINT64_C(0x7fefffffffffffff), 7, AME_FFLAG_NV, false },
        { AME_OP_RDEXPACC, 0x00000080, 0x7fffffff,
          1, 7, AME_FFLAG_NX, false },
        { AME_OP_LDEXP, 0x60000008, 0x7fffffff,
          UINT64_C(0x7fefffffffffffff), 127, AME_FFLAG_NV, true },
        { AME_OP_LDEXP, 0x60000008, 0x7fffffff,
          UINT64_C(0xffefffffffffffff), 128, AME_FFLAG_NV, true },
        { AME_OP_LDEXP, 0x18000080, 0x80000000,
          1, 1, AME_FFLAG_NX, false },
        { AME_OP_LDEXP, 0x50000008, 0x80000000,
          UINT64_C(0x8000000000000001), 255, AME_FFLAG_NX, false },
    };

    for (unsigned i = 0; i < G_N_ELEMENTS(cases); i++) {
        const AmeDTypeDesc *dest = dtype(cases[i].dest);
        const AmeValue source = { .lo = cases[i].source };
        const AmeValue exponent = { .lo = cases[i].exponent };
        const AmeValue seven = { .lo = 7 };
        const AmeDTypeDesc *int32 = dtype(0x40000020);

        g_assert_true(riscv_ame_fp_status_prepare(&status, dest));
        if (cases[i].op == AME_OP_LDEXPACC ||
            cases[i].op == AME_OP_RDEXPACC) {
            riscv_ame_exec_fp_scale_acc(
                cases[i].op, dest, fp64, &source, int32, &exponent,
                &seven, &result, &status);
        } else {
            riscv_ame_exec_fp_binary(
                cases[i].op, dest, fp64, &source, int32, &exponent,
                &result, &status);
        }
        g_assert_cmphex(result.value.lo, ==, cases[i].expected);
        g_assert_cmphex(result.value.hi, ==, 0);
        g_assert_cmphex(result.fflags, ==, cases[i].flags);
        g_assert_cmpint(result.saturated, ==, cases[i].saturated);
    }
}

static void test_full_width_addsub(void)
{
    const AmeDTypeDesc *u128 = dtype(0x00000080);
    const AmeDTypeDesc *i32 = dtype(0x40000020);
    const AmeValue maximum = { .lo = UINT64_MAX, .hi = UINT64_MAX };
    const AmeValue half = { .lo = UINT64_C(0x3fe0000000000000) };
    const AmeValue negative_half = { .lo = UINT64_C(0xbfe0000000000000) };
    const AmeValue zero = { 0 };
    const AmeDTypeDesc *f64 = fp_dtype(AME_FP_FORMAT_FP64, AME_FP_RNE);
    const AmeOperation ops[] = {
        AME_OP_ADD, AME_OP_SUB, AME_OP_ABSDIFF, AME_OP_MEAN, AME_OP_HDIFF,
    };

    for (unsigned rm = AME_FP_RNE; rm <= AME_FP_RNO; rm++) {
        bool down = rm == AME_FP_RTZ || rm == AME_FP_RDN || rm == AME_FP_RNO;
        const AmeDTypeDesc *d = fp_dtype(AME_FP_FORMAT_FP64, rm);
        AmeExecResult result;
        float_status status;

        for (unsigned i = 0; i < ARRAY_SIZE(ops); i++) {
            bool subtract = ops[i] == AME_OP_SUB || ops[i] == AME_OP_HDIFF;
            bool halve = ops[i] == AME_OP_MEAN || ops[i] == AME_OP_HDIFF;
            const AmeValue *b = ops[i] == AME_OP_ABSDIFF ?
                                &negative_half : &half;
            uint64_t expected = UINT64_C(0x47f0000000000000) - down;

            expected -= halve ? BIT_ULL(52) : 0;
            g_assert_true(riscv_ame_fp_status_prepare(&status, d));
            riscv_ame_exec_fp_binary(
                ops[i], d, subtract ? f64 : u128,
                subtract ? &negative_half : &maximum,
                subtract ? u128 : f64, subtract ? &maximum : b,
                &result, &status);
            g_assert_cmphex(result.value.lo, ==, expected);
            g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);
        }
        for (unsigned op = 0; op < 2; op++) {
            g_assert_true(riscv_ame_fp_status_prepare(&status, d));
            riscv_ame_exec_fp_scale_acc(
                op ? AME_OP_RDEXPACC : AME_OP_LDEXPACC,
                d, u128, &maximum, i32, &zero, &half, &result, &status);
            g_assert_cmphex(result.value.lo, ==,
                            UINT64_C(0x47f0000000000000) - down);
            g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);
        }

        d = fp_dtype(AME_FP_FORMAT_FP32, rm);
        g_assert_true(riscv_ame_fp_status_prepare(&status, d));
        riscv_ame_exec_fp_binary(
            AME_OP_ADD, d, u128, &maximum, f64, &half, &result, &status);
        g_assert_cmphex(result.value.lo, ==, down ? 0x7f7fffff : 0x7f800000);
        g_assert_cmphex(result.fflags, ==,
                        AME_FFLAG_NX | (down ? 0 : AME_FFLAG_OF));
    }
}

/* Exact FP64 encoding of significand * 2^scale, without host FP arithmetic. */
static uint64_t exact_f64(uint64_t significand, int scale)
{
    unsigned msb = 63 - clz64(significand);

    g_assert_cmpuint(msb, <=, 52);
    return ((uint64_t)(1023 + msb + scale) << 52) |
           ((significand << (52 - msb)) & MAKE_64BIT_MASK(0, 52));
}

static void test_full_width_muladd(void)
{
    const AmeDTypeDesc *u128 = dtype(0x00000080);

    for (unsigned m = 102; m <= 126; m++) {
        uint64_t p = BIT_ULL(128 - m) - 1;
        uint64_t k = p * p;
        Int128 base = int128_lshift(int128_make64(p), m);
        Int128 a = int128_sub(base, int128_make64(p));
        Int128 b = int128_add(base, int128_make64(p));
        AmeValue left = { int128_getlo(a), int128_gethi(a) };
        AmeValue right = { int128_getlo(b), int128_gethi(b) };
        uint64_t boundary = exact_f64(k, 2 * m);
        uint64_t addend = exact_f64(2 * k - 1, -1);

        /* A*B = k*2^(2m)-k. Adding k-0.5 must not become the exact boundary. */
        for (unsigned rm = AME_FP_RNE; rm <= AME_FP_RNO; rm++) {
            const AmeDTypeDesc *d = fp_dtype(AME_FP_FORMAT_FP64, rm);
            AmeExecResult result;
            float_status status;

            for (unsigned negative = 0; negative < 2; negative++) {
                bool down_magnitude = rm == AME_FP_RTZ || rm == AME_FP_RNO ||
                    rm == (negative ? AME_FP_RUP : AME_FP_RDN);
                uint64_t sign = negative ? BIT_ULL(63) : 0;
                AmeValue current = { .lo = addend | sign };

                g_assert_true(riscv_ame_fp_status_prepare(&status, d));
                riscv_ame_exec_fp_ternary(
                    negative ? AME_OP_MULNACC : AME_OP_MULACC,
                    d, u128, &left, u128, &right, &current, &result, &status);
                g_assert_cmphex(result.value.lo, ==,
                                (boundary - down_magnitude) | sign);
                g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);
            }

            /* Cancellation exposes the low product bits: the answer is -k. */
            AmeValue current = { .lo = boundary | BIT_ULL(63) };

            g_assert_true(riscv_ame_fp_status_prepare(&status, d));
            riscv_ame_exec_fp_ternary(
                AME_OP_MULACC, d, u128, &left, u128, &right,
                &current, &result, &status);
            g_assert_cmphex(result.value.lo, ==, exact_f64(k, 0) | BIT_ULL(63));
            g_assert_cmphex(result.fflags, ==, 0);
        }
    }
}

static void test_frint_flags(void)
{
    const AmeDTypeDesc *f64 = fp_dtype(AME_FP_FORMAT_FP64, AME_FP_RNE);
    const AmeDTypeDesc *f16 = fp_dtype(AME_FP_FORMAT_FP16, AME_FP_RUP);
    const struct {
        uint64_t source, expected;
        uint8_t flags;
    } cases[] = {
        { UINT64_C(0x3ff8000000000000), 0x3c00, 0 }, /* 1.5 -> 1 */
        { UINT64_C(0x40a0010000000000), 0x6800, 0 }, /* 2048.5 -> 2048 */
        { UINT64_C(0x40a0030000000000), 0x6800, AME_FFLAG_NX }, /* 2049.5 */
        { UINT64_C(0x8000000000000000), 0x8000, 0 },
        { UINT64_C(0x7ff0000000000000), 0x7c00, 0 },
        { UINT64_C(0xfff0000000000000), 0xfc00, 0 },
        { UINT64_C(0x7ff0000000000001), 0x7e00, AME_FFLAG_NV },
        { UINT64_C(0x7ff8000000000001), 0x7e00, 0 },
    };

    for (unsigned i = 0; i < ARRAY_SIZE(cases); i++) {
        for (unsigned sticky = 0; sticky < 2; sticky++) {
            AmeValue source = { .lo = cases[i].source };
            AmeExecResult result;
            float_status status;
            uint8_t expected_flags = cases[i].flags;

            g_assert_true(riscv_ame_fp_status_prepare(&status, f16));
            if (sticky) {
                float_raise(float_flag_inexact | float_flag_divbyzero, &status);
                expected_flags |= AME_FFLAG_NX | AME_FFLAG_DZ;
            }
            riscv_ame_exec_fp_unary(
                AME_OP_FRINTM, f16, f64, &source, &result, &status);
            g_assert_cmphex(result.value.lo, ==, cases[i].expected);
            g_assert_cmphex(result.fflags, ==, expected_flags);
            g_assert_cmpint(get_float_rounding_mode(&status), ==,
                            float_round_up);
        }
    }
}

static void test_primitive_types(void)
{
    const AmeDTypeDesc *h = fp_dtype(AME_FP_FORMAT_FP16, AME_FP_RNE);
    const AmeDTypeDesc *s = fp_dtype(AME_FP_FORMAT_FP32, AME_FP_RNE);
    const AmeDTypeDesc *e4 = fp_dtype(AME_FP_FORMAT_OCP_E4M3, AME_FP_RNE);
    const AmeDTypeDesc *e5 = fp_dtype(AME_FP_FORMAT_OCP_E5M2, AME_FP_RNE);
    const AmeDTypeDesc *i8 = dtype(0x40000008);
    const AmeDTypeDesc *u128 = dtype(0x00000080);
    AmeDTypeDesc invalid = *h;

    g_assert_true(riscv_ame_fp_supports(AME_OP_CONV, h, u128, NULL));
    g_assert_true(riscv_ame_fp_supports(AME_OP_CONV, e4, s, NULL));
    g_assert_true(riscv_ame_fp_supports(AME_OP_CONV, i8, h, NULL));
    g_assert_true(riscv_ame_fp_supports(AME_OP_ABS, e4, s, NULL));
    g_assert_true(riscv_ame_fp_supports(AME_OP_SQRT, s, s, NULL));
    g_assert_true(riscv_ame_fp_supports(AME_OP_SQRT, h, s, NULL));
    g_assert_true(riscv_ame_fp_supports(AME_OP_SIN, h, h, NULL));
    g_assert_true(riscv_ame_fp_supports(AME_OP_COS, e4, e4, NULL));
    g_assert_true(riscv_ame_fp_supports(AME_OP_SIN, s, s, NULL));
    g_assert_true(riscv_ame_fp_supports(AME_OP_ADD, e4, h, s));
    g_assert_true(riscv_ame_fp_supports(AME_OP_HDIFF, e4, e4, e5));
    g_assert_true(riscv_ame_fp_supports(AME_OP_LOG2SUB, h, s, e4));
    g_assert_true(riscv_ame_fp_supports(AME_OP_CMPGE, u128, e4, s));
    g_assert_false(riscv_ame_fp_supports(AME_OP_CMPGE, h, e4, s));
    g_assert_true(riscv_ame_fp_supports(AME_OP_LDEXP, h, s, i8));
    g_assert_false(riscv_ame_fp_supports(AME_OP_LDEXP, h, s, e4));
    g_assert_true(riscv_ame_fp_supports(AME_OP_RDEXPACC, e4, s, u128));
    g_assert_true(riscv_ame_fp_supports(AME_OP_MULADD, e4, s, h));
    g_assert_true(riscv_ame_fp_supports(AME_OP_MULADD, e4, s, i8));
    g_assert_false(riscv_ame_fp_supports(AME_OP_AND, h, h, h));
    g_assert_false(riscv_ame_fp_supports(AME_OP_ADD, h, h, NULL));
    g_assert_false(riscv_ame_fp_supports(AME_OP_CONV, h, NULL, NULL));
    invalid.bits = 32;
    g_assert_false(riscv_ame_fp_supports(AME_OP_CONV, &invalid, h,
                                        NULL));
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);

    g_test_add_func("/riscv/ame/fp/primitive-types", test_primitive_types);
    g_test_add_func("/riscv/ame/fp/status", test_ieee_status);
    g_test_add_func("/riscv/ame/fp/conversion", test_ieee_conversion);
    g_test_add_func("/riscv/ame/fp/int128-conversion",
                    test_int128_fp_conversion);
    g_test_add_func("/riscv/ame/fp/fp-int-exhaustive", test_fp_int_exhaustive);
    g_test_add_func("/riscv/ame/fp/fp-int-boundaries", test_fp_int_boundaries);
    g_test_add_func("/riscv/ame/fp/fp-int-accrued-flags",
                    test_fp_int_accrued_flags);
    g_test_add_func("/riscv/ame/fp/unary", test_ieee_unary);
    g_test_add_func("/riscv/ame/fp/mixed-unary", test_mixed_unary);
    g_test_add_func("/riscv/ame/fp/mixed-unary-rounding",
                    test_mixed_unary_rounding);
    g_test_add_func("/riscv/ame/fp/mixed-unary-special",
                    test_mixed_unary_special);
    g_test_add_func("/riscv/ame/fp/nonlinear-special",
                    test_ieee_nonlinear_special);
    g_test_add_func("/riscv/ame/fp/tanh-accrued-flags",
                    test_tanh_accrued_flags);
    g_test_add_func("/riscv/ame/fp/nonlinear-exhaustive",
                    test_ieee_nonlinear_exhaustive);
    g_test_add_func("/riscv/ame/fp/sincos", test_ieee_sincos);
    g_test_add_func("/riscv/ame/fp/sincos-full-range", test_sincos_full_range);
    g_test_add_func("/riscv/ame/fp/sincos-tiny-rounding",
                    test_sincos_tiny_rounding);
    g_test_add_func("/riscv/ame/fp/sincos-exhaustive",
                    test_ieee_sincos_exhaustive);
    g_test_add_func("/riscv/ame/fp/log2-composite",
                    test_ieee_log2_composite);
    g_test_add_func("/riscv/ame/fp/binary", test_ieee_binary);
    g_test_add_func("/riscv/ame/fp/fused", test_ieee_fused);
    g_test_add_func("/riscv/ame/fp/scale-acc", test_ieee_scale_acc);
    g_test_add_func("/riscv/ame/fp/mixed-operations",
                    test_fp_mixed_operations);
    g_test_add_func("/riscv/ame/fp/half", test_ieee_half_operations);
    g_test_add_func("/riscv/ame/fp/invalid", test_ieee_invalid_requests);
    g_test_add_func("/riscv/ame/fp/ocp-conversion", test_ocp_conversion);
    g_test_add_func("/riscv/ame/fp/ocp-conversion-exhaustive",
                    test_ocp_conversion_exhaustive);
    g_test_add_func("/riscv/ame/fp/ocp-binary-exhaustive",
                    test_ocp_binary_exhaustive);
    g_test_add_func("/riscv/ame/fp/ocp-basic", test_ocp_basic_operations);
    g_test_add_func("/riscv/ame/fp/ocp-fused-scale-acc",
                    test_ocp_fused_and_scale_acc);
    g_test_add_func("/riscv/ame/fp/ocp-exp2-overflow", test_ocp_exp2_overflow);
    g_test_add_func("/riscv/ame/fp/ocp-nonlinear-exhaustive",
                    test_ocp_nonlinear_exhaustive);
    g_test_add_func("/riscv/ame/fp/mixed-log2-composite",
                    test_fp_mixed_log2_composite);
    g_test_add_func("/riscv/ame/fp/integer-log2-bias", test_integer_log2_bias);
    g_test_add_func("/riscv/ame/fp/integer-float-cancellation",
                    test_integer_float_cancellation);
    g_test_add_func("/riscv/ame/fp/mixed-numeric-small",
                    test_mixed_numeric_small);
    g_test_add_func("/riscv/ame/fp/mixed-numeric-special",
                    test_mixed_numeric_special);
    g_test_add_func("/riscv/ame/fp/full-width-addsub", test_full_width_addsub);
    g_test_add_func("/riscv/ame/fp/full-width-muladd", test_full_width_muladd);
    g_test_add_func("/riscv/ame/fp/frint-flags", test_frint_flags);

    return g_test_run();
}
