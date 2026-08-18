/*
 * Tests for QEMU SoftFloat extensions
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef HW_POISON_H
#error Must define HW_POISON_H to work around TARGET_* poisoning
#endif

#include "qemu/osdep.h"
#include "fpu/softfloat.h"

typedef uint64_t (*Float128NarrowFn)(float128 input, float_status *status);

typedef struct {
    const char *name;
    Float128NarrowFn direct;
    Float128NarrowFn via_float32;
    uint64_t one;
    uint64_t next_one;
    uint64_t max;
    uint64_t inf;
    uint64_t default_nan;
    uint64_t tie_high;
    uint64_t max_high;
    uint64_t overflow_high;
    uint64_t minsub_high;
    uint64_t half_minsub_high;
} NarrowFormat;

typedef uint8_t (*Float128FP8Fn)(float128 input, bool saturate,
                                 float_status *status);
typedef uint8_t (*Float32FP8Fn)(float32 input, bool saturate,
                                float_status *status);
typedef uint8_t (*BFloat16FP8Fn)(bfloat16 input, bool saturate,
                                 float_status *status);

typedef struct {
    const char *name;
    Float128FP8Fn direct;
    Float128FP8Fn via_float32;
    Float32FP8Fn from_float32;
    BFloat16FP8Fn from_bfloat16;
    unsigned int exp_bits;
    unsigned int frac_bits;
    int bias;
    bool e4m3;
    uint8_t one;
    uint8_t next_one;
    uint8_t min_normal;
    uint8_t max;
    uint8_t overflow;
    uint8_t default_nan;
    uint64_t tie_high;
    uint64_t max_high;
    uint64_t overflow_high;
    uint64_t minsub_high;
    uint64_t half_minsub_high;
    uint64_t tiny_to_normal_high;
    uint64_t normal_after_rounding_high;
} FP8Format;

static uint64_t narrow_f16(float128 input, float_status *status)
{
    return float128_to_float16(input, status);
}

static uint64_t narrow_bf16(float128 input, float_status *status)
{
    return float128_to_bfloat16(input, status);
}

static uint64_t narrow_e4m3(float128 input, float_status *status)
{
    return float128_to_float8_e4m3(input, false, status);
}

static uint64_t narrow_e5m2(float128 input, float_status *status)
{
    return float128_to_float8_e5m2(input, false, status);
}

static uint64_t narrow_f16_via_float32(float128 input,
                                       float_status *status)
{
    float32 intermediate = float128_to_float32(input, status);

    return float32_to_float16(intermediate, true, status);
}

static uint64_t narrow_bf16_via_float32(float128 input,
                                        float_status *status)
{
    float32 intermediate = float128_to_float32(input, status);

    return float32_to_bfloat16(intermediate, status);
}

static uint8_t fp8_e4m3_via_float32(float128 input, bool saturate,
                                    float_status *status)
{
    float32 intermediate = float128_to_float32(input, status);

    return float32_to_float8_e4m3(intermediate, saturate, status);
}

static uint8_t fp8_e5m2_via_float32(float128 input, bool saturate,
                                    float_status *status)
{
    float32 intermediate = float128_to_float32(input, status);

    return float32_to_float8_e5m2(intermediate, saturate, status);
}

static const NarrowFormat narrow_formats[] = {
    {
        .name = "float16",
        .direct = narrow_f16,
        .via_float32 = narrow_f16_via_float32,
        .one = 0x3c00,
        .next_one = 0x3c01,
        .max = 0x7bff,
        .inf = 0x7c00,
        .default_nan = 0x7e00,
        .tie_high = UINT64_C(0x3fff002000000000),
        .max_high = UINT64_C(0x400effc000000000),
        .overflow_high = UINT64_C(0x400f000000000000),
        .minsub_high = UINT64_C(0x3fe7000000000000),
        .half_minsub_high = UINT64_C(0x3fe6000000000000),
    }, {
        .name = "bfloat16",
        .direct = narrow_bf16,
        .via_float32 = narrow_bf16_via_float32,
        .one = 0x3f80,
        .next_one = 0x3f81,
        .max = 0x7f7f,
        .inf = 0x7f80,
        .default_nan = 0x7fc0,
        .tie_high = UINT64_C(0x3fff010000000000),
        .max_high = UINT64_C(0x407efe0000000000),
        .overflow_high = UINT64_C(0x407f000000000000),
        .minsub_high = UINT64_C(0x3f7a000000000000),
        .half_minsub_high = UINT64_C(0x3f79000000000000),
    },
};

static const FP8Format fp8_formats[] = {
    {
        .name = "e4m3",
        .direct = float128_to_float8_e4m3,
        .via_float32 = fp8_e4m3_via_float32,
        .from_float32 = float32_to_float8_e4m3,
        .from_bfloat16 = bfloat16_to_float8_e4m3,
        .exp_bits = 4,
        .frac_bits = 3,
        .bias = 7,
        .e4m3 = true,
        .one = 0x38,
        .next_one = 0x39,
        .min_normal = 0x08,
        .max = 0x7e,
        .overflow = 0x7f,
        .default_nan = 0x7f,
        .tie_high = UINT64_C(0x3fff100000000000),
        .max_high = UINT64_C(0x4007c00000000000),
        .overflow_high = UINT64_C(0x4008000000000000),
        .minsub_high = UINT64_C(0x3ff6000000000000),
        .half_minsub_high = UINT64_C(0x3ff5000000000000),
        .tiny_to_normal_high = UINT64_C(0x3ff8e00000000000),
        .normal_after_rounding_high = UINT64_C(0x3ff8f00000000000),
    }, {
        .name = "e5m2",
        .direct = float128_to_float8_e5m2,
        .via_float32 = fp8_e5m2_via_float32,
        .from_float32 = float32_to_float8_e5m2,
        .from_bfloat16 = bfloat16_to_float8_e5m2,
        .exp_bits = 5,
        .frac_bits = 2,
        .bias = 15,
        .one = 0x3c,
        .next_one = 0x3d,
        .min_normal = 0x04,
        .max = 0x7b,
        .overflow = 0x7c,
        .default_nan = 0x7e,
        .tie_high = UINT64_C(0x3fff200000000000),
        .max_high = UINT64_C(0x400ec00000000000),
        .overflow_high = UINT64_C(0x400f000000000000),
        .minsub_high = UINT64_C(0x3fef000000000000),
        .half_minsub_high = UINT64_C(0x3fee000000000000),
        .tiny_to_normal_high = UINT64_C(0x3ff0c00000000000),
        .normal_after_rounding_high = UINT64_C(0x3ff0e00000000000),
    },
};

static const FloatRoundMode all_rounding_modes[] = {
    float_round_nearest_even,
    float_round_to_zero,
    float_round_down,
    float_round_up,
    float_round_ties_away,
    float_round_to_odd,
    float_round_to_odd_inf,
    float_round_nearest_even_max,
};

static float_status make_status(void)
{
    float_status status = { };

    set_float_rounding_mode(float_round_nearest_even, &status);
    set_float_detect_tininess(float_tininess_after_rounding, &status);
    set_default_nan_mode(true, &status);
    set_float_default_nan_pattern(0x40, &status);
    return status;
}

static float_status make_accrued_status(FloatRoundMode mode)
{
    float_status status = make_status();

    set_float_rounding_mode(mode, &status);
    float_raise(float_flag_divbyzero, &status);
    return status;
}

static void assert_narrow(const NarrowFormat *fmt, FloatRoundMode rmode,
                          uint64_t expected,
                          FloatExceptionFlags expected_flags,
                          uint64_t high, uint64_t low)
{
    float_status status = make_status();
    float128 input = make_float128(high, low);
    uint64_t result;

    set_float_rounding_mode(rmode, &status);
    result = fmt->direct(input, &status);
    g_test_message("format: %s", fmt->name);
    g_assert_cmphex(result, ==, expected);
    g_assert_cmphex(get_float_exception_flags(&status), ==, expected_flags);
}

static void test_float128_narrow_boundaries(void)
{
    for (size_t i = 0; i < ARRAY_SIZE(narrow_formats); i++) {
        const NarrowFormat *fmt = &narrow_formats[i];

        assert_narrow(fmt, float_round_nearest_even, fmt->one, 0,
                      UINT64_C(0x3fff000000000000), 0);
        assert_narrow(fmt, float_round_nearest_even, fmt->max, 0,
                      fmt->max_high, 0);
        assert_narrow(fmt, float_round_nearest_even, fmt->inf,
                      float_flag_overflow | float_flag_inexact,
                      fmt->overflow_high, 0);
        assert_narrow(fmt, float_round_nearest_even, 1, 0,
                      fmt->minsub_high, 0);
        assert_narrow(fmt, float_round_nearest_even, 0,
                      float_flag_underflow | float_flag_inexact,
                      fmt->half_minsub_high, 0);
        assert_narrow(fmt, float_round_nearest_even, fmt->default_nan, 0,
                      UINT64_C(0x7fff800000000000), 0);
        assert_narrow(fmt, float_round_nearest_even, fmt->default_nan,
                      float_flag_invalid | float_flag_invalid_snan,
                      UINT64_C(0x7fff000000000000), 1);
    }
}

static void test_float128_narrow_ties(void)
{
    for (size_t i = 0; i < ARRAY_SIZE(narrow_formats); i++) {
        const NarrowFormat *fmt = &narrow_formats[i];

        assert_narrow(fmt, float_round_nearest_even, fmt->one,
                      float_flag_inexact, fmt->tie_high, 0);
        assert_narrow(fmt, float_round_up, fmt->next_one,
                      float_flag_inexact, fmt->tie_high, 0);
    }
}

static void test_float128_narrow_single_rounding(void)
{
    for (size_t i = 0; i < ARRAY_SIZE(narrow_formats); i++) {
        const NarrowFormat *fmt = &narrow_formats[i];
        float_status status = make_status();
        float128 input = make_float128(fmt->tie_high, 1);

        assert_narrow(fmt, float_round_nearest_even, fmt->next_one,
                      float_flag_inexact, fmt->tie_high, 1);
        g_assert_cmphex(fmt->via_float32(input, &status), ==, fmt->one);

        status = make_status();
        input = make_float128(fmt->half_minsub_high, 1);
        assert_narrow(fmt, float_round_nearest_even, 1,
                      float_flag_underflow | float_flag_inexact,
                      fmt->half_minsub_high, 1);
        g_assert_cmphex(fmt->via_float32(input, &status), ==, 0);
    }
}

static void test_float128_narrow_status(void)
{
    static const struct {
        Float128NarrowFn convert;
        unsigned frac_bits;
        uint16_t exp_mask;
        uint16_t sign;
        uint64_t minsub_high;
        bool single_nan;
    } formats[] = {
        { narrow_f16, 10, 0x7c00, 0x8000, UINT64_C(0x3fe7000000000000) },
        { narrow_bf16, 7, 0x7f80, 0x8000, UINT64_C(0x3f7a000000000000) },
        { narrow_e4m3, 3, 0x78, 0x80, UINT64_C(0x3ff6000000000000), true },
        { narrow_e5m2, 2, 0x7c, 0x80, UINT64_C(0x3fef000000000000) },
    };
    static const uint64_t nan_high[] = {
        UINT64_C(0xffffc00000000000), /* Negative qNaN with payload. */
        UINT64_C(0xffff000000000000), /* Negative sNaN, low-word payload. */
    };

    for (size_t f = 0; f < ARRAY_SIZE(formats); f++) {
        float_status status;

        for (size_t n = 0; n < ARRAY_SIZE(nan_high); n++) {
            uint16_t fraction = nan_high[n] >> (48 - formats[f].frac_bits);
            uint16_t expected = formats[f].sign | formats[f].exp_mask |
                (fraction & ((1U << formats[f].frac_bits) - 1)) |
                (1U << (formats[f].frac_bits - 1));
            FloatExceptionFlags flags = float_flag_divbyzero |
                (n == 1 ? float_flag_invalid | float_flag_invalid_snan : 0);

            if (formats[f].single_nan) {
                expected = 0xff;
            }
            status = make_accrued_status(float_round_nearest_even);
            set_default_nan_mode(false, &status);
            g_assert_cmphex(formats[f].convert(
                make_float128(nan_high[n], 1), &status), ==, expected);
            g_assert_cmphex(get_float_exception_flags(&status), ==, flags);
        }

        status = make_accrued_status(float_round_nearest_even);
        set_flush_inputs_to_zero(true, &status);
        g_assert_cmphex(formats[f].convert(make_float128(0, 1), &status),
                        ==, 0);
        g_assert_cmphex(get_float_exception_flags(&status), ==,
                        float_flag_divbyzero |
                        float_flag_input_denormal_flushed);

        for (unsigned before = 0; before < 2; before++) {
            status = make_accrued_status(float_round_nearest_even);
            set_flush_to_zero(true, &status);
            set_float_ftz_detection(before, &status);
            g_assert_cmphex(formats[f].convert(
                make_float128(formats[f].minsub_high, 0), &status), ==, 0);
            g_assert_cmphex(get_float_exception_flags(&status), ==,
                            float_flag_divbyzero |
                            float_flag_output_denormal_flushed);
        }
    }
}

static bool fp8_encoding_is_finite(const FP8Format *fmt, uint8_t raw)
{
    unsigned int exp_mask = (1U << fmt->exp_bits) - 1;
    unsigned int frac_mask = (1U << fmt->frac_bits) - 1;
    unsigned int exp = (raw >> fmt->frac_bits) & exp_mask;
    unsigned int frac = raw & frac_mask;

    if (fmt->e4m3) {
        return exp != exp_mask || frac != frac_mask;
    }
    return exp != exp_mask;
}

static float128 fp8_encoding_to_float128(const FP8Format *fmt, uint8_t raw)
{
    unsigned int exp_mask = (1U << fmt->exp_bits) - 1;
    unsigned int frac_mask = (1U << fmt->frac_bits) - 1;
    unsigned int exp = (raw >> fmt->frac_bits) & exp_mask;
    unsigned int frac = raw & frac_mask;
    unsigned int msb;
    unsigned int remainder;
    uint64_t high = (uint64_t)(raw & 0x80) << 56;
    int unbiased;

    g_assert_true(fp8_encoding_is_finite(fmt, raw));
    if (exp == 0) {
        if (frac == 0) {
            return make_float128(high, 0);
        }
        msb = 31 - __builtin_clz(frac);
        remainder = frac ^ (1U << msb);
        unbiased = 1 - fmt->bias - fmt->frac_bits + msb;
        high |= (uint64_t)(unbiased + 16383) << 48;
        high |= (uint64_t)remainder << (48 - msb);
    } else {
        unbiased = exp - fmt->bias;
        high |= (uint64_t)(unbiased + 16383) << 48;
        high |= (uint64_t)frac << (48 - fmt->frac_bits);
    }
    return make_float128(high, 0);
}

static void assert_fp8(const FP8Format *fmt, FloatRoundMode rmode,
                       bool saturate, uint8_t expected,
                       FloatExceptionFlags expected_flags,
                       uint64_t high, uint64_t low)
{
    float_status status = make_status();
    uint8_t result;

    set_float_rounding_mode(rmode, &status);
    result = fmt->direct(make_float128(high, low), saturate, &status);
    g_test_message("format: %s", fmt->name);
    g_assert_cmphex(result, ==, expected);
    g_assert_cmphex(get_float_exception_flags(&status), ==, expected_flags);
}

static void test_float128_fp8_exact_encodings(void)
{
    for (size_t i = 0; i < ARRAY_SIZE(fp8_formats); i++) {
        const FP8Format *fmt = &fp8_formats[i];
        unsigned int count = 0;

        for (unsigned int raw = 0; raw <= UINT8_MAX; raw++) {
            float_status status;
            uint8_t result;

            if (!fp8_encoding_is_finite(fmt, raw)) {
                continue;
            }
            status = make_status();
            result = fmt->direct(fp8_encoding_to_float128(fmt, raw), false,
                                 &status);
            g_assert_cmphex(result, ==, raw);
            g_assert_cmphex(get_float_exception_flags(&status), ==, 0);
            count++;
        }
        g_assert_cmpuint(count, ==, fmt->e4m3 ? 254 : 248);
    }
}

static void test_float128_fp8_boundaries(void)
{
    const FloatExceptionFlags overflow_flags =
        float_flag_overflow | float_flag_inexact;

    for (size_t i = 0; i < ARRAY_SIZE(fp8_formats); i++) {
        const FP8Format *fmt = &fp8_formats[i];

        assert_fp8(fmt, float_round_nearest_even, false, fmt->one, 0,
                   UINT64_C(0x3fff000000000000), 0);
        assert_fp8(fmt, float_round_nearest_even, false, fmt->max, 0,
                   fmt->max_high, 0);
        assert_fp8(fmt, float_round_nearest_even, false, fmt->overflow,
                   overflow_flags, fmt->overflow_high, 0);
        assert_fp8(fmt, float_round_nearest_even, true, fmt->max,
                   overflow_flags, fmt->overflow_high, 0);
        assert_fp8(fmt, float_round_nearest_even, false, 1, 0,
                   fmt->minsub_high, 0);
        assert_fp8(fmt, float_round_nearest_even, false, 0,
                   float_flag_underflow | float_flag_inexact,
                   fmt->half_minsub_high, 0);
        assert_fp8(fmt, float_round_nearest_even, false,
                   fmt->default_nan, 0,
                   UINT64_C(0x7fff800000000000), 0);
        assert_fp8(fmt, float_round_nearest_even, false,
                   fmt->default_nan,
                   float_flag_invalid | float_flag_invalid_snan,
                   UINT64_C(0x7fff000000000000), 1);
    }
}

static void test_fp8_overflow_modes(void)
{
    const FloatExceptionFlags flags = float_flag_divbyzero |
                                     float_flag_overflow | float_flag_inexact;

    for (size_t i = 0; i < ARRAY_SIZE(fp8_formats); i++) {
        const FP8Format *fmt = &fp8_formats[i];
        uint32_t raw32 = fmt->e4m3 ? UINT32_C(0x44000000) :
                                   UINT32_C(0x47800000);

        /* 512 and 65536 have exact FP8 significands but exceed its range. */
        for (size_t m = 0; m < ARRAY_SIZE(all_rounding_modes); m++) {
            FloatRoundMode mode = all_rounding_modes[m];

            for (unsigned sat = 0; sat < 2; sat++) {
                for (unsigned sign = 0; sign < 2; sign++) {
                    uint8_t expected = (sat ? fmt->max : fmt->overflow) |
                                       (sign << 7);
                    float_status status = make_accrued_status(mode);
                    float128 input = make_float128(
                        fmt->overflow_high | (uint64_t)sign << 63, 0);
                    uint32_t source32 = raw32 | (sign << 31);

                    g_test_message("%s mode=%d sat=%u sign=%u",
                                   fmt->name, mode, sat, sign);
                    g_assert_cmphex(fmt->direct(input, sat, &status), ==,
                                    expected);
                    g_assert_cmphex(get_float_exception_flags(&status), ==,
                                    flags);

                    status = make_accrued_status(mode);
                    g_assert_cmphex(fmt->from_float32(source32, sat, &status),
                                    ==, expected);
                    g_assert_cmphex(get_float_exception_flags(&status), ==,
                                    flags);

                    status = make_accrued_status(mode);
                    g_assert_cmphex(fmt->from_bfloat16(source32 >> 16, sat,
                                                      &status), ==, expected);
                    g_assert_cmphex(get_float_exception_flags(&status), ==,
                                    flags);

                }
            }
        }
    }
}

static void test_fp8_special_values(void)
{
    static const uint64_t high[] = {
        UINT64_C(0x7fff000000000000), /* Infinity */
        UINT64_C(0x7fffc00000000000), /* qNaN */
        UINT64_C(0x7fff400000000000), /* sNaN */
    };
    static const uint32_t raw32[] = {
        UINT32_C(0x7f800000), UINT32_C(0x7fe00000), UINT32_C(0x7fa00000),
    };

    for (size_t i = 0; i < ARRAY_SIZE(fp8_formats); i++) {
        const FP8Format *fmt = &fp8_formats[i];

        for (size_t m = 0; m < ARRAY_SIZE(all_rounding_modes); m++) {
            for (unsigned sat = 0; sat < 2; sat++) {
                for (unsigned sign = 0; sign < 2; sign++) {
                    for (size_t v = 0; v < ARRAY_SIZE(high); v++) {
                        uint8_t expected = v ? fmt->default_nan :
                            (sat ? fmt->max : fmt->overflow) | (sign << 7);
                        FloatExceptionFlags flags = float_flag_divbyzero |
                            (v == 2 ? float_flag_invalid |
                                      float_flag_invalid_snan : 0);
                        float_status status = make_accrued_status(
                            all_rounding_modes[m]);
                        float128 input = make_float128(
                            high[v] | (uint64_t)sign << 63, 0);
                        uint32_t source32 = raw32[v] | (sign << 31);

                        g_test_message("%s mode=%d sat=%u sign=%u special=%zu",
                                       fmt->name, all_rounding_modes[m],
                                       sat, sign, v);
                        g_assert_cmphex(fmt->direct(input, sat, &status), ==,
                                        expected);
                        g_assert_cmphex(get_float_exception_flags(&status), ==,
                                        flags);

                        status = make_accrued_status(all_rounding_modes[m]);
                        g_assert_cmphex(fmt->from_float32(source32, sat,
                                                          &status), ==,
                                        expected);
                        g_assert_cmphex(get_float_exception_flags(&status), ==,
                                        flags);

                        status = make_accrued_status(all_rounding_modes[m]);
                        g_assert_cmphex(fmt->from_bfloat16(source32 >> 16, sat,
                                                           &status), ==,
                                        expected);
                        g_assert_cmphex(get_float_exception_flags(&status), ==,
                                        flags);
                    }
                }
            }
        }
    }
}

static void test_float128_fp8_ties(void)
{
    for (size_t i = 0; i < ARRAY_SIZE(fp8_formats); i++) {
        const FP8Format *fmt = &fp8_formats[i];

        assert_fp8(fmt, float_round_nearest_even, false, fmt->one,
                   float_flag_inexact, fmt->tie_high, 0);
        assert_fp8(fmt, float_round_up, false, fmt->next_one,
                   float_flag_inexact, fmt->tie_high, 0);
    }
}

static void test_float128_fp8_single_rounding(void)
{
    for (size_t i = 0; i < ARRAY_SIZE(fp8_formats); i++) {
        const FP8Format *fmt = &fp8_formats[i];
        float_status status = make_status();
        float128 input = make_float128(fmt->tie_high, 1);

        assert_fp8(fmt, float_round_nearest_even, false, fmt->next_one,
                   float_flag_inexact, fmt->tie_high, 1);
        g_assert_cmphex(fmt->via_float32(input, false, &status), ==, fmt->one);
        g_assert_cmphex(get_float_exception_flags(&status), ==,
                        float_flag_inexact);

        status = make_status();
        input = make_float128(fmt->half_minsub_high, 1);
        assert_fp8(fmt, float_round_nearest_even, false, 1,
                   float_flag_underflow | float_flag_inexact,
                   fmt->half_minsub_high, 1);
        g_assert_cmphex(fmt->via_float32(input, false, &status), ==, 0);
        g_assert_cmphex(get_float_exception_flags(&status), ==,
                        float_flag_underflow | float_flag_inexact);
    }
}

static void assert_fp8_tininess(const FP8Format *fmt, uint64_t high,
                                FloatExceptionFlags after_flags)
{
    float128 input = make_float128(high, 0);
    float_status direct_status = make_status();
    float_status via_status = make_status();
    float_status before_status = make_status();
    uint8_t direct;
    uint8_t via;

    direct = fmt->direct(input, false, &direct_status);
    via = fmt->via_float32(input, false, &via_status);
    g_assert_cmphex(direct, ==, fmt->min_normal);
    g_assert_cmphex(direct, ==, via);
    g_assert_cmphex(get_float_exception_flags(&direct_status), ==,
                    after_flags);
    g_assert_cmphex(get_float_exception_flags(&via_status), ==,
                    after_flags);

    set_float_detect_tininess(float_tininess_before_rounding,
                              &before_status);
    direct = fmt->direct(input, false, &before_status);
    g_assert_cmphex(direct, ==, fmt->min_normal);
    g_assert_cmphex(get_float_exception_flags(&before_status), ==,
                    float_flag_underflow | float_flag_inexact);
}

static void test_float128_fp8_tininess_boundaries(void)
{
    for (size_t i = 0; i < ARRAY_SIZE(fp8_formats); i++) {
        const FP8Format *fmt = &fp8_formats[i];

        /* The encoded result is normal, but the unbounded result is tiny. */
        assert_fp8_tininess(fmt, fmt->tiny_to_normal_high,
                            float_flag_underflow | float_flag_inexact);

        /* Unbounded rounding reaches the minimum normal value. */
        assert_fp8_tininess(fmt, fmt->normal_after_rounding_high,
                            float_flag_inexact);
    }
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);
    g_test_add_func("/softfloat/float128-narrow/boundaries",
                    test_float128_narrow_boundaries);
    g_test_add_func("/softfloat/float128-narrow/ties",
                    test_float128_narrow_ties);
    g_test_add_func("/softfloat/float128-narrow/single-rounding",
                    test_float128_narrow_single_rounding);
    g_test_add_func("/softfloat/float128-narrow/status",
                    test_float128_narrow_status);
    g_test_add_func("/softfloat/float128-fp8/exact-encodings",
                    test_float128_fp8_exact_encodings);
    g_test_add_func("/softfloat/float128-fp8/boundaries",
                    test_float128_fp8_boundaries);
    g_test_add_func("/softfloat/float128-fp8/ties",
                    test_float128_fp8_ties);
    g_test_add_func("/softfloat/float128-fp8/single-rounding",
                    test_float128_fp8_single_rounding);
    g_test_add_func("/softfloat/float128-fp8/tininess-boundaries",
                    test_float128_fp8_tininess_boundaries);
    g_test_add_func("/softfloat/fp8/overflow-modes", test_fp8_overflow_modes);
    g_test_add_func("/softfloat/fp8/special-values", test_fp8_special_values);
    return g_test_run();
}
