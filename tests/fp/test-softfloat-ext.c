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

static uint64_t narrow_f16(float128 input, float_status *status)
{
    return float128_to_float16(input, status);
}

static uint64_t narrow_bf16(float128 input, float_status *status)
{
    return float128_to_bfloat16(input, status);
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
    } formats[] = {
        { narrow_f16, 10, 0x7c00, 0x8000, UINT64_C(0x3fe7000000000000) },
        { narrow_bf16, 7, 0x7f80, 0x8000, UINT64_C(0x3f7a000000000000) },
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
    return g_test_run();
}
