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
#include "fpu/softfloat-parts.h"

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
typedef uint8_t (*Int128FP8Fn)(Int128 input, bool saturate,
                               float_status *status);

typedef struct {
    const char *name;
    Float128FP8Fn direct;
    Float128FP8Fn via_float32;
    Float32FP8Fn from_float32;
    BFloat16FP8Fn from_bfloat16;
    Int128FP8Fn from_int128;
    Int128FP8Fn from_uint128;
    unsigned int exp_bits;
    unsigned int frac_bits;
    int bias;
    bool e4m3;
    uint8_t one;
    uint8_t next_one;
    uint8_t two;
    uint8_t four;
    uint8_t min_normal;
    uint8_t max;
    uint8_t overflow;
    uint8_t default_nan;
    uint8_t sqrt_two_rne;
    uint8_t sqrt_two_down;
    uint8_t sqrt_two_up;
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
        .from_int128 = int128_to_float8_e4m3,
        .from_uint128 = uint128_to_float8_e4m3,
        .exp_bits = 4,
        .frac_bits = 3,
        .bias = 7,
        .e4m3 = true,
        .one = 0x38,
        .next_one = 0x39,
        .two = 0x40,
        .four = 0x48,
        .min_normal = 0x08,
        .max = 0x7e,
        .overflow = 0x7f,
        .default_nan = 0x7f,
        .sqrt_two_rne = 0x3b,
        .sqrt_two_down = 0x3b,
        .sqrt_two_up = 0x3c,
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
        .from_int128 = int128_to_float8_e5m2,
        .from_uint128 = uint128_to_float8_e5m2,
        .exp_bits = 5,
        .frac_bits = 2,
        .bias = 15,
        .one = 0x3c,
        .next_one = 0x3d,
        .two = 0x40,
        .four = 0x44,
        .min_normal = 0x04,
        .max = 0x7b,
        .overflow = 0x7c,
        .default_nan = 0x7e,
        .sqrt_two_rne = 0x3e,
        .sqrt_two_down = 0x3d,
        .sqrt_two_up = 0x3e,
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

static void test_int128_direct_conversions(void)
{
    Int128 signed_min = int128_make128(0, UINT64_C(0x8000000000000000));
    Int128 unsigned_max = int128_make128(UINT64_MAX, UINT64_MAX);
    Int128 signed_double_round = int128_make128(
        1, UINT64_C(0x4000000000000200));
    Int128 unsigned_double_round = int128_make128(
        1, UINT64_C(0x8000000000000400));
    float_status status;

    status = make_status();
    g_assert_cmphex(float64_val(int128_to_float64(signed_min, &status)), ==,
                    UINT64_C(0xc7e0000000000000));
    g_assert_cmphex(get_float_exception_flags(&status), ==, 0);

    status = make_status();
    g_assert_cmphex(float64_val(uint128_to_float64(unsigned_max, &status)),
                    ==, UINT64_C(0x47f0000000000000));
    g_assert_cmphex(get_float_exception_flags(&status), ==,
                    float_flag_inexact);

    /*
     * The low one bit is lost by an intermediate binary128 rounding.  A
     * direct conversion must still see that the FP64 halfway value is
     * exceeded and round upward.
     */
    status = make_status();
    g_assert_cmphex(float64_val(int128_to_float64(signed_double_round,
                                                  &status)), ==,
                    UINT64_C(0x47d0000000000001));
    g_assert_cmphex(get_float_exception_flags(&status), ==,
                    float_flag_inexact);

    status = make_status();
    g_assert_cmphex(float64_val(uint128_to_float64(unsigned_double_round,
                                                   &status)), ==,
                    UINT64_C(0x47e0000000000001));
    g_assert_cmphex(get_float_exception_flags(&status), ==,
                    float_flag_inexact);

    status = make_status();
    set_float_rounding_mode(float_round_to_zero, &status);
    g_assert_cmphex(float16_val(uint128_to_float16(unsigned_max, &status)),
                    ==, UINT16_C(0x7bff));
    g_assert_cmphex(get_float_exception_flags(&status), ==,
                    float_flag_overflow | float_flag_inexact);

    status = make_status();
    set_float_rounding_mode(float_round_to_zero, &status);
    g_assert_cmphex(uint128_to_bfloat16(unsigned_max, &status), ==,
                    UINT16_C(0x7f7f));
    g_assert_cmphex(get_float_exception_flags(&status), ==,
                    float_flag_inexact);

    status = make_status();
    set_float_rounding_mode(float_round_to_zero, &status);
    g_assert_cmphex(float32_val(uint128_to_float32(unsigned_max, &status)),
                    ==, UINT32_C(0x7f7fffff));
    g_assert_cmphex(get_float_exception_flags(&status), ==,
                    float_flag_inexact);

    status = make_status();
    g_assert_cmphex(uint128_to_float8_e4m3(int128_make64(1), true, &status),
                    ==, UINT8_C(0x38));
    g_assert_cmphex(get_float_exception_flags(&status), ==, 0);

    status = make_status();
    g_assert_cmphex(uint128_to_float8_e5m2(int128_make64(1), false, &status),
                    ==, UINT8_C(0x3c));
    g_assert_cmphex(get_float_exception_flags(&status), ==, 0);

    status = make_status();
    g_assert_cmphex(int128_to_float8_e4m3(int128_makes64(-1), true,
                                         &status), ==, UINT8_C(0xb8));
    g_assert_cmphex(int128_to_float8_e5m2(int128_makes64(-1), false,
                                         &status), ==, UINT8_C(0xbc));
    g_assert_cmphex(get_float_exception_flags(&status), ==, 0);
}

typedef uint64_t (*Int128ConvertFn)(Int128, float_status *);

typedef struct Int128RoundFormat {
    const char *name;
    Int128ConvertFn convert_unsigned;
    Int128ConvertFn convert_signed;
    Int128 halfway;
    uint64_t even;
    uint64_t odd;
    uint64_t sign;
    uint64_t negative_one;
    uint64_t probe_halfway;
    uint64_t probe_even;
    uint64_t probe_odd;
} Int128RoundFormat;

static uint64_t uint128_to_e4m3_bits(Int128 value, float_status *status)
{
    return uint128_to_float8_e4m3(value, true, status);
}

static uint64_t uint128_to_e5m2_bits(Int128 value, float_status *status)
{
    return uint128_to_float8_e5m2(value, false, status);
}

static uint64_t uint128_to_f16_bits(Int128 value, float_status *status)
{
    return float16_val(uint128_to_float16(value, status));
}

static uint64_t uint128_to_bf16_bits(Int128 value, float_status *status)
{
    return uint128_to_bfloat16(value, status);
}

static uint64_t uint128_to_f32_bits(Int128 value, float_status *status)
{
    return float32_val(uint128_to_float32(value, status));
}

static uint64_t uint128_to_f64_bits(Int128 value, float_status *status)
{
    return float64_val(uint128_to_float64(value, status));
}

static uint64_t int128_to_e4m3_bits(Int128 value, float_status *status)
{
    return int128_to_float8_e4m3(value, true, status);
}

static uint64_t int128_to_e5m2_bits(Int128 value, float_status *status)
{
    return int128_to_float8_e5m2(value, false, status);
}

static uint64_t int128_to_f16_bits(Int128 value, float_status *status)
{
    return float16_val(int128_to_float16(value, status));
}

static uint64_t int128_to_bf16_bits(Int128 value, float_status *status)
{
    return int128_to_bfloat16(value, status);
}

static uint64_t int128_to_f32_bits(Int128 value, float_status *status)
{
    return float32_val(int128_to_float32(value, status));
}

static uint64_t int128_to_f64_bits(Int128 value, float_status *status)
{
    return float64_val(int128_to_float64(value, status));
}

static void test_int128_direct_rounding(void)
{
    static const FloatRoundMode modes[] = {
        float_round_nearest_even,
        float_round_to_zero,
        float_round_down,
        float_round_up,
        float_round_ties_away,
        float_round_to_odd,
    };
    static const bool rounds_up[] = {
        false, false, false, true, true, true,
    };
    static const bool negative_rounds_up[] = {
        false, false, true, false, true, true,
    };
    const Int128RoundFormat formats[] = {
        { "e4m3", uint128_to_e4m3_bits, int128_to_e4m3_bits,
          int128_make64(17), 0x58, 0x59, 0x80, 0xb8,
          34, 0x60, 0x61 },
        { "e5m2", uint128_to_e5m2_bits, int128_to_e5m2_bits,
          int128_make64(9), 0x48, 0x49, 0x80, 0xbc,
          18, 0x4c, 0x4d },
        { "float16", uint128_to_f16_bits, int128_to_f16_bits,
          int128_make64(2049), 0x6800, 0x6801, 0x8000, 0xbc00,
          4098, 0x6c00, 0x6c01 },
        { "bfloat16", uint128_to_bf16_bits, int128_to_bf16_bits,
          int128_make64(257), 0x4380, 0x4381, 0x8000, 0xbf80,
          514, 0x4400, 0x4401 },
        { "float32", uint128_to_f32_bits, int128_to_f32_bits,
          int128_make64(16777217), 0x4b800000, 0x4b800001, 0x80000000,
          0xbf800000,
          UINT64_C(33554434), 0x4c000000, 0x4c000001 },
        { "float64", uint128_to_f64_bits, int128_to_f64_bits,
          int128_make64(UINT64_C(9007199254740993)),
          UINT64_C(0x4340000000000000),
          UINT64_C(0x4340000000000001),
          UINT64_C(0x8000000000000000),
          UINT64_C(0xbff0000000000000),
          UINT64_C(18014398509481986),
          UINT64_C(0x4350000000000000),
          UINT64_C(0x4350000000000001) },
    };

    for (size_t i = 0; i < ARRAY_SIZE(formats); i++) {
        for (size_t j = 0; j < ARRAY_SIZE(modes); j++) {
            float_status status = make_status();
            uint64_t result;

            set_float_rounding_mode(modes[j], &status);
            result = formats[i].convert_unsigned(formats[i].halfway,
                                                 &status);
            g_test_message("format: %s, round: %zu", formats[i].name, j);
            g_assert_cmphex(result, ==,
                            rounds_up[j] ? formats[i].odd : formats[i].even);
            g_assert_cmphex(get_float_exception_flags(&status), ==,
                            float_flag_inexact);

            status = make_status();
            set_float_rounding_mode(modes[j], &status);
            result = formats[i].convert_signed(
                int128_neg(formats[i].halfway), &status);
            g_assert_cmphex(
                result, ==,
                formats[i].sign |
                (negative_rounds_up[j] ? formats[i].odd : formats[i].even));
            g_assert_cmphex(get_float_exception_flags(&status), ==,
                            float_flag_inexact);
        }

        {
            float_status status = make_status();
            uint64_t result = formats[i].convert_unsigned(
                int128_zero(), &status);

            g_assert_cmphex(result, ==, 0);
            g_assert_cmphex(get_float_exception_flags(&status), ==, 0);

            status = make_status();
            result = formats[i].convert_signed(int128_zero(), &status);
            g_assert_cmphex(result, ==, 0);
            g_assert_cmphex(get_float_exception_flags(&status), ==, 0);
        }
        {
            float_status status = make_status();
            uint64_t result = formats[i].convert_signed(
                int128_makes64(-1), &status);

            g_assert_cmphex(result, ==, formats[i].negative_one);
            g_assert_cmphex(get_float_exception_flags(&status), ==, 0);
        }
        {
            float_status status = make_status();
            uint64_t result = formats[i].convert_unsigned(
                int128_make64(formats[i].probe_halfway - 2), &status);

            g_assert_cmphex(result, ==, formats[i].probe_even);
            g_assert_cmphex(get_float_exception_flags(&status), ==, 0);
        }
        for (int side = -1; side <= 1; side += 2) {
            float_status status = make_status();
            uint64_t result = formats[i].convert_unsigned(
                int128_make64(formats[i].probe_halfway + side), &status);

            g_assert_cmphex(result, ==, side < 0 ? formats[i].probe_even :
                                                  formats[i].probe_odd);
            g_assert_cmphex(get_float_exception_flags(&status), ==,
                            float_flag_inexact);
        }
    }
}

static void assert_int128_to_float128(Int128 input, bool is_signed,
                                      FloatRoundMode mode, uint64_t high,
                                      uint64_t low, FloatExceptionFlags flags)
{
    float_status status = make_status();
    float128 result;

    set_float_rounding_mode(mode, &status);
    /* Conversion must accrue exceptions, not clear a previous exception. */
    set_float_exception_flags(float_flag_divbyzero, &status);
    result = is_signed ? int128_to_float128(input, &status) :
                         uint128_to_float128(input, &status);
    g_assert_cmphex(result.high, ==, high);
    g_assert_cmphex(result.low, ==, low);
    g_assert_cmphex(get_float_exception_flags(&status), ==,
                    flags | float_flag_divbyzero);
}

static void test_int128_word_boundaries(void)
{
    const FloatRoundMode mode = float_round_nearest_even;

    assert_int128_to_float128(int128_zero(), false, mode, 0, 0, 0);
    assert_int128_to_float128(int128_zero(), true, mode, 0, 0, 0);
    assert_int128_to_float128(int128_make64(UINT64_MAX), false, mode,
                              UINT64_C(0x403effffffffffff),
                              UINT64_C(0xfffe000000000000), 0);
    assert_int128_to_float128(int128_make128(1, 1), true, mode,
                              UINT64_C(0x403f000000000000),
                              UINT64_C(0x0001000000000000), 0);

    /* Every normalization shift, including the boundary between the words. */
    for (unsigned int bit = 0; bit < 128; bit++) {
        uint64_t lo = bit < 64 ? UINT64_C(1) << bit : 0;
        uint64_t hi = bit >= 64 ? UINT64_C(1) << (bit - 64) : 0;
        Int128 value = int128_make128(lo, hi);
        Int128 negative = bit == 127 ? value : int128_neg(value);
        uint64_t f128_high = (uint64_t)(16383 + bit) << 48;
        uint64_t f64_bits = (uint64_t)(1023 + bit) << 52;
        uint64_t sign = UINT64_C(1) << 63;
        float_status status = make_status();

        assert_int128_to_float128(value, false, mode, f128_high, 0, 0);
        assert_int128_to_float128(negative, true, mode,
                                  f128_high | sign, 0, 0);
        if (bit < 127) {
            assert_int128_to_float128(value, true, mode, f128_high, 0, 0);
        }
        g_assert_cmphex(uint128_to_float64(value, &status), ==, f64_bits);
        g_assert_cmphex(int128_to_float64(negative, &status), ==,
                        f64_bits | sign);
        g_assert_cmphex(get_float_exception_flags(&status), ==, 0);
    }
}

static void test_int128_float128_rounding(void)
{
    static const struct {
        FloatRoundMode mode;
        bool positive_up;
        bool negative_up;
        bool maximum_up;
    } rounding[] = {
        { float_round_nearest_even, false, false, true },
        { float_round_to_zero, false, false, false },
        { float_round_down, false, true, false },
        { float_round_up, true, false, true },
        { float_round_ties_away, true, true, true },
        { float_round_to_odd, true, true, false },
        { float_round_to_odd_inf, true, true, false },
    };
    /* At 2^113 binary128's ULP is 2, so adding 1 is an exact halfway. */
    Int128 halfway = int128_make128(1, UINT64_C(1) << 49);

    for (size_t i = 0; i < ARRAY_SIZE(rounding); i++) {
        FloatRoundMode mode = rounding[i].mode;
        uint64_t max_low = rounding[i].maximum_up ? 0 : UINT64_MAX;
        uint64_t smax_high = rounding[i].maximum_up ?
                            UINT64_C(0x407e000000000000) :
                            UINT64_C(0x407dffffffffffff);
        uint64_t umax_high = rounding[i].maximum_up ?
                            UINT64_C(0x407f000000000000) :
                            UINT64_C(0x407effffffffffff);

        assert_int128_to_float128(halfway, false, mode,
                                  UINT64_C(0x4070000000000000),
                                  rounding[i].positive_up, float_flag_inexact);
        assert_int128_to_float128(halfway, true, mode,
                                  UINT64_C(0x4070000000000000),
                                  rounding[i].positive_up, float_flag_inexact);
        assert_int128_to_float128(int128_neg(halfway), true, mode,
                                  UINT64_C(0xc070000000000000),
                                  rounding[i].negative_up, float_flag_inexact);
        assert_int128_to_float128(INT128_MAX, true, mode,
                                  smax_high, max_low,
                                  float_flag_inexact);
        assert_int128_to_float128(UINT128_MAX, false, mode,
                                  umax_high, max_low,
                                  float_flag_inexact);
        assert_int128_to_float128(INT128_MIN, true, mode,
                                  UINT64_C(0xc07e000000000000), 0, 0);
    }
}

typedef struct Int128RefFormat {
    Int128ConvertFn convert_unsigned;
    Int128ConvertFn convert_signed;
    unsigned exp_bits;
    unsigned frac_bits;
    unsigned bias;
    bool ocp;
    bool e4m3;
} Int128RefFormat;

/* Round the exact integer at unbounded exponent, then apply format limits. */
static Int128 int128_float_reference(const Int128RefFormat *fmt, Int128 raw,
                                     bool is_signed, FloatRoundMode mode,
                                     FloatExceptionFlags *flags)
{
    uint64_t lo = int128_getlo(raw), hi = int128_gethi(raw);
    bool negative = is_signed && (hi >> 63);
    unsigned exp_max = (1U << fmt->exp_bits) - 1;
    Int128 value, rounded, remainder = int128_zero();
    Int128 frac_mask = int128_sub(int128_lshift(int128_one(), fmt->frac_bits),
                                  int128_one());
    bool increment = false, overflow;
    int exponent, shift;

    *flags = 0;
    if (negative) {
        lo = ~lo + 1;
        hi = ~hi + (lo == 0);
    }
    value = int128_make128(lo, hi);
    if (!int128_nz(value)) {
        return int128_zero();
    }
    exponent = hi ? 127 - clz64(hi) : 63 - clz64(lo);
    shift = exponent - (int)fmt->frac_bits;
    if (shift > 0) {
        Int128 mask = int128_sub(int128_lshift(int128_one(), shift),
                                  int128_one());
        Int128 halfway = int128_lshift(int128_one(), shift - 1);

        rounded = int128_urshift(value, shift);
        remainder = int128_and(value, mask);
        switch (mode) {
        case float_round_nearest_even:
        case float_round_nearest_even_max:
        case float_round_ties_away:
            increment = int128_ult(halfway, remainder) ||
                (int128_eq(halfway, remainder) &&
                 (mode == float_round_ties_away ||
                  (int128_getlo(rounded) & 1)));
            break;
        case float_round_down:
            increment = negative && int128_nz(remainder);
            break;
        case float_round_up:
            increment = !negative && int128_nz(remainder);
            break;
        case float_round_to_zero:
            break;
        case float_round_to_odd:
        case float_round_to_odd_inf:
            if (int128_nz(remainder)) {
                rounded = int128_or(rounded, int128_one());
            }
            break;
        default:
            g_assert_not_reached();
        }
        if (increment) {
            rounded = int128_add(rounded, int128_one());
        }
    } else {
        rounded = int128_lshift(value, -shift);
    }
    if (int128_nz(int128_urshift(rounded, fmt->frac_bits + 1))) {
        rounded = int128_urshift(rounded, 1);
        exponent++;
    }

    exponent += fmt->bias;
    overflow = fmt->e4m3 ? exponent > exp_max ||
        (exponent == exp_max && int128_getlo(rounded) == 15) :
        exponent >= exp_max;
    if (overflow) {
        bool bounded = fmt->ocp ? fmt->e4m3 :
            mode == float_round_to_zero || mode == float_round_to_odd ||
            mode == float_round_nearest_even_max ||
            (mode == float_round_down && !negative) ||
            (mode == float_round_up && negative);

        *flags = float_flag_overflow | float_flag_inexact;
        if (fmt->e4m3) {
            value = int128_make64(0x7e);
        } else {
            value = int128_lshift(int128_make64(exp_max - bounded),
                                   fmt->frac_bits);
            if (bounded) {
                value = int128_or(value, frac_mask);
            }
        }
    } else {
        value = int128_or(
            int128_lshift(int128_make64(exponent), fmt->frac_bits),
            int128_and(rounded, frac_mask));
        if (int128_nz(remainder)) {
            *flags = float_flag_inexact;
        }
    }
    if (negative) {
        unsigned sign_bit = fmt->exp_bits + fmt->frac_bits;
        Int128 sign = sign_bit < 64 ? int128_make64(BIT_ULL(sign_bit)) :
                     int128_make128(0, BIT_ULL(sign_bit - 64));

        value = int128_or(value, sign);
    }
    return value;
}

static void check_int128_reference(Int128 raw)
{
    static const Int128RefFormat formats[] = {
        { uint128_to_f16_bits, int128_to_f16_bits, 5, 10, 15 },
        { uint128_to_bf16_bits, int128_to_bf16_bits, 8, 7, 127 },
        { uint128_to_f32_bits, int128_to_f32_bits, 8, 23, 127 },
        { uint128_to_f64_bits, int128_to_f64_bits, 11, 52, 1023 },
        { uint128_to_e4m3_bits, int128_to_e4m3_bits, 4, 3, 7, true, true },
        { uint128_to_e5m2_bits, int128_to_e5m2_bits, 5, 2, 15, true },
        { NULL, NULL, 15, 112, 16383 },
    };

    for (size_t f = 0; f < ARRAY_SIZE(formats); f++) {
        const Int128RefFormat *fmt = &formats[f];

        for (size_t m = 0; m < ARRAY_SIZE(all_rounding_modes); m++) {
            for (unsigned is_signed = 0; is_signed < 2; is_signed++) {
                FloatExceptionFlags flags;
                Int128 expected = int128_float_reference(
                    fmt, raw, is_signed, all_rounding_modes[m], &flags);
                float_status status = make_accrued_status(
                    all_rounding_modes[m]);
                Int128 result;

                if (fmt->frac_bits == 112) {
                    float128 f128 = is_signed ?
                        int128_to_float128(raw, &status) :
                        uint128_to_float128(raw, &status);

                    result = int128_make128(f128.low, f128.high);
                } else {
                    result = int128_make64(is_signed ?
                        fmt->convert_signed(raw, &status) :
                        fmt->convert_unsigned(raw, &status));
                }
                g_assert_cmphex(int128_gethi(result), ==,
                                int128_gethi(expected));
                g_assert_cmphex(int128_getlo(result), ==,
                                int128_getlo(expected));
                g_assert_cmphex(get_float_exception_flags(&status), ==,
                                flags | float_flag_divbyzero);
            }
        }
    }
}

static uint64_t int128_reference_random(uint64_t *seed)
{
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    return *seed;
}

static void test_int128_reference(void)
{
    uint64_t seed = UINT64_C(0x6a09e667f3bcc909);

    check_int128_reference(int128_zero());
    check_int128_reference(INT128_MIN);
    check_int128_reference(INT128_MAX);
    check_int128_reference(UINT128_MAX);
    for (unsigned bit = 0; bit < 128; bit++) {
        Int128 power = bit < 64 ? int128_make64(BIT_ULL(bit)) :
                       int128_make128(0, BIT_ULL(bit - 64));

        check_int128_reference(power);
        check_int128_reference(int128_sub(power, int128_one()));
        check_int128_reference(int128_add(power, int128_one()));
    }
    for (unsigned i = 0; i < 1024; i++) {
        uint64_t lo = int128_reference_random(&seed);
        uint64_t hi = int128_reference_random(&seed);

        check_int128_reference(int128_make128(lo, hi));
    }
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
        uint64_t integer = fmt->e4m3 ? 512 : 65536;

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
                    Int128 signed_input = int128_makes64(
                        sign ? -(int64_t)integer : (int64_t)integer);

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

                    status = make_accrued_status(mode);
                    g_assert_cmphex(fmt->from_int128(signed_input, sat,
                                                     &status), ==, expected);
                    g_assert_cmphex(get_float_exception_flags(&status), ==,
                                    flags);

                    if (!sign) {
                        status = make_accrued_status(mode);
                        g_assert_cmphex(fmt->from_uint128(
                            int128_make64(integer), sat, &status), ==,
                            expected);
                        g_assert_cmphex(get_float_exception_flags(&status), ==,
                                        flags);
                    }
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

static uint8_t fp8_sqrt(const FP8Format *fmt, uint8_t input,
                        FloatRoundMode rmode, FloatExceptionFlags *flags)
{
    float_status status = make_status();
    uint8_t result;

    set_float_rounding_mode(rmode, &status);
    if (fmt->e4m3) {
        result = float8_e4m3_sqrt(input, &status);
    } else {
        result = float8_e5m2_sqrt(input, &status);
    }
    *flags = get_float_exception_flags(&status);
    return result;
}

static void test_float8_sqrt(void)
{
    for (size_t i = 0; i < ARRAY_SIZE(fp8_formats); i++) {
        const FP8Format *fmt = &fp8_formats[i];
        FloatExceptionFlags flags;

        g_assert_cmphex(fp8_sqrt(fmt, fmt->four, float_round_nearest_even,
                                &flags), ==, fmt->two);
        g_assert_cmphex(flags, ==, 0);

        g_assert_cmphex(fp8_sqrt(fmt, fmt->two, float_round_nearest_even,
                                &flags), ==, fmt->sqrt_two_rne);
        g_assert_cmphex(flags, ==, float_flag_inexact);
        g_assert_cmphex(fp8_sqrt(fmt, fmt->two, float_round_down, &flags),
                        ==, fmt->sqrt_two_down);
        g_assert_cmphex(flags, ==, float_flag_inexact);
        g_assert_cmphex(fp8_sqrt(fmt, fmt->two, float_round_up, &flags),
                        ==, fmt->sqrt_two_up);
        g_assert_cmphex(flags, ==, float_flag_inexact);

        g_assert_cmphex(fp8_sqrt(fmt, 0, float_round_nearest_even, &flags),
                        ==, 0);
        g_assert_cmphex(flags, ==, 0);
        g_assert_cmphex(fp8_sqrt(fmt, 0x80, float_round_nearest_even, &flags),
                        ==, 0x80);
        g_assert_cmphex(flags, ==, 0);

        g_assert_cmphex(fp8_sqrt(fmt, fmt->one | 0x80,
                                float_round_nearest_even, &flags),
                        ==, fmt->default_nan);
        g_assert_cmphex(flags, ==,
                        float_flag_invalid | float_flag_invalid_sqrt);
        g_assert_cmphex(fp8_sqrt(fmt, fmt->default_nan,
                                float_round_nearest_even, &flags),
                        ==, fmt->default_nan);
        g_assert_cmphex(flags, ==, 0);

        if (!fmt->e4m3) {
            g_assert_cmphex(fp8_sqrt(fmt, 0x7c, float_round_nearest_even,
                                    &flags), ==, 0x7c);
            g_assert_cmphex(flags, ==, 0);
            g_assert_cmphex(fp8_sqrt(fmt, 0xfc, float_round_nearest_even,
                                    &flags), ==, fmt->default_nan);
            g_assert_cmphex(flags, ==,
                            float_flag_invalid | float_flag_invalid_sqrt);
            g_assert_cmphex(fp8_sqrt(fmt, 0x7d, float_round_nearest_even,
                                    &flags), ==, fmt->default_nan);
            g_assert_cmphex(flags, ==,
                            float_flag_invalid | float_flag_invalid_snan);
        }
    }
}

/* Value in units of the format's minimum subnormal, for finite positive raw. */
static uint64_t fp8_scaled_value(const FP8Format *fmt, uint8_t raw)
{
    unsigned fraction = raw & ((1U << fmt->frac_bits) - 1);
    unsigned exponent = raw >> fmt->frac_bits;

    return exponent ? (uint64_t)((1U << fmt->frac_bits) | fraction) <<
                      (exponent - 1) : fraction;
}

static uint8_t fp8_sqrt_reference(const FP8Format *fmt, uint8_t raw,
                                 FloatRoundMode mode,
                                 FloatExceptionFlags *flags)
{
    uint8_t positive = raw & 0x7f;
    unsigned low = 0, high = fmt->max;
    uint64_t input, lower, upper, midpoint;

    *flags = 0;
    if (!fp8_encoding_is_finite(fmt, raw) &&
        (fmt->e4m3 || positive > 0x7c)) {
        if (!fmt->e4m3 && positive == 0x7d) {
            *flags = float_flag_invalid | float_flag_invalid_snan;
        }
        return fmt->default_nan;
    }
    if (positive == 0) {
        return raw;
    }
    if (raw & 0x80) {
        *flags = float_flag_invalid | float_flag_invalid_sqrt;
        return fmt->default_nan;
    }
    if (!fmt->e4m3 && positive == 0x7c) {
        return raw;
    }
    if (positive < (1U << fmt->frac_bits)) {
        *flags = float_flag_input_denormal_used;
    }

    /* Compare y*y with x in units of (minimum subnormal)^2. */
    input = fp8_scaled_value(fmt, raw) << (fmt->bias + fmt->frac_bits - 1);
    while (low < high) {
        unsigned middle = (low + high + 1) / 2;
        uint64_t value = fp8_scaled_value(fmt, middle);

        if (value * value <= input) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    lower = fp8_scaled_value(fmt, low);
    if (lower * lower == input) {
        return low;
    }
    *flags |= float_flag_inexact;
    switch (mode) {
    case float_round_down:
    case float_round_to_zero:
        return low;
    case float_round_up:
        return low + 1;
    case float_round_to_odd:
    case float_round_to_odd_inf:
        return low | 1;
    case float_round_nearest_even:
    case float_round_nearest_even_max:
    case float_round_ties_away:
        upper = fp8_scaled_value(fmt, low + 1);
        midpoint = (lower + upper) * (lower + upper);
        return low + (input * 4 > midpoint ||
                      (input * 4 == midpoint &&
                       (mode == float_round_ties_away || (low & 1))));
    default:
        g_assert_not_reached();
    }
}

static void test_float8_sqrt_exhaustive(void)
{
    for (size_t f = 0; f < ARRAY_SIZE(fp8_formats); f++) {
        const FP8Format *fmt = &fp8_formats[f];

        for (size_t m = 0; m < ARRAY_SIZE(all_rounding_modes); m++) {
            for (unsigned raw = 0; raw <= UINT8_MAX; raw++) {
                FloatExceptionFlags flags;
                uint8_t expected = fp8_sqrt_reference(
                    fmt, raw, all_rounding_modes[m], &flags);
                float_status status = make_accrued_status(
                    all_rounding_modes[m]);
                uint8_t result = fmt->e4m3 ? float8_e4m3_sqrt(raw, &status) :
                                           float8_e5m2_sqrt(raw, &status);

                g_assert_cmphex(result, ==, expected);
                g_assert_cmphex(get_float_exception_flags(&status), ==,
                                flags | float_flag_divbyzero);
            }
        }
    }
}

static void test_int128_canonical(void)
{
    FloatParts128 zero = uint128_unpack_canonical(int128_zero());

    g_assert_cmpint(zero.cls, ==, float_class_zero);
    g_assert_false(zero.sign);
    for (unsigned bit = 0; bit < 128; bit++) {
        Int128 input = int128_or(int128_lshift(int128_one(), bit),
                                int128_one());
        FloatParts128 p = uint128_unpack_canonical(input);
        Int128 fraction = int128_make128(p.frac_lo, p.frac_hi);
        Int128 recovered = int128_urshift(fraction, 127 - p.exp);

        g_assert_cmpint(p.cls, ==, float_class_normal);
        g_assert_false(p.sign);
        g_assert_cmpint(p.exp, ==, bit);
        g_assert_true(int128_eq(input, recovered));
        if (bit < 127) {
            p = int128_unpack_canonical(int128_neg(input));
            fraction = int128_make128(p.frac_lo, p.frac_hi);
            recovered = int128_urshift(fraction, 127 - p.exp);
            g_assert_cmpint(p.cls, ==, float_class_normal);
            g_assert_true(p.sign);
            g_assert_true(int128_eq(input, recovered));
        }
    }
    zero = int128_unpack_canonical(int128_lshift(int128_one(), 127));
    g_assert_cmpint(zero.cls, ==, float_class_normal);
    g_assert_true(zero.sign);
    g_assert_cmpint(zero.exp, ==, 127);
    g_assert_cmphex(zero.frac_hi, ==, UINT64_C(0x8000000000000000));
    g_assert_cmphex(zero.frac_lo, ==, 0);
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
    g_test_add_func("/softfloat/float8/sqrt", test_float8_sqrt);
    g_test_add_func("/softfloat/float8/sqrt-exhaustive",
                    test_float8_sqrt_exhaustive);
    g_test_add_func("/softfloat/int128/direct-conversions",
                    test_int128_direct_conversions);
    g_test_add_func("/softfloat/int128/direct-rounding",
                    test_int128_direct_rounding);
    g_test_add_func("/softfloat/int128/word-boundaries",
                    test_int128_word_boundaries);
    g_test_add_func("/softfloat/int128/float128-rounding",
                    test_int128_float128_rounding);
    g_test_add_func("/softfloat/int128/reference", test_int128_reference);
    g_test_add_func("/softfloat/int128/canonical", test_int128_canonical);
    return g_test_run();
}
