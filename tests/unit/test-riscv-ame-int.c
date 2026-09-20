/*
 * RISC-V Attached Matrix Extension integer primitive tests
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/bitops.h"
#include "target/riscv/ame/int.h"

static const AmeDTypeDesc *dtype(uint32_t encoding)
{
    const AmeDTypeDesc *desc = riscv_ame_dtype_by_encoding(encoding);

    g_assert_nonnull(desc);
    return desc;
}

static void assert_acc_words(const AmeIntAccumulator *acc,
                             uint64_t word0, uint64_t word1,
                             uint64_t word2, uint64_t word3,
                             uint64_t word4)
{
    g_assert_cmphex(acc->word[0], ==, word0);
    g_assert_cmphex(acc->word[1], ==, word1);
    g_assert_cmphex(acc->word[2], ==, word2);
    g_assert_cmphex(acc->word[3], ==, word3);
    g_assert_cmphex(acc->word[4], ==, word4);
}

static void assert_product_words(const AmeIntProduct *product,
                                 uint64_t word0, uint64_t word1,
                                 uint64_t word2, uint64_t word3)
{
    g_assert_cmphex(product->word[0], ==, word0);
    g_assert_cmphex(product->word[1], ==, word1);
    g_assert_cmphex(product->word[2], ==, word2);
    g_assert_cmphex(product->word[3], ==, word3);
}

static void test_value_extension(void)
{
    const AmeDTypeDesc *int4 = dtype(UINT32_C(0x40000004));
    const AmeDTypeDesc *int64 = dtype(UINT32_C(0x40000040));
    const AmeDTypeDesc *int128 = dtype(UINT32_C(0x40000080));
    const AmeDTypeDesc *uint4 = dtype(UINT32_C(0x00000004));
    const AmeDTypeDesc *uint64 = dtype(UINT32_C(0x00000040));
    AmeValue value;
    AmeIntAccumulator acc;

    value = (AmeValue) { .lo = UINT64_MAX, .hi = UINT64_MAX };
    riscv_ame_int_acc_init(&acc, int4, &value);
    assert_acc_words(&acc, UINT64_MAX, UINT64_MAX, UINT64_MAX,
                     UINT64_MAX, UINT64_MAX);

    riscv_ame_int_acc_init(&acc, uint4, &value);
    assert_acc_words(&acc, 0xf, 0, 0, 0, 0);

    riscv_ame_int_acc_init(&acc, int64, &value);
    assert_acc_words(&acc, UINT64_MAX, UINT64_MAX, UINT64_MAX,
                     UINT64_MAX, UINT64_MAX);

    riscv_ame_int_acc_init(&acc, uint64, &value);
    assert_acc_words(&acc, UINT64_MAX, 0, 0, 0, 0);

    riscv_ame_int_acc_init(&acc, int128, &value);
    assert_acc_words(&acc, UINT64_MAX, UINT64_MAX, UINT64_MAX,
                     UINT64_MAX, UINT64_MAX);

    value = (AmeValue) { .lo = 0, .hi = UINT64_C(0x8000000000000000) };
    riscv_ame_int_acc_init(&acc, int128, &value);
    assert_acc_words(&acc, 0, UINT64_C(0x8000000000000000),
                     UINT64_MAX, UINT64_MAX, UINT64_MAX);
}

static void test_accumulator_carry_and_finish(void)
{
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    const AmeDTypeDesc *int128 = dtype(UINT32_C(0x40000080));
    const AmeDTypeDesc *int128_sat = dtype(UINT32_C(0x60000080));
    const AmeDTypeDesc *uint128 = dtype(UINT32_C(0x00000080));
    const AmeDTypeDesc *uint128_sat = dtype(UINT32_C(0x20000080));
    AmeValue signed_maximum = { .lo = UINT64_MAX, .hi = INT64_MAX };
    AmeValue signed_minimum = { .hi = UINT64_C(0x8000000000000000) };
    AmeValue unsigned_maximum = { .lo = UINT64_MAX, .hi = UINT64_MAX };
    AmeValue one = { .lo = 1 };
    AmeValue minus_one = { .lo = UINT8_MAX };
    AmeValue zero = { 0 };
    AmeIntAccumulator acc;
    AmeExecResult result;

    riscv_ame_int_acc_init(&acc, int128, &signed_maximum);
    riscv_ame_int_acc_add(&acc, int8, &one);
    assert_acc_words(&acc, 0, UINT64_C(0x8000000000000000), 0, 0, 0);
    riscv_ame_int_acc_finish(&acc, int128, &result);
    g_assert_cmphex(result.value.lo, ==, 0);
    g_assert_cmphex(result.value.hi, ==, UINT64_C(0x8000000000000000));
    g_assert_false(result.saturated);
    g_assert_cmphex(result.fflags, ==, 0);
    riscv_ame_int_acc_finish(&acc, int128_sat, &result);
    g_assert_cmphex(result.value.lo, ==, UINT64_MAX);
    g_assert_cmphex(result.value.hi, ==, INT64_MAX);
    g_assert_true(result.saturated);

    riscv_ame_int_acc_init(&acc, int128, &signed_minimum);
    riscv_ame_int_acc_add(&acc, int8, &minus_one);
    riscv_ame_int_acc_finish(&acc, int128_sat, &result);
    g_assert_cmphex(result.value.lo, ==, 0);
    g_assert_cmphex(result.value.hi, ==, UINT64_C(0x8000000000000000));
    g_assert_true(result.saturated);

    riscv_ame_int_acc_init(&acc, uint128, &unsigned_maximum);
    riscv_ame_int_acc_add(&acc, int8, &one);
    assert_acc_words(&acc, 0, 0, 1, 0, 0);
    riscv_ame_int_acc_finish(&acc, uint128, &result);
    g_assert_cmphex(result.value.lo, ==, 0);
    g_assert_cmphex(result.value.hi, ==, 0);
    g_assert_false(result.saturated);
    riscv_ame_int_acc_finish(&acc, uint128_sat, &result);
    g_assert_cmphex(result.value.lo, ==, UINT64_MAX);
    g_assert_cmphex(result.value.hi, ==, UINT64_MAX);
    g_assert_true(result.saturated);

    riscv_ame_int_acc_init(&acc, uint128, &zero);
    riscv_ame_int_acc_add(&acc, int8, &minus_one);
    riscv_ame_int_acc_finish(&acc, uint128_sat, &result);
    g_assert_cmphex(result.value.lo, ==, 0);
    g_assert_cmphex(result.value.hi, ==, 0);
    g_assert_true(result.saturated);
}

static void assert_finish(const AmeIntAccumulator *acc,
                          const AmeDTypeDesc *d, AmeValue expected,
                          bool saturated)
{
    AmeIntAccumulator before = *acc;
    AmeExecResult result;

    memset(&result, 0xa5, sizeof(result));
    riscv_ame_int_acc_finish(acc, d, &result);
    g_assert_cmphex(result.value.lo, ==, expected.lo);
    g_assert_cmphex(result.value.hi, ==, expected.hi);
    g_assert_cmpint(result.saturated, ==, saturated);
    g_assert_cmphex(result.fflags, ==, 0);
    g_assert_cmpmem(acc, sizeof(*acc), &before, sizeof(before));
}

static void test_accumulator_finish_boundaries(void)
{
    const AmeDTypeDesc *int4 = dtype(0x40000004);

    for (unsigned bits = 4; bits <= AME_VALUE_BITS; bits *= 2) {
        AmeValue mask = {
            .lo = bits < 64 ? UINT64_MAX >> (64 - bits) : UINT64_MAX,
            .hi = bits > 64 ? UINT64_MAX >> (128 - bits) : 0,
        };
        AmeValue sign = {
            .lo = bits <= 64 ? BIT_ULL(bits - 1) : 0,
            .hi = bits > 64 ? BIT_ULL(bits - 65) : 0,
        };

        for (unsigned variant = 0; variant < AME_INT_VARIANTS_PER_WIDTH;
             variant++) {
            const AmeDTypeDesc *d = dtype(bits | (variant << 27));
            AmeValue limits[] = {
                d->integer.is_signed ? sign : (AmeValue) { 0 },
                { .lo = mask.lo ^ (d->integer.is_signed ? sign.lo : 0),
                  .hi = mask.hi ^ (d->integer.is_signed ? sign.hi : 0) },
            };

            for (unsigned limit = 0; limit < ARRAY_SIZE(limits); limit++) {
                for (int delta = -1; delta <= 1; delta++) {
                    AmeValue addend = { .lo = delta & 0xf };
                    AmeValue expected;
                    AmeIntAccumulator acc;
                    bool overflow = (limit == 0 && delta < 0) ||
                                    (limit == 1 && delta > 0);

                    riscv_ame_int_acc_init(&acc, d, &limits[limit]);
                    riscv_ame_int_acc_add(&acc, int4, &addend);
                    expected = d->integer.saturating && overflow ?
                               limits[limit] : (AmeValue) {
                                   .lo = acc.word[0] & mask.lo,
                                   .hi = acc.word[1] & mask.hi,
                               };
                    assert_finish(&acc, d, expected,
                                  d->integer.saturating && overflow);
                }
            }

            /* Every accumulator limb participates in the range check. */
            for (unsigned i = 2; i < AME_INT_ACC_WORDS; i++) {
                for (unsigned negative = 0; negative < 2; negative++) {
                    AmeIntAccumulator acc = { 0 };
                    AmeValue expected = { 0 };

                    acc.word[i] = 1;
                    if (negative) {
                        /* -2^(64*i), represented in two's complement. */
                        acc.word[i] = UINT64_MAX;
                        for (unsigned j = i + 1; j < AME_INT_ACC_WORDS; j++) {
                            acc.word[j] = UINT64_MAX;
                        }
                    }
                    if (d->integer.saturating) {
                        expected = limits[!negative];
                    }
                    assert_finish(&acc, d, expected,
                                  d->integer.saturating);
                }
            }
        }
    }
}

static void test_full_width_product(void)
{
    const AmeDTypeDesc *int128 = dtype(UINT32_C(0x40000080));
    const AmeDTypeDesc *uint128 = dtype(UINT32_C(0x00000080));
    AmeValue maximum = { .lo = UINT64_MAX, .hi = UINT64_MAX };
    AmeValue minimum = { .hi = UINT64_C(0x8000000000000000) };
    AmeValue minus_one = { .lo = UINT64_MAX, .hi = UINT64_MAX };
    AmeValue one = { .lo = 1 };
    AmeValue zero = { 0 };
    AmeIntProduct product;
    AmeIntAccumulator acc = { 0 };

    riscv_ame_int_product_init(&product, uint128, &maximum,
                               uint128, &maximum, false);
    assert_product_words(&product, 1, 0, UINT64_MAX - 1, UINT64_MAX);
    g_assert_false(product.negative);
    riscv_ame_int_acc_add_product(&acc, &product);
    assert_acc_words(&acc, 1, 0, UINT64_MAX - 1, UINT64_MAX, 0);

    riscv_ame_int_product_init(&product, int128, &minimum,
                               int128, &minus_one, false);
    assert_product_words(&product, 0, UINT64_C(0x8000000000000000), 0, 0);
    g_assert_false(product.negative);

    riscv_ame_int_product_init(&product, int128, &minimum,
                               int128, &one, false);
    assert_product_words(&product, 0, UINT64_C(0x8000000000000000), 0, 0);
    g_assert_true(product.negative);

    riscv_ame_int_product_init(&product, int128, &minimum,
                               int128, &one, true);
    g_assert_false(product.negative);

    riscv_ame_int_product_init(&product, uint128, &zero,
                               int128, &minus_one, true);
    assert_product_words(&product, 0, 0, 0, 0);
    g_assert_false(product.negative);
}

static void test_word_product_boundaries(void)
{
    const AmeDTypeDesc *int64 = dtype(UINT32_C(0x40000040));
    const AmeDTypeDesc *uint64 = dtype(UINT32_C(0x00000040));
    const AmeDTypeDesc *int128 = dtype(UINT32_C(0x40000080));
    const AmeDTypeDesc *uint128 = dtype(UINT32_C(0x00000080));
    AmeValue maximum = { .lo = UINT64_MAX };
    AmeValue minimum = { .lo = UINT64_C(0x8000000000000000) };
    AmeValue minus_one = { .lo = UINT64_MAX, .hi = UINT64_MAX };
    AmeValue two = { .lo = 2 };
    AmeValue high_word = { .hi = 1 };
    AmeIntProduct product;

    riscv_ame_int_product_init(&product, uint64, &maximum,
                               uint64, &maximum, false);
    assert_product_words(&product, 1, UINT64_MAX - 1, 0, 0);
    g_assert_false(product.negative);

    /* The magnitude of INT64_MIN fits an unsigned word. */
    riscv_ame_int_product_init(&product, int64, &minimum,
                               int64, &minus_one, false);
    assert_product_words(&product, minimum.lo, 0, 0, 0);
    g_assert_false(product.negative);
    riscv_ame_int_product_init(&product, int64, &minimum,
                               uint64, &maximum, true);
    assert_product_words(&product, minimum.lo, INT64_MAX, 0, 0);
    g_assert_false(product.negative);

    /* Int128 uses the same path when both magnitudes fit one word. */
    riscv_ame_int_product_init(&product, int128, &minus_one,
                               uint128, &maximum, false);
    assert_product_words(&product, UINT64_MAX, 0, 0, 0);
    g_assert_true(product.negative);
    riscv_ame_int_product_init(&product, uint128, &maximum,
                               uint128, &maximum, false);
    assert_product_words(&product, 1, UINT64_MAX - 1, 0, 0);
    g_assert_false(product.negative);

    /* Either nonzero high word must retain the full product path. */
    riscv_ame_int_product_init(&product, uint128, &high_word,
                               uint128, &two, false);
    assert_product_words(&product, 0, 2, 0, 0);
    riscv_ame_int_product_init(&product, uint128, &two,
                               uint128, &high_word, false);
    assert_product_words(&product, 0, 2, 0, 0);
    riscv_ame_int_product_init(&product, uint128, &high_word,
                               uint128, &high_word, false);
    assert_product_words(&product, 0, 0, 1, 0);
    g_assert_false(product.negative);
}

static int test_int4_value(unsigned int raw, bool is_signed)
{
    return is_signed && (raw & 8) ? raw - 16 : raw;
}

static void test_small_product_exhaustive(void)
{
    const AmeDTypeDesc *dtypes[] = {
        dtype(UINT32_C(0x00000004)),
        dtype(UINT32_C(0x40000004)),
    };
    size_t left_type, right_type;

    for (left_type = 0; left_type < ARRAY_SIZE(dtypes); left_type++) {
        for (right_type = 0; right_type < ARRAY_SIZE(dtypes); right_type++) {
            unsigned int left_raw, right_raw;

            for (left_raw = 0; left_raw < 16; left_raw++) {
                for (right_raw = 0; right_raw < 16; right_raw++) {
                    AmeValue left = { .lo = left_raw };
                    AmeValue right = { .lo = right_raw };
                    int expected = test_int4_value(
                        left_raw, dtypes[left_type]->integer.is_signed) *
                        test_int4_value(
                            right_raw, dtypes[right_type]->integer.is_signed);
                    AmeIntProduct product;
                    unsigned int magnitude = expected < 0 ?
                                             -expected : expected;

                    riscv_ame_int_product_init(
                        &product, dtypes[left_type], &left,
                        dtypes[right_type], &right, false);
                    assert_product_words(&product, magnitude, 0, 0, 0);
                    g_assert_cmpint(product.negative, ==, expected < 0);

                    riscv_ame_int_product_init(
                        &product, dtypes[left_type], &left,
                        dtypes[right_type], &right, true);
                    assert_product_words(&product, magnitude, 0, 0, 0);
                    g_assert_cmpint(product.negative, ==,
                                    expected != 0 && expected >= 0);
                }
            }
        }
    }
}

static uint64_t next_test_value(uint64_t *state)
{
    *state = *state * UINT64_C(6364136223846793005) + 1;
    return *state;
}

static void reference_unsigned_product(const AmeValue *left,
                                       const AmeValue *right,
                                       uint64_t result[AME_INT_PRODUCT_WORDS])
{
    uint32_t left_words[4] = {
        (uint32_t)left->lo, (uint32_t)(left->lo >> 32),
        (uint32_t)left->hi, (uint32_t)(left->hi >> 32),
    };
    uint32_t right_words[4] = {
        (uint32_t)right->lo, (uint32_t)(right->lo >> 32),
        (uint32_t)right->hi, (uint32_t)(right->hi >> 32),
    };
    uint32_t product[8] = { 0 };
    size_t i, j;

    for (i = 0; i < ARRAY_SIZE(left_words); i++) {
        uint64_t carry = 0;

        for (j = 0; j < ARRAY_SIZE(right_words); j++) {
            uint64_t partial = (uint64_t)left_words[i] * right_words[j] +
                               product[i + j] + carry;

            product[i + j] = (uint32_t)partial;
            carry = partial >> 32;
        }
        for (j = i + ARRAY_SIZE(right_words); carry != 0; j++) {
            uint64_t sum;

            g_assert_cmpuint(j, <, ARRAY_SIZE(product));
            sum = (uint64_t)product[j] + carry;
            product[j] = (uint32_t)sum;
            carry = sum >> 32;
        }
    }
    for (i = 0; i < AME_INT_PRODUCT_WORDS; i++) {
        result[i] = product[2 * i] |
                    (uint64_t)product[2 * i + 1] << 32;
    }
}

static void test_product_randomized(void)
{
    const AmeDTypeDesc *uint128 = dtype(UINT32_C(0x00000080));
    uint64_t state = UINT64_C(0x5a17c9e3d4b628f1);
    size_t iteration;

    for (iteration = 0; iteration < 4096; iteration++) {
        AmeValue left = {
            .lo = next_test_value(&state),
            .hi = next_test_value(&state),
        };
        AmeValue right = {
            .lo = next_test_value(&state),
            .hi = next_test_value(&state),
        };
        uint64_t expected[AME_INT_PRODUCT_WORDS];
        AmeIntProduct product;
        size_t word;

        for (unsigned zero_high = 0; zero_high < 4; zero_high++) {
            AmeValue a = left, b = right;

            if (zero_high & 1) {
                a.hi = 0;
            }
            if (zero_high & 2) {
                b.hi = 0;
            }
            reference_unsigned_product(&a, &b, expected);
            riscv_ame_int_product_init(&product, uint128, &a,
                                       uint128, &b, false);
            for (word = 0; word < AME_INT_PRODUCT_WORDS; word++) {
                g_assert_cmphex(product.word[word], ==, expected[word]);
            }
            g_assert_false(product.negative);
        }
    }
}

static void test_maximum_dot_product(void)
{
    const AmeDTypeDesc *int128_sat = dtype(UINT32_C(0x60000080));
    const AmeDTypeDesc *uint128 = dtype(UINT32_C(0x00000080));
    const AmeDTypeDesc *uint128_sat = dtype(UINT32_C(0x20000080));
    AmeValue maximum = { .lo = UINT64_MAX, .hi = UINT64_MAX };
    AmeIntProduct product;
    AmeIntAccumulator acc = { 0 };
    AmeExecResult result;
    size_t i;

    g_assert_cmpuint(AME_INT_MAX_DOT_TERMS, ==, 2048);
    riscv_ame_int_product_init(&product, uint128, &maximum,
                               uint128, &maximum, false);
    for (i = 0; i < AME_INT_MAX_DOT_TERMS; i++) {
        riscv_ame_int_acc_add_product(&acc, &product);
    }

    /* 2048 * (2^128 - 1)^2 = 2^267 - 2^140 + 2^11. */
    assert_acc_words(&acc, UINT64_C(0x800), 0,
                     UINT64_C(0xfffffffffffff000), UINT64_MAX,
                     UINT64_C(0x7ff));
    riscv_ame_int_acc_finish(&acc, uint128_sat, &result);
    g_assert_cmphex(result.value.lo, ==, UINT64_MAX);
    g_assert_cmphex(result.value.hi, ==, UINT64_MAX);
    g_assert_true(result.saturated);

    acc = (AmeIntAccumulator) { 0 };
    product.negative = true;
    for (i = 0; i < AME_INT_MAX_DOT_TERMS; i++) {
        riscv_ame_int_acc_add_product(&acc, &product);
    }
    assert_acc_words(&acc, UINT64_C(0xfffffffffffff800), UINT64_MAX,
                     UINT64_C(0xfff), 0,
                     UINT64_C(0xfffffffffffff800));
    riscv_ame_int_acc_finish(&acc, uint128_sat, &result);
    g_assert_cmphex(result.value.lo, ==, 0);
    g_assert_cmphex(result.value.hi, ==, 0);
    g_assert_true(result.saturated);
    riscv_ame_int_acc_finish(&acc, int128_sat, &result);
    g_assert_cmphex(result.value.lo, ==, 0);
    g_assert_cmphex(result.value.hi, ==, UINT64_C(0x8000000000000000));
    g_assert_true(result.saturated);
}

static uint64_t round_half(uint8_t raw, uint32_t encoding)
{
    const AmeDTypeDesc *source_dtype = dtype(UINT32_C(0x40000008));
    const AmeDTypeDesc *dest_dtype = dtype(encoding);
    AmeValue value = { .lo = raw };
    AmeIntAccumulator acc;
    AmeExecResult result;

    riscv_ame_int_acc_init(&acc, source_dtype, &value);
    riscv_ame_int_acc_round_shift(&acc, dest_dtype, 1);
    riscv_ame_int_acc_finish(&acc, dest_dtype, &result);
    return result.value.lo;
}

static void test_rounding(void)
{
    const AmeDTypeDesc *int4_sat = dtype(UINT32_C(0x60000004));
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    const AmeDTypeDesc *int128_rne = dtype(UINT32_C(0x48000080));
    const AmeDTypeDesc *int128_rdn = dtype(UINT32_C(0x50000080));
    const AmeDTypeDesc *int128_rod = dtype(UINT32_C(0x58000080));
    const AmeDTypeDesc *uint8 = dtype(UINT32_C(0x00000008));
    const AmeDTypeDesc *uint128 = dtype(UINT32_C(0x00000080));
    AmeValue value;
    AmeIntAccumulator acc;
    AmeExecResult result;

    g_assert_cmphex(round_half(5, UINT32_C(0x40000008)), ==, 3);
    g_assert_cmphex(round_half(5, UINT32_C(0x48000008)), ==, 2);
    g_assert_cmphex(round_half(5, UINT32_C(0x50000008)), ==, 2);
    g_assert_cmphex(round_half(5, UINT32_C(0x58000008)), ==, 3);

    g_assert_cmphex(round_half(UINT8_C(0xfb), UINT32_C(0x40000008)), ==,
                    UINT8_C(0xfe));
    g_assert_cmphex(round_half(UINT8_C(0xfb), UINT32_C(0x48000008)), ==,
                    UINT8_C(0xfe));
    g_assert_cmphex(round_half(UINT8_C(0xfb), UINT32_C(0x50000008)), ==,
                    UINT8_C(0xfd));
    g_assert_cmphex(round_half(UINT8_C(0xfb), UINT32_C(0x58000008)), ==,
                    UINT8_C(0xfd));

    value = (AmeValue) {
        .lo = UINT64_C(1) << (AME_VALUE_WORD_BITS - 1),
        .hi = 1,
    };
    riscv_ame_int_acc_init(&acc, uint128, &value);
    riscv_ame_int_acc_round_shift(&acc, int128_rne,
                                  AME_VALUE_WORD_BITS);
    riscv_ame_int_acc_finish(&acc, int128_rne, &result);
    g_assert_cmphex(result.value.lo, ==, 2);
    g_assert_cmphex(result.value.hi, ==, 0);

    value = (AmeValue) {
        .lo = UINT64_C(1) << (AME_VALUE_WORD_BITS - 1),
        .hi = 2,
    };
    riscv_ame_int_acc_init(&acc, uint128, &value);
    riscv_ame_int_acc_round_shift(&acc, int128_rne,
                                  AME_VALUE_WORD_BITS);
    riscv_ame_int_acc_finish(&acc, int128_rne, &result);
    g_assert_cmphex(result.value.lo, ==, 2);
    value.lo++;
    riscv_ame_int_acc_init(&acc, uint128, &value);
    riscv_ame_int_acc_round_shift(&acc, int128_rne,
                                  AME_VALUE_WORD_BITS);
    riscv_ame_int_acc_finish(&acc, int128_rne, &result);
    g_assert_cmphex(result.value.lo, ==, 3);

    value = (AmeValue) { .lo = 1 };
    riscv_ame_int_acc_init(&acc, uint128, &value);
    riscv_ame_int_acc_round_shift(&acc, int128_rod, AME_VALUE_BITS);
    riscv_ame_int_acc_finish(&acc, int128_rod, &result);
    g_assert_cmphex(result.value.lo, ==, 1);
    riscv_ame_int_acc_init(&acc, uint128, &value);
    riscv_ame_int_acc_round_shift(&acc, int128_rne, AME_VALUE_BITS);
    riscv_ame_int_acc_finish(&acc, int128_rne, &result);
    g_assert_cmphex(result.value.lo, ==, 0);

    value = (AmeValue) { .lo = UINT64_MAX, .hi = UINT64_MAX };
    riscv_ame_int_acc_init(&acc, int128_rne, &value);
    riscv_ame_int_acc_round_shift(&acc, int128_rne, 400);
    riscv_ame_int_acc_finish(&acc, int128_rne, &result);
    g_assert_cmphex(result.value.lo, ==, 0);
    g_assert_cmphex(result.value.hi, ==, 0);
    riscv_ame_int_acc_init(&acc, int128_rdn, &value);
    riscv_ame_int_acc_round_shift(&acc, int128_rdn, 400);
    riscv_ame_int_acc_finish(&acc, int128_rdn, &result);
    g_assert_cmphex(result.value.lo, ==, UINT64_MAX);
    g_assert_cmphex(result.value.hi, ==, UINT64_MAX);
    riscv_ame_int_acc_init(&acc, int128_rod, &value);
    riscv_ame_int_acc_round_shift(&acc, int128_rod, 400);
    riscv_ame_int_acc_finish(&acc, int128_rod, &result);
    g_assert_cmphex(result.value.lo, ==, UINT64_MAX);
    g_assert_cmphex(result.value.hi, ==, UINT64_MAX);

    value = (AmeValue) { .lo = 15 };
    riscv_ame_int_acc_init(&acc, uint8, &value);
    riscv_ame_int_acc_round_shift(&acc, int4_sat, 1);
    riscv_ame_int_acc_finish(&acc, int4_sat, &result);
    g_assert_cmphex(result.value.lo, ==, 7);
    g_assert_true(result.saturated);

    value = (AmeValue) { .lo = UINT8_C(0xa5) };
    riscv_ame_int_acc_init(&acc, int8, &value);
    riscv_ame_int_acc_round_shift(&acc, int8, 0);
    riscv_ame_int_acc_finish(&acc, int8, &result);
    g_assert_cmphex(result.value.lo, ==, UINT8_C(0xa5));
}

static int64_t reference_round_shift(int64_t value, unsigned int amount,
                                     AmeIntRoundMode mode)
{
    uint64_t raw = value;
    int64_t scale, floor;
    bool round_bit, sticky, discarded;

    if (amount == 0) {
        return value;
    }
    scale = INT64_C(1) << amount;
    floor = value >= 0 ? value / scale :
            -((-value + scale - 1) / scale);
    round_bit = extract64(raw, amount - 1, 1);
    sticky = amount > 1 &&
             (raw & MAKE_64BIT_MASK(0, amount - 1)) != 0;
    discarded = round_bit || sticky;

    switch (mode) {
    case AME_INT_RNU:
        return floor + round_bit;
    case AME_INT_RNE:
        return floor + (round_bit && (sticky || (floor & 1)));
    case AME_INT_RDN:
        return floor;
    case AME_INT_ROD:
        return floor | discarded;
    case AME_INT_ROUND_MAX:
    default:
        g_assert_not_reached();
    }
}

static void test_rounding_exhaustive_small(void)
{
    static const uint32_t encodings[AME_INT_ROUND_MAX] = {
        [AME_INT_RNU] = UINT32_C(0x40000008),
        [AME_INT_RNE] = UINT32_C(0x48000008),
        [AME_INT_RDN] = UINT32_C(0x50000008),
        [AME_INT_ROD] = UINT32_C(0x58000008),
    };
    const AmeDTypeDesc *source_dtype = dtype(UINT32_C(0x40000008));
    unsigned int raw, amount, mode;

    for (raw = 0; raw <= UINT8_MAX; raw++) {
        for (amount = 0; amount <= 12; amount++) {
            for (mode = 0; mode < AME_INT_ROUND_MAX; mode++) {
                const AmeDTypeDesc *dest_dtype = dtype(encodings[mode]);
                AmeValue value = { .lo = raw };
                AmeIntAccumulator acc;
                AmeExecResult result;
                int64_t expected = reference_round_shift(
                    (int8_t)raw, amount, mode);

                riscv_ame_int_acc_init(&acc, source_dtype, &value);
                riscv_ame_int_acc_round_shift(&acc, dest_dtype, amount);
                riscv_ame_int_acc_finish(&acc, dest_dtype, &result);
                g_assert_cmphex(result.value.lo, ==, (uint8_t)expected);
                g_assert_false(result.saturated);
            }
        }
    }
}

static void test_small_saturation(void)
{
    const AmeDTypeDesc *int4 = dtype(UINT32_C(0x40000004));
    const AmeDTypeDesc *int4_sat = dtype(UINT32_C(0x60000004));
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    const AmeDTypeDesc *uint4 = dtype(UINT32_C(0x00000004));
    const AmeDTypeDesc *uint4_sat = dtype(UINT32_C(0x20000004));
    const AmeDTypeDesc *uint8 = dtype(UINT32_C(0x00000008));
    AmeValue value;
    AmeIntAccumulator acc;
    AmeExecResult result;

    value = (AmeValue) { .lo = 8 };
    riscv_ame_int_acc_init(&acc, uint8, &value);
    riscv_ame_int_acc_finish(&acc, int4, &result);
    g_assert_cmphex(result.value.lo, ==, 8);
    g_assert_false(result.saturated);
    riscv_ame_int_acc_finish(&acc, int4_sat, &result);
    g_assert_cmphex(result.value.lo, ==, 7);
    g_assert_true(result.saturated);

    value.lo = UINT8_C(0xf7); /* -9 */
    riscv_ame_int_acc_init(&acc, int8, &value);
    riscv_ame_int_acc_finish(&acc, int4_sat, &result);
    g_assert_cmphex(result.value.lo, ==, 8);
    g_assert_true(result.saturated);

    value.lo = 16;
    riscv_ame_int_acc_init(&acc, uint8, &value);
    riscv_ame_int_acc_finish(&acc, uint4, &result);
    g_assert_cmphex(result.value.lo, ==, 0);
    g_assert_false(result.saturated);
    riscv_ame_int_acc_finish(&acc, uint4_sat, &result);
    g_assert_cmphex(result.value.lo, ==, 0xf);
    g_assert_true(result.saturated);
}

static void test_conversion_exhaustive_small(void)
{
    const AmeDTypeDesc *sources[] = {
        dtype(UINT32_C(0x40000008)),
        dtype(UINT32_C(0x00000008)),
    };
    size_t source_index;

    for (source_index = 0; source_index < ARRAY_SIZE(sources);
         source_index++) {
        const AmeDTypeDesc *source_dtype = sources[source_index];
        unsigned int raw;

        for (raw = 0; raw <= UINT8_MAX; raw++) {
            int64_t value = source_dtype->integer.is_signed ?
                            (int64_t)(int8_t)raw : (int64_t)raw;
            size_t id;

            for (id = 0; id < AME_DTYPE_COUNT; id++) {
                const AmeDTypeDesc *dest_dtype = &riscv_ame_dtypes[id];
                AmeValue source = { .lo = raw };
                AmeExecResult result;
                int64_t converted = value;
                bool saturated = false;

                if (dest_dtype->class != AME_DTYPE_CLASS_INTEGER ||
                    dest_dtype->bits != 4) {
                    continue;
                }
                if (dest_dtype->integer.saturating) {
                    int64_t minimum = dest_dtype->integer.is_signed ? -8 : 0;
                    int64_t maximum = dest_dtype->integer.is_signed ? 7 : 15;

                    if (converted < minimum) {
                        converted = minimum;
                        saturated = true;
                    } else if (converted > maximum) {
                        converted = maximum;
                        saturated = true;
                    }
                }

                riscv_ame_exec_int_convert(dest_dtype, source_dtype,
                                           &source, &result);
                g_assert_cmphex(result.value.lo, ==,
                                (uint64_t)converted & UINT64_C(0xf));
                g_assert_cmphex(result.value.hi, ==, 0);
                g_assert_cmpint(result.saturated, ==, saturated);
                g_assert_cmphex(result.fflags, ==, 0);
            }
        }
    }
}

static void test_conversion_int128_boundaries(void)
{
    const AmeDTypeDesc *int128 = dtype(UINT32_C(0x40000080));
    const AmeDTypeDesc *int128_sat = dtype(UINT32_C(0x60000080));
    const AmeDTypeDesc *uint128 = dtype(UINT32_C(0x00000080));
    const AmeDTypeDesc *uint128_sat = dtype(UINT32_C(0x20000080));
    AmeValue value;
    AmeExecResult result;

    value = (AmeValue) { .lo = UINT64_MAX, .hi = UINT64_MAX };
    riscv_ame_exec_int_convert(int128, uint128, &value, &result);
    g_assert_cmphex(result.value.lo, ==, UINT64_MAX);
    g_assert_cmphex(result.value.hi, ==, UINT64_MAX);
    g_assert_false(result.saturated);
    riscv_ame_exec_int_convert(int128_sat, uint128, &value, &result);
    g_assert_cmphex(result.value.lo, ==, UINT64_MAX);
    g_assert_cmphex(result.value.hi, ==,
                    UINT64_C(0x7fffffffffffffff));
    g_assert_true(result.saturated);

    value = (AmeValue) { .hi = UINT64_C(0x8000000000000000) };
    riscv_ame_exec_int_convert(uint128, int128, &value, &result);
    g_assert_cmphex(result.value.lo, ==, 0);
    g_assert_cmphex(result.value.hi, ==,
                    UINT64_C(0x8000000000000000));
    g_assert_false(result.saturated);
    riscv_ame_exec_int_convert(uint128_sat, int128, &value, &result);
    g_assert_cmphex(result.value.lo, ==, 0);
    g_assert_cmphex(result.value.hi, ==, 0);
    g_assert_true(result.saturated);
}

static void test_primitive_types(void)
{
    const AmeDTypeDesc *i8 = dtype(0x40000008);
    const AmeDTypeDesc *i16 = dtype(0x40000010);
    const AmeDTypeDesc *u8 = dtype(0x00000008);
    const AmeDTypeDesc *sat8 = dtype(0x60000008);
    const AmeDTypeDesc *fp16 = dtype(0x14300110);
    AmeDTypeDesc invalid = *i8;

    g_assert_true(riscv_ame_int_supports(AME_OP_CONV, u8, i16, NULL));
    g_assert_true(riscv_ame_int_supports(AME_OP_ABS, i16, i8, NULL));
    g_assert_true(riscv_ame_int_supports(AME_OP_ABS, i8, i16, NULL));
    g_assert_true(riscv_ame_int_supports(AME_OP_ADD, i16, i8, i8));
    g_assert_true(riscv_ame_int_supports(AME_OP_ADD, i8, i16, i8));
    g_assert_true(riscv_ame_int_supports(AME_OP_ADD, i16, i8, u8));
    g_assert_true(riscv_ame_int_supports(AME_OP_MAX, i8, i8, i8));
    g_assert_false(riscv_ame_int_supports(AME_OP_MAX, sat8, i8, i8));
    g_assert_true(riscv_ame_int_supports(AME_OP_AND, i8, i8, i8));
    g_assert_false(riscv_ame_int_supports(AME_OP_AND, i16, i8, i8));
    g_assert_true(riscv_ame_int_supports(AME_OP_CMPGE, u8, i8, i16));
    g_assert_true(riscv_ame_int_supports(AME_OP_CMPGE, u8, i8, u8));
    g_assert_true(riscv_ame_int_supports(AME_OP_SRA, i8, i8, u8));
    g_assert_false(riscv_ame_int_supports(AME_OP_SRA, u8, u8, i8));
    g_assert_true(riscv_ame_int_supports(AME_OP_SLL, u8, u8, i8));
    g_assert_true(riscv_ame_int_supports(AME_OP_LDEXP, i16, i8, u8));
    g_assert_true(riscv_ame_int_supports(AME_OP_LDEXP, i8, i16, u8));
    g_assert_true(riscv_ame_int_supports(AME_OP_RDEXPACC, i16, i8, u8));
    g_assert_true(riscv_ame_int_supports(AME_OP_MULADD, i16, i8, i8));
    g_assert_true(riscv_ame_int_supports(AME_OP_MULADD, i16, i8, u8));
    g_assert_false(riscv_ame_int_supports(AME_OP_ADD, i8, fp16, i8));
    g_assert_false(riscv_ame_int_supports(AME_OP_SQRT, i8, i8, NULL));
    g_assert_false(riscv_ame_int_supports(AME_OP_ADD, i8, i8, NULL));
    g_assert_false(riscv_ame_int_supports(AME_OP_CONV, NULL, i8, NULL));
    invalid.bits = 12;
    g_assert_false(riscv_ame_int_supports(AME_OP_CONV, i8, &invalid,
                                          NULL));
}

static void test_mixed_boundaries(void)
{
    for (unsigned bits = 4; bits <= 128; bits *= 2) {
        const AmeDTypeDesc *s = dtype(0x40000000 | bits);
        const AmeDTypeDesc *u = dtype(bits);
        const AmeDTypeDesc *usat = dtype(0x20000000 | bits);
        AmeValue maximum = {
            .lo = bits >= 64 ? UINT64_MAX : (UINT64_C(1) << bits) - 1,
            .hi = bits == 128 ? UINT64_MAX : 0,
        };
        AmeValue minus_one = maximum;
        AmeValue one = { .lo = 1 };
        AmeExecResult result;

        /* Unsigned maximum is positive even when its top input bit is set. */
        riscv_ame_exec_int_binary(
            AME_OP_ADD, usat, u, &maximum, s, &minus_one, &result);
        g_assert_cmphex(result.value.lo, ==, maximum.lo - 1);
        g_assert_cmphex(result.value.hi, ==, maximum.hi);
        g_assert_false(result.saturated);

        riscv_ame_exec_int_binary(
            AME_OP_ABSDIFF, usat, s, &minus_one, u, &one, &result);
        g_assert_cmphex(result.value.lo, ==, 2);
        g_assert_cmphex(result.value.hi, ==, 0);
        g_assert_false(result.saturated);

        riscv_ame_exec_int_binary(
            AME_OP_CMPGE, usat, u, &maximum, s, &minus_one, &result);
        g_assert_cmphex(result.value.lo, ==, maximum.lo);
        g_assert_cmphex(result.value.hi, ==, maximum.hi);
        g_assert_false(result.saturated);
        riscv_ame_exec_int_binary(
            AME_OP_CMPLT, u, s, &minus_one, u, &maximum, &result);
        g_assert_cmphex(result.value.lo, ==, maximum.lo);
        g_assert_cmphex(result.value.hi, ==, maximum.hi);
        g_assert_false(result.saturated);

        /* Convert only the result, not the negative source or product. */
        riscv_ame_exec_int_binary(
            AME_OP_MUL, u, s, &minus_one, u, &maximum, &result);
        g_assert_cmphex(result.value.lo, ==, 1);
        g_assert_cmphex(result.value.hi, ==, 0);
        g_assert_false(result.saturated);
        riscv_ame_exec_int_binary(
            AME_OP_MUL, usat, s, &minus_one, u, &maximum, &result);
        g_assert_cmphex(result.value.lo, ==, 0);
        g_assert_cmphex(result.value.hi, ==, 0);
        g_assert_true(result.saturated);
        riscv_ame_exec_int_ternary(
            AME_OP_MULACC, usat, s, &minus_one, u, &maximum,
            &maximum, &result);
        g_assert_cmphex(result.value.lo, ==, 0);
        g_assert_cmphex(result.value.hi, ==, 0);
        g_assert_false(result.saturated);
    }
}

static void assert_narrow_result(const AmeExecResult *result,
                                 const AmeDTypeDesc *dest, int64_t expected)
{
    int64_t minimum = dest->integer.is_signed ?
                      -(INT64_C(1) << (dest->bits - 1)) : 0;
    int64_t maximum = dest->integer.is_signed ?
                      (INT64_C(1) << (dest->bits - 1)) - 1 :
                      (INT64_C(1) << dest->bits) - 1;
    bool saturated = dest->integer.saturating &&
                     (expected < minimum || expected > maximum);

    if (dest->integer.saturating) {
        expected = CLAMP(expected, minimum, maximum);
    }
    g_assert_cmphex(result->value.lo, ==,
                    (uint64_t)expected & MAKE_64BIT_MASK(0, dest->bits));
    g_assert_cmphex(result->value.hi, ==, 0);
    g_assert_cmpint(result->saturated, ==, saturated);
    g_assert_cmphex(result->fflags, ==, 0);
}

static void test_mixed_shifts(void)
{
    static const AmeOperation ops[] = { AME_OP_SLL, AME_OP_SRL, AME_OP_SRA };
    const AmeDTypeDesc *amount_type = dtype(0x00000080);

    for (unsigned bits = 4; bits <= 8; bits *= 2) {
        for (unsigned sign = 0; sign < 2; sign++) {
            const AmeDTypeDesc *a = dtype(sign << 30 | bits);

            for (unsigned id = 0; id < AME_INT_DTYPE_COUNT; id++) {
                const AmeDTypeDesc *d = &riscv_ame_dtypes[id];

                if (d->bits > 8) {
                    continue;
                }
                for (unsigned op = 0; op < G_N_ELEMENTS(ops); op++) {
                    bool supported = ops[op] != AME_OP_SRA || sign;

                    g_assert_cmpint(riscv_ame_int_supports(
                        ops[op], d, a, amount_type), ==, supported);
                    if (!supported) {
                        continue;
                    }
                    for (unsigned raw = 0; raw < (1u << bits); raw++) {
                        int64_t input = raw;
                        AmeValue data = { .lo = raw };

                        if (sign && (raw & (1u << (bits - 1)))) {
                            input -= INT64_C(1) << bits;
                        }

                        for (unsigned n = 0; n < 2 * bits; n++) {
                            AmeValue amount = { .lo = n, .hi = UINT64_MAX };
                            unsigned shift = n % bits;
                            int64_t divisor = INT64_C(1) << shift;
                            int64_t expected;
                            AmeExecResult result;

                            if (ops[op] == AME_OP_SLL) {
                                expected = input * divisor;
                            } else if (ops[op] == AME_OP_SRL) {
                                expected = raw / divisor;
                            } else {
                                expected = input / divisor;
                                if (input < 0 && input % divisor) {
                                    expected--;
                                }
                            }
                            riscv_ame_exec_int_binary(
                                ops[op], d, a, &data, amount_type, &amount,
                                &result);
                            assert_narrow_result(&result, d, expected);
                        }
                    }
                }
            }
        }
    }
}

static void test_mixed_shift_boundaries(void)
{
    const AmeDTypeDesc *s128 = dtype(0x40000080);
    const AmeDTypeDesc *u128 = dtype(0x00000080);
    AmeValue minus_one = { .lo = UINT64_MAX, .hi = UINT64_MAX };
    AmeValue amount = { 0 };
    AmeExecResult result;

    riscv_ame_exec_int_binary(
        AME_OP_SRA, dtype(0x20000008), s128, &minus_one, u128, &amount,
        &result);
    assert_narrow_result(&result, dtype(0x20000008), -1);
    amount.lo = 127;
    riscv_ame_exec_int_binary(
        AME_OP_SRA, u128, s128, &minus_one, u128, &amount, &result);
    g_assert_cmphex(result.value.lo, ==, UINT64_MAX);
    g_assert_cmphex(result.value.hi, ==, UINT64_MAX);
    g_assert_false(result.saturated);
    riscv_ame_exec_int_binary(
        AME_OP_SRL, dtype(0x60000008), s128, &minus_one, u128, &amount,
        &result);
    assert_narrow_result(&result, dtype(0x60000008), 1);
    amount.lo = 0;
    riscv_ame_exec_int_binary(
        AME_OP_SRL, s128, u128, &minus_one, u128, &amount, &result);
    g_assert_cmphex(result.value.lo, ==, UINT64_MAX);
    g_assert_cmphex(result.value.hi, ==, UINT64_MAX);
    g_assert_false(result.saturated);
    riscv_ame_exec_int_binary(
        AME_OP_SRL, dtype(0x60000080), u128, &minus_one, u128, &amount,
        &result);
    g_assert_cmphex(result.value.lo, ==, UINT64_MAX);
    g_assert_cmphex(result.value.hi, ==, INT64_MAX);
    g_assert_true(result.saturated);
}

static void test_scaled_sum(void)
{
    static const unsigned magnitudes[] = { 0, 1, 3, 31, 127, 255 };
    static const int exponents[] = { -9, -1, 0, 1, 9 };

    for (unsigned id = 0; id < AME_INT_DTYPE_COUNT; id++) {
        const AmeDTypeDesc *d = &riscv_ame_dtypes[id];

        if (d->bits > 8) {
            continue;
        }
        for (unsigned a = 0; a < G_N_ELEMENTS(magnitudes); a++) {
            for (unsigned b = 0; b < G_N_ELEMENTS(magnitudes); b++) {
                for (unsigned signs = 0; signs < 4; signs++) {
                    AmeIntProduct left = {
                        .word = { magnitudes[a] }, .negative = signs & 1,
                    };
                    AmeIntProduct right = {
                        .word = { magnitudes[b] }, .negative = signs & 2,
                    };

                    for (unsigned ae = 0; ae < G_N_ELEMENTS(exponents); ae++) {
                        for (unsigned be = 0; be < G_N_ELEMENTS(exponents);
                             be++) {
                            int base = MIN(0, MIN(exponents[ae],
                                                  exponents[be]));
                            int64_t x = (int64_t)magnitudes[a] *
                                        (INT64_C(1) << (exponents[ae] - base));
                            int64_t y = (int64_t)magnitudes[b] *
                                        (INT64_C(1) << (exponents[be] - base));
                            int64_t exact = (left.negative ? -x : x) +
                                            (right.negative ? -y : y);
                            int64_t minimum = d->integer.is_signed ?
                                -(INT64_C(1) << (d->bits - 1)) : 0;
                            int64_t maximum = (INT64_C(1) <<
                                (d->bits - d->integer.is_signed)) - 1;

                            for (unsigned absolute = 0; absolute < 2;
                                 absolute++) {
                                int64_t value = absolute && exact < 0 ?
                                                -exact : exact;
                                int64_t expected = reference_round_shift(
                                    value, -base, d->integer.rounding);
                                bool invalid = expected < minimum ||
                                               expected > maximum;
                                bool inexact = value %
                                               (INT64_C(1) << -base) != 0;
                                AmeExecResult result;

                                riscv_ame_int_scaled_sum(
                                    &left, exponents[ae], &right,
                                    exponents[be], absolute, d, &result);
                                g_assert_cmphex(result.fflags, ==,
                                    invalid ? AME_FFLAG_NV :
                                    inexact ? AME_FFLAG_NX : 0);
                                result.fflags = 0;
                                assert_narrow_result(&result, d, expected);
                            }
                        }
                    }
                }
            }
        }
    }
}

static void test_scaled_sum_remote_terms(void)
{
    const AmeDTypeDesc *s128 = dtype(0x50000080);
    const AmeDTypeDesc *u128 = dtype(0x00000080);
    AmeIntProduct left = { .word = { 1 } };
    AmeIntProduct right = { .word = { 1 }, .negative = true };
    AmeExecResult result;

    /* 2^128 minus a remote fraction is Uint128_MAX under RDN. */
    riscv_ame_int_scaled_sum(&left, 128, &right, -1074, false,
                              dtype(0x10000080), &result);
    g_assert_cmphex(result.value.lo, ==, UINT64_MAX);
    g_assert_cmphex(result.value.hi, ==, UINT64_MAX);
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);
    g_assert_false(result.saturated);
    /* Nearest instead crosses the Uint128 range, then wraps. */
    riscv_ame_int_scaled_sum(&left, 128, &right, -1074, false, u128, &result);
    g_assert_cmphex(result.value.lo, ==, 0);
    g_assert_cmphex(result.value.hi, ==, 0);
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NV);

    /* High integral bits cannot destroy the low bits used by wrapping. */
    riscv_ame_int_scaled_sum(&left, 1023, &right, 0, false, u128, &result);
    g_assert_cmphex(result.value.lo, ==, UINT64_MAX);
    g_assert_cmphex(result.value.hi, ==, UINT64_MAX);
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NV);
    riscv_ame_int_scaled_sum(&left, 1023, &right, 0, false,
                              dtype(0x20000080), &result);
    g_assert_cmphex(result.value.lo, ==, UINT64_MAX);
    g_assert_cmphex(result.value.hi, ==, UINT64_MAX);
    g_assert_true(result.saturated);

    /* Exact cancellation is performed before applying a huge exponent. */
    riscv_ame_int_scaled_sum(&left, INT_MAX, &right, INT_MAX, false,
                              s128, &result);
    g_assert_cmphex(result.value.lo, ==, 0);
    g_assert_cmphex(result.value.hi, ==, 0);
    g_assert_cmphex(result.fflags, ==, 0);
    /* | -0.5 + epsilon | is strictly below a tie. */
    riscv_ame_int_scaled_sum(&right, -1, &left, INT_MIN, true, u128, &result);
    g_assert_cmphex(result.value.lo, ==, 0);
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);
    /* A remote negative fraction rounds down to -1. */
    riscv_ame_int_scaled_sum(&right, INT_MIN, NULL, 0, false, s128, &result);
    g_assert_cmphex(result.value.lo, ==, UINT64_MAX);
    g_assert_cmphex(result.value.hi, ==, UINT64_MAX);
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NX);
}

static void test_narrowing_arithmetic(void)
{
    static const AmeOperation ops[] = {
        AME_OP_ADD, AME_OP_SUB, AME_OP_ABSDIFF,
        AME_OP_MUL, AME_OP_NMUL, AME_OP_HDIFF, AME_OP_MEAN,
    };

    /* All Int/Uint8 inputs, all Int/Uint4 saturation and rounding modes. */
    for (unsigned signs = 0; signs < 4; signs++) {
        const AmeDTypeDesc *a = dtype(8 | ((signs & 1) << 30));
        const AmeDTypeDesc *b = dtype(8 | ((signs >> 1) << 30));

        for (unsigned ar = 0; ar <= UINT8_MAX; ar++) {
            int64_t av = a->integer.is_signed ? (int8_t)ar : (int64_t)ar;
            AmeValue left = { .lo = ar };

            for (unsigned br = 0; br <= UINT8_MAX; br++) {
                int64_t bv = b->integer.is_signed ? (int8_t)br : (int64_t)br;
                AmeValue right = { .lo = br };
                int64_t values[] = {
                    av + bv, bv - av, llabs(av - bv),
                    av * bv, -av * bv, bv - av, av + bv,
                };

                for (unsigned variant = 0; variant < AME_INT_VARIANTS_PER_WIDTH;
                     variant++) {
                    const AmeDTypeDesc *d = dtype(4 | (variant << 27));

                    for (unsigned op = 0; op < ARRAY_SIZE(ops); op++) {
                        AmeExecResult result;
                        int64_t expected = values[op];
                        bool half = ops[op] == AME_OP_HDIFF ||
                                    ops[op] == AME_OP_MEAN;

                        /* Other arithmetic does not use the rounding field. */
                        if (!half && d->integer.rounding != AME_INT_RNU) {
                            continue;
                        }
                        if (half) {
                            expected = reference_round_shift(
                                expected, 1, d->integer.rounding);
                        }
                        riscv_ame_exec_int_binary(
                            ops[op], d, a, &left, b, &right, &result);
                        assert_narrow_result(&result, d, expected);
                    }
                }
            }
        }
    }
}

static void test_narrowing_boundaries(void)
{
    const AmeDTypeDesc *uint128 = dtype(0x00000080);
    const AmeDTypeDesc *int128 = dtype(0x40000080);
    const AmeDTypeDesc *int4_sat = dtype(0x60000004);
    const AmeDTypeDesc *uint4_sat = dtype(0x20000004);
    AmeValue maximum = { .lo = UINT64_MAX, .hi = UINT64_MAX };
    AmeValue minimum = { .hi = BIT_ULL(63) };
    AmeValue one = { .lo = 1 };
    AmeExecResult result;

    for (unsigned bits = 4; bits < AME_VALUE_BITS; bits *= 2) {
        const AmeDTypeDesc *d = dtype(0x60000000 | bits);
        const AmeDTypeDesc *a = dtype(2 * bits);
        const AmeDTypeDesc *b = dtype(0x40000000 | (2 * bits));
        AmeValue power = {
            .lo = bits < 64 ? BIT_ULL(bits) : 0,
            .hi = bits == 64 ? 1 : 0,
        };
        AmeValue complement = {
            .lo = 3 - power.lo,
            .hi = -power.hi - (3 < power.lo),
        };
        AmeValue half = { .lo = BIT_ULL(bits - 1) };
        AmeValue negative_half = { .lo = -half.lo, .hi = UINT64_MAX };
        AmeValue exponent = { .lo = bits };

        /* Wide terms cancel before conversion, without accruing SAT. */
        riscv_ame_exec_int_binary(
            AME_OP_ADD, d, a, &power, b, &complement, &result);
        g_assert_cmphex(result.value.lo, ==, 3);
        g_assert_false(result.saturated);
        riscv_ame_exec_int_binary(
            AME_OP_MEAN, d, a, &power, b, &complement, &result);
        g_assert_cmphex(result.value.lo, ==, 2);
        g_assert_false(result.saturated);

        /* A product above D's maximum cancels with the old signed D. */
        riscv_ame_exec_int_ternary(
            AME_OP_MULACC, d, a, &half, b, &one, &half, &result);
        g_assert_cmphex(result.value.lo, ==, 0);
        g_assert_false(result.saturated);
        riscv_ame_exec_int_ternary(
            AME_OP_MULNACC, d, b, &negative_half, a, &one, &half, &result);
        g_assert_cmphex(result.value.lo, ==, 0);
        g_assert_false(result.saturated);
        riscv_ame_exec_int_ternary(
            AME_OP_MULADD, d, a, &half, b, &one, &half, &result);
        g_assert_cmphex(result.value.lo, ==, 0);
        g_assert_false(result.saturated);
        riscv_ame_exec_int_ternary(
            AME_OP_MULSUB, d, b, &negative_half, a, &one, &half, &result);
        g_assert_cmphex(result.value.lo, ==, 0);
        g_assert_false(result.saturated);

        /* Right scaling must use the entire source, including its high word. */
        riscv_ame_exec_int_binary(
            AME_OP_RDEXP, d, a, &power, uint128, &exponent, &result);
        g_assert_cmphex(result.value.lo, ==, 1);
        g_assert_cmphex(result.value.hi, ==, 0);
        g_assert_false(result.saturated);
    }

    riscv_ame_exec_int_binary(
        AME_OP_MUL, dtype(4), uint128, &maximum,
        uint128, &maximum, &result);
    assert_narrow_result(&result, dtype(4), 1);
    riscv_ame_exec_int_binary(
        AME_OP_MUL, uint4_sat, uint128, &maximum,
        uint128, &maximum, &result);
    g_assert_cmphex(result.value.lo, ==, 15);
    g_assert_cmphex(result.value.hi, ==, 0);
    g_assert_true(result.saturated);
    riscv_ame_exec_int_binary(
        AME_OP_MUL, int4_sat, int128, &minimum,
        uint128, &one, &result);
    g_assert_cmphex(result.value.lo, ==, 8);
    g_assert_cmphex(result.value.hi, ==, 0);
    g_assert_true(result.saturated);
    riscv_ame_exec_int_unary(
        AME_OP_ABS, uint4_sat, int128, &minimum, &result);
    g_assert_cmphex(result.value.lo, ==, 15);
    g_assert_cmphex(result.value.hi, ==, 0);
    g_assert_true(result.saturated);
}

static void test_narrowing_scale(void)
{
    static const int exponents[] = { -12, -8, -4, -1, 0, 1, 4, 8, 12 };
    static const unsigned currents[] = { 0, 7, 8, 15 };
    static const AmeOperation ops[] = {
        AME_OP_LDEXP, AME_OP_RDEXP,
        AME_OP_LDEXPACC, AME_OP_RDEXPACC,
    };
    const AmeDTypeDesc *int8 = dtype(0x40000008);

    for (unsigned variant = 0; variant < AME_INT_VARIANTS_PER_WIDTH;
         variant++) {
        const AmeDTypeDesc *d = dtype(4 | (variant << 27));

        for (unsigned sign = 0; sign < 2; sign++) {
            const AmeDTypeDesc *a = dtype(8 | (sign << 30));

            for (unsigned raw = 0; raw <= UINT8_MAX; raw++) {
                AmeValue data = { .lo = raw };
                int64_t value = sign ? (int8_t)raw : (int64_t)raw;

                for (unsigned e = 0; e < ARRAY_SIZE(exponents); e++) {
                    AmeValue exponent = { .lo = (uint8_t)exponents[e] };

                    for (unsigned op = 0; op < ARRAY_SIZE(ops); op++) {
                        bool accumulate = ops[op] == AME_OP_LDEXPACC ||
                                          ops[op] == AME_OP_RDEXPACC;
                        bool reverse = ops[op] == AME_OP_RDEXP ||
                                       ops[op] == AME_OP_RDEXPACC;
                        int shift = reverse ? -exponents[e] : exponents[e];

                        for (unsigned c = 0; c < ARRAY_SIZE(currents); c++) {
                            AmeValue current = { .lo = currents[c] };
                            AmeExecResult result;
                            int64_t addend = accumulate ?
                                test_int4_value(currents[c],
                                                d->integer.is_signed) : 0;
                            int64_t expected;

                            if (!accumulate && c != 0) {
                                continue;
                            }
                            if (shift >= 0) {
                                expected = value * (INT64_C(1) << shift) +
                                           addend;
                            } else {
                                expected = reference_round_shift(
                                    value + addend * (INT64_C(1) << -shift),
                                    -shift, d->integer.rounding);
                            }
                            if (accumulate) {
                                riscv_ame_exec_int_ternary(
                                    ops[op], d, a, &data, int8, &exponent,
                                    &current, &result);
                            } else {
                                riscv_ame_exec_int_binary(
                                    ops[op], d, a, &data, int8, &exponent,
                                    &result);
                            }
                            assert_narrow_result(&result, d, expected);
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);

    g_test_add_func("/riscv/ame/int/primitive-types", test_primitive_types);
    g_test_add_func("/riscv/ame/int/mixed-boundaries", test_mixed_boundaries);
    g_test_add_func("/riscv/ame/int/narrowing-arithmetic",
                    test_narrowing_arithmetic);
    g_test_add_func("/riscv/ame/int/narrowing-boundaries",
                    test_narrowing_boundaries);
    g_test_add_func("/riscv/ame/int/narrowing-scale", test_narrowing_scale);
    g_test_add_func("/riscv/ame/int/mixed-shifts", test_mixed_shifts);
    g_test_add_func("/riscv/ame/int/mixed-shift-boundaries",
                    test_mixed_shift_boundaries);
    g_test_add_func("/riscv/ame/int/scaled-sum", test_scaled_sum);
    g_test_add_func("/riscv/ame/int/scaled-sum-remote",
                    test_scaled_sum_remote_terms);
    g_test_add_func("/riscv/ame/int/value-extension", test_value_extension);
    g_test_add_func("/riscv/ame/int/accumulator-carry",
                    test_accumulator_carry_and_finish);
    g_test_add_func("/riscv/ame/int/accumulator-finish-boundaries",
                    test_accumulator_finish_boundaries);
    g_test_add_func("/riscv/ame/int/full-width-product",
                    test_full_width_product);
    g_test_add_func("/riscv/ame/int/word-product-boundaries",
                    test_word_product_boundaries);
    g_test_add_func("/riscv/ame/int/small-product-exhaustive",
                    test_small_product_exhaustive);
    g_test_add_func("/riscv/ame/int/product-randomized",
                    test_product_randomized);
    g_test_add_func("/riscv/ame/int/maximum-dot-product",
                    test_maximum_dot_product);
    g_test_add_func("/riscv/ame/int/rounding", test_rounding);
    g_test_add_func("/riscv/ame/int/rounding-exhaustive-small",
                    test_rounding_exhaustive_small);
    g_test_add_func("/riscv/ame/int/saturation", test_small_saturation);
    g_test_add_func("/riscv/ame/int/conversion-exhaustive-small",
                    test_conversion_exhaustive_small);
    g_test_add_func("/riscv/ame/int/conversion-int128-boundaries",
                    test_conversion_int128_boundaries);

    return g_test_run();
}
