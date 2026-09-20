/*
 * RISC-V Attached Matrix Extension (AME) integer primitives
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/bitops.h"
#include "qemu/host-utils.h"
#include "target/riscv/ame/int.h"

#define AME_INT_ACC_BITS (AME_INT_ACC_WORDS * AME_VALUE_WORD_BITS)

QEMU_BUILD_BUG_ON(AME_INT_MAX_DOT_TERMS >
                  (UINT64_C(1) << (AME_VALUE_WORD_BITS - 1)));

bool riscv_ame_int_supports(AmeOperation op, const AmeDTypeDesc *dest,
                            const AmeDTypeDesc *a, const AmeDTypeDesc *b)
{
    if (!riscv_ame_dtype_is_integer(dest) ||
        !riscv_ame_dtype_is_integer(a)) {
        return false;
    }
    switch (op) {
    case AME_OP_CONV:
        return b == NULL;
    case AME_OP_ABS:
        return b == NULL;
    default:
        break;
    }
    if (!riscv_ame_dtype_is_integer(b)) {
        return false;
    }
    switch (op) {
    case AME_OP_AND:
    case AME_OP_ANDNOT:
    case AME_OP_OR:
    case AME_OP_ORNOT:
    case AME_OP_XOR:
    case AME_OP_MIN:
    case AME_OP_MAX:
        return dest == a && dest == b;
    case AME_OP_CMPGE:
    case AME_OP_CMPLT:
        return true;
    case AME_OP_SLL:
    case AME_OP_SRL:
        return true;
    case AME_OP_SRA:
        return a->integer.is_signed;
    case AME_OP_LDEXP:
    case AME_OP_RDEXP:
    case AME_OP_LDEXPACC:
    case AME_OP_RDEXPACC:
        return true;
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
        return true;
    default:
        return false;
    }
}

void riscv_ame_exec_int_convert(const AmeDTypeDesc *dest_dtype,
                                const AmeDTypeDesc *source_dtype,
                                const AmeValue *source,
                                AmeExecResult *result)
{
    AmeIntAccumulator acc;

    riscv_ame_int_acc_init(&acc, source_dtype, source);
    riscv_ame_int_acc_finish(&acc, dest_dtype, result);
}

static void ame_value_truncate(AmeValue *value, unsigned int bits)
{
    g_assert(bits >= AME_DTYPE_BITS_MIN && bits <= AME_VALUE_BITS);

    if (bits < AME_VALUE_WORD_BITS) {
        value->lo &= MAKE_64BIT_MASK(0, bits);
        value->hi = 0;
    } else if (bits == AME_VALUE_WORD_BITS) {
        value->hi = 0;
    } else if (bits < AME_VALUE_BITS) {
        value->hi &= MAKE_64BIT_MASK(0, bits - AME_VALUE_WORD_BITS);
    }
}

static bool ame_value_sign(const AmeValue *value, unsigned int bits)
{
    g_assert(bits >= AME_DTYPE_BITS_MIN && bits <= AME_VALUE_BITS);

    if (bits <= AME_VALUE_WORD_BITS) {
        return extract64(value->lo, bits - 1, 1);
    }
    return extract64(value->hi, bits - AME_VALUE_WORD_BITS - 1, 1);
}

static AmeValue ame_value_sub(AmeValue left, AmeValue right)
{
    AmeValue result;
    uint64_t borrow;

    result.lo = left.lo - right.lo;
    borrow = left.lo < right.lo;
    result.hi = left.hi - right.hi - borrow;
    return result;
}

static AmeValue ame_value_negate(AmeValue value)
{
    return ame_value_sub((AmeValue) { 0 }, value);
}

static AmeValue ame_value_rshift(AmeValue value, unsigned int amount,
                                 bool arithmetic)
{
    uint64_t fill;

    g_assert(amount < AME_VALUE_BITS);
    fill = arithmetic &&
           (value.hi >> (AME_VALUE_WORD_BITS - 1)) ? UINT64_MAX : 0;
    if (amount == 0) {
        return value;
    }
    if (amount < AME_VALUE_WORD_BITS) {
        return (AmeValue) {
            .lo = value.lo >> amount |
                  value.hi << (AME_VALUE_WORD_BITS - amount),
            .hi = value.hi >> amount |
                  fill << (AME_VALUE_WORD_BITS - amount),
        };
    }
    if (amount == AME_VALUE_WORD_BITS) {
        return (AmeValue) { .lo = value.hi, .hi = fill };
    }

    amount -= AME_VALUE_WORD_BITS;
    return (AmeValue) {
        .lo = value.hi >> amount |
              fill << (AME_VALUE_WORD_BITS - amount),
        .hi = fill,
    };
}

static AmeValue ame_value_extend(const AmeValue *value, unsigned int bits,
                                 bool is_signed)
{
    AmeValue result = *value;

    ame_value_truncate(&result, bits);
    if (is_signed && ame_value_sign(&result, bits)) {
        if (bits < AME_VALUE_WORD_BITS) {
            result.lo |= ~MAKE_64BIT_MASK(0, bits);
            result.hi = UINT64_MAX;
        } else if (bits == AME_VALUE_WORD_BITS) {
            result.hi = UINT64_MAX;
        }
    }
    return result;
}

static AmeValue ame_value_unsigned_max(unsigned int bits)
{
    AmeValue value = { .lo = UINT64_MAX, .hi = UINT64_MAX };

    ame_value_truncate(&value, bits);
    return value;
}

static AmeValue ame_value_signed_limit(unsigned int bits, bool negative)
{
    AmeValue value = { 0 };

    if (bits <= AME_VALUE_WORD_BITS) {
        value.lo = UINT64_C(1) << (bits - 1);
        if (!negative) {
            value.lo--;
        }
    } else {
        value.hi = UINT64_C(1) <<
                   (bits - AME_VALUE_WORD_BITS - 1);
        if (!negative) {
            value.hi--;
            value.lo = UINT64_MAX;
        }
    }
    return value;
}

static bool ame_words_add_word(uint64_t *words, size_t num_words,
                               size_t index, uint64_t value)
{
    while (value != 0 && index < num_words) {
        uint64_t old = words[index];

        words[index] += value;
        value = words[index] < old;
        index++;
    }
    return value != 0;
}

static void ame_words_negate(uint64_t *words, size_t num_words)
{
    uint64_t carry = 1;
    size_t i;

    for (i = 0; i < num_words; i++) {
        uint64_t inverted = ~words[i];

        words[i] = inverted + carry;
        carry &= words[i] == 0;
    }
}

static bool ame_words_are_zero(const uint64_t *words, size_t num_words)
{
    size_t i;

    for (i = 0; i < num_words; i++) {
        if (words[i] != 0) {
            return false;
        }
    }
    return true;
}

static void ame_unsigned_product(AmeIntProduct *product,
                                 AmeValue left, AmeValue right)
{
    uint64_t left_words[AME_VALUE_WORDS] = { left.lo, left.hi };
    uint64_t right_words[AME_VALUE_WORDS] = { right.lo, right.hi };
    size_t i, j;

    *product = (AmeIntProduct) { 0 };
    if ((left.hi | right.hi) == 0) {
        mulu64(&product->word[0], &product->word[1], left.lo, right.lo);
        return;
    }
    for (i = 0; i < ARRAY_SIZE(left_words); i++) {
        for (j = 0; j < ARRAY_SIZE(right_words); j++) {
            uint64_t low, high;

            mulu64(&low, &high, left_words[i], right_words[j]);
            g_assert(!ame_words_add_word(product->word,
                                         AME_INT_PRODUCT_WORDS,
                                         i + j, low));
            g_assert(!ame_words_add_word(product->word,
                                         AME_INT_PRODUCT_WORDS,
                                         i + j + 1, high));
        }
    }
}

void riscv_ame_int_product_init(AmeIntProduct *product,
                                const AmeDTypeDesc *left_dtype,
                                const AmeValue *left,
                                const AmeDTypeDesc *right_dtype,
                                const AmeValue *right, bool negate)
{
    AmeValue left_magnitude, right_magnitude;
    bool left_negative, right_negative;

    g_assert(product != NULL && left != NULL && right != NULL &&
             riscv_ame_dtype_is_integer(left_dtype) &&
             riscv_ame_dtype_is_integer(right_dtype));

    left_magnitude = ame_value_extend(left, left_dtype->bits,
                                      left_dtype->integer.is_signed);
    right_magnitude = ame_value_extend(right, right_dtype->bits,
                                       right_dtype->integer.is_signed);
    left_negative = left_dtype->integer.is_signed &&
                    left_magnitude.hi >> (AME_VALUE_WORD_BITS - 1);
    right_negative = right_dtype->integer.is_signed &&
                     right_magnitude.hi >> (AME_VALUE_WORD_BITS - 1);
    if (left_negative) {
        left_magnitude = ame_value_negate(left_magnitude);
    }
    if (right_negative) {
        right_magnitude = ame_value_negate(right_magnitude);
    }

    ame_unsigned_product(product, left_magnitude, right_magnitude);
    product->negative = !ame_words_are_zero(product->word,
                                             AME_INT_PRODUCT_WORDS) &&
                        ((left_negative != right_negative) != negate);
}

static bool ame_int_acc_is_negative(const AmeIntAccumulator *acc)
{
    return acc->word[AME_INT_ACC_WORDS - 1] >>
           (AME_VALUE_WORD_BITS - 1);
}

static void ame_int_acc_add_acc(AmeIntAccumulator *acc,
                                const AmeIntAccumulator *addend)
{
    bool left_negative = ame_int_acc_is_negative(acc);
    bool right_negative = ame_int_acc_is_negative(addend);
    uint64_t carry = 0;
    size_t i;

    for (i = 0; i < AME_INT_ACC_WORDS; i++) {
        uint64_t sum = acc->word[i] + addend->word[i];
        uint64_t first_carry = sum < acc->word[i];
        uint64_t result = sum + carry;

        carry = first_carry || result < sum;
        acc->word[i] = result;
    }

    /* Architectural bounds keep every supported accumulation in five limbs. */
    g_assert(left_negative != right_negative ||
             ame_int_acc_is_negative(acc) == left_negative);
}

static void ame_int_acc_sub_acc(AmeIntAccumulator *acc,
                                const AmeIntAccumulator *subtrahend)
{
    AmeIntAccumulator negated = *subtrahend;

    ame_words_negate(negated.word, AME_INT_ACC_WORDS);
    ame_int_acc_add_acc(acc, &negated);
}

static int ame_int_acc_unsigned_compare(const AmeIntAccumulator *left,
                                        const AmeIntAccumulator *right)
{
    size_t i = AME_INT_ACC_WORDS;

    while (i-- > 0) {
        if (left->word[i] != right->word[i]) {
            return left->word[i] < right->word[i] ? -1 : 1;
        }
    }
    return 0;
}

static int ame_int_acc_signed_compare(const AmeIntAccumulator *left,
                                      const AmeIntAccumulator *right)
{
    bool left_negative = ame_int_acc_is_negative(left);
    bool right_negative = ame_int_acc_is_negative(right);

    if (left_negative != right_negative) {
        return left_negative ? -1 : 1;
    }
    return ame_int_acc_unsigned_compare(left, right);
}

void riscv_ame_int_acc_init(AmeIntAccumulator *acc,
                            const AmeDTypeDesc *dtype,
                            const AmeValue *value)
{
    AmeValue extended;
    uint64_t fill;
    size_t i;

    g_assert(acc != NULL && value != NULL && riscv_ame_dtype_is_integer(dtype));
    extended = ame_value_extend(value, dtype->bits, dtype->integer.is_signed);
    fill = dtype->integer.is_signed &&
           extended.hi >> (AME_VALUE_WORD_BITS - 1) ? UINT64_MAX : 0;
    acc->word[0] = extended.lo;
    acc->word[1] = extended.hi;
    for (i = 2; i < AME_INT_ACC_WORDS; i++) {
        acc->word[i] = fill;
    }
}

void riscv_ame_int_acc_add(AmeIntAccumulator *acc,
                           const AmeDTypeDesc *dtype,
                           const AmeValue *value)
{
    AmeIntAccumulator addend;

    g_assert(acc != NULL);
    riscv_ame_int_acc_init(&addend, dtype, value);
    ame_int_acc_add_acc(acc, &addend);
}

void riscv_ame_int_acc_add_product(AmeIntAccumulator *acc,
                                   const AmeIntProduct *product)
{
    AmeIntAccumulator addend = { 0 };

    g_assert(acc != NULL && product != NULL);
    memcpy(addend.word, product->word, sizeof(product->word));
    if (product->negative) {
        ame_words_negate(addend.word, AME_INT_ACC_WORDS);
    }
    ame_int_acc_add_acc(acc, &addend);
}

static void ame_int_acc_lshift(AmeIntAccumulator *acc,
                               unsigned int amount)
{
    AmeIntAccumulator original = *acc;
    unsigned int word_shift = amount / AME_VALUE_WORD_BITS;
    unsigned int bit_shift = amount % AME_VALUE_WORD_BITS;
    size_t i = AME_INT_ACC_WORDS;

    g_assert(amount < AME_INT_ACC_BITS);
    while (i-- > 0) {
        uint64_t value = 0;

        if (i >= word_shift) {
            size_t source = i - word_shift;

            value = original.word[source] << bit_shift;
            if (bit_shift != 0 && source != 0) {
                value |= original.word[source - 1] >>
                         (AME_VALUE_WORD_BITS - bit_shift);
            }
        }
        acc->word[i] = value;
    }
}

static bool ame_int_acc_bit(const AmeIntAccumulator *acc,
                            unsigned int bit)
{
    if (bit >= AME_INT_ACC_BITS) {
        return ame_int_acc_is_negative(acc);
    }
    return extract64(acc->word[bit / AME_VALUE_WORD_BITS],
                     bit % AME_VALUE_WORD_BITS, 1);
}

static bool ame_int_acc_low_bits_nonzero(const AmeIntAccumulator *acc,
                                         unsigned int bits)
{
    unsigned int complete_words, remaining_bits;
    size_t i;

    if (bits == 0) {
        return false;
    }
    if (bits > AME_INT_ACC_BITS && ame_int_acc_is_negative(acc)) {
        return true;
    }
    bits = MIN(bits, (unsigned int)AME_INT_ACC_BITS);
    complete_words = bits / AME_VALUE_WORD_BITS;
    remaining_bits = bits % AME_VALUE_WORD_BITS;
    for (i = 0; i < complete_words; i++) {
        if (acc->word[i] != 0) {
            return true;
        }
    }
    return remaining_bits != 0 &&
           (acc->word[complete_words] &
            MAKE_64BIT_MASK(0, remaining_bits)) != 0;
}

static void ame_int_acc_arithmetic_rshift(AmeIntAccumulator *acc,
                                           unsigned int amount)
{
    AmeIntAccumulator original = *acc;
    uint64_t fill = ame_int_acc_is_negative(acc) ? UINT64_MAX : 0;
    unsigned int word_shift, bit_shift;
    size_t i;

    if (amount >= AME_INT_ACC_BITS) {
        for (i = 0; i < AME_INT_ACC_WORDS; i++) {
            acc->word[i] = fill;
        }
        return;
    }

    word_shift = amount / AME_VALUE_WORD_BITS;
    bit_shift = amount % AME_VALUE_WORD_BITS;
    for (i = 0; i < AME_INT_ACC_WORDS; i++) {
        size_t source = i + word_shift;
        uint64_t low = source < AME_INT_ACC_WORDS ?
                       original.word[source] : fill;

        if (bit_shift == 0) {
            acc->word[i] = low;
        } else {
            uint64_t high = source + 1 < AME_INT_ACC_WORDS ?
                            original.word[source + 1] : fill;

            acc->word[i] = low >> bit_shift |
                           high << (AME_VALUE_WORD_BITS - bit_shift);
        }
    }
}

static void ame_int_acc_round(AmeIntAccumulator *acc,
                              const AmeDTypeDesc *dest_dtype,
                              bool round_bit, bool sticky)
{
    bool discarded = round_bit || sticky;
    bool increment;

    g_assert(acc != NULL && riscv_ame_dtype_is_integer(dest_dtype) &&
             dest_dtype->integer.rounding < AME_INT_ROUND_MAX);

    switch (dest_dtype->integer.rounding) {
    case AME_INT_RNU:
        increment = round_bit;
        break;
    case AME_INT_RNE:
        increment = round_bit && (sticky || (acc->word[0] & 1));
        break;
    case AME_INT_RDN:
        increment = false;
        break;
    case AME_INT_ROD:
        acc->word[0] |= discarded;
        return;
    case AME_INT_ROUND_MAX:
    default:
        g_assert_not_reached();
    }
    if (increment) {
        /* Adding one to -1 intentionally discards the unsigned carry. */
        ame_words_add_word(acc->word, AME_INT_ACC_WORDS, 0, 1);
    }
}

void riscv_ame_int_acc_round_shift(AmeIntAccumulator *acc,
                                   const AmeDTypeDesc *dest_dtype,
                                   unsigned int amount)
{
    bool round_bit, sticky;

    g_assert(acc != NULL && riscv_ame_dtype_is_integer(dest_dtype));
    if (amount == 0) {
        return;
    }

    round_bit = ame_int_acc_bit(acc, amount - 1);
    sticky = ame_int_acc_low_bits_nonzero(acc, amount - 1);
    ame_int_acc_arithmetic_rshift(acc, amount);
    ame_int_acc_round(acc, dest_dtype, round_bit, sticky);
}

void riscv_ame_int_acc_finish(const AmeIntAccumulator *acc,
                              const AmeDTypeDesc *dest_dtype,
                              AmeExecResult *result)
{
    bool negative;
    uint64_t fill, excess;
    unsigned int bits, word;

    g_assert(acc != NULL && result != NULL &&
             riscv_ame_dtype_is_integer(dest_dtype));
    *result = (AmeExecResult) {
        .value = { .lo = acc->word[0], .hi = acc->word[1] },
    };
    if (!dest_dtype->integer.saturating) {
        ame_value_truncate(&result->value, dest_dtype->bits);
        return;
    }

    negative = ame_int_acc_is_negative(acc);
    fill = dest_dtype->integer.is_signed && negative ? UINT64_MAX : 0;
    bits = dest_dtype->bits - dest_dtype->integer.is_signed;
    word = bits / AME_VALUE_WORD_BITS;
    excess = (acc->word[word] ^ fill) >> (bits % AME_VALUE_WORD_BITS);

    /* Above the value bits, a fitting result has only zero/sign extension. */
    for (unsigned int i = word + 1; i < AME_INT_ACC_WORDS; i++) {
        excess |= acc->word[i] ^ fill;
    }
    if (excess) {
        result->value = dest_dtype->integer.is_signed ?
            ame_value_signed_limit(dest_dtype->bits, negative) :
            (negative ? (AmeValue) { 0 } :
                        ame_value_unsigned_max(dest_dtype->bits));
        result->saturated = true;
    } else {
        ame_value_truncate(&result->value, dest_dtype->bits);
    }
}

static int ame_int_product_msb(const AmeIntProduct *value)
{
    for (int i = AME_INT_PRODUCT_WORDS - 1; i >= 0; i--) {
        if (value->word[i]) {
            return i * AME_VALUE_WORD_BITS +
                   AME_VALUE_WORD_BITS - 1 - clz64(value->word[i]);
        }
    }
    return -1;
}

static int ame_int_acc_magnitude_msb(const AmeIntAccumulator *value)
{
    AmeIntAccumulator magnitude = *value;

    if (ame_int_acc_is_negative(value)) {
        ame_words_negate(magnitude.word, AME_INT_ACC_WORDS);
    }
    for (int i = AME_INT_ACC_WORDS - 1; i >= 0; i--) {
        if (magnitude.word[i]) {
            return i * AME_VALUE_WORD_BITS +
                   AME_VALUE_WORD_BITS - 1 - clz64(magnitude.word[i]);
        }
    }
    return -1;
}

void riscv_ame_int_scaled_sum(const AmeIntProduct *left, int left_exp,
                              const AmeIntProduct *right, int right_exp,
                              bool absolute, const AmeDTypeDesc *dest_dtype,
                              AmeExecResult *result)
{
    const AmeIntProduct *terms[] = { left, right };
    const int exponents[] = { left_exp, right_exp };
    const int acc_bits = AME_INT_ACC_BITS;
    AmeIntAccumulator sum = { 0 };
    AmeDTypeDesc clamped = *dest_dtype;
    int64_t lowest = INT64_MAX, highest = INT64_MIN, base;
    bool negative = false, lost_fraction = false, overflow, inexact = false;
    unsigned dominant = 0;
    int msb;

    for (unsigned i = 0; i < G_N_ELEMENTS(terms); i++) {
        if (!terms[i]) {
            continue;
        }
        msb = ame_int_product_msb(terms[i]);
        if (msb < 0) {
            continue;
        }
        lowest = MIN(lowest, exponents[i]);
        if ((int64_t)exponents[i] + msb > highest) {
            highest = (int64_t)exponents[i] + msb;
            dominant = i;
        }
    }
    if (lowest == INT64_MAX) {
        riscv_ame_int_acc_finish(&sum, dest_dtype, result);
        return;
    }

    /*
     * Two 256-bit terms fit exactly when their combined span fits 318 bits.
     * Otherwise they cannot nearly cancel: the larger term determines sign.
     * Keep that term and jam the remote fraction. For very large integers,
     * retain low bits for wrapping and track the true overflow sign separately.
     */
    base = highest - lowest < acc_bits - 2 ? lowest :
           MIN(-2, highest - (acc_bits - 3));
    overflow = highest - base >= acc_bits - 2;
    for (unsigned i = 0; i < G_N_ELEMENTS(terms); i++) {
        AmeIntAccumulator term = { 0 };
        int64_t shift = (int64_t)exponents[i] - base;

        if (!terms[i]) {
            continue;
        }
        memcpy(term.word, terms[i]->word, sizeof(terms[i]->word));
        if (shift >= acc_bits) {
            memset(&term, 0, sizeof(term));
        } else if (shift >= 0) {
            ame_int_acc_lshift(&term, shift);
        }
        if (overflow) {
            term.word[AME_INT_ACC_WORDS - 1] &= UINT64_MAX >> 2;
        }
        if (terms[i]->negative) {
            ame_words_negate(term.word, AME_INT_ACC_WORDS);
        }
        if (shift < 0) {
            unsigned amount = MIN(-shift, acc_bits + 1);

            lost_fraction |= ame_int_acc_low_bits_nonzero(&term, amount);
            ame_int_acc_arithmetic_rshift(&term, amount);
        }
        ame_int_acc_add_acc(&sum, &term);
    }
    negative = overflow ? terms[dominant]->negative :
               ame_int_acc_is_negative(&sum);
    if (absolute && negative) {
        ame_words_negate(sum.word, AME_INT_ACC_WORDS);
        if (lost_fraction) {
            AmeIntAccumulator one = { .word = { 1 } };

            ame_int_acc_sub_acc(&sum, &one);
        }
        negative = false;
    }
    if (lost_fraction) {
        sum.word[0] |= 1;
    }
    if (base < 0) {
        unsigned amount = MIN(-base, acc_bits + 1);

        inexact = ame_int_acc_low_bits_nonzero(&sum, amount);
        riscv_ame_int_acc_round_shift(&sum, dest_dtype, amount);
    } else {
        msb = ame_int_acc_magnitude_msb(&sum);
        overflow = msb >= 0 && base + msb >= AME_VALUE_BITS;
        if (base >= AME_VALUE_BITS) {
            memset(&sum, 0, sizeof(sum));
        } else {
            ame_int_acc_lshift(&sum, base);
        }
    }

    clamped.integer.saturating = true;
    if (overflow) {
        AmeIntAccumulator sentinel = { 0 };
        AmeIntProduct large = { .negative = negative };

        large.word[AME_VALUE_WORDS] = 1;
        riscv_ame_int_acc_add_product(&sentinel, &large);
        riscv_ame_int_acc_finish(&sentinel, &clamped, result);
    } else {
        riscv_ame_int_acc_finish(&sum, &clamped, result);
        overflow = result->saturated;
    }
    if (!dest_dtype->integer.saturating) {
        riscv_ame_int_acc_finish(&sum, dest_dtype, result);
    }
    result->fflags = overflow ? AME_FFLAG_NV : inexact ? AME_FFLAG_NX : 0;
}

void riscv_ame_exec_int_unary(AmeOperation op,
                              const AmeDTypeDesc *dest_dtype,
                              const AmeDTypeDesc *source_dtype,
                              const AmeValue *source,
                              AmeExecResult *result)
{
    AmeIntAccumulator value;

    riscv_ame_int_acc_init(&value, source_dtype, source);
    if (ame_int_acc_is_negative(&value)) {
        ame_words_negate(value.word, AME_INT_ACC_WORDS);
    }
    riscv_ame_int_acc_finish(&value, dest_dtype, result);
}

static void ame_exec_int_arithmetic(AmeOperation op,
                                    const AmeDTypeDesc *dest_dtype,
                                    const AmeDTypeDesc *src1_dtype,
                                    const AmeValue *src1,
                                    const AmeDTypeDesc *src2_dtype,
                                    const AmeValue *src2,
                                    AmeExecResult *result)
{
    AmeIntAccumulator left, right, value;

    if (op == AME_OP_MUL || op == AME_OP_NMUL) {
        AmeIntProduct product;

        value = (AmeIntAccumulator) { 0 };
        riscv_ame_int_product_init(&product, src1_dtype, src1,
                                   src2_dtype, src2, op == AME_OP_NMUL);
        riscv_ame_int_acc_add_product(&value, &product);
        riscv_ame_int_acc_finish(&value, dest_dtype, result);
        return;
    }

    riscv_ame_int_acc_init(&left, src1_dtype, src1);
    riscv_ame_int_acc_init(&right, src2_dtype, src2);

    switch (op) {
    case AME_OP_ABSDIFF:
        if (ame_int_acc_signed_compare(&left, &right) < 0) {
            value = right;
            ame_int_acc_sub_acc(&value, &left);
        } else {
            value = left;
            ame_int_acc_sub_acc(&value, &right);
        }
        break;
    case AME_OP_ADD:
        value = left;
        ame_int_acc_add_acc(&value, &right);
        break;
    case AME_OP_SUB:
        value = right;
        ame_int_acc_sub_acc(&value, &left);
        break;
    case AME_OP_HDIFF:
        value = right;
        ame_int_acc_sub_acc(&value, &left);
        riscv_ame_int_acc_round_shift(&value, dest_dtype, 1);
        break;
    case AME_OP_MEAN:
        value = left;
        ame_int_acc_add_acc(&value, &right);
        riscv_ame_int_acc_round_shift(&value, dest_dtype, 1);
        break;
    case AME_OP_MAX:
        value = ame_int_acc_signed_compare(&left, &right) >= 0 ? left : right;
        break;
    case AME_OP_MIN:
        value = ame_int_acc_signed_compare(&left, &right) <= 0 ? left : right;
        break;
    default:
        g_assert_not_reached();
    }

    riscv_ame_int_acc_finish(&value, dest_dtype, result);
}

static void ame_exec_int_bitwise(AmeOperation op,
                                 const AmeDTypeDesc *dest_dtype,
                                 const AmeValue *src1,
                                 const AmeValue *src2,
                                 AmeExecResult *result)
{
    *result = (AmeExecResult) { 0 };
    switch (op) {
    case AME_OP_AND:
        result->value.lo = src1->lo & src2->lo;
        result->value.hi = src1->hi & src2->hi;
        break;
    case AME_OP_ANDNOT:
        result->value.lo = src1->lo & ~src2->lo;
        result->value.hi = src1->hi & ~src2->hi;
        break;
    case AME_OP_OR:
        result->value.lo = src1->lo | src2->lo;
        result->value.hi = src1->hi | src2->hi;
        break;
    case AME_OP_ORNOT:
        result->value.lo = src1->lo | ~src2->lo;
        result->value.hi = src1->hi | ~src2->hi;
        break;
    case AME_OP_XOR:
        result->value.lo = src1->lo ^ src2->lo;
        result->value.hi = src1->hi ^ src2->hi;
        break;
    default:
        g_assert_not_reached();
    }
    ame_value_truncate(&result->value, dest_dtype->bits);
}

static void ame_exec_int_compare(AmeOperation op,
                                 const AmeDTypeDesc *dest_dtype,
                                 const AmeDTypeDesc *src1_dtype,
                                 const AmeValue *src1,
                                 const AmeDTypeDesc *src2_dtype,
                                 const AmeValue *src2,
                                 AmeExecResult *result)
{
    AmeIntAccumulator left, right;
    int compare;
    bool predicate;

    riscv_ame_int_acc_init(&left, src1_dtype, src1);
    riscv_ame_int_acc_init(&right, src2_dtype, src2);
    compare = ame_int_acc_signed_compare(&left, &right);
    predicate = op == AME_OP_CMPGE ? compare >= 0 : compare < 0;
    *result = (AmeExecResult) {
        .value = predicate ? ame_value_unsigned_max(dest_dtype->bits) :
                             (AmeValue) { 0 },
    };
}

static void ame_exec_int_shift(AmeOperation op,
                               const AmeDTypeDesc *dest_dtype,
                               const AmeDTypeDesc *data_dtype,
                               const AmeValue *data,
                               const AmeValue *amount_value,
                               AmeExecResult *result)
{
    AmeIntAccumulator acc;
    AmeValue value;
    unsigned int amount;

    amount = amount_value->lo & (data_dtype->bits - 1);
    if (op == AME_OP_SLL) {
        riscv_ame_int_acc_init(&acc, data_dtype, data);
        ame_int_acc_lshift(&acc, amount);
    } else {
        AmeDTypeDesc shifted_dtype = *data_dtype;

        value = ame_value_extend(data, data_dtype->bits, op == AME_OP_SRA);
        value = ame_value_rshift(value, amount, op == AME_OP_SRA);
        shifted_dtype.bits = AME_VALUE_BITS;
        shifted_dtype.integer.is_signed = op == AME_OP_SRA;
        riscv_ame_int_acc_init(&acc, &shifted_dtype, &value);
    }
    riscv_ame_int_acc_finish(&acc, dest_dtype, result);
}

static unsigned int ame_int_scale_amount(
    const AmeDTypeDesc *exponent_dtype, const AmeValue *exponent,
    bool reverse, bool *right_shift)
{
    AmeValue value = ame_value_extend(exponent, exponent_dtype->bits,
                                      exponent_dtype->integer.is_signed);
    bool negative = exponent_dtype->integer.is_signed &&
                    ame_value_sign(exponent, exponent_dtype->bits);
    AmeValue magnitude = negative ? ame_value_negate(value) : value;
    bool nonzero = magnitude.lo != 0 || magnitude.hi != 0;

    if (reverse && nonzero) {
        negative = !negative;
    }
    *right_shift = negative;
    if (magnitude.hi != 0 || magnitude.lo > AME_INT_ACC_BITS + 1) {
        return AME_INT_ACC_BITS + 1;
    }
    return magnitude.lo;
}

static void ame_int_scale_large_left(const AmeDTypeDesc *dest_dtype,
                                     const AmeIntAccumulator *data,
                                     const AmeValue *current,
                                     AmeExecResult *result)
{
    bool data_is_zero = ame_words_are_zero(data->word,
                                            AME_INT_ACC_WORDS);
    bool negative = ame_int_acc_is_negative(data);

    if (data_is_zero || !dest_dtype->integer.saturating) {
        AmeIntAccumulator value = { 0 };

        if (current != NULL) {
            riscv_ame_int_acc_init(&value, dest_dtype, current);
        }
        riscv_ame_int_acc_finish(&value, dest_dtype, result);
        return;
    }

    *result = (AmeExecResult) {
        .value = dest_dtype->integer.is_signed ?
                 ame_value_signed_limit(dest_dtype->bits, negative) :
                 negative ? (AmeValue) { 0 } :
                            ame_value_unsigned_max(dest_dtype->bits),
        .saturated = true,
    };
}

static void ame_exec_int_scale(AmeOperation op,
                               const AmeDTypeDesc *dest_dtype,
                               const AmeDTypeDesc *data_dtype,
                               const AmeValue *data,
                               const AmeDTypeDesc *exponent_dtype,
                               const AmeValue *exponent,
                               const AmeValue *current,
                               AmeExecResult *result)
{
    bool accumulate = op == AME_OP_LDEXPACC ||
                      op == AME_OP_RDEXPACC;
    bool reverse = op == AME_OP_RDEXP || op == AME_OP_RDEXPACC;
    AmeIntAccumulator value;
    bool right_shift;
    unsigned int amount;

    riscv_ame_int_acc_init(&value, data_dtype, data);
    amount = ame_int_scale_amount(exponent_dtype, exponent, reverse,
                                  &right_shift);
    if (!right_shift) {
        if (amount >= dest_dtype->bits) {
            ame_int_scale_large_left(dest_dtype, &value,
                                     accumulate ? current : NULL, result);
            return;
        }
        ame_int_acc_lshift(&value, amount);
        if (accumulate) {
            riscv_ame_int_acc_add(&value, dest_dtype, current);
        }
        riscv_ame_int_acc_finish(&value, dest_dtype, result);
        return;
    }

    if (amount != 0) {
        bool round_bit = ame_int_acc_bit(&value, amount - 1);
        bool sticky = ame_int_acc_low_bits_nonzero(&value, amount - 1);

        ame_int_acc_arithmetic_rshift(&value, amount);
        if (accumulate) {
            riscv_ame_int_acc_add(&value, dest_dtype, current);
        }
        ame_int_acc_round(&value, dest_dtype, round_bit, sticky);
    } else if (accumulate) {
        riscv_ame_int_acc_add(&value, dest_dtype, current);
    }
    riscv_ame_int_acc_finish(&value, dest_dtype, result);
}

void riscv_ame_exec_int_ternary(AmeOperation op,
                                const AmeDTypeDesc *dest_dtype,
                                const AmeDTypeDesc *src1_dtype,
                                const AmeValue *src1,
                                const AmeDTypeDesc *src2_dtype,
                                const AmeValue *src2,
                                const AmeValue *current,
                                AmeExecResult *result)
{
    AmeIntAccumulator value;
    AmeIntProduct product;

    if (op == AME_OP_LDEXPACC || op == AME_OP_RDEXPACC) {
        ame_exec_int_scale(op, dest_dtype, src1_dtype, src1,
                           src2_dtype, src2, current, result);
        return;
    }

    switch (op) {
    case AME_OP_MULACC:
    case AME_OP_MULNACC:
        riscv_ame_int_acc_init(&value, dest_dtype, current);
        riscv_ame_int_product_init(&product, src1_dtype, src1,
                                   src2_dtype, src2,
                                   op == AME_OP_MULNACC);
        break;
    case AME_OP_MULADD:
    case AME_OP_MULSUB:
        riscv_ame_int_acc_init(&value, src1_dtype, src1);
        riscv_ame_int_product_init(&product, src2_dtype, src2,
                                   dest_dtype, current,
                                   op == AME_OP_MULSUB);
        break;
    default:
        g_assert_not_reached();
    }
    riscv_ame_int_acc_add_product(&value, &product);
    riscv_ame_int_acc_finish(&value, dest_dtype, result);
}

void riscv_ame_exec_int_binary(AmeOperation op,
                               const AmeDTypeDesc *dest_dtype,
                               const AmeDTypeDesc *src1_dtype,
                               const AmeValue *src1,
                               const AmeDTypeDesc *src2_dtype,
                               const AmeValue *src2,
                               AmeExecResult *result)
{
    switch (op) {
    case AME_OP_ABSDIFF:
    case AME_OP_ADD:
    case AME_OP_HDIFF:
    case AME_OP_MAX:
    case AME_OP_MEAN:
    case AME_OP_MIN:
    case AME_OP_MUL:
    case AME_OP_NMUL:
    case AME_OP_SUB:
        ame_exec_int_arithmetic(op, dest_dtype, src1_dtype, src1,
                                src2_dtype, src2, result);
        break;
    case AME_OP_AND:
    case AME_OP_ANDNOT:
    case AME_OP_OR:
    case AME_OP_ORNOT:
    case AME_OP_XOR:
        ame_exec_int_bitwise(op, dest_dtype, src1, src2, result);
        break;
    case AME_OP_CMPGE:
    case AME_OP_CMPLT:
        ame_exec_int_compare(op, dest_dtype, src1_dtype, src1,
                             src2_dtype, src2, result);
        break;
    case AME_OP_SLL:
    case AME_OP_SRA:
    case AME_OP_SRL:
        ame_exec_int_shift(op, dest_dtype, src1_dtype, src1, src2, result);
        break;
    case AME_OP_LDEXP:
    case AME_OP_RDEXP:
        ame_exec_int_scale(op, dest_dtype, src1_dtype, src1,
                           src2_dtype, src2, NULL, result);
        break;
    default:
        g_assert_not_reached();
    }
}
