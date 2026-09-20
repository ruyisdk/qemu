/*
 * RISC-V Attached Matrix Extension (AME) integer primitives
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef TARGET_RISCV_AME_INT_H
#define TARGET_RISCV_AME_INT_H

#include <stdbool.h>
#include <stdint.h>

#include "target/riscv/ame/ame.h"
#include "target/riscv/ame/datatype.h"

#define AME_INT_PRODUCT_WORDS (2 * AME_VALUE_WORDS)

/* Exact product of two integer elements, including mixed signedness. */
typedef struct AmeIntProduct {
    /* Little-endian unsigned magnitude; sign is kept separately. */
    uint64_t word[AME_INT_PRODUCT_WORDS];
    bool negative;
} AmeIntProduct;

/*
 * The largest dot product has N_max * (U_max / 4) = 2048 products.
 * Five limbs hold a 256-bit magnitude, eleven carry bits, and its sign.
 */
#define AME_INT_MAX_DOT_TERMS \
    (RV_AME_N_MAX * (RV_AME_UDSZ_MAX / AME_DTYPE_BITS_MIN))
#define AME_INT_ACC_WORDS (AME_INT_PRODUCT_WORDS + 1)

/* Exact intermediate integer sum within AME's architectural size bounds. */
typedef struct AmeIntAccumulator {
    /* Little-endian limbs containing a signed two's-complement value. */
    uint64_t word[AME_INT_ACC_WORDS];
} AmeIntAccumulator;

/**
 * riscv_ame_int_supports: Check an integer primitive's operation and datatypes
 * @op: Numerical operation
 * @dest: Destination datatype
 * @a: First source datatype
 * @b: Second source datatype; NULL for a unary operation or conversion
 *
 * Returns: true for an implemented primitive/type combination; otherwise false.
 */
bool riscv_ame_int_supports(AmeOperation op, const AmeDTypeDesc *dest,
                            const AmeDTypeDesc *a, const AmeDTypeDesc *b);

/**
 * riscv_ame_exec_int_convert: Convert one integer element
 * @dest_dtype: Integer destination datatype, including saturation behavior
 * @source_dtype: Integer source datatype, including signedness
 * @source: Raw source element bits
 * @result: Converted raw bits and saturation flag
 *
 * Extend according to source signedness, then clamp or wrap to the destination.
 * The caller must validate the integer pair and supply non-NULL values/output.
 */
void riscv_ame_exec_int_convert(const AmeDTypeDesc *dest_dtype,
                                const AmeDTypeDesc *source_dtype,
                                const AmeValue *source,
                                AmeExecResult *result);
/**
 * riscv_ame_exec_int_unary: Compute one unary integer result
 * @op: Unary operation
 * @dest_dtype: Destination datatype
 * @source_dtype: Source datatype
 * @source: Raw source element bits
 * @result: Output raw bits and saturation flag
 *
 * The caller must validate the operation and integer pair, and supply
 * non-NULL values/output.
 */
void riscv_ame_exec_int_unary(AmeOperation op,
                              const AmeDTypeDesc *dest_dtype,
                              const AmeDTypeDesc *source_dtype,
                              const AmeValue *source,
                              AmeExecResult *result);
/**
 * riscv_ame_exec_int_binary: Compute one result from two integer elements
 * @op: Binary operation, including comparisons, shifts and exponent scaling
 * @dest_dtype: Destination datatype and final rounding/saturation rules
 * @src1_dtype: First source datatype
 * @src1: First source's raw bits
 * @src2_dtype: Second source datatype
 * @src2: Second source's raw bits
 * @result: Output raw bits and saturation flag
 *
 * Interpret each source with its own signedness. Arithmetic keeps the full
 * intermediate value until the operation's destination conversion.
 *
 * The caller must validate the binary operation and integer tuple, and supply
 * non-NULL values/output.
 */
void riscv_ame_exec_int_binary(AmeOperation op,
                               const AmeDTypeDesc *dest_dtype,
                               const AmeDTypeDesc *src1_dtype,
                               const AmeValue *src1,
                               const AmeDTypeDesc *src2_dtype,
                               const AmeValue *src2,
                               AmeExecResult *result);
/**
 * riscv_ame_exec_int_ternary: Compute an integer result with old destination
 * @op: Multiply/add or exponent-scaling accumulation operation
 * @dest_dtype: Destination datatype and final rounding/saturation rules
 * @src1_dtype: First source datatype
 * @src1: First source's raw bits
 * @src2_dtype: Second source datatype
 * @src2: Second source's raw bits
 * @current: Old destination element, interpreted using @dest_dtype
 * @result: Output raw bits and saturation flag
 *
 * The caller must validate the ternary operation and integer tuple, and supply
 * non-NULL values/output.
 */
void riscv_ame_exec_int_ternary(AmeOperation op,
                                const AmeDTypeDesc *dest_dtype,
                                const AmeDTypeDesc *src1_dtype,
                                const AmeValue *src1,
                                const AmeDTypeDesc *src2_dtype,
                                const AmeValue *src2,
                                const AmeValue *current,
                                AmeExecResult *result);

/**
 * riscv_ame_int_product_init: Form an exact signed-magnitude integer product
 * @product: Output product, completely overwritten
 * @left_dtype: Left integer datatype
 * @left: Left element's raw bits
 * @right_dtype: Right integer datatype
 * @right: Right element's raw bits
 * @negate: Negate the mathematical product
 *
 * Each operand uses its own width and signedness. No rounding, saturation
 * or destination-width truncation is performed. Inputs must be valid integers.
 */
void riscv_ame_int_product_init(AmeIntProduct *product,
                                const AmeDTypeDesc *left_dtype,
                                const AmeValue *left,
                                const AmeDTypeDesc *right_dtype,
                                const AmeValue *right, bool negate);

/**
 * riscv_ame_int_acc_init: Extend one integer element into an exact accumulator
 * @acc: Accumulator to overwrite
 * @dtype: Valid integer datatype, supplying width and signedness
 * @value: Raw element bits
 *
 * Sign-extend signed inputs and zero-extend unsigned inputs.
 */
void riscv_ame_int_acc_init(AmeIntAccumulator *acc,
                            const AmeDTypeDesc *dtype,
                            const AmeValue *value);
/**
 * riscv_ame_int_acc_add: Add an integer element without narrowing
 * @acc: Initialized accumulator to update
 * @dtype: Valid integer datatype, supplying width and signedness
 * @value: Raw element bits
 *
 * The sum must fit the accumulator; AME's supported reduction bounds ensure it.
 */
void riscv_ame_int_acc_add(AmeIntAccumulator *acc,
                           const AmeDTypeDesc *dtype,
                           const AmeValue *value);
/**
 * riscv_ame_int_acc_add_product: Add an exact product without narrowing
 * @acc: Initialized accumulator to update
 * @product: Signed-magnitude product
 *
 * The sum must fit the accumulator, as ensured by supported dot-product bounds.
 */
void riscv_ame_int_acc_add_product(AmeIntAccumulator *acc,
                                   const AmeIntProduct *product);

/**
 * riscv_ame_int_acc_round_shift: Divide an intermediate by a power of two
 * @acc: Initialized accumulator to update
 * @dest_dtype: Valid integer datatype whose rounding mode is used
 * @amount: Right-shift count; zero leaves the value unchanged
 *
 * Round discarded bits without destination-width narrowing or saturation.
 */
void riscv_ame_int_acc_round_shift(AmeIntAccumulator *acc,
                                   const AmeDTypeDesc *dest_dtype,
                                   unsigned int amount);
/**
 * riscv_ame_int_acc_finish: Convert an exact intermediate to an integer result
 * @acc: Accumulator to read, left unchanged
 * @dest_dtype: Destination integer datatype
 * @result: Output raw bits and saturation flag; floating flags are zero
 *
 * Clamp to the destination range for saturating datatypes; otherwise keep the
 * low destination-width bits. This does not perform a rounded right shift.
 */
void riscv_ame_int_acc_finish(const AmeIntAccumulator *acc,
                              const AmeDTypeDesc *dest_dtype,
                              AmeExecResult *result);

/**
 * riscv_ame_int_scaled_sum: Convert a sum of two exact binary values to integer
 * @left: Signed magnitude of the first term
 * @left_exp: First term's power-of-two exponent
 * @right: Signed magnitude of the second term; NULL for a single term
 * @right_exp: Second term's power-of-two exponent
 * @absolute: Take the absolute value of the complete sum
 * @dest_dtype: Integer destination, including rounding and saturation
 * @result: Converted bits, saturation and NV/NX flags for this result only
 *
 * Evaluate left * 2^left_exp + right * 2^right_exp without input rounding.
 * Round fraction bits once, then clamp or wrap. A finite out-of-range result
 * sets NV; an in-range inexact result sets NX. This is the integer-result
 * conversion used by mixed integer/floating arithmetic.
 */
void riscv_ame_int_scaled_sum(const AmeIntProduct *left, int left_exp,
                              const AmeIntProduct *right, int right_exp,
                              bool absolute, const AmeDTypeDesc *dest_dtype,
                              AmeExecResult *result);

#endif /* TARGET_RISCV_AME_INT_H */
