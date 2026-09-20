/*
 * RISC-V Attached Matrix Extension (AME) floating-point primitives
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef TARGET_RISCV_AME_FP_H
#define TARGET_RISCV_AME_FP_H

#include <stdbool.h>
#include <stdint.h>

#include "fpu/softfloat-types.h"
#include "target/riscv/ame/ame.h"
#include "target/riscv/ame/datatype.h"

/*
 * Floating primitives operate on raw element bits.
 * Execution calls require an operation and datatypes validated by exec.c,
 * non-NULL value and result pointers, and an initialized float_status.
 * Floating results use its selected rounding mode; integer results use the
 * destination's integer mode. Integer operands retain their full precision
 * until final rounding.
 * They accrue exceptions in that status; result.fflags includes exceptions
 * already present on entry.
 */

/**
 * riscv_ame_fp_supports: Check a floating primitive's operation and datatypes
 * @op: Numerical operation
 * @dest: Integer or floating destination datatype, as permitted by @op
 * @a: Integer or floating first source datatype, as permitted by @op
 * @b: Second source datatype; NULL for a unary operation or conversion
 *
 * Returns: true for an implemented primitive/type combination; otherwise false.
 */
bool riscv_ame_fp_supports(AmeOperation op, const AmeDTypeDesc *dest,
                           const AmeDTypeDesc *a, const AmeDTypeDesc *b);

/**
 * riscv_ame_fp_status_init: Initialize local AME SoftFloat state
 * @status: Storage to overwrite
 *
 * Clear exceptions and install AME NaN and tininess settings. Select rounding
 * separately with riscv_ame_fp_status_set_rounding().
 */
void riscv_ame_fp_status_init(float_status *status);

/**
 * riscv_ame_fp_status_prepare: Initialize SoftFloat state for a datatype
 * @status: Storage to overwrite on success
 * @dest_dtype: Floating or integer destination datatype
 *
 * Select floating destination rounding. An integer destination initializes
 * exception state only; the final integer result uses its integer mode.
 *
 * Returns: true on success; false for an invalid datatype/mode or NULL status,
 * leaving @status unchanged.
 */
bool riscv_ame_fp_status_prepare(float_status *status,
                                 const AmeDTypeDesc *dest_dtype);
/**
 * riscv_ame_fp_status_set_rounding: Change rounding without clearing exceptions
 * @status: Previously initialized SoftFloat state
 * @dest_dtype: Floating datatype whose rounding mode is selected
 *
 * Returns: true on success; false for an invalid datatype/mode or NULL status,
 * leaving @status unchanged.
 */
bool riscv_ame_fp_status_set_rounding(float_status *status,
                                      const AmeDTypeDesc *dest_dtype);

/**
 * riscv_ame_fp_status_flags: Map SoftFloat exceptions to amefflags bits
 * @status: SoftFloat state to read; exceptions are not cleared
 *
 * Returns: AME_FFLAG_* mask for invalid, divide-by-zero, overflow, underflow
 * and inexact exceptions.
 */
uint8_t riscv_ame_fp_status_flags(const float_status *status);

/**
 * riscv_ame_fp_compare_zero: Compare a floating predicate element with zero
 * @dtype: Valid floating datatype
 * @value: Raw predicate bits
 * @ordered: Output; false for an unordered NaN comparison
 * @negative: Output; true only if the value compares less than zero
 * @status: Initialized SoftFloat state in which exceptions accrue
 *
 * Negative zero is not less than zero. This supports conditional selection
 * without converting the selected data to the predicate's format.
 */
void riscv_ame_fp_compare_zero(const AmeDTypeDesc *dtype,
                               const AmeValue *value,
                               bool *ordered, bool *negative,
                               float_status *status);

/**
 * riscv_ame_exec_fp_convert: Convert an element involving floating-point
 * @dest_dtype: Integer or floating destination datatype
 * @source_dtype: Integer or floating source datatype
 * @source: Raw source element bits
 * @result: Output raw bits, saturation and accrued exception flags
 * @status: Initialized SoftFloat state; floating destinations use its rounding
 *
 * Integer destinations round using their integer mode, then clamp or wrap.
 * Finite out-of-range results set NV and clamping sets SAT. NaN and infinity
 * return integer bounds with NV only. An invalid element does not add NX or
 * discard previously accrued exceptions. These exception and nonfinite-result
 * rules supplement the v0.6 draft, which does not fully specify them.
 */
void riscv_ame_exec_fp_convert(const AmeDTypeDesc *dest_dtype,
                               const AmeDTypeDesc *source_dtype,
                               const AmeValue *source,
                               AmeExecResult *result,
                               float_status *status);
/**
 * riscv_ame_exec_fp_unary: Compute unary arithmetic involving floating-point
 * @op: Unary operation, excluding datatype conversion
 * @dest_dtype: Destination datatype; ABS also allows integer results
 * @source_dtype: Source datatype; ABS also allows integer inputs
 * @source: Raw source element bits
 * @result: Output raw bits, saturation and accrued exception flags
 * @status: Initialized SoftFloat state with destination rounding
 */
void riscv_ame_exec_fp_unary(AmeOperation op,
                             const AmeDTypeDesc *dest_dtype,
                             const AmeDTypeDesc *source_dtype,
                             const AmeValue *source,
                             AmeExecResult *result,
                             float_status *status);
/**
 * riscv_ame_exec_fp_binary: Compute binary floating arithmetic or comparison
 * @op: Binary arithmetic, comparison or exponent-scaling operation
 * @dest_dtype: Integer or floating destination; integer for comparison masks
 * @src1_dtype: Integer or floating first source datatype
 * @src1: First source's raw bits
 * @src2_dtype: Integer or floating second source; integer for exponent scaling
 * @src2: Second source's raw bits
 * @result: Output raw bits, saturation and accrued exception flags
 * @status: Initialized SoftFloat state with the applicable rounding mode
 *
 * Comparison results are all destination bits set for true, zero for false.
 */
void riscv_ame_exec_fp_binary(AmeOperation op,
                              const AmeDTypeDesc *dest_dtype,
                              const AmeDTypeDesc *src1_dtype,
                              const AmeValue *src1,
                              const AmeDTypeDesc *src2_dtype,
                              const AmeValue *src2,
                              AmeExecResult *result,
                              float_status *status);
/**
 * riscv_ame_exec_fp_ternary: Compute a multiply/add with old destination
 * @op: MULACC, MULNACC, MULADD or MULSUB operation
 * @dest_dtype: Integer or floating destination datatype
 * @src1_dtype: Integer or floating first source datatype
 * @src1: First source's raw bits
 * @src2_dtype: Integer or floating second source datatype
 * @src2: Second source's raw bits
 * @current: Old destination element, interpreted using @dest_dtype
 * @result: Output raw bits, saturation and accrued exception flags
 * @status: Initialized SoftFloat state with destination rounding
 */
void riscv_ame_exec_fp_ternary(AmeOperation op,
                               const AmeDTypeDesc *dest_dtype,
                               const AmeDTypeDesc *src1_dtype,
                               const AmeValue *src1,
                               const AmeDTypeDesc *src2_dtype,
                               const AmeValue *src2,
                               const AmeValue *current,
                               AmeExecResult *result,
                               float_status *status);
/**
 * riscv_ame_exec_fp_scale_acc: Scale an element and add the old destination
 * @op: LDEXPACC or RDEXPACC operation
 * @dest_dtype: Integer or floating destination datatype
 * @data_dtype: Integer or floating source datatype
 * @data: Raw source element bits
 * @exponent_dtype: Integer datatype supplying exponent width and signedness
 * @exponent: Raw exponent bits
 * @current: Old destination element, interpreted using @dest_dtype
 * @result: Output raw bits, saturation and accrued exception flags
 * @status: Initialized SoftFloat state with destination rounding
 */
void riscv_ame_exec_fp_scale_acc(AmeOperation op,
                                 const AmeDTypeDesc *dest_dtype,
                                 const AmeDTypeDesc *data_dtype,
                                 const AmeValue *data,
                                 const AmeDTypeDesc *exponent_dtype,
                                 const AmeValue *exponent,
                                 const AmeValue *current,
                                 AmeExecResult *result,
                                 float_status *status);

#endif /* TARGET_RISCV_AME_FP_H */
