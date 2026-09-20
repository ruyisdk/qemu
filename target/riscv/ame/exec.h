/*
 * RISC-V Attached Matrix Extension (AME) execution model
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef TARGET_RISCV_AME_EXEC_H
#define TARGET_RISCV_AME_EXEC_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "fpu/softfloat-types.h"
#include "target/riscv/ame/layout.h"

/*
 * Numerical/structural implementation selected after capability authorization.
 * INT uses integer arithmetic only; FP uses SoftFloat for at least one
 * datatype and also handles mixed integer/floating tuples. The suffix
 * selects the kernel family.
 * COPY moves raw bits, ZERO fills zeros, and SELECT tests a predicate while
 * preserving selected data bits.
 */
typedef enum AmeImplClass {
    AME_IMPL_UNSUPPORTED = 0,
    AME_IMPL_COPY = 1,
    AME_IMPL_ZERO = 2,
    AME_IMPL_INT_UNARY = 3,
    AME_IMPL_INT_BINARY = 4,
    AME_IMPL_INT_TERNARY = 5,
    AME_IMPL_INT_COMPARE = 6,
    AME_IMPL_INT_CONVERT = 7,
    AME_IMPL_INT_INDEX = 8,
    AME_IMPL_INT_SHIFT = 9,
    AME_IMPL_INT_REDUCE = 10,
    AME_IMPL_INT_SCATTER = 11,
    AME_IMPL_INT_MATMUL = 12,
    AME_IMPL_FP_UNARY = 13,
    AME_IMPL_FP_BINARY = 14,
    AME_IMPL_FP_TERNARY = 15,
    AME_IMPL_FP_COMPARE = 16,
    AME_IMPL_FP_CONVERT = 17,
    AME_IMPL_FP_REDUCE = 18,
    AME_IMPL_FP_SCATTER = 19,
    AME_IMPL_FP_MATMUL = 20,
    AME_IMPL_FP_SCALE = 21,
    AME_IMPL_SELECT = 22,
    AME_IMPL_MAX,
} AmeImplClass;

/**
 * riscv_ame_exec_impl: Select an implementation for an authorized tuple
 * @op: Operation to execute
 * @dest: Destination datatype
 * @source1: First source datatype; NULL if unused
 * @source2: Second source datatype; NULL if unused
 *
 * Call after checking backend capabilities. Custom operands enter the custom
 * selector even when only the second source is custom.
 *
 * Returns: implementation family, or UNSUPPORTED if no selector accepts it.
 */
AmeImplClass riscv_ame_exec_impl(AmeOperation op,
                                const AmeDTypeDesc *dest,
                                const AmeDTypeDesc *source1,
                                const AmeDTypeDesc *source2);

/**
 * riscv_ame_impl_is_fp: Identify a floating-point implementation family
 * @impl: Selected implementation
 *
 * SELECT is not an FP family, even when it uses a floating predicate.
 *
 * Returns: true for AME_IMPL_FP_* values; otherwise false.
 */
bool riscv_ame_impl_is_fp(AmeImplClass impl);

/*
 * Single-element calls validate the primitive operation, datatypes and value
 * pointers before invoking fp/int arithmetic. Failure leaves output and status
 * unchanged. Tensor kernels perform the same validation once per instruction.
 * A floating path requires initialized SoftFloat state with the applicable
 * rounding mode; integer paths allow NULL status. Exceptions accrue in status.
 * The caller authorizes the operation/datatype tuple before dispatching.
 */

/**
 * riscv_ame_exec_convert: Dispatch a single-element datatype conversion
 * @impl: Selected conversion implementation
 * @dest_dtype: Destination datatype
 * @source_dtype: Source datatype
 * @source: Raw source bits
 * @result: Output bits and flags, valid on success
 * @status: Prepared SoftFloat state for floating conversion; otherwise NULL
 *
 * Returns: true on success; false for unsupported input or missing pointers.
 */
bool riscv_ame_exec_convert(AmeImplClass impl,
                            const AmeDTypeDesc *dest_dtype,
                            const AmeDTypeDesc *source_dtype,
                            const AmeValue *source,
                            AmeExecResult *result,
                            float_status *status);
/**
 * riscv_ame_exec_unary: Dispatch a single-element unary operation
 * @impl: Selected unary implementation
 * @op: Operation, excluding conversion
 * @dest_dtype: Destination datatype
 * @source_dtype: Source datatype
 * @source: Raw source bits
 * @result: Output bits and flags, valid on success
 * @status: Prepared SoftFloat state for a floating path; otherwise NULL
 *
 * Returns: true on success; false for unsupported input or missing pointers.
 */
bool riscv_ame_exec_unary(AmeImplClass impl, AmeOperation op,
                          const AmeDTypeDesc *dest_dtype,
                          const AmeDTypeDesc *source_dtype,
                          const AmeValue *source,
                          AmeExecResult *result,
                          float_status *status);
/**
 * riscv_ame_exec_binary: Dispatch a single-element binary operation
 * @impl: Selected binary, comparison, shift, scale or selection implementation
 * @op: Operation to execute
 * @dest_dtype: Destination datatype
 * @src1_dtype: First source datatype
 * @src1: First source's raw bits
 * @src2_dtype: Second source datatype
 * @src2: Second source's raw bits
 * @result: Output bits and flags, valid on success
 * @status: Prepared SoftFloat state if floating arithmetic/predicates are used
 *
 * For selection, source 1 is the predicate and source 2 is the selected data.
 *
 * Returns: true on success; false for unsupported input or missing pointers.
 */
bool riscv_ame_exec_binary(AmeImplClass impl, AmeOperation op,
                           const AmeDTypeDesc *dest_dtype,
                           const AmeDTypeDesc *src1_dtype,
                           const AmeValue *src1,
                           const AmeDTypeDesc *src2_dtype,
                           const AmeValue *src2,
                           AmeExecResult *result,
                           float_status *status);
/**
 * riscv_ame_exec_ternary: Dispatch an operation that reads the old destination
 * @impl: Selected multiply/add, scaling-accumulate or selection implementation
 * @op: Operation to execute
 * @dest_dtype: Destination datatype
 * @src1_dtype: First source datatype
 * @src1: First source's raw bits
 * @src2_dtype: Second source datatype
 * @src2: Second source's raw bits
 * @current: Old destination bits, interpreted using @dest_dtype
 * @result: Output bits and flags, valid on success
 * @status: Prepared SoftFloat state if floating arithmetic/predicates are used
 *
 * Returns: true on success; false for unsupported input or missing pointers.
 */
bool riscv_ame_exec_ternary(AmeImplClass impl, AmeOperation op,
                            const AmeDTypeDesc *dest_dtype,
                            const AmeDTypeDesc *src1_dtype,
                            const AmeValue *src1,
                            const AmeDTypeDesc *src2_dtype,
                            const AmeValue *src2,
                            const AmeValue *current,
                            AmeExecResult *result,
                            float_status *status);

/*
 * Tensor calls below read the complete physical register files through
 * validated views. result_backing is a separate host buffer and must not
 * overlap either source file. Results start at buffer offset zero regardless
 * of the architectural destination register number.
 *
 * On success, copy the staged destination span back and accrue returned flags.
 * On failure, the output buffer and flags are unchanged.
 */

/**
 * riscv_ame_exec_m_convert: Convert all elements of an M span
 * @impl: Selected conversion implementation
 * @layout: Backend geometry
 * @source_mregs: Complete M register file to read
 * @destination: Validated destination view
 * @source: Validated source view with the same element count
 * @result_backing: Separate output buffer for the destination span
 * @result_bytes: Output capacity in bytes
 * @flags: Output exception/saturation flags combined across all elements
 *
 * Returns: true if the full result is staged; false for invalid input or size.
 */
bool riscv_ame_exec_m_convert(AmeImplClass impl,
                              const RISCVAMELayout *layout,
                              const uint8_t *source_mregs,
                              const AmeMView *destination,
                              const AmeMView *source,
                              uint8_t *result_backing,
                              size_t result_bytes,
                              AmeExecFlags *flags);
/**
 * riscv_ame_exec_m_unary: Apply a unary operation to every M element
 * @impl: Selected unary implementation
 * @op: Operation to execute
 * @layout: Backend geometry
 * @source_mregs: Complete M register file to read
 * @destination: Validated destination view
 * @source: Validated source view with the same element count
 * @result_backing: Separate output buffer for the destination span
 * @result_bytes: Output capacity in bytes
 * @flags: Output exception/saturation flags combined across all elements
 *
 * Returns: true if the full result is staged; false for invalid input or size.
 */
bool riscv_ame_exec_m_unary(AmeImplClass impl, AmeOperation op,
                            const RISCVAMELayout *layout,
                            const uint8_t *source_mregs,
                            const AmeMView *destination,
                            const AmeMView *source,
                            uint8_t *result_backing,
                            size_t result_bytes,
                            AmeExecFlags *flags);
/**
 * riscv_ame_exec_m_binary: Apply a binary operation to corresponding M elements
 * @impl: Selected binary implementation
 * @op: Operation to execute
 * @layout: Backend geometry
 * @source_mregs: Complete M register file to read
 * @destination: Validated destination view
 * @source1: Validated first source with the destination's element count
 * @source2: Validated second source with the destination's element count
 * @result_backing: Separate output buffer for the destination span
 * @result_bytes: Output capacity in bytes
 * @flags: Output exception/saturation flags combined across all elements
 *
 * Returns: true if the full result is staged; false for invalid input or size.
 */
bool riscv_ame_exec_m_binary(AmeImplClass impl, AmeOperation op,
                             const RISCVAMELayout *layout,
                             const uint8_t *source_mregs,
                             const AmeMView *destination,
                             const AmeMView *source1,
                             const AmeMView *source2,
                             uint8_t *result_backing,
                             size_t result_bytes,
                             AmeExecFlags *flags);
/**
 * riscv_ame_exec_m_binary_scalar: Combine each M element with one scalar value
 * @impl: Selected binary implementation
 * @op: Operation to execute
 * @layout: Backend geometry
 * @source_mregs: Complete M register file to read
 * @destination: Validated destination view
 * @source: Validated source with the destination's element count
 * @scalar_dtype: Datatype interpreting @scalar at this execution stage
 * @scalar: Raw scalar bits, already converted if the instruction requires it
 * @scalar_first: True places the scalar at source 1; false at source 2
 * @result_backing: Separate output buffer for the destination span
 * @result_bytes: Output capacity in bytes
 * @flags: Output exception/saturation flags; excludes any earlier conversion
 *
 * Returns: true if the full result is staged; false for invalid input or size.
 */
bool riscv_ame_exec_m_binary_scalar(AmeImplClass impl, AmeOperation op,
                                    const RISCVAMELayout *layout,
                                    const uint8_t *source_mregs,
                                    const AmeMView *destination,
                                    const AmeMView *source,
                                    const AmeDTypeDesc *scalar_dtype,
                                    const AmeValue *scalar,
                                    bool scalar_first,
                                    uint8_t *result_backing,
                                    size_t result_bytes,
                                    AmeExecFlags *flags);
/**
 * riscv_ame_exec_m_ternary: Combine two M sources with the old destination
 * @impl: Selected ternary implementation
 * @op: Operation to execute
 * @layout: Backend geometry
 * @source_mregs: Complete M register file, including the old destination
 * @destination: Validated destination view; also identifies old values to read
 * @source1: Validated first source with the destination's element count
 * @source2: Validated second source with the destination's element count
 * @result_backing: Separate output buffer for the destination span
 * @result_bytes: Output capacity in bytes
 * @flags: Output exception/saturation flags combined across all elements
 *
 * Returns: true if the full result is staged; false for invalid input or size.
 */
bool riscv_ame_exec_m_ternary(AmeImplClass impl, AmeOperation op,
                              const RISCVAMELayout *layout,
                              const uint8_t *source_mregs,
                              const AmeMView *destination,
                              const AmeMView *source1,
                              const AmeMView *source2,
                              uint8_t *result_backing,
                              size_t result_bytes,
                              AmeExecFlags *flags);
/**
 * riscv_ame_exec_m_ternary_scalar: Stage an M/scalar/old-destination operation
 * @impl: Selected ternary implementation
 * @op: Operation to execute
 * @layout: Backend geometry
 * @source_mregs: Complete M register file, including the old destination
 * @destination: Validated destination view; also identifies old values to read
 * @source: Validated source with the destination's element count
 * @scalar_dtype: Datatype interpreting @scalar at this execution stage
 * @scalar: Raw scalar bits, already converted if the instruction requires it
 * @scalar_first: True places the scalar at source 1; false at source 2
 * @result_backing: Separate output buffer for the destination span
 * @result_bytes: Output capacity in bytes
 * @flags: Output exception/saturation flags; excludes any earlier conversion
 *
 * Returns: true if the full result is staged; false for invalid input or size.
 */
bool riscv_ame_exec_m_ternary_scalar(AmeImplClass impl, AmeOperation op,
                                     const RISCVAMELayout *layout,
                                     const uint8_t *source_mregs,
                                     const AmeMView *destination,
                                     const AmeMView *source,
                                     const AmeDTypeDesc *scalar_dtype,
                                     const AmeValue *scalar,
                                     bool scalar_first,
                                     uint8_t *result_backing,
                                     size_t result_bytes,
                                     AmeExecFlags *flags);

#endif /* TARGET_RISCV_AME_EXEC_H */
