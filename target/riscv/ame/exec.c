/*
 * RISC-V Attached Matrix Extension (AME) execution model
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/bitops.h"
#include "fpu/softfloat.h"
#include "target/riscv/ame/exec.h"
#include "target/riscv/ame/fp.h"
#include "target/riscv/ame/int.h"

/*
 * Custom numerical paths are selected once per instruction, not per element.
 * A concrete implementation supplies the complete tuple semantics here and
 * admits the matching tuples in cap-set.c. No custom tuples are implemented.
 */
static AmeImplClass ame_custom_exec_impl(AmeOperation op,
                                         const AmeDTypeDesc *dest,
                                         const AmeDTypeDesc *source1,
                                         const AmeDTypeDesc *source2)
{
    return AME_IMPL_UNSUPPORTED;
}

AmeImplClass riscv_ame_exec_impl(AmeOperation op,
                                const AmeDTypeDesc *dest,
                                const AmeDTypeDesc *source1,
                                const AmeDTypeDesc *source2)
{
    bool fp;

    if (!dest || (unsigned)op >= AME_OP_COUNT) {
        return AME_IMPL_UNSUPPORTED;
    }
    if (dest->class == AME_DTYPE_CLASS_CUSTOM ||
        (source1 && source1->class == AME_DTYPE_CLASS_CUSTOM) ||
        (source2 && source2->class == AME_DTYPE_CLASS_CUSTOM)) {
        return ame_custom_exec_impl(op, dest, source1, source2);
    }
    if (!riscv_ame_dtype_is_standard(dest) ||
        (source1 && !riscv_ame_dtype_is_standard(source1)) ||
        (source2 && !riscv_ame_dtype_is_standard(source2))) {
        return AME_IMPL_UNSUPPORTED;
    }
    fp = riscv_ame_dtype_is_fp(dest) || riscv_ame_dtype_is_fp(source1) ||
         riscv_ame_dtype_is_fp(source2);

    switch (op) {
    case AME_OP_ZERO:
        return AME_IMPL_ZERO;
    case AME_OP_COLID:
    case AME_OP_ROWID:
        return AME_IMPL_INT_INDEX;
    case AME_OP_COLBCAST:
    case AME_OP_ROWBCAST:
    case AME_OP_SHIFT:
    case AME_OP_ROWSHIFT:
    case AME_OP_COLZIP:
    case AME_OP_COLUNZIP:
    case AME_OP_ROWZIP:
    case AME_OP_ROWUNZIP:
    case AME_OP_GATHER:
    case AME_OP_GATHERROW:
        return AME_IMPL_COPY;
    case AME_OP_SEL:
    case AME_OP_SELGE:
    case AME_OP_MUXGE:
    case AME_OP_MUXLT:
        return AME_IMPL_SELECT;
    case AME_OP_CONV:
        return fp ? AME_IMPL_FP_CONVERT : AME_IMPL_INT_CONVERT;
    case AME_OP_CMPGE:
    case AME_OP_CMPLT:
        if (!source1) {
            return AME_IMPL_UNSUPPORTED;
        }
        return fp ? AME_IMPL_FP_COMPARE : AME_IMPL_INT_COMPARE;
    case AME_OP_SLL:
    case AME_OP_SRL:
    case AME_OP_SRA:
        return AME_IMPL_INT_SHIFT;
    case AME_OP_ABS:
    case AME_OP_COS:
    case AME_OP_SIN:
    case AME_OP_EXP2:
    case AME_OP_LOG2:
    case AME_OP_FRINTM:
    case AME_OP_FRINTN:
    case AME_OP_FRINTP:
    case AME_OP_FRINTZ:
    case AME_OP_REC:
    case AME_OP_RSQRT:
    case AME_OP_SQRT:
    case AME_OP_TANH:
        return fp ? AME_IMPL_FP_UNARY : AME_IMPL_INT_UNARY;
    case AME_OP_ADD:
    case AME_OP_SUB:
    case AME_OP_MUL:
    case AME_OP_NMUL:
    case AME_OP_ABSDIFF:
    case AME_OP_HDIFF:
    case AME_OP_MEAN:
    case AME_OP_AND:
    case AME_OP_ANDNOT:
    case AME_OP_OR:
    case AME_OP_ORNOT:
    case AME_OP_XOR:
    case AME_OP_MIN:
    case AME_OP_MAX:
    case AME_OP_LOG2SUB:
    case AME_OP_SUBLOG2:
        return fp ? AME_IMPL_FP_BINARY : AME_IMPL_INT_BINARY;
    case AME_OP_LDEXP:
    case AME_OP_RDEXP:
        return fp ? AME_IMPL_FP_SCALE : AME_IMPL_INT_BINARY;
    case AME_OP_LDEXPACC:
    case AME_OP_RDEXPACC:
    case AME_OP_MULACC:
    case AME_OP_MULNACC:
    case AME_OP_MULADD:
    case AME_OP_MULSUB:
        return fp ? AME_IMPL_FP_TERNARY : AME_IMPL_INT_TERNARY;
    case AME_OP_PREFIXADD:
    case AME_OP_PREFIXADDCOL:
    case AME_OP_PREFIXMAX:
    case AME_OP_PREFIXMAXCOL:
    case AME_OP_REDUCEADD:
    case AME_OP_REDUCEADDCOL:
    case AME_OP_REDUCEMAX:
    case AME_OP_REDUCEMAXCOL:
    case AME_OP_REDUCEMIN:
    case AME_OP_REDUCEMINCOL:
        return fp ? AME_IMPL_FP_REDUCE : AME_IMPL_INT_REDUCE;
    case AME_OP_COLSCATADD:
    case AME_OP_COLSCATMAX:
    case AME_OP_ROWSCATADD:
    case AME_OP_ROWSCATMAX:
        return fp ? AME_IMPL_FP_SCATTER : AME_IMPL_INT_SCATTER;
    case AME_OP_MMACC:
    case AME_OP_MMACCNEG:
    case AME_OP_MMATACC:
    case AME_OP_MMATACCNEG:
    case AME_OP_MMBTACC:
    case AME_OP_MMBTACCNEG:
        return fp ? AME_IMPL_FP_MATMUL : AME_IMPL_INT_MATMUL;
    default:
        g_assert_not_reached();
    }
}

bool riscv_ame_impl_is_fp(AmeImplClass impl)
{
    switch (impl) {
    case AME_IMPL_FP_UNARY:
    case AME_IMPL_FP_BINARY:
    case AME_IMPL_FP_TERNARY:
    case AME_IMPL_FP_COMPARE:
    case AME_IMPL_FP_CONVERT:
    case AME_IMPL_FP_REDUCE:
    case AME_IMPL_FP_SCATTER:
    case AME_IMPL_FP_MATMUL:
    case AME_IMPL_FP_SCALE:
        return true;
    default:
        return false;
    }
}

static bool ame_exec_status_prepare(AmeImplClass impl,
                                    const AmeDTypeDesc *dest_dtype,
                                    const AmeDTypeDesc *src1_dtype,
                                    float_status *storage,
                                    float_status **status)
{
    const AmeDTypeDesc *round_dtype;

    *status = NULL;
    if (impl == AME_IMPL_SELECT && riscv_ame_dtype_is_fp(src1_dtype)) {
        if (!riscv_ame_fp_status_prepare(storage, src1_dtype)) {
            return false;
        }
        *status = storage;
        return true;
    }
    if (!riscv_ame_impl_is_fp(impl)) {
        return true;
    }

    round_dtype = impl == AME_IMPL_FP_COMPARE ? src1_dtype : dest_dtype;
    if (!riscv_ame_fp_status_prepare(storage, round_dtype)) {
        return false;
    }
    *status = storage;
    return true;
}

static bool ame_exec_convert_supported(AmeImplClass impl,
                                       const AmeDTypeDesc *d,
                                       const AmeDTypeDesc *a)
{
    switch (impl) {
    case AME_IMPL_INT_CONVERT:
        return riscv_ame_int_supports(AME_OP_CONV, d, a, NULL);
    case AME_IMPL_FP_CONVERT:
        return riscv_ame_fp_supports(AME_OP_CONV, d, a, NULL);
    default:
        return false;
    }
}

static bool ame_exec_unary_supported(AmeImplClass impl, AmeOperation op,
                                     const AmeDTypeDesc *d,
                                     const AmeDTypeDesc *a)
{
    if (op == AME_OP_CONV) {
        return false;
    }
    switch (impl) {
    case AME_IMPL_INT_UNARY:
        return riscv_ame_int_supports(op, d, a, NULL);
    case AME_IMPL_FP_UNARY:
        return riscv_ame_fp_supports(op, d, a, NULL);
    default:
        return false;
    }
}

static bool ame_exec_select_supported(const AmeDTypeDesc *d,
                                      const AmeDTypeDesc *predicate,
                                      const AmeDTypeDesc *selected)
{
    if (d == NULL || predicate == NULL || selected != d ||
        d->bits < AME_DTYPE_BITS_MIN || d->bits > AME_VALUE_BITS) {
        return false;
    }
    /* Compare-zero needs only the predicate format, not a data conversion. */
    return riscv_ame_int_supports(AME_OP_CONV, predicate, predicate, NULL) ||
           riscv_ame_fp_supports(AME_OP_CONV, predicate, predicate, NULL);
}

static bool ame_exec_binary_supported(AmeImplClass impl, AmeOperation op,
                                      const AmeDTypeDesc *d,
                                      const AmeDTypeDesc *a,
                                      const AmeDTypeDesc *b)
{
    switch (impl) {
    case AME_IMPL_SELECT:
        return (op == AME_OP_SEL || op == AME_OP_SELGE) &&
               ame_exec_select_supported(d, a, b);
    case AME_IMPL_INT_COMPARE:
        return (op == AME_OP_CMPGE || op == AME_OP_CMPLT) &&
               riscv_ame_int_supports(op, d, a, b);
    case AME_IMPL_FP_COMPARE:
        return (op == AME_OP_CMPGE || op == AME_OP_CMPLT) &&
               riscv_ame_fp_supports(op, d, a, b);
    case AME_IMPL_FP_SCALE:
        return (op == AME_OP_LDEXP || op == AME_OP_RDEXP) &&
               riscv_ame_fp_supports(op, d, a, b);
    case AME_IMPL_INT_SHIFT:
        return (op == AME_OP_SLL || op == AME_OP_SRL ||
                op == AME_OP_SRA) && riscv_ame_int_supports(op, d, a, b);
    case AME_IMPL_INT_BINARY:
    case AME_IMPL_FP_BINARY:
        break;
    default:
        return false;
    }

    switch (op) {
    case AME_OP_ABSDIFF:
    case AME_OP_ADD:
    case AME_OP_SUB:
    case AME_OP_HDIFF:
    case AME_OP_MEAN:
    case AME_OP_MIN:
    case AME_OP_MAX:
    case AME_OP_MUL:
    case AME_OP_NMUL:
    case AME_OP_AND:
    case AME_OP_ANDNOT:
    case AME_OP_OR:
    case AME_OP_ORNOT:
    case AME_OP_XOR:
    case AME_OP_LDEXP:
    case AME_OP_RDEXP:
    case AME_OP_LOG2SUB:
    case AME_OP_SUBLOG2:
        return impl == AME_IMPL_INT_BINARY ?
               riscv_ame_int_supports(op, d, a, b) :
               riscv_ame_fp_supports(op, d, a, b);
    case AME_OP_CMPGE:
    case AME_OP_CMPLT:
        return impl != AME_IMPL_INT_BINARY &&
               riscv_ame_fp_supports(op, d, a, b);
    default:
        return false;
    }
}

static bool ame_exec_ternary_supported(AmeImplClass impl, AmeOperation op,
                                       const AmeDTypeDesc *d,
                                       const AmeDTypeDesc *a,
                                       const AmeDTypeDesc *b)
{
    if (impl == AME_IMPL_SELECT) {
        return (op == AME_OP_MUXLT || op == AME_OP_MUXGE) &&
               ame_exec_select_supported(d, a, b);
    }
    switch (op) {
    case AME_OP_LDEXPACC:
    case AME_OP_RDEXPACC:
    case AME_OP_MULACC:
    case AME_OP_MULNACC:
    case AME_OP_MULADD:
    case AME_OP_MULSUB:
        switch (impl) {
        case AME_IMPL_INT_TERNARY:
            return riscv_ame_int_supports(op, d, a, b);
        case AME_IMPL_FP_TERNARY:
            return riscv_ame_fp_supports(op, d, a, b);
        default:
            return false;
        }
    default:
        return false;
    }
}

/* The checked execution entry has validated the operation and datatypes. */
static void ame_exec_select(AmeOperation op,
                            const AmeDTypeDesc *dest_dtype,
                            const AmeDTypeDesc *predicate_dtype,
                            const AmeValue *predicate,
                            const AmeValue *selected,
                            const AmeValue *current,
                            AmeExecResult *result,
                            float_status *status)
{
    bool ordered = true;
    bool negative = false;
    bool select;

    if (predicate_dtype->class == AME_DTYPE_CLASS_INTEGER) {
        if (predicate_dtype->integer.is_signed) {
            negative = predicate_dtype->bits <= AME_VALUE_WORD_BITS ?
                extract64(predicate->lo, predicate_dtype->bits - 1, 1) :
                extract64(predicate->hi,
                          predicate_dtype->bits -
                          AME_VALUE_WORD_BITS - 1, 1);
        }
    } else {
        riscv_ame_fp_compare_zero(predicate_dtype, predicate,
                                  &ordered, &negative, status);
    }
    select = op == AME_OP_SELGE || op == AME_OP_MUXGE ?
             ordered && !negative : negative;
    *result = (AmeExecResult) {
        .value = select ? *selected :
                 current != NULL ? *current : (AmeValue) { 0 },
        .fflags = status != NULL ?
                  riscv_ame_fp_status_flags(status) : 0,
    };
    if (dest_dtype->bits < AME_VALUE_WORD_BITS) {
        result->value.lo &= MAKE_64BIT_MASK(0, dest_dtype->bits);
        result->value.hi = 0;
    } else if (dest_dtype->bits == AME_VALUE_WORD_BITS) {
        result->value.hi = 0;
    } else if (dest_dtype->bits < AME_VALUE_BITS) {
        result->value.hi &= MAKE_64BIT_MASK(
            0, dest_dtype->bits - AME_VALUE_WORD_BITS);
    }
}

static void ame_exec_convert(AmeImplClass impl,
                              const AmeDTypeDesc *dest_dtype,
                              const AmeDTypeDesc *source_dtype,
                              const AmeValue *source,
                              AmeExecResult *result,
                              float_status *status)
{
    if (impl == AME_IMPL_FP_CONVERT) {
        riscv_ame_exec_fp_convert(dest_dtype, source_dtype, source,
                                  result, status);
    } else {
        riscv_ame_exec_int_convert(dest_dtype, source_dtype, source, result);
    }
}

bool riscv_ame_exec_convert(AmeImplClass impl,
                            const AmeDTypeDesc *dest_dtype,
                            const AmeDTypeDesc *source_dtype,
                            const AmeValue *source,
                            AmeExecResult *result,
                            float_status *status)
{
    if (source == NULL || result == NULL ||
        (impl == AME_IMPL_FP_CONVERT && status == NULL) ||
        !ame_exec_convert_supported(impl, dest_dtype, source_dtype)) {
        return false;
    }
    ame_exec_convert(impl, dest_dtype, source_dtype, source, result, status);
    return true;
}

bool riscv_ame_exec_unary(AmeImplClass impl, AmeOperation op,
                          const AmeDTypeDesc *dest_dtype,
                          const AmeDTypeDesc *source_dtype,
                          const AmeValue *source,
                          AmeExecResult *result,
                          float_status *status)
{
    if (source == NULL || result == NULL ||
        (impl == AME_IMPL_FP_UNARY && status == NULL) ||
        !ame_exec_unary_supported(impl, op, dest_dtype, source_dtype)) {
        return false;
    }
    if (impl == AME_IMPL_FP_UNARY) {
        riscv_ame_exec_fp_unary(op, dest_dtype, source_dtype,
                               source, result, status);
    } else {
        riscv_ame_exec_int_unary(op, dest_dtype, source_dtype, source, result);
    }
    return true;
}

bool riscv_ame_exec_binary(AmeImplClass impl, AmeOperation op,
                           const AmeDTypeDesc *dest_dtype,
                           const AmeDTypeDesc *src1_dtype,
                           const AmeValue *src1,
                           const AmeDTypeDesc *src2_dtype,
                           const AmeValue *src2,
                           AmeExecResult *result,
                           float_status *status)
{
    if (src1 == NULL || src2 == NULL || result == NULL ||
        ((riscv_ame_impl_is_fp(impl) ||
          (impl == AME_IMPL_SELECT && riscv_ame_dtype_is_fp(src1_dtype))) &&
         status == NULL) ||
        !ame_exec_binary_supported(impl, op, dest_dtype, src1_dtype,
                                   src2_dtype)) {
        return false;
    }
    if (impl == AME_IMPL_SELECT) {
        ame_exec_select(op, dest_dtype, src1_dtype, src1,
                         src2, NULL, result, status);
    } else if (riscv_ame_impl_is_fp(impl)) {
        riscv_ame_exec_fp_binary(op, dest_dtype, src1_dtype, src1,
                                 src2_dtype, src2, result, status);
    } else {
        riscv_ame_exec_int_binary(op, dest_dtype, src1_dtype, src1,
                                  src2_dtype, src2, result);
    }
    return true;
}

bool riscv_ame_exec_ternary(AmeImplClass impl, AmeOperation op,
                            const AmeDTypeDesc *dest_dtype,
                            const AmeDTypeDesc *src1_dtype,
                            const AmeValue *src1,
                            const AmeDTypeDesc *src2_dtype,
                            const AmeValue *src2,
                            const AmeValue *current,
                            AmeExecResult *result,
                            float_status *status)
{
    if (src1 == NULL || src2 == NULL || current == NULL || result == NULL ||
        ((riscv_ame_impl_is_fp(impl) ||
          (impl == AME_IMPL_SELECT && riscv_ame_dtype_is_fp(src1_dtype))) &&
         status == NULL) ||
        !ame_exec_ternary_supported(impl, op, dest_dtype, src1_dtype,
                                    src2_dtype)) {
        return false;
    }
    if (impl == AME_IMPL_SELECT) {
        ame_exec_select(op, dest_dtype, src1_dtype, src1,
                         src2, current, result, status);
    } else if (impl == AME_IMPL_INT_TERNARY) {
        riscv_ame_exec_int_ternary(
            op, dest_dtype, src1_dtype, src1, src2_dtype, src2,
            current, result);
    } else if (op == AME_OP_LDEXPACC || op == AME_OP_RDEXPACC) {
        riscv_ame_exec_fp_scale_acc(
            op, dest_dtype, src1_dtype, src1, src2_dtype, src2,
            current, result, status);
    } else {
        riscv_ame_exec_fp_ternary(
            op, dest_dtype, src1_dtype, src1, src2_dtype, src2,
            current, result, status);
    }
    return true;
}

static bool ame_exec_m_result_init(const RISCVAMELayout *layout,
                                   const AmeMView *destination,
                                   uint8_t *result_backing,
                                   size_t result_bytes,
                                   AmeMView *staged_destination)
{
    size_t bytes;

    if (layout == NULL || destination == NULL || result_backing == NULL ||
        staged_destination == NULL || destination->dtype == NULL) {
        return false;
    }
    bytes = riscv_ame_m_view_bytes(layout, destination);
    if (bytes > result_bytes) {
        return false;
    }
    *staged_destination = *destination;
    staged_destination->base = 0;
    memset(result_backing, 0, bytes);
    return true;
}

bool riscv_ame_exec_m_convert(AmeImplClass impl,
                              const RISCVAMELayout *layout,
                              const uint8_t *source_mregs,
                              const AmeMView *destination,
                              const AmeMView *source,
                              uint8_t *result_backing,
                              size_t result_bytes,
                              AmeExecFlags *flags)
{
    AmeMView staged_destination;
    AmeExecFlags accrued = { 0 };
    float_status fp_status;
    float_status *status;
    size_t element;

    if (layout == NULL || source_mregs == NULL || destination == NULL ||
        source == NULL || flags == NULL) {
        return false;
    }
    if (destination->total_elements != source->total_elements) {
        return false;
    }
    if (!ame_exec_convert_supported(
            impl, destination->dtype, source->dtype) ||
        !ame_exec_status_prepare(impl, destination->dtype, source->dtype,
                                 &fp_status, &status)) {
        return false;
    }

    if (!ame_exec_m_result_init(layout, destination, result_backing,
                                result_bytes, &staged_destination)) {
        return false;
    }

    /* Equal-width integer conversion without saturation preserves every bit. */
    if (impl == AME_IMPL_INT_CONVERT &&
        destination->dtype->bits == source->dtype->bits &&
        !destination->dtype->integer.saturating) {
        memcpy(result_backing,
               source_mregs + source->base * layout->mreg_bytes,
               riscv_ame_m_view_bytes(layout, destination));
        *flags = accrued;
        return true;
    }

    for (element = 0; element < destination->total_elements; element++) {
        AmeValue value = riscv_ame_m_read(layout, source_mregs, source,
                                         element);
        AmeExecResult converted;

        ame_exec_convert(impl, destination->dtype, source->dtype,
                          &value, &converted, status);
        riscv_ame_m_write(layout, result_backing, &staged_destination,
                          element, &converted.value);
        accrued.fflags |= converted.fflags;
        accrued.saturated |= converted.saturated;
    }

    *flags = accrued;
    return true;
}

bool riscv_ame_exec_m_unary(AmeImplClass impl, AmeOperation op,
                            const RISCVAMELayout *layout,
                            const uint8_t *source_mregs,
                            const AmeMView *destination,
                            const AmeMView *source,
                            uint8_t *result_backing,
                            size_t result_bytes,
                            AmeExecFlags *flags)
{
    AmeMView staged_destination;
    AmeExecFlags accrued = { 0 };
    float_status fp_status;
    float_status *status;
    size_t element;

    if (layout == NULL || source_mregs == NULL || source == NULL ||
        flags == NULL || destination == NULL) {
        return false;
    }
    if (destination->total_elements != source->total_elements) {
        return false;
    }
    if (!ame_exec_unary_supported(
            impl, op, destination->dtype, source->dtype) ||
        !ame_exec_status_prepare(impl, destination->dtype, source->dtype,
                                 &fp_status, &status)) {
        return false;
    }
    if (!ame_exec_m_result_init(layout, destination, result_backing,
                                result_bytes, &staged_destination)) {
        return false;
    }

    for (element = 0; element < destination->total_elements; element++) {
        AmeValue value = riscv_ame_m_read(layout, source_mregs, source,
                                         element);
        AmeExecResult executed;

        if (impl == AME_IMPL_INT_UNARY) {
            riscv_ame_exec_int_unary(
                op, destination->dtype, source->dtype, &value, &executed);
        } else {
            riscv_ame_exec_fp_unary(
                op, destination->dtype, source->dtype, &value,
                &executed, status);
        }
        riscv_ame_m_write(layout, result_backing, &staged_destination,
                          element, &executed.value);
        accrued.fflags |= executed.fflags;
        accrued.saturated |= executed.saturated;
    }

    *flags = accrued;
    return true;
}

/* All views are validated; destination names the non-aliasing staging bank. */
#define AME_BINARY_VALUE_LOOP(expression)                                  \
    do {                                                                   \
        for (size_t i = 0; i < destination->total_elements; i++) {         \
            AmeValue a = riscv_ame_m_read(layout, mregs, source1, i);      \
            AmeValue b = riscv_ame_m_read(layout, mregs, source2, i);      \
            AmeValue value = { .lo = (expression) };                       \
                                                                           \
            riscv_ame_m_write(layout, result, destination, i, &value);     \
        }                                                                  \
    } while (0)

/* Boolean operations preserve packed element boundaries, including Int4. */
static bool ame_binary_bits_fast(AmeOperation op, const uint8_t *a,
                                 const uint8_t *b, uint8_t *restrict result,
                                 size_t bytes)
{
#define AME_BINARY_BYTE_LOOP(expression)        \
    do {                                       \
        for (size_t i = 0; i < bytes; i++) {     \
            result[i] = (expression);          \
        }                                      \
    } while (0)

    switch (op) {
    case AME_OP_AND:
        AME_BINARY_BYTE_LOOP(a[i] & b[i]);
        break;
    case AME_OP_ANDNOT:
        AME_BINARY_BYTE_LOOP(a[i] & ~b[i]);
        break;
    case AME_OP_OR:
        AME_BINARY_BYTE_LOOP(a[i] | b[i]);
        break;
    case AME_OP_ORNOT:
        AME_BINARY_BYTE_LOOP(a[i] | ~b[i]);
        break;
    case AME_OP_XOR:
        AME_BINARY_BYTE_LOOP(a[i] ^ b[i]);
        break;
    default:
        return false;
    }
#undef AME_BINARY_BYTE_LOOP
    return true;
}

/* Return false without writing anything when the generic path is needed. */
static bool ame_binary_int_fast(AmeOperation op, const RISCVAMELayout *layout,
                                const uint8_t *mregs,
                                const AmeMView *destination,
                                const AmeMView *source1,
                                const AmeMView *source2, uint8_t *result)
{
    const AmeDTypeDesc *dtype = destination->dtype;

    if (dtype->bits != source1->dtype->bits ||
        dtype->bits != source2->dtype->bits ||
        dtype->bits > AME_VALUE_WORD_BITS || dtype->integer.saturating) {
        return false;
    }

    /* Source signedness changes only multiples of 2^bits, not the low bits. */
    switch (op) {
    case AME_OP_ADD:
        AME_BINARY_VALUE_LOOP(a.lo + b.lo);
        break;
    case AME_OP_SUB:
        AME_BINARY_VALUE_LOOP(b.lo - a.lo);
        break;
    case AME_OP_MUL:
        AME_BINARY_VALUE_LOOP(a.lo * b.lo);
        break;
    case AME_OP_NMUL:
        AME_BINARY_VALUE_LOOP(-(a.lo * b.lo));
        break;
    case AME_OP_AND:
    case AME_OP_ANDNOT:
    case AME_OP_OR:
    case AME_OP_ORNOT:
    case AME_OP_XOR:
        return ame_binary_bits_fast(
            op, mregs + source1->base * layout->mreg_bytes,
            mregs + source2->base * layout->mreg_bytes,
            result, riscv_ame_m_view_bytes(layout, destination));
    default:
        return false;
    }
    return true;
}

static bool ame_binary_fp_fast(AmeOperation op, const RISCVAMELayout *layout,
                               const uint8_t *mregs,
                               const AmeMView *destination,
                               const AmeMView *source1,
                               const AmeMView *source2, uint8_t *result,
                               float_status *status)
{
    const AmeDTypeDesc *dtype = destination->dtype;
    uint64_t sign;

    if (!riscv_ame_dtype_is_fp(dtype) ||
        !riscv_ame_dtype_is_fp(source1->dtype) ||
        !riscv_ame_dtype_is_fp(source2->dtype) ||
        dtype->fp.format != source1->dtype->fp.format ||
        dtype->fp.format != source2->dtype->fp.format) {
        return false;
    }
    sign = BIT_ULL(dtype->bits - 1);

    /* One format/operation dispatch per tensor, no per-element callback. */
#define AME_FP_BINARY_LOOP(type, make, val)                                \
    do {                                                                   \
        switch (op) {                                                      \
        case AME_OP_ADD:                                                 \
            AME_BINARY_VALUE_LOOP(val(type##_add(make(a.lo), make(b.lo),   \
                                            status)));                     \
            break;                                                         \
        case AME_OP_SUB:                                                 \
            AME_BINARY_VALUE_LOOP(val(type##_sub(make(b.lo), make(a.lo),   \
                                            status)));                     \
            break;                                                         \
        case AME_OP_MUL:                                                 \
            AME_BINARY_VALUE_LOOP(val(type##_mul(make(a.lo), make(b.lo),   \
                                            status)));                     \
            break;                                                         \
        case AME_OP_NMUL:                                                \
            AME_BINARY_VALUE_LOOP(val(type##_mul(make(a.lo ^ sign),        \
                                            make(b.lo), status)));         \
            break;                                                         \
        default:                                                           \
            return false;                                                  \
        }                                                                  \
    } while (0)

    switch (dtype->fp.format) {
    case AME_FP_FORMAT_FP16:
        AME_FP_BINARY_LOOP(float16, make_float16, float16_val);
        break;
    case AME_FP_FORMAT_BF16:
        AME_FP_BINARY_LOOP(bfloat16, make_float16, float16_val);
        break;
    case AME_FP_FORMAT_FP32:
        AME_FP_BINARY_LOOP(float32, make_float32, float32_val);
        break;
    case AME_FP_FORMAT_FP64:
        AME_FP_BINARY_LOOP(float64, make_float64, float64_val);
        break;
    default:
        return false;
    }
#undef AME_FP_BINARY_LOOP
    return true;
}
#undef AME_BINARY_VALUE_LOOP

bool riscv_ame_exec_m_binary(AmeImplClass impl, AmeOperation op,
                             const RISCVAMELayout *layout,
                             const uint8_t *source_mregs,
                             const AmeMView *destination,
                             const AmeMView *source1,
                             const AmeMView *source2,
                             uint8_t *result_backing,
                             size_t result_bytes,
                             AmeExecFlags *flags)
{
    AmeMView staged_destination;
    AmeExecFlags accrued = { 0 };
    float_status fp_status;
    float_status *status;
    size_t element;
    bool fp;

    if (layout == NULL || source_mregs == NULL || source1 == NULL ||
        source2 == NULL || flags == NULL || destination == NULL) {
        return false;
    }
    if (destination->total_elements != source1->total_elements ||
        destination->total_elements != source2->total_elements) {
        return false;
    }
    if (!ame_exec_binary_supported(
            impl, op, destination->dtype, source1->dtype, source2->dtype) ||
        !ame_exec_status_prepare(impl, destination->dtype, source1->dtype,
                                 &fp_status, &status)) {
        return false;
    }
    if (!ame_exec_m_result_init(layout, destination, result_backing,
                                result_bytes, &staged_destination)) {
        return false;
    }

    if ((impl == AME_IMPL_INT_BINARY &&
         ame_binary_int_fast(op, layout, source_mregs, &staged_destination,
                             source1, source2, result_backing)) ||
        (impl == AME_IMPL_FP_BINARY &&
         ame_binary_fp_fast(op, layout, source_mregs, &staged_destination,
                            source1, source2, result_backing, status))) {
        *flags = (AmeExecFlags) {
            .fflags = status ? riscv_ame_fp_status_flags(status) : 0,
        };
        return true;
    }

    fp = riscv_ame_impl_is_fp(impl);
    for (element = 0; element < destination->total_elements; element++) {
        AmeValue left = riscv_ame_m_read(layout, source_mregs, source1,
                                        element);
        AmeValue right = riscv_ame_m_read(layout, source_mregs, source2,
                                         element);
        AmeExecResult executed;

        if (impl == AME_IMPL_SELECT) {
            ame_exec_select(
                op, destination->dtype, source1->dtype, &left,
                &right, NULL, &executed, status);
        } else if (fp) {
            riscv_ame_exec_fp_binary(
                op, destination->dtype, source1->dtype, &left,
                source2->dtype, &right, &executed, status);
        } else {
            riscv_ame_exec_int_binary(
                op, destination->dtype, source1->dtype, &left,
                source2->dtype, &right, &executed);
        }
        riscv_ame_m_write(layout, result_backing, &staged_destination,
                          element, &executed.value);
        accrued.fflags |= executed.fflags;
        accrued.saturated |= executed.saturated;
    }

    *flags = accrued;
    return true;
}

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
                                    AmeExecFlags *flags)
{
    AmeMView staged_destination;
    AmeExecFlags accrued = { 0 };
    float_status fp_status;
    float_status *status;
    size_t element;
    bool fp;
    const AmeDTypeDesc *left_dtype, *right_dtype;

    if (layout == NULL || source_mregs == NULL || source == NULL ||
        scalar == NULL || flags == NULL || destination == NULL) {
        return false;
    }
    if (destination->total_elements != source->total_elements) {
        return false;
    }
    left_dtype = scalar_first ? scalar_dtype : source->dtype;
    right_dtype = scalar_first ? source->dtype : scalar_dtype;
    if (!ame_exec_binary_supported(impl, op, destination->dtype,
                                   left_dtype, right_dtype) ||
        !ame_exec_status_prepare(impl, destination->dtype, left_dtype,
                                 &fp_status, &status)) {
        return false;
    }
    if (!ame_exec_m_result_init(layout, destination, result_backing,
                                result_bytes, &staged_destination)) {
        return false;
    }

    fp = riscv_ame_impl_is_fp(impl);
    for (element = 0; element < destination->total_elements; element++) {
        AmeValue tensor = riscv_ame_m_read(layout, source_mregs, source,
                                          element);
        const AmeValue *left = scalar_first ? scalar : &tensor;
        const AmeValue *right = scalar_first ? &tensor : scalar;
        AmeExecResult executed;

        if (impl == AME_IMPL_SELECT) {
            ame_exec_select(
                op, destination->dtype, left_dtype, left, right,
                NULL, &executed, status);
        } else if (fp) {
            riscv_ame_exec_fp_binary(
                op, destination->dtype, left_dtype, left, right_dtype, right,
                &executed, status);
        } else {
            riscv_ame_exec_int_binary(
                op, destination->dtype, left_dtype, left, right_dtype, right,
                &executed);
        }
        riscv_ame_m_write(layout, result_backing, &staged_destination,
                          element, &executed.value);
        accrued.fflags |= executed.fflags;
        accrued.saturated |= executed.saturated;
    }

    *flags = accrued;
    return true;
}

bool riscv_ame_exec_m_ternary(AmeImplClass impl, AmeOperation op,
                              const RISCVAMELayout *layout,
                              const uint8_t *source_mregs,
                              const AmeMView *destination,
                              const AmeMView *source1,
                              const AmeMView *source2,
                              uint8_t *result_backing,
                              size_t result_bytes,
                              AmeExecFlags *flags)
{
    AmeMView staged_destination;
    AmeExecFlags accrued = { 0 };
    float_status fp_status;
    float_status *status;
    size_t element;
    bool fp_scale;

    if (layout == NULL || source_mregs == NULL || source1 == NULL ||
        source2 == NULL || flags == NULL || destination == NULL) {
        return false;
    }
    if (destination->total_elements != source1->total_elements ||
        destination->total_elements != source2->total_elements) {
        return false;
    }
    if (!ame_exec_ternary_supported(
            impl, op, destination->dtype, source1->dtype, source2->dtype) ||
        !ame_exec_status_prepare(impl, destination->dtype, source1->dtype,
                                 &fp_status, &status)) {
        return false;
    }
    if (!ame_exec_m_result_init(layout, destination, result_backing,
                                result_bytes, &staged_destination)) {
        return false;
    }

    fp_scale = op == AME_OP_LDEXPACC || op == AME_OP_RDEXPACC;
    for (element = 0; element < destination->total_elements; element++) {
        AmeValue left = riscv_ame_m_read(layout, source_mregs, source1,
                                        element);
        AmeValue right = riscv_ame_m_read(layout, source_mregs, source2,
                                         element);
        AmeValue current = riscv_ame_m_read(layout, source_mregs,
                                           destination, element);
        AmeExecResult executed;

        if (impl == AME_IMPL_SELECT) {
            ame_exec_select(
                op, destination->dtype, source1->dtype, &left,
                &right, &current, &executed, status);
        } else if (impl == AME_IMPL_INT_TERNARY) {
            riscv_ame_exec_int_ternary(
                op, destination->dtype, source1->dtype, &left,
                source2->dtype, &right, &current, &executed);
        } else if (fp_scale) {
            riscv_ame_exec_fp_scale_acc(
                op, destination->dtype, source1->dtype, &left,
                source2->dtype, &right, &current, &executed, status);
        } else {
            riscv_ame_exec_fp_ternary(
                op, destination->dtype, source1->dtype, &left,
                source2->dtype, &right, &current, &executed, status);
        }
        riscv_ame_m_write(layout, result_backing, &staged_destination,
                          element, &executed.value);
        accrued.fflags |= executed.fflags;
        accrued.saturated |= executed.saturated;
    }

    *flags = accrued;
    return true;
}

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
                                     AmeExecFlags *flags)
{
    AmeMView staged_destination;
    AmeExecFlags accrued = { 0 };
    float_status fp_status;
    float_status *status;
    size_t element;
    bool fp_scale;
    const AmeDTypeDesc *left_dtype, *right_dtype;

    if (layout == NULL || source_mregs == NULL || source == NULL ||
        scalar == NULL || flags == NULL || destination == NULL) {
        return false;
    }
    if (destination->total_elements != source->total_elements) {
        return false;
    }
    left_dtype = scalar_first ? scalar_dtype : source->dtype;
    right_dtype = scalar_first ? source->dtype : scalar_dtype;
    if (!ame_exec_ternary_supported(impl, op, destination->dtype,
                                    left_dtype, right_dtype) ||
        !ame_exec_status_prepare(impl, destination->dtype, left_dtype,
                                 &fp_status, &status)) {
        return false;
    }
    if (!ame_exec_m_result_init(layout, destination, result_backing,
                                result_bytes, &staged_destination)) {
        return false;
    }

    fp_scale = op == AME_OP_LDEXPACC || op == AME_OP_RDEXPACC;
    for (element = 0; element < destination->total_elements; element++) {
        AmeValue tensor = riscv_ame_m_read(layout, source_mregs, source,
                                          element);
        AmeValue current = riscv_ame_m_read(layout, source_mregs,
                                           destination, element);
        const AmeValue *left = scalar_first ? scalar : &tensor;
        const AmeValue *right = scalar_first ? &tensor : scalar;
        AmeExecResult executed;

        if (impl == AME_IMPL_SELECT) {
            ame_exec_select(
                op, destination->dtype, left_dtype, left, right,
                &current, &executed, status);
        } else if (impl == AME_IMPL_INT_TERNARY) {
            riscv_ame_exec_int_ternary(
                op, destination->dtype, left_dtype, left, right_dtype, right,
                &current, &executed);
        } else if (fp_scale) {
            riscv_ame_exec_fp_scale_acc(
                op, destination->dtype, left_dtype, left, right_dtype, right,
                &current, &executed, status);
        } else {
            riscv_ame_exec_fp_ternary(
                op, destination->dtype, left_dtype, left, right_dtype, right,
                &current, &executed, status);
        }
        riscv_ame_m_write(layout, result_backing, &staged_destination,
                          element, &executed.value);
        accrued.fflags |= executed.fflags;
        accrued.saturated |= executed.saturated;
    }

    *flags = accrued;
    return true;
}
