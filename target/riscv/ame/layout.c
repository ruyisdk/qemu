/*
 * RISC-V Attached Matrix Extension (AME) tensor layout model
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/bitops.h"
#include "qemu/bswap.h"
#include "qemu/host-utils.h"
#include "target/riscv/ame/layout.h"

G_STATIC_ASSERT(RV_AME_N_MAX <= UINT32_MAX / RV_AME_N_MAX /
                MAX_CONST(RV_AME_UDSZ_MAX, AME_VALUE_BITS) /
                MAX_CONST(RV_AME_M_REGS_MAX, RV_AME_ACC_REGS_MAX));

static bool ame_size_mul(size_t left, size_t right, size_t *result)
{
    size_t value;

    if (__builtin_mul_overflow(left, right, &value)) {
        return false;
    }
    *result = value;
    return true;
}

static bool ame_bits_to_bytes(size_t bits, size_t *bytes)
{
    if (bits == 0 || bits > SIZE_MAX - (CHAR_BIT - 1)) {
        return false;
    }
    *bytes = DIV_ROUND_UP(bits, CHAR_BIT);
    return true;
}

bool riscv_ame_compute_layout(uint64_t n, uint64_t u, uint32_t num_mregs,
                              uint32_t num_accregs,
                              const AmeCapabilities *capabilities,
                              RISCVAMELayout *layout, Error **errp)
{
    RISCVAMELayout geometry;
    uint32_t elements, mreg_bits, acc_bits;
    uint16_t max_dtype_bits = 0;
    size_t i;

    if (layout == NULL) {
        error_setg(errp, "Ztt layout output must be specified");
        return false;
    }
    if (!is_power_of_2(n) || n > RV_AME_N_MAX) {
        error_setg(errp, "Ztt N must be a power of two in [1, %u]",
                   RV_AME_N_MAX);
        return false;
    }
    if (!is_power_of_2(u) || u > RV_AME_UDSZ_MAX) {
        error_setg(errp, "Ztt U must be a power of two in [1, %u]",
                   RV_AME_UDSZ_MAX);
        return false;
    }
    if (num_mregs != 16 && num_mregs != RV_AME_M_REGS_MAX) {
        error_setg(errp, "Ztt NM must be 16 or %u", RV_AME_M_REGS_MAX);
        return false;
    }
    if (!is_power_of_2(num_accregs) ||
        num_accregs > RV_AME_ACC_REGS_MAX) {
        error_setg(errp, "Ztt NA must be a power of two in [1, %u]",
                   RV_AME_ACC_REGS_MAX);
        return false;
    }
    if (capabilities == NULL ||
        riscv_ame_dtype_set_empty(&capabilities->dtypes)) {
        error_setg(errp, "Ztt capability set must be initialized");
        return false;
    }

    geometry = (RISCVAMELayout) {
        .n = n,
        .u = u,
        .num_mregs = num_mregs,
        .num_accregs = num_accregs,
    };
    for (i = 0; i < AME_DTYPE_COUNT; i++) {
        const AmeDTypeDesc *dtype = &riscv_ame_dtypes[i];
        AmeDTypeShape shape;

        if (!riscv_ame_dtype_set_has(&capabilities->dtypes, dtype->id)) {
            continue;
        }
        if (!riscv_ame_dtype_shape(&geometry, dtype, &shape)) {
            error_setg(errp,
                       "Ztt datatype 0x%08" PRIx32
                       " is incompatible with U=%" PRIu64,
                       dtype->encoding, u);
            return false;
        }
        max_dtype_bits = MAX(max_dtype_bits, dtype->bits);
    }
    g_assert(max_dtype_bits != 0);

    elements = geometry.n * geometry.n;
    mreg_bits = elements * geometry.u;
    if (mreg_bits % CHAR_BIT != 0) {
        error_setg(errp, "Ztt N squared times U must be byte aligned");
        return false;
    }
    acc_bits = elements * max_dtype_bits;
    geometry.mreg_bytes = mreg_bits / CHAR_BIT;
    geometry.mregs_bytes = geometry.mreg_bytes * num_mregs;
    geometry.acc_bytes = DIV_ROUND_UP(acc_bits, CHAR_BIT);
    geometry.accs_bytes = geometry.acc_bytes * num_accregs;
    *layout = geometry;
    return true;
}

bool riscv_ame_layout_equal(const RISCVAMELayout *left,
                            const RISCVAMELayout *right)
{
    return left == right ||
           (left != NULL && right != NULL &&
            left->n == right->n && left->u == right->u &&
            left->num_mregs == right->num_mregs &&
            left->num_accregs == right->num_accregs &&
            left->mreg_bytes == right->mreg_bytes &&
            left->mregs_bytes == right->mregs_bytes &&
            left->acc_bytes == right->acc_bytes &&
            left->accs_bytes == right->accs_bytes);
}

bool riscv_ame_dtype_shape(const RISCVAMELayout *layout,
                           const AmeDTypeDesc *dtype,
                           AmeDTypeShape *shape)
{
    uint32_t pack_factor, group_size;

    if (layout == NULL || dtype == NULL || shape == NULL || layout->u == 0 ||
        dtype->bits < AME_DTYPE_BITS_MIN || dtype->bits > AME_VALUE_BITS ||
        !is_power_of_2(dtype->bits)) {
        return false;
    }
    if (dtype->bits < layout->u) {
        if (layout->u % dtype->bits != 0) {
            return false;
        }
        pack_factor = layout->u / dtype->bits;
        group_size = 1;
    } else {
        if (dtype->bits % layout->u != 0) {
            return false;
        }
        pack_factor = 1;
        group_size = dtype->bits / layout->u;
    }
    *shape = (AmeDTypeShape) {
        .pack_factor = pack_factor,
        .group_size = group_size,
    };
    return true;
}

bool riscv_ame_n_sq(const RISCVAMELayout *layout,
                    const AmeDTypeDesc *const *dtypes, size_t num_dtypes,
                    uint32_t *n_sq)
{
    uint32_t result = 1;
    size_t i;

    if (layout == NULL || dtypes == NULL || num_dtypes == 0 || n_sq == NULL) {
        return false;
    }
    for (i = 0; i < num_dtypes; i++) {
        AmeDTypeShape shape;

        if (!riscv_ame_dtype_shape(layout, dtypes[i], &shape)) {
            return false;
        }
        result = MAX(result, shape.pack_factor);
    }
    *n_sq = result;
    return true;
}

bool riscv_ame_m_view_init(const RISCVAMELayout *layout, AmeMView *view,
                           uint32_t base, const AmeDTypeDesc *dtype,
                           uint32_t n_sq)
{
    AmeDTypeShape shape;
    uint32_t span, elements_per_square, total_elements;

    if (layout == NULL || view == NULL || base >= layout->num_mregs ||
        !is_power_of_2(n_sq) ||
        !riscv_ame_dtype_shape(layout, dtype, &shape) ||
        n_sq % shape.pack_factor != 0) {
        return false;
    }
    if (shape.pack_factor > 1) {
        span = n_sq / shape.pack_factor;
    } else if (umul32_overflow(n_sq, shape.group_size, &span)) {
        return false;
    }
    if (span == 0 || base % shape.group_size != 0 ||
        span > layout->num_mregs - base ||
        umul32_overflow(layout->n, layout->n, &elements_per_square) ||
        umul32_overflow(elements_per_square, n_sq, &total_elements)) {
        return false;
    }

    *view = (AmeMView) {
        .dtype = dtype,
        .base = base,
        .pack_factor = shape.pack_factor,
        .group_size = shape.group_size,
        .n_sq = n_sq,
        .span = span,
        .elements_per_square = elements_per_square,
        .total_elements = total_elements,
    };
    return true;
}

size_t riscv_ame_m_view_bytes(const RISCVAMELayout *layout,
                              const AmeMView *view)
{
    g_assert(layout != NULL && view != NULL);
    g_assert(view->span <= SIZE_MAX / layout->mreg_bytes);
    return view->span * layout->mreg_bytes;
}

static size_t ame_m_element_bit_offset(const RISCVAMELayout *layout,
                                       const AmeMView *view,
                                       size_t element)
{
    g_assert(element < view->total_elements);
    /* View validation makes both packed and grouped squares contiguous. */
    return view->base * layout->mreg_bytes * CHAR_BIT +
           element * view->dtype->bits;
}

static inline AmeValue ame_bits_load(const uint8_t *bytes, size_t bit_offset,
                                     unsigned int bits)
{
    const uint8_t *ptr = bytes + bit_offset / CHAR_BIT;

    if (bits == 4) {
        return (AmeValue) {
            .lo = extract8(*ptr, bit_offset % CHAR_BIT, 4),
        };
    }

    g_assert(bit_offset % CHAR_BIT == 0);
    switch (bits) {
    case 8:
        return (AmeValue) { .lo = *ptr };
    case 16:
        return (AmeValue) { .lo = lduw_le_p(ptr) };
    case 32:
        return (AmeValue) { .lo = (uint32_t)ldl_le_p(ptr) };
    case AME_VALUE_WORD_BITS:
        return (AmeValue) { .lo = ldq_le_p(ptr) };
    case AME_VALUE_BITS:
        return (AmeValue) {
            .lo = ldq_le_p(ptr),
            .hi = ldq_le_p(ptr + sizeof(uint64_t)),
        };
    default:
        g_assert_not_reached();
    }
}

static inline void ame_bits_store(uint8_t *bytes, size_t bit_offset,
                                  unsigned int bits, const AmeValue *value)
{
    uint8_t *ptr = bytes + bit_offset / CHAR_BIT;

    if (bits == 4) {
        *ptr = deposit32(*ptr, bit_offset % CHAR_BIT, 4, value->lo);
        return;
    }

    g_assert(bit_offset % CHAR_BIT == 0);
    switch (bits) {
    case 8:
        *ptr = value->lo;
        break;
    case 16:
        stw_le_p(ptr, value->lo);
        break;
    case 32:
        stl_le_p(ptr, value->lo);
        break;
    case AME_VALUE_WORD_BITS:
        stq_le_p(ptr, value->lo);
        break;
    case AME_VALUE_BITS:
        stq_le_p(ptr, value->lo);
        stq_le_p(ptr + sizeof(uint64_t), value->hi);
        break;
    default:
        g_assert_not_reached();
    }
}

AmeValue riscv_ame_m_read(const RISCVAMELayout *layout,
                          const uint8_t *mregs, const AmeMView *view,
                          size_t element)
{
    g_assert(layout != NULL && mregs != NULL && view != NULL);
    return ame_bits_load(mregs,
                         ame_m_element_bit_offset(layout, view, element),
                         view->dtype->bits);
}

void riscv_ame_m_write(const RISCVAMELayout *layout, uint8_t *mregs,
                       const AmeMView *view, size_t element,
                       const AmeValue *value)
{
    g_assert(layout != NULL && mregs != NULL && view != NULL && value != NULL);
    ame_bits_store(mregs, ame_m_element_bit_offset(layout, view, element),
                   view->dtype->bits, value);
}

bool riscv_ame_acc_view_init(const RISCVAMELayout *layout, AmeAccView *view,
                             uint32_t index, const AmeDTypeDesc *dtype)
{
    AmeDTypeShape shape;
    uint32_t elements_per_square, bits;
    size_t bytes;

    if (layout == NULL || view == NULL || index >= layout->num_accregs ||
        !riscv_ame_dtype_shape(layout, dtype, &shape) ||
        umul32_overflow(layout->n, layout->n, &elements_per_square) ||
        umul32_overflow(elements_per_square, dtype->bits, &bits) ||
        !ame_bits_to_bytes(bits, &bytes) || bytes > layout->acc_bytes) {
        return false;
    }

    *view = (AmeAccView) {
        .dtype = dtype,
        .index = index,
        .elements_per_square = elements_per_square,
    };
    return true;
}

static size_t ame_acc_element_bit_offset(const RISCVAMELayout *layout,
                                         const AmeAccView *view,
                                         size_t element)
{
    g_assert(element < view->elements_per_square);
    return view->index * layout->acc_bytes * CHAR_BIT +
           element * view->dtype->bits;
}

AmeValue riscv_ame_acc_read(const RISCVAMELayout *layout,
                            const uint8_t *accs, const AmeAccView *view,
                            size_t element)
{
    g_assert(layout != NULL && accs != NULL && view != NULL);
    return ame_bits_load(accs,
                         ame_acc_element_bit_offset(layout, view, element),
                         view->dtype->bits);
}

void riscv_ame_acc_write(const RISCVAMELayout *layout, uint8_t *accs,
                         const AmeAccView *view, size_t element,
                         const AmeValue *value)
{
    g_assert(layout != NULL && accs != NULL && view != NULL && value != NULL);
    ame_bits_store(accs, ame_acc_element_bit_offset(layout, view, element),
                   view->dtype->bits, value);
}

static AmeMView ame_staged_m_view(const RISCVAMELayout *layout,
                                  const AmeMView *destination,
                                  size_t scratch_bytes)
{
    AmeMView staged = *destination;

    g_assert(riscv_ame_m_view_bytes(layout, destination) <= scratch_bytes);
    staged.base = 0;
    return staged;
}

void riscv_ame_move_m_to_m(const RISCVAMELayout *layout, uint8_t *mregs,
                           const AmeMView *destination,
                           const AmeMView *source)
{
    size_t bytes;

    g_assert(layout != NULL && mregs != NULL && destination != NULL);
    g_assert(source != NULL);
    g_assert(destination->dtype->id == source->dtype->id);
    g_assert(destination->n_sq == source->n_sq);
    bytes = riscv_ame_m_view_bytes(layout, destination);
    g_assert(bytes == riscv_ame_m_view_bytes(layout, source));

    memmove(mregs + destination->base * layout->mreg_bytes,
            mregs + source->base * layout->mreg_bytes, bytes);
}

void riscv_ame_move_acc_to_m(const RISCVAMELayout *layout, uint8_t *mregs,
                             const AmeMView *destination,
                             const uint8_t *accs, uint32_t acc_base,
                             uint32_t count, uint8_t *scratch,
                             size_t scratch_bytes)
{
    AmeMView staged;
    size_t element, bytes;
    uint32_t square;

    g_assert(layout != NULL && mregs != NULL && destination != NULL);
    g_assert(accs != NULL && scratch != NULL && count > 0);
    g_assert(count == destination->n_sq);
    g_assert(acc_base < layout->num_accregs);
    g_assert(count <= layout->num_accregs - acc_base);
    staged = ame_staged_m_view(layout, destination, scratch_bytes);
    bytes = riscv_ame_m_view_bytes(layout, destination);
    memset(scratch, 0, bytes);

    for (square = 0; square < count; square++) {
        AmeAccView source;
        size_t square_bits;

        g_assert(riscv_ame_acc_view_init(layout, &source, acc_base + square,
                                         destination->dtype));
        square_bits = source.elements_per_square * source.dtype->bits;
        if (square_bits % CHAR_BIT == 0) {
            size_t square_bytes = square_bits / CHAR_BIT;

            memcpy(scratch + square * square_bytes,
                   accs + (acc_base + square) * layout->acc_bytes,
                   square_bytes);
            continue;
        }
        for (element = 0; element < source.elements_per_square; element++) {
            AmeValue value = riscv_ame_acc_read(layout, accs, &source,
                                                element);

            riscv_ame_m_write(
                layout, scratch, &staged,
                square * source.elements_per_square + element, &value);
        }
    }
    memcpy(mregs + destination->base * layout->mreg_bytes, scratch, bytes);
}

void riscv_ame_move_m_to_acc(const RISCVAMELayout *layout, uint8_t *accs,
                             uint32_t acc_base, uint32_t count,
                             const uint8_t *mregs, const AmeMView *source,
                             uint8_t *scratch, size_t scratch_bytes)
{
    size_t element, bytes;
    uint32_t square;

    g_assert(layout != NULL && accs != NULL && mregs != NULL);
    g_assert(source != NULL && scratch != NULL && count > 0);
    g_assert(count == source->n_sq);
    g_assert(acc_base < layout->num_accregs);
    g_assert(count <= layout->num_accregs - acc_base);
    g_assert(count <= SIZE_MAX / layout->acc_bytes);
    bytes = count * layout->acc_bytes;
    g_assert(bytes <= scratch_bytes);
    memcpy(scratch, accs + acc_base * layout->acc_bytes, bytes);

    for (square = 0; square < count; square++) {
        AmeAccView destination;
        size_t square_bits;

        g_assert(riscv_ame_acc_view_init(layout, &destination, square,
                                         source->dtype));
        square_bits = destination.elements_per_square * destination.dtype->bits;
        if (square_bits % CHAR_BIT == 0) {
            size_t square_bytes = square_bits / CHAR_BIT;

            memcpy(scratch + square * layout->acc_bytes,
                   mregs + source->base * layout->mreg_bytes +
                   square * square_bytes, square_bytes);
            continue;
        }
        for (element = 0; element < destination.elements_per_square;
             element++) {
            AmeValue value = riscv_ame_m_read(
                layout, mregs, source,
                square * destination.elements_per_square + element);

            riscv_ame_acc_write(layout, scratch, &destination, element,
                                &value);
        }
    }
    memcpy(accs + acc_base * layout->acc_bytes, scratch, bytes);
}

bool riscv_ame_mem_plan_init(const RISCVAMELayout *layout,
                             const AmeDTypeDesc *dtype, AmeMemFormat format,
                             AmeMemPlan *plan)
{
    AmeDTypeShape shape;
    uint32_t num_segments;
    size_t segment_bits, segment_bytes, total_bytes;

    if (layout == NULL || plan == NULL || format >= AME_MEM_FORMAT_MAX) {
        return false;
    }
    if (format == AME_MEM_WHOLE_REGISTER) {
        *plan = (AmeMemPlan) {
            .format = format,
            .num_segments = 1,
            .pack_factor = 1,
            .element_bits = layout->u,
            .segment_bits = layout->mreg_bytes * CHAR_BIT,
            .segment_bytes = layout->mreg_bytes,
            .total_bytes = layout->mreg_bytes,
            .alignment = DIV_ROUND_UP(layout->u, CHAR_BIT),
        };
        return true;
    }
    if (!riscv_ame_dtype_shape(layout, dtype, &shape)) {
        return false;
    }

    switch (format) {
    case AME_MEM_ROW_MAJOR:
    case AME_MEM_COLUMN_MAJOR:
        num_segments = 1;
        if (!ame_size_mul(shape.pack_factor, layout->n, &segment_bits) ||
            !ame_size_mul(segment_bits, layout->n, &segment_bits) ||
            !ame_size_mul(segment_bits, dtype->bits, &segment_bits)) {
            return false;
        }
        break;
    case AME_MEM_STRIDED:
        num_segments = layout->n;
        if (!ame_size_mul(shape.pack_factor, layout->n, &segment_bits) ||
            !ame_size_mul(segment_bits, dtype->bits, &segment_bits)) {
            return false;
        }
        break;
    case AME_MEM_TRANSPOSED_STRIDED:
        if (umul32_overflow(shape.pack_factor, layout->n, &num_segments) ||
            !ame_size_mul(layout->n, dtype->bits, &segment_bits)) {
            return false;
        }
        break;
    case AME_MEM_WHOLE_REGISTER:
    case AME_MEM_FORMAT_MAX:
    default:
        g_assert_not_reached();
    }

    if (!ame_bits_to_bytes(segment_bits, &segment_bytes) ||
        !ame_size_mul(num_segments, segment_bytes, &total_bytes)) {
        return false;
    }
    *plan = (AmeMemPlan) {
        .format = format,
        .num_segments = num_segments,
        .pack_factor = shape.pack_factor,
        .element_bits = dtype->bits,
        .segment_bits = segment_bits,
        .segment_bytes = segment_bytes,
        .total_bytes = total_bytes,
        .alignment = DIV_ROUND_UP(dtype->bits, CHAR_BIT),
    };
    return true;
}

uint64_t riscv_ame_xlen_madd(uint64_t base, uint64_t index, uint64_t stride,
                             unsigned int xlen)
{
    g_assert(xlen == 32 || xlen == 64);
    if (xlen == 32) {
        return (uint32_t)base + (uint32_t)index * (uint32_t)stride;
    }
    return base + index * stride;
}

static bool ame_mem_plan_matches_view(const RISCVAMELayout *layout,
                                      const AmeMView *view,
                                      const AmeMemPlan *plan)
{
    AmeMemPlan expected;

    if (view == NULL || plan == NULL ||
        plan->format <= AME_MEM_WHOLE_REGISTER ||
        plan->format >= AME_MEM_FORMAT_MAX ||
        view->n_sq != view->pack_factor ||
        !riscv_ame_mem_plan_init(layout, view->dtype, plan->format,
                                 &expected)) {
        return false;
    }
    return plan->num_segments == expected.num_segments &&
           plan->pack_factor == expected.pack_factor &&
           plan->element_bits == expected.element_bits &&
           plan->segment_bits == expected.segment_bits &&
           plan->segment_bytes == expected.segment_bytes &&
           plan->total_bytes == expected.total_bytes &&
           plan->alignment == expected.alignment;
}

static size_t ame_memory_element_bit_offset(const RISCVAMELayout *layout,
                                            const AmeMView *view,
                                            const AmeMemPlan *plan,
                                            size_t element)
{
    size_t square = element / view->elements_per_square;
    size_t local = element % view->elements_per_square;
    size_t row = local / layout->n;
    size_t column = local % layout->n;
    size_t memory_element;

    switch (plan->format) {
    case AME_MEM_ROW_MAJOR:
        memory_element = square * view->elements_per_square + local;
        return memory_element * view->dtype->bits;
    case AME_MEM_COLUMN_MAJOR:
        memory_element = square * view->elements_per_square +
                         column * layout->n + row;
        return memory_element * view->dtype->bits;
    case AME_MEM_STRIDED:
        return row * plan->segment_bytes * CHAR_BIT +
               (square * layout->n + column) * view->dtype->bits;
    case AME_MEM_TRANSPOSED_STRIDED:
        return (square * layout->n + column) *
                   plan->segment_bytes * CHAR_BIT +
               row * view->dtype->bits;
    case AME_MEM_WHOLE_REGISTER:
    case AME_MEM_FORMAT_MAX:
    default:
        g_assert_not_reached();
    }
}

bool riscv_ame_m_import(const RISCVAMELayout *layout, const AmeMView *view,
                        uint8_t *mregs, const AmeMemPlan *plan,
                        const uint8_t *memory)
{
    size_t element;

    if (layout == NULL || mregs == NULL || memory == NULL ||
        !ame_mem_plan_matches_view(layout, view, plan)) {
        return false;
    }
    if (plan->format == AME_MEM_ROW_MAJOR) {
        memcpy(mregs + view->base * layout->mreg_bytes, memory,
               plan->total_bytes);
        return true;
    }
    for (element = 0; element < view->total_elements; element++) {
        AmeValue value = ame_bits_load(
            memory,
            ame_memory_element_bit_offset(layout, view, plan, element),
            view->dtype->bits);

        riscv_ame_m_write(layout, mregs, view, element, &value);
    }
    return true;
}

bool riscv_ame_m_export(const RISCVAMELayout *layout, const AmeMView *view,
                        const uint8_t *mregs, const AmeMemPlan *plan,
                        uint8_t *memory)
{
    size_t element;

    if (layout == NULL || mregs == NULL || memory == NULL ||
        !ame_mem_plan_matches_view(layout, view, plan)) {
        return false;
    }
    if (plan->format == AME_MEM_ROW_MAJOR) {
        memcpy(memory, mregs + view->base * layout->mreg_bytes,
               plan->total_bytes);
        return true;
    }
    memset(memory, 0, plan->total_bytes);
    for (element = 0; element < view->total_elements; element++) {
        AmeValue value = riscv_ame_m_read(layout, mregs, view, element);

        ame_bits_store(
            memory,
            ame_memory_element_bit_offset(layout, view, plan, element),
            view->dtype->bits, &value);
    }
    return true;
}
