/*
 * RISC-V Attached Matrix Extension execution tests
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qapi/error.h"
#include "fpu/softfloat.h"
#include "target/riscv/ame/exec.h"
#include "target/riscv/ame/fp.h"
#include "target/riscv/ame/int.h"

static const AmeDTypeDesc *dtype(uint32_t encoding)
{
    const AmeDTypeDesc *desc = riscv_ame_dtype_by_encoding(encoding);

    g_assert_nonnull(desc);
    return desc;
}

static uint32_t fp32_bits(float value)
{
    union {
        float value;
        uint32_t bits;
    } converted = { .value = value };

    G_STATIC_ASSERT(sizeof(float) == sizeof(uint32_t));
    return converted.bits;
}

static RISCVAMELayout test_layout_geometry(uint64_t n, uint64_t u)
{
    RISCVAMELayout layout;

    g_assert_true(riscv_ame_compute_layout(
        n, u, RV_AME_M_REGS_MAX, RV_AME_ACC_REGS_MAX,
        riscv_ame_maximum_capabilities(), &layout, &error_abort));
    return layout;
}

static RISCVAMELayout test_layout(void)
{
    return test_layout_geometry(2, 8);
}

static void init_views(const RISCVAMELayout *layout,
                       AmeMView *destination, uint32_t destination_base,
                       const AmeDTypeDesc *destination_dtype,
                       AmeMView *source, uint32_t source_base,
                       const AmeDTypeDesc *source_dtype)
{
    const AmeDTypeDesc *dtypes[] = {
        destination_dtype,
        source_dtype,
    };
    uint32_t n_sq;

    g_assert_true(riscv_ame_n_sq(layout, dtypes, ARRAY_SIZE(dtypes),
                                &n_sq));
    g_assert_true(riscv_ame_m_view_init(layout, destination,
                                       destination_base,
                                       destination_dtype, n_sq));
    g_assert_true(riscv_ame_m_view_init(layout, source, source_base,
                                       source_dtype, n_sq));
    g_assert_cmpuint(destination->total_elements, ==,
                     source->total_elements);
}

static AmeMView staged_view(const AmeMView *view)
{
    AmeMView staged = *view;

    staged.base = 0;
    return staged;
}

static void test_convert_scalar_dispatch(void)
{
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    const AmeDTypeDesc *uint4_sat = dtype(UINT32_C(0x20000004));
    const AmeDTypeDesc *fp16 = dtype(UINT32_C(0x14300110));
    AmeValue source = { .lo = UINT8_C(0xff) };
    AmeExecResult result;

    g_assert_true(riscv_ame_exec_convert(AME_IMPL_INT_CONVERT,
                                         uint4_sat, int8, &source,
                                         &result, NULL));
    g_assert_cmphex(result.value.lo, ==, 0);
    g_assert_true(result.saturated);
    g_assert_false(riscv_ame_exec_convert(AME_IMPL_FP_CONVERT,
                                          uint4_sat, int8, &source,
                                          &result, NULL));
    g_assert_false(riscv_ame_exec_convert(AME_IMPL_INT_CONVERT,
                                          fp16, int8, &source, &result,
                                          NULL));
}

static void test_checked_execution_rejects(void)
{
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    const AmeDTypeDesc *fp32 = dtype(UINT32_C(0x20300120));
    AmeDTypeDesc malformed = *fp32;
    const struct {
        unsigned arity;
        AmeImplClass impl;
        AmeOperation op;
        const AmeDTypeDesc *d, *a, *b;
    } invalid[] = {
        { 1, AME_IMPL_FP_UNARY, AME_OP_COS, fp32, int8 },
        { 2, AME_IMPL_FP_BINARY, AME_OP_LOG2SUB, int8, fp32, fp32 },
        { 0, AME_IMPL_FP_CONVERT, AME_OP_CONV, &malformed, fp32 },
        { 3, AME_IMPL_FP_TERNARY, AME_OP_ADD, fp32, fp32, fp32 },
        { 2, AME_IMPL_FP_SCALE, AME_OP_ADD, fp32, fp32, int8 },
        { 2, AME_IMPL_FP_BINARY, AME_OP_AND, fp32, fp32, fp32 },
        { 1, AME_IMPL_FP_UNARY, AME_OP_CONV, fp32, fp32 },
        { 1, AME_IMPL_INT_UNARY, AME_OP_CONV, int8, int8 },
        { 2, AME_IMPL_FP_BINARY, AME_OP_MULACC, fp32, fp32, fp32 },
        { 2, AME_IMPL_INT_BINARY, AME_OP_MULACC, int8, int8, int8 },
        { 3, AME_IMPL_INT_TERNARY, AME_OP_ABS, int8, int8, int8 },
        { 2, AME_IMPL_FP_REDUCE, AME_OP_ADD, fp32, fp32, fp32 },
        { 2, AME_IMPL_FP_SCATTER, AME_OP_ADD, fp32, fp32, fp32 },
        { 3, AME_IMPL_FP_MATMUL, AME_OP_MULACC, fp32, fp32, fp32 },
        { 2, AME_IMPL_FP_COMPARE, AME_OP_ADD, int8, fp32, fp32 },
        { 2, AME_IMPL_SELECT, AME_OP_SEL, int8, &malformed, int8 },
        { 2, AME_IMPL_INT_BINARY, AME_OP_ADD, int8, int8, NULL },
        { 3, AME_IMPL_INT_TERNARY, AME_OP_MULACC, int8, int8, NULL },
    };
    AmeValue value = { .lo = 1 };
    AmeExecResult result;
    uint8_t original_result[sizeof(result)];
    float_status status;
    uint8_t original_status[sizeof(status)];

    malformed.bits++;
    memset(&result, 0xa5, sizeof(result));
    memcpy(original_result, &result, sizeof(result));
    g_assert_true(riscv_ame_fp_status_prepare(&status, fp32));
    float_raise(float_flag_divbyzero, &status);
    memcpy(original_status, &status, sizeof(status));

    for (size_t i = 0; i < ARRAY_SIZE(invalid); i++) {
        bool valid;

        switch (invalid[i].arity) {
        case 0:
            valid = riscv_ame_exec_convert(invalid[i].impl, invalid[i].d,
                invalid[i].a, &value, &result, &status);
            break;
        case 1:
            valid = riscv_ame_exec_unary(invalid[i].impl, invalid[i].op,
                invalid[i].d, invalid[i].a, &value, &result, &status);
            break;
        case 2:
            valid = riscv_ame_exec_binary(invalid[i].impl, invalid[i].op,
                invalid[i].d, invalid[i].a, &value,
                invalid[i].b, &value, &result, &status);
            break;
        case 3:
            valid = riscv_ame_exec_ternary(invalid[i].impl, invalid[i].op,
                invalid[i].d, invalid[i].a, &value,
                invalid[i].b, &value, &value, &result, &status);
            break;
        default:
            g_assert_not_reached();
        }
        g_assert_false(valid);
        g_assert_cmpmem(&result, sizeof(result),
                        original_result, sizeof(original_result));
        g_assert_cmpmem(&status, sizeof(status),
                        original_status, sizeof(original_status));
    }

    /* Each checked entry rejects missing values, outputs and FP state. */
    for (unsigned missing = 0; missing < 5; missing++) {
        const AmeValue *a = missing == 0 ? NULL : &value;
        const AmeValue *b = missing == 1 ? NULL : &value;
        const AmeValue *current = missing == 2 ? NULL : &value;
        AmeExecResult *output = missing == 3 ? NULL : &result;
        float_status *state = missing == 4 ? NULL : &status;

        if (missing == 0 || missing >= 3) {
            g_assert_false(riscv_ame_exec_convert(AME_IMPL_FP_CONVERT,
                fp32, fp32, a, output, state));
            g_assert_false(riscv_ame_exec_unary(AME_IMPL_FP_UNARY,
                AME_OP_ABS, fp32, fp32, a, output, state));
        }
        if (missing != 2) {
            g_assert_false(riscv_ame_exec_binary(AME_IMPL_FP_BINARY,
                AME_OP_ADD, fp32, fp32, a, fp32, b, output, state));
            g_assert_false(riscv_ame_exec_binary(AME_IMPL_SELECT,
                AME_OP_SELGE, int8, fp32, a, int8, b, output, state));
        }
        g_assert_false(riscv_ame_exec_ternary(AME_IMPL_FP_TERNARY,
            AME_OP_MULACC, fp32, fp32, a, fp32, b, current, output, state));
        g_assert_false(riscv_ame_exec_ternary(AME_IMPL_SELECT,
            AME_OP_MUXGE, int8, fp32, a, int8, b, current, output, state));
        g_assert_cmpmem(&result, sizeof(result),
                        original_result, sizeof(original_result));
        g_assert_cmpmem(&status, sizeof(status),
                        original_status, sizeof(original_status));
    }
}

static void test_elementwise_rejects_invalid_requests(void)
{
    RISCVAMELayout layout = test_layout();
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    g_autofree uint8_t *mregs = g_malloc0(layout.mregs_bytes);
    g_autofree uint8_t *result = g_malloc(layout.mregs_bytes);
    AmeMView d, a, b;
    AmeExecFlags rejected_flags = { .fflags = 7, .saturated = true };
    AmeValue scalar_value = { .lo = 1 };

    g_assert_true(riscv_ame_m_view_init(&layout, &d, 0, int8, 1));
    g_assert_true(riscv_ame_m_view_init(&layout, &a, 1, int8, 1));
    g_assert_true(riscv_ame_m_view_init(&layout, &b, 2, int8, 1));

    for (unsigned invalid = 0; invalid < 10; invalid++) {
        const RISCVAMELayout *geometry = invalid == 0 ? NULL : &layout;
        const uint8_t *input = invalid == 1 ? NULL : mregs;
        AmeMView dest = d, source = a;
        const AmeMView *dest_view = invalid == 2 ? NULL : &dest;
        const AmeMView *source_view = invalid == 3 ? NULL : &source;
        uint8_t *output = invalid == 4 ? NULL : result;
        AmeExecFlags flags = { .fflags = 7, .saturated = true };
        AmeExecFlags *output_flags = invalid == 5 ? NULL : &flags;
        size_t bytes = invalid == 6 ? 0 : layout.mregs_bytes;
        AmeValue scalar = { .lo = 1 };

        if (invalid == 7) {
            dest.dtype = NULL;
        } else if (invalid == 8) {
            source.dtype = NULL;
        } else if (invalid == 9) {
            source.total_elements++;
        }
        memset(result, 0xa5, layout.mregs_bytes);
        g_assert_false(riscv_ame_exec_m_convert(
            AME_IMPL_INT_CONVERT, geometry, input, dest_view, source_view,
            output, bytes, output_flags));
        g_assert_false(riscv_ame_exec_m_unary(
            AME_IMPL_INT_UNARY, AME_OP_ABS, geometry, input,
            dest_view, source_view, output, bytes, output_flags));
        g_assert_false(riscv_ame_exec_m_binary(
            AME_IMPL_INT_BINARY, AME_OP_ADD, geometry, input,
            dest_view, source_view, &b, output, bytes, output_flags));
        g_assert_false(riscv_ame_exec_m_binary_scalar(
            AME_IMPL_INT_BINARY, AME_OP_ADD, geometry, input,
            dest_view, source_view, int8, &scalar, false,
            output, bytes, output_flags));
        g_assert_false(riscv_ame_exec_m_ternary(
            AME_IMPL_INT_TERNARY, AME_OP_MULACC, geometry, input,
            dest_view, source_view, &b, output, bytes, output_flags));
        g_assert_false(riscv_ame_exec_m_ternary_scalar(
            AME_IMPL_INT_TERNARY, AME_OP_MULACC, geometry, input,
            dest_view, source_view, int8, &scalar, false,
            output, bytes, output_flags));
        g_assert_cmphex(flags.fflags, ==, 7);
        g_assert_true(flags.saturated);
        for (size_t i = 0; i < layout.mregs_bytes; i++) {
            g_assert_cmphex(result[i], ==, 0xa5);
        }
    }

    b.dtype = NULL;
    g_assert_false(riscv_ame_exec_m_binary(
        AME_IMPL_INT_BINARY, AME_OP_ADD, &layout, mregs,
        &d, &a, &b, result, layout.mregs_bytes, &rejected_flags));
    g_assert_false(riscv_ame_exec_m_ternary(
        AME_IMPL_INT_TERNARY, AME_OP_MULACC, &layout, mregs,
        &d, &a, &b, result, layout.mregs_bytes, &rejected_flags));
    for (unsigned scalar_first = 0; scalar_first < 2; scalar_first++) {
        g_assert_false(riscv_ame_exec_m_binary_scalar(
            AME_IMPL_INT_BINARY, AME_OP_ADD, &layout, mregs, &d, &a,
            NULL, &scalar_value, scalar_first, result, layout.mregs_bytes,
            &rejected_flags));
        g_assert_false(riscv_ame_exec_m_ternary_scalar(
            AME_IMPL_INT_TERNARY, AME_OP_MULACC, &layout, mregs, &d, &a,
            NULL, &scalar_value, scalar_first, result, layout.mregs_bytes,
            &rejected_flags));
    }
    g_assert_cmphex(rejected_flags.fflags, ==, 7);
    g_assert_true(rejected_flags.saturated);
    for (size_t i = 0; i < layout.mregs_bytes; i++) {
        g_assert_cmphex(result[i], ==, 0xa5);
    }
}

static void test_convert_packed_to_unit(void)
{
    RISCVAMELayout layout = test_layout();
    const AmeDTypeDesc *int4 = dtype(UINT32_C(0x40000004));
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    g_autofree uint8_t *mregs = g_malloc0(layout.mregs_bytes);
    g_autofree uint8_t *result = g_malloc0(layout.mregs_bytes);
    AmeMView destination, source, staged;
    AmeExecFlags flags = { .saturated = true };
    size_t element;

    init_views(&layout, &destination, 2, int8, &source, 0, int4);
    for (element = 0; element < source.total_elements; element++) {
        AmeValue value = { .lo = element + 4 };

        riscv_ame_m_write(&layout, mregs, &source, element, &value);
    }

    g_assert_true(riscv_ame_exec_m_convert(
        AME_IMPL_INT_CONVERT, &layout, mregs, &destination, &source,
        result, layout.mregs_bytes, &flags));
    g_assert_false(flags.saturated);
    g_assert_cmphex(flags.fflags, ==, 0);
    staged = staged_view(&destination);
    for (element = 0; element < staged.total_elements; element++) {
        uint8_t raw = element + 4;
        uint8_t expected = (raw & 8) ? raw | UINT8_C(0xf0) : raw;
        AmeValue value = riscv_ame_m_read(&layout, result, &staged,
                                          element);

        g_assert_cmphex(value.lo, ==, expected);
        g_assert_cmphex(value.hi, ==, 0);
    }
}

static void test_convert_unit_to_packed_saturation(void)
{
    static const int8_t inputs[] = { -9, -8, -1, 0, 7, 8, 15, -16 };
    static const uint8_t expected[] = { 8, 8, 15, 0, 7, 7, 7, 8 };
    RISCVAMELayout layout = test_layout();
    const AmeDTypeDesc *int4_sat = dtype(UINT32_C(0x60000004));
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    g_autofree uint8_t *mregs = g_malloc0(layout.mregs_bytes);
    g_autofree uint8_t *result = g_malloc0(layout.mregs_bytes);
    AmeMView destination, source, staged;
    AmeExecFlags flags = { 0 };
    size_t element;

    init_views(&layout, &destination, 8, int4_sat, &source, 4, int8);
    g_assert_cmpuint(source.total_elements, ==, ARRAY_SIZE(inputs));
    for (element = 0; element < ARRAY_SIZE(inputs); element++) {
        AmeValue value = { .lo = (uint8_t)inputs[element] };

        riscv_ame_m_write(&layout, mregs, &source, element, &value);
    }

    g_assert_true(riscv_ame_exec_m_convert(
        AME_IMPL_INT_CONVERT, &layout, mregs, &destination, &source,
        result, layout.mregs_bytes, &flags));
    g_assert_true(flags.saturated);
    g_assert_cmphex(flags.fflags, ==, 0);
    staged = staged_view(&destination);
    for (element = 0; element < ARRAY_SIZE(expected); element++) {
        AmeValue value = riscv_ame_m_read(&layout, result, &staged,
                                          element);

        g_assert_cmphex(value.lo, ==, expected[element]);
    }
}

static void test_convert_overlapping_views(void)
{
    static const uint8_t inputs[] = { 1, 2, UINT8_C(0xff), UINT8_C(0x7f) };
    static const uint16_t expected[] = { 1, 2, UINT16_C(0xffff), 127 };
    RISCVAMELayout layout = test_layout();
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    const AmeDTypeDesc *int16 = dtype(UINT32_C(0x40000010));
    g_autofree uint8_t *mregs = g_malloc0(layout.mregs_bytes);
    g_autofree uint8_t *result = g_malloc0(layout.mregs_bytes);
    AmeMView destination, source, staged;
    AmeExecFlags flags = { .saturated = true };
    size_t element;
    size_t bytes;

    init_views(&layout, &destination, 0, int16, &source, 1, int8);
    g_assert_cmpuint(destination.span, ==, 2);
    g_assert_cmpuint(source.span, ==, 1);
    for (element = 0; element < ARRAY_SIZE(inputs); element++) {
        AmeValue value = { .lo = inputs[element] };

        riscv_ame_m_write(&layout, mregs, &source, element, &value);
    }

    g_assert_true(riscv_ame_exec_m_convert(
        AME_IMPL_INT_CONVERT, &layout, mregs, &destination, &source,
        result, layout.mregs_bytes, &flags));
    g_assert_false(flags.saturated);
    g_assert_cmphex(flags.fflags, ==, 0);
    bytes = riscv_ame_m_view_bytes(&layout, &destination);
    memcpy(mregs + destination.base * layout.mreg_bytes, result, bytes);
    staged = destination;
    for (element = 0; element < ARRAY_SIZE(expected); element++) {
        AmeValue value = riscv_ame_m_read(&layout, mregs, &staged,
                                          element);

        g_assert_cmphex(value.lo, ==, expected[element]);
    }
}

static void test_convert_int128_saturation(void)
{
    const AmeValue inputs[] = {
        { .lo = UINT64_MAX, .hi = UINT64_MAX },
        { .hi = UINT64_C(0x8000000000000000) },
        { .lo = UINT64_MAX, .hi = UINT64_C(0x7fffffffffffffff) },
        { .lo = 127 },
    };
    static const uint8_t expected[] = { 0, 0, UINT8_MAX, 127 };
    RISCVAMELayout layout = test_layout();
    const AmeDTypeDesc *int128 = dtype(UINT32_C(0x40000080));
    const AmeDTypeDesc *uint8_sat = dtype(UINT32_C(0x20000008));
    g_autofree uint8_t *mregs = g_malloc0(layout.mregs_bytes);
    g_autofree uint8_t *result = g_malloc0(layout.mregs_bytes);
    AmeMView destination, source, staged;
    AmeExecFlags flags = { 0 };
    size_t element;

    init_views(&layout, &destination, 0, uint8_sat, &source, 16, int128);
    g_assert_cmpuint(source.group_size, ==, 16);
    for (element = 0; element < ARRAY_SIZE(inputs); element++) {
        riscv_ame_m_write(&layout, mregs, &source, element,
                          &inputs[element]);
    }

    g_assert_true(riscv_ame_exec_m_convert(
        AME_IMPL_INT_CONVERT, &layout, mregs, &destination, &source,
        result, layout.mregs_bytes, &flags));
    g_assert_true(flags.saturated);
    g_assert_cmphex(flags.fflags, ==, 0);
    staged = staged_view(&destination);
    for (element = 0; element < ARRAY_SIZE(expected); element++) {
        AmeValue value = riscv_ame_m_read(&layout, result, &staged,
                                          element);

        g_assert_cmphex(value.lo, ==, expected[element]);
    }
}

static void test_convert_fp_to_packed_int(void)
{
    static const uint16_t raw[] = { 0x3e00, 0xbc00, 0x7c00, 0x7e00, 0x4900 };
    static const uint8_t wrapped[] = { 2, 0xf, 7, 7, 0xa };
    static const uint8_t clamped[] = { 2, 0xf, 7, 7, 7 };
    RISCVAMELayout layout = test_layout_geometry(2, 32);
    const AmeDTypeDesc *fp16 = dtype(0x14300110);
    g_autofree uint8_t *mregs = g_malloc0(layout.mregs_bytes);
    g_autofree uint8_t *snapshot = g_malloc(layout.mregs_bytes);
    g_autofree uint8_t *result = g_malloc(layout.mregs_bytes);
    AmeMView destination, source, staged;
    AmeExecFlags flags;

    for (unsigned sat = 0; sat < 2; sat++) {
        const AmeDTypeDesc *int4 = dtype(0x40000004 | (sat << 29));

        /* The first source register overlaps the packed destination. */
        init_views(&layout, &destination, 0, int4, &source, 0, fp16);
        for (size_t i = 0; i < source.total_elements; i++) {
            AmeValue value = { .lo = raw[i % G_N_ELEMENTS(raw)] };

            riscv_ame_m_write(&layout, mregs, &source, i, &value);
        }
        memcpy(snapshot, mregs, layout.mregs_bytes);
        g_assert_cmpint(riscv_ame_exec_impl(AME_OP_CONV, int4, fp16, NULL),
                        ==, AME_IMPL_FP_CONVERT);
        g_assert_true(riscv_ame_exec_m_convert(
            AME_IMPL_FP_CONVERT, &layout, mregs, &destination, &source,
            result, layout.mregs_bytes, &flags));
        g_assert_cmpmem(mregs, layout.mregs_bytes,
                        snapshot, layout.mregs_bytes);
        g_assert_cmphex(flags.fflags, ==, AME_FFLAG_NV | AME_FFLAG_NX);
        g_assert_cmpint(flags.saturated, ==, sat);
        staged = staged_view(&destination);
        for (size_t i = 0; i < staged.total_elements; i++) {
            AmeValue actual = riscv_ame_m_read(&layout, result, &staged, i);

            g_assert_cmphex(actual.lo, ==,
                            sat ? clamped[i % G_N_ELEMENTS(raw)] :
                                  wrapped[i % G_N_ELEMENTS(raw)]);
            g_assert_cmphex(actual.hi, ==, 0);
        }
    }
}

static void test_convert_rejects_invalid_executor(void)
{
    RISCVAMELayout layout = test_layout();
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    g_autofree uint8_t *mregs = g_malloc0(layout.mregs_bytes);
    g_autofree uint8_t *result = g_malloc0(layout.mregs_bytes);
    AmeMView destination, source;
    AmeExecFlags flags = { .saturated = true };

    init_views(&layout, &destination, 0, int8, &source, 1, int8);
    memset(result, 0xa5, layout.mregs_bytes);
    g_assert_false(riscv_ame_exec_m_convert(
        AME_IMPL_INT_CONVERT, &layout, mregs, &destination, &source,
        result, riscv_ame_m_view_bytes(&layout, &destination) - 1,
        &flags));
    g_assert_true(flags.saturated);
    g_assert_cmphex(flags.fflags, ==, 0);
    g_assert_false(riscv_ame_exec_m_convert(
        AME_IMPL_UNSUPPORTED, &layout, mregs, &destination, &source,
        result, layout.mregs_bytes, &flags));
    g_assert_true(flags.saturated);
    g_assert_cmphex(flags.fflags, ==, 0);
    for (size_t i = 0; i < layout.mregs_bytes; i++) {
        g_assert_cmphex(result[i], ==, UINT8_C(0xa5));
    }
}

static void test_convert_same_width_integer_bits(void)
{
    static const unsigned widths[] = { 4, 8, 16, 32, 64, 128 };
    RISCVAMELayout layout = test_layout_geometry(2, 64);
    g_autofree uint8_t *mregs = g_malloc(layout.mregs_bytes);
    g_autofree uint8_t *result = g_malloc(layout.mregs_bytes);

    for (unsigned w = 0; w < ARRAY_SIZE(widths); w++) {
        for (unsigned source_variant = 0; source_variant < 16;
             source_variant++) {
            const AmeDTypeDesc *a_type = dtype(widths[w] |
                (source_variant << 27));

            for (unsigned dest_variant = 0; dest_variant < 8;
                 dest_variant++) {
                const AmeDTypeDesc *d_type = dtype(widths[w] |
                    ((dest_variant & 3) << 27) |
                    ((dest_variant >> 2) << 30));
                AmeMView d, a, staged;
                AmeExecFlags flags = {
                    .fflags = UINT8_MAX, .saturated = true,
                };
                size_t bytes;

                init_views(&layout, &d, 0, d_type, &a, 4, a_type);
                staged = staged_view(&d);
                bytes = riscv_ame_m_view_bytes(&layout, &d);
                memset(mregs, 0, layout.mregs_bytes);
                for (size_t e = 0; e < a.total_elements; e++) {
                    AmeValue raw = {
                        .lo = e % 2 ? UINT64_MAX : BIT_ULL(63),
                        .hi = e % 2 ? BIT_ULL(63) : UINT64_MAX,
                    };

                    riscv_ame_m_write(&layout, mregs, &a, e, &raw);
                }
                memset(result, 0xcd, layout.mregs_bytes);
                g_assert_true(riscv_ame_exec_m_convert(AME_IMPL_INT_CONVERT,
                    &layout, mregs, &d, &a, result, layout.mregs_bytes,
                    &flags));
                g_assert_cmphex(flags.fflags, ==, 0);
                g_assert_false(flags.saturated);
                for (size_t e = 0; e < d.total_elements; e++) {
                    AmeValue expected = riscv_ame_m_read(&layout, mregs, &a, e);
                    AmeValue actual = riscv_ame_m_read(&layout, result,
                                                       &staged, e);

                    g_assert_cmphex(actual.lo, ==, expected.lo);
                    g_assert_cmphex(actual.hi, ==, expected.hi);
                }
                for (size_t byte = bytes; byte < layout.mregs_bytes; byte++) {
                    g_assert_cmphex(result[byte], ==, 0xcd);
                }
            }
        }

        /* A saturating destination cannot use the raw-copy conversion. */
        const AmeDTypeDesc *a_type = dtype(widths[w]);
        const AmeDTypeDesc *d_type = dtype(widths[w] | UINT32_C(0x60000000));
        AmeMView d, a, staged;
        AmeExecFlags flags = { 0 };
        AmeValue maximum = {
            .lo = widths[w] >= 64 ? UINT64_MAX : BIT_ULL(widths[w]) - 1,
            .hi = widths[w] == 128 ? UINT64_MAX : 0,
        };
        AmeValue expected = {
            .lo = widths[w] == 128 ? UINT64_MAX : maximum.lo >> 1,
            .hi = widths[w] == 128 ? INT64_MAX : 0,
        };

        init_views(&layout, &d, 0, d_type, &a, 4, a_type);
        for (size_t e = 0; e < a.total_elements; e++) {
            riscv_ame_m_write(&layout, mregs, &a, e, &maximum);
        }
        g_assert_true(riscv_ame_exec_m_convert(AME_IMPL_INT_CONVERT,
            &layout, mregs, &d, &a, result, layout.mregs_bytes, &flags));
        g_assert_cmphex(flags.fflags, ==, 0);
        g_assert_true(flags.saturated);
        staged = staged_view(&d);
        for (size_t e = 0; e < d.total_elements; e++) {
            AmeValue actual = riscv_ame_m_read(&layout, result, &staged, e);

            g_assert_cmphex(actual.lo, ==, expected.lo);
            g_assert_cmphex(actual.hi, ==, expected.hi);
        }
    }

    /* Same-format floating conversion must still quiet sNaN and raise NV. */
    const AmeDTypeDesc *fp32 = dtype(UINT32_C(0x20300120));
    AmeMView d, a, staged;
    AmeExecFlags flags = { 0 };

    init_views(&layout, &d, 0, fp32, &a, 4, fp32);
    memset(mregs, 0, layout.mregs_bytes);
    riscv_ame_m_write(&layout, mregs, &a, 0,
                      &(AmeValue) { .lo = UINT32_C(0x7f800001) });
    g_assert_true(riscv_ame_exec_m_convert(AME_IMPL_FP_CONVERT,
        &layout, mregs, &d, &a, result, layout.mregs_bytes, &flags));
    staged = staged_view(&d);
    g_assert_cmphex(riscv_ame_m_read(&layout, result, &staged, 0).lo,
                    ==, UINT32_C(0x7fc00000));
    g_assert_cmphex(flags.fflags, ==, AME_FFLAG_NV);
    g_assert_false(flags.saturated);
}

static void assert_value(const AmeExecResult *result,
                         uint64_t low, uint64_t high, bool saturated)
{
    g_assert_cmphex(result->value.lo, ==, low);
    g_assert_cmphex(result->value.hi, ==, high);
    g_assert_cmpint(result->saturated, ==, saturated);
    g_assert_cmphex(result->fflags, ==, 0);
}

static AmeExecResult exec_unary(AmeOperation op,
                                const AmeDTypeDesc *destination,
                                const AmeDTypeDesc *source_dtype,
                                AmeValue source)
{
    AmeExecResult result;

    g_assert_true(riscv_ame_exec_unary(AME_IMPL_INT_UNARY, op,
                                       destination, source_dtype,
                                       &source, &result, NULL));
    return result;
}

static AmeExecResult exec_binary(AmeImplClass impl, AmeOperation op,
                                 const AmeDTypeDesc *destination,
                                 const AmeDTypeDesc *src1_dtype,
                                 AmeValue src1,
                                 const AmeDTypeDesc *src2_dtype,
                                 AmeValue src2)
{
    AmeExecResult result;
    float_status fp_status;
    float_status *status = NULL;

    if (impl == AME_IMPL_SELECT &&
        (src1_dtype->class == AME_DTYPE_CLASS_IEEE ||
         src1_dtype->class == AME_DTYPE_CLASS_OCP)) {
        g_assert_true(riscv_ame_fp_status_prepare(&fp_status, src1_dtype));
        status = &fp_status;
    }

    g_assert_true(riscv_ame_exec_binary(impl, op, destination,
                                        src1_dtype, &src1,
                                        src2_dtype, &src2, &result, status));
    return result;
}

static AmeExecResult exec_ternary(AmeImplClass impl, AmeOperation op,
                                  const AmeDTypeDesc *destination,
                                  const AmeDTypeDesc *src1_dtype,
                                  AmeValue src1,
                                  const AmeDTypeDesc *src2_dtype,
                                  AmeValue src2, AmeValue current)
{
    AmeExecResult result;
    float_status fp_status;
    float_status *status = NULL;

    if (impl == AME_IMPL_SELECT &&
        (src1_dtype->class == AME_DTYPE_CLASS_IEEE ||
         src1_dtype->class == AME_DTYPE_CLASS_OCP)) {
        g_assert_true(riscv_ame_fp_status_prepare(&fp_status, src1_dtype));
        status = &fp_status;
    }

    g_assert_true(riscv_ame_exec_ternary(
        impl, op, destination, src1_dtype, &src1, src2_dtype, &src2,
        &current, &result, status));
    return result;
}

static void test_integer_arithmetic_dispatch(void)
{
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    const AmeDTypeDesc *int8_sat = dtype(UINT32_C(0x60000008));
    const AmeDTypeDesc *int16 = dtype(UINT32_C(0x40000010));
    const AmeDTypeDesc *int128_sat = dtype(UINT32_C(0x60000080));
    const AmeDTypeDesc *uint128 = dtype(UINT32_C(0x00000080));
    const AmeDTypeDesc *uint128_sat = dtype(UINT32_C(0x20000080));
    AmeExecResult result;

    result = exec_unary(AME_OP_ABS, int8, int8,
                        (AmeValue) { .lo = UINT8_C(0x80) });
    assert_value(&result, UINT8_C(0x80), 0, false);
    result = exec_unary(AME_OP_ABS, int8_sat, int8,
                        (AmeValue) { .lo = UINT8_C(0x80) });
    assert_value(&result, INT8_MAX, 0, true);
    result = exec_unary(
        AME_OP_ABS, int128_sat, dtype(UINT32_C(0x40000080)),
        (AmeValue) { .hi = UINT64_C(0x8000000000000000) });
    assert_value(&result, UINT64_MAX, INT64_MAX, true);

    result = exec_binary(AME_IMPL_INT_BINARY, AME_OP_ADD,
                         int16, int8, (AmeValue) { .lo = INT8_MAX },
                         int8, (AmeValue) { .lo = 1 });
    assert_value(&result, 128, 0, false);
    result = exec_binary(AME_IMPL_INT_BINARY, AME_OP_ADD,
                         int8_sat, int8, (AmeValue) { .lo = INT8_MAX },
                         int8, (AmeValue) { .lo = 1 });
    assert_value(&result, INT8_MAX, 0, true);

    /* SUB is B - A. */
    result = exec_binary(AME_IMPL_INT_BINARY, AME_OP_SUB,
                         int16, int8, (AmeValue) { .lo = 5 },
                         int8, (AmeValue) { .lo = 2 });
    assert_value(&result, UINT16_C(0xfffd), 0, false);
    result = exec_binary(AME_IMPL_INT_BINARY, AME_OP_ABSDIFF,
                         int16, int8,
                         (AmeValue) { .lo = UINT8_C(0x80) },
                         int8, (AmeValue) { .lo = INT8_MAX });
    assert_value(&result, 255, 0, false);

    result = exec_binary(AME_IMPL_INT_BINARY, AME_OP_NMUL,
                         int16, int8, (AmeValue) { .lo = 3 },
                         int8, (AmeValue) { .lo = 4 });
    assert_value(&result, UINT16_C(0xfff4), 0, false);
    result = exec_binary(
        AME_IMPL_INT_BINARY, AME_OP_MUL, uint128,
        uint128, (AmeValue) { .lo = UINT64_MAX, .hi = UINT64_MAX },
        uint128, (AmeValue) { .lo = 2 });
    assert_value(&result, UINT64_C(0xfffffffffffffffe), UINT64_MAX,
                 false);
    result = exec_binary(
        AME_IMPL_INT_BINARY, AME_OP_MUL, uint128_sat,
        uint128, (AmeValue) { .lo = UINT64_MAX, .hi = UINT64_MAX },
        uint128, (AmeValue) { .lo = 2 });
    assert_value(&result, UINT64_MAX, UINT64_MAX, true);

    result = exec_binary(AME_IMPL_INT_BINARY, AME_OP_MAX,
                         int8, int8,
                         (AmeValue) { .lo = UINT8_C(0xf0) },
                         int8, (AmeValue) { .lo = 7 });
    assert_value(&result, 7, 0, false);
    result = exec_binary(AME_IMPL_INT_BINARY, AME_OP_MIN,
                         int8, int8,
                         (AmeValue) { .lo = UINT8_C(0xf0) },
                         int8, (AmeValue) { .lo = 7 });
    assert_value(&result, UINT8_C(0xf0), 0, false);
}

static void test_integer_bitwise_compare_select(void)
{
    const AmeDTypeDesc *int4 = dtype(UINT32_C(0x40000004));
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    const AmeDTypeDesc *uint8 = dtype(UINT32_C(0x00000008));
    const AmeDTypeDesc *int128 = dtype(UINT32_C(0x40000080));
    const AmeDTypeDesc *fp32 = dtype(UINT32_C(0x20300120));
    const AmeValue left = { .lo = UINT8_C(0x5a) };
    const AmeValue right = { .lo = UINT8_C(0x0f) };
    static const struct {
        AmeOperation op;
        uint8_t expected;
    } bitwise[] = {
        { AME_OP_AND, UINT8_C(0x0a) },
        { AME_OP_ANDNOT, UINT8_C(0x50) },
        { AME_OP_OR, UINT8_C(0x5f) },
        { AME_OP_ORNOT, UINT8_C(0xfa) },
        { AME_OP_XOR, UINT8_C(0x55) },
    };
    AmeExecResult result;
    size_t i;

    for (i = 0; i < ARRAY_SIZE(bitwise); i++) {
        result = exec_binary(AME_IMPL_INT_BINARY, bitwise[i].op,
                             int8, int8, left, int8, right);
        assert_value(&result, bitwise[i].expected, 0, false);
    }

    result = exec_binary(AME_IMPL_INT_COMPARE, AME_OP_CMPGE,
                         int128, int4,
                         (AmeValue) { .lo = UINT8_C(0xf) },
                         int8, (AmeValue) { .lo = UINT8_C(0xfe) });
    assert_value(&result, UINT64_MAX, UINT64_MAX, false);
    result = exec_binary(AME_IMPL_INT_COMPARE, AME_OP_CMPLT,
                         int128, int4,
                         (AmeValue) { .lo = UINT8_C(0xf) },
                         int8, (AmeValue) { .lo = UINT8_C(0xfe) });
    assert_value(&result, 0, 0, false);

    result = exec_binary(AME_IMPL_SELECT, AME_OP_SEL,
                         int8, int4, (AmeValue) { .lo = 8 },
                         int8, (AmeValue) { .lo = UINT8_C(0xa5) });
    assert_value(&result, UINT8_C(0xa5), 0, false);
    result = exec_binary(AME_IMPL_SELECT, AME_OP_SELGE,
                         int8, int4, (AmeValue) { .lo = 8 },
                         int8, (AmeValue) { .lo = UINT8_C(0xa5) });
    assert_value(&result, 0, 0, false);

    /* Predicates use an ordered numeric comparison with semantic zero. */
    result = exec_binary(AME_IMPL_SELECT, AME_OP_SELGE,
                         int8, uint8, (AmeValue) { .lo = UINT8_MAX },
                         int8, (AmeValue) { .lo = UINT8_C(0xa5) });
    assert_value(&result, UINT8_C(0xa5), 0, false);
    result = exec_binary(AME_IMPL_SELECT, AME_OP_SEL,
                         int8, uint8, (AmeValue) { .lo = UINT8_MAX },
                         int8, (AmeValue) { .lo = UINT8_C(0xa5) });
    assert_value(&result, 0, 0, false);
    result = exec_binary(AME_IMPL_SELECT, AME_OP_SELGE,
                         int8, fp32,
                         (AmeValue) { .lo = UINT32_C(0x80000000) },
                         int8, (AmeValue) { .lo = UINT8_C(0xa5) });
    assert_value(&result, UINT8_C(0xa5), 0, false);
    result = exec_binary(AME_IMPL_SELECT, AME_OP_SELGE,
                         int8, fp32,
                         (AmeValue) { .lo = UINT32_C(0x7fc00000) },
                         int8, (AmeValue) { .lo = UINT8_C(0xa5) });
    assert_value(&result, 0, 0, false);
    g_assert_cmphex(result.fflags, ==, 0);
    result = exec_binary(AME_IMPL_SELECT, AME_OP_SEL,
                         int8, fp32,
                         (AmeValue) { .lo = UINT32_C(0x7f800001) },
                         int8, (AmeValue) { .lo = UINT8_C(0xa5) });
    g_assert_cmphex(result.value.lo, ==, 0);
    g_assert_cmphex(result.value.hi, ==, 0);
    g_assert_false(result.saturated);
    g_assert_cmphex(result.fflags, ==, AME_FFLAG_NV);
}

static void test_integer_shift_dispatch(void)
{
    const AmeDTypeDesc *int4 = dtype(UINT32_C(0x40000004));
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    const AmeDTypeDesc *int8_sat = dtype(UINT32_C(0x60000008));
    const AmeDTypeDesc *uint128 = dtype(UINT32_C(0x00000080));
    AmeValue amount = { .lo = 9 };
    AmeExecResult result;

    result = exec_binary(AME_IMPL_INT_SHIFT, AME_OP_SLL,
                         int8, int8,
                         (AmeValue) { .lo = UINT8_C(0x81) },
                         int4, amount);
    assert_value(&result, 2, 0, false);
    result = exec_binary(AME_IMPL_INT_SHIFT, AME_OP_SLL,
                         int8_sat, int8_sat,
                         (AmeValue) { .lo = UINT8_C(0x81) },
                         int4, amount);
    assert_value(&result, UINT8_C(0x80), 0, true);
    result = exec_binary(AME_IMPL_INT_SHIFT, AME_OP_SRA,
                         int8, int8,
                         (AmeValue) { .lo = UINT8_C(0x81) },
                         int4, amount);
    assert_value(&result, UINT8_C(0xc0), 0, false);
    result = exec_binary(AME_IMPL_INT_SHIFT, AME_OP_SRL,
                         int8, int8,
                         (AmeValue) { .lo = UINT8_C(0x81) },
                         int4, amount);
    assert_value(&result, UINT8_C(0x40), 0, false);

    result = exec_binary(AME_IMPL_INT_SHIFT, AME_OP_SLL,
                         uint128, uint128, (AmeValue) { .lo = 1 },
                         int8, (AmeValue) { .lo = 127 });
    assert_value(&result, 0, UINT64_C(0x8000000000000000), false);
}

static int int4_value(unsigned int raw, bool is_signed)
{
    return is_signed && (raw & 8) ? (int)raw - 16 : (int)raw;
}

static unsigned int int4_finish(int value, bool is_signed, bool saturating,
                                bool *saturated)
{
    int minimum = is_signed ? -8 : 0;
    int maximum = is_signed ? 7 : 15;

    *saturated = saturating && (value < minimum || value > maximum);
    if (saturating) {
        value = CLAMP(value, minimum, maximum);
    }
    return value & 15;
}

static void test_integer_elementwise_exhaustive_small(void)
{
    static const uint32_t plain_encodings[] = {
        UINT32_C(0x00000004), UINT32_C(0x40000004),
    };
    static const uint32_t saturating_encodings[] = {
        UINT32_C(0x20000004), UINT32_C(0x60000004),
    };
    const AmeDTypeDesc *predicate_dest = dtype(UINT32_C(0x00000004));
    size_t type;

    for (type = 0; type < ARRAY_SIZE(plain_encodings); type++) {
        const AmeDTypeDesc *plain = dtype(plain_encodings[type]);
        const AmeDTypeDesc *sat = dtype(saturating_encodings[type]);
        unsigned int left_raw;

        for (left_raw = 0; left_raw < 16; left_raw++) {
            int left = int4_value(left_raw, plain->integer.is_signed);
            int absolute = left < 0 ? -left : left;
            bool expected_sat;
            unsigned int expected;
            AmeExecResult result;
            unsigned int right_raw;

            expected = int4_finish(absolute, plain->integer.is_signed, false,
                                   &expected_sat);
            result = exec_unary(AME_OP_ABS, plain, plain,
                                (AmeValue) { .lo = left_raw });
            assert_value(&result, expected, 0, expected_sat);
            expected = int4_finish(absolute, plain->integer.is_signed, true,
                                   &expected_sat);
            result = exec_unary(AME_OP_ABS, sat, plain,
                                (AmeValue) { .lo = left_raw });
            assert_value(&result, expected, 0, expected_sat);

            for (right_raw = 0; right_raw < 16; right_raw++) {
                int right = int4_value(right_raw, plain->integer.is_signed);
                static const AmeOperation arithmetic_ops[] = {
                    AME_OP_ABSDIFF, AME_OP_ADD, AME_OP_MUL,
                    AME_OP_NMUL, AME_OP_SUB,
                };
                int arithmetic_values[] = {
                    ABS(left - right), left + right, left * right,
                    -(left * right), right - left,
                };
                size_t i;

                for (i = 0; i < ARRAY_SIZE(arithmetic_ops); i++) {
                    expected = int4_finish(arithmetic_values[i],
                                           plain->integer.is_signed, false,
                                           &expected_sat);
                    result = exec_binary(
                        AME_IMPL_INT_BINARY, arithmetic_ops[i], plain,
                        plain, (AmeValue) { .lo = left_raw },
                        plain, (AmeValue) { .lo = right_raw });
                    assert_value(&result, expected, 0, expected_sat);

                    expected = int4_finish(arithmetic_values[i],
                                           plain->integer.is_signed, true,
                                           &expected_sat);
                    result = exec_binary(
                        AME_IMPL_INT_BINARY, arithmetic_ops[i], sat,
                        plain, (AmeValue) { .lo = left_raw },
                        plain, (AmeValue) { .lo = right_raw });
                    assert_value(&result, expected, 0, expected_sat);
                }

                result = exec_binary(AME_IMPL_INT_BINARY, AME_OP_MAX,
                                     plain, plain,
                                     (AmeValue) { .lo = left_raw }, plain,
                                     (AmeValue) { .lo = right_raw });
                assert_value(&result, left >= right ? left_raw : right_raw,
                             0, false);
                result = exec_binary(AME_IMPL_INT_BINARY, AME_OP_MIN,
                                     plain, plain,
                                     (AmeValue) { .lo = left_raw }, plain,
                                     (AmeValue) { .lo = right_raw });
                assert_value(&result, left <= right ? left_raw : right_raw,
                             0, false);

                result = exec_binary(AME_IMPL_INT_COMPARE,
                                     AME_OP_CMPGE, predicate_dest,
                                     plain, (AmeValue) { .lo = left_raw },
                                     plain, (AmeValue) { .lo = right_raw });
                assert_value(&result, left >= right ? 15 : 0, 0, false);
                result = exec_binary(AME_IMPL_INT_COMPARE,
                                     AME_OP_CMPLT, predicate_dest,
                                     plain, (AmeValue) { .lo = left_raw },
                                     plain, (AmeValue) { .lo = right_raw });
                assert_value(&result, left < right ? 15 : 0, 0, false);

                result = exec_binary(AME_IMPL_INT_SHIFT, AME_OP_SLL,
                                     plain, plain,
                                     (AmeValue) { .lo = left_raw },
                                     predicate_dest,
                                     (AmeValue) { .lo = right_raw });
                expected = (left_raw << (right_raw & 3)) & 15;
                assert_value(&result, expected, 0, false);
                result = exec_binary(AME_IMPL_INT_SHIFT, AME_OP_SLL,
                                     sat, sat,
                                     (AmeValue) { .lo = left_raw },
                                     predicate_dest,
                                     (AmeValue) { .lo = right_raw });
                expected = int4_finish(
                    left * (1 << (right_raw & 3)),
                    plain->integer.is_signed, true,
                    &expected_sat);
                assert_value(&result, expected, 0, expected_sat);

                result = exec_binary(AME_IMPL_INT_SHIFT, AME_OP_SRL,
                                     plain, plain,
                                     (AmeValue) { .lo = left_raw },
                                     predicate_dest,
                                     (AmeValue) { .lo = right_raw });
                assert_value(&result, left_raw >> (right_raw & 3),
                             0, false);
                if (plain->integer.is_signed) {
                    result = exec_binary(
                        AME_IMPL_INT_SHIFT, AME_OP_SRA, plain, plain,
                        (AmeValue) { .lo = left_raw }, predicate_dest,
                        (AmeValue) { .lo = right_raw });
                    assert_value(&result,
                                 (left >> (right_raw & 3)) & 15,
                                 0, false);
                }
            }
        }
    }
}

static int round_half_reference(int value, AmeIntRoundMode rounding)
{
    int floor = value / 2;
    int remainder;

    if (value < 0 && value % 2 != 0) {
        floor--;
    }
    remainder = value - floor * 2;
    switch (rounding) {
    case AME_INT_RNU:
        floor += remainder != 0;
        break;
    case AME_INT_RNE:
        floor += remainder != 0 && (floor & 1);
        break;
    case AME_INT_RDN:
        break;
    case AME_INT_ROD:
        if (remainder != 0) {
            floor |= 1;
        }
        break;
    case AME_INT_ROUND_MAX:
    default:
        g_assert_not_reached();
    }
    return floor;
}

static void test_integer_mixed_exhaustive_small(void)
{
    static const AmeOperation ops[] = {
        AME_OP_ABSDIFF, AME_OP_ADD, AME_OP_SUB,
        AME_OP_MUL, AME_OP_NMUL,
    };

    /* D, A, and B have independent signedness; only D controls saturation. */
    for (unsigned signs = 0; signs < 8; signs++) {
        bool ds = signs & 1, as = signs & 2, bs = signs & 4;
        const AmeDTypeDesc *a = dtype(4 | ((uint32_t)as << 30));
        const AmeDTypeDesc *b = dtype(4 | ((uint32_t)bs << 30));

        for (unsigned sat = 0; sat < 2; sat++) {
            const AmeDTypeDesc *d = dtype(4 | ((uint32_t)ds << 30) |
                                          (sat << 29));

            for (unsigned ar = 0; ar < 16; ar++) {
                int av = int4_value(ar, as);
                bool saturated;
                unsigned expected = int4_finish(ABS(av), ds, sat, &saturated);
                AmeExecResult result = exec_unary(
                    AME_OP_ABS, d, a, (AmeValue) { .lo = ar });

                assert_value(&result, expected, 0, saturated);
                for (unsigned br = 0; br < 16; br++) {
                    int bv = int4_value(br, bs);
                    int exact[] = {
                        ABS(av - bv), av + bv, bv - av, av * bv, -av * bv,
                    };

                    for (unsigned op = 0; op < ARRAY_SIZE(ops); op++) {
                        expected = int4_finish(exact[op], ds, sat, &saturated);
                        result = exec_binary(
                            AME_IMPL_INT_BINARY, ops[op], d,
                            a, (AmeValue) { .lo = ar },
                            b, (AmeValue) { .lo = br });
                        assert_value(&result, expected, 0, saturated);
                    }
                    result = exec_binary(
                        AME_IMPL_INT_COMPARE, AME_OP_CMPGE, d,
                        a, (AmeValue) { .lo = ar },
                        b, (AmeValue) { .lo = br });
                    assert_value(&result, av >= bv ? 15 : 0, 0, false);
                    result = exec_binary(
                        AME_IMPL_INT_COMPARE, AME_OP_CMPLT, d,
                        a, (AmeValue) { .lo = ar },
                        b, (AmeValue) { .lo = br });
                    assert_value(&result, av < bv ? 15 : 0, 0, false);
                }
            }
        }
    }
}

static void test_integer_capability_primitives(void)
{
    static const AmeOperation ops[] = {
        AME_OP_ABSDIFF, AME_OP_ADD, AME_OP_SUB,
        AME_OP_MUL, AME_OP_MULACC, AME_OP_CMPGE,
        AME_OP_LDEXP, AME_OP_LDEXPACC,
        AME_OP_MIN, AME_OP_AND, AME_OP_SLL, AME_OP_SRA,
    };
    const AmeCapabilities *caps = riscv_ame_maximum_capabilities();

    for (unsigned d = 0; d < AME_DTYPE_COUNT; d++) {
        const AmeDTypeDesc *dd = &riscv_ame_dtypes[d];

        if (dd->class != AME_DTYPE_CLASS_INTEGER) {
            continue;
        }
        for (unsigned a = 0; a < AME_DTYPE_COUNT; a++) {
            const AmeDTypeDesc *ad = &riscv_ame_dtypes[a];

            if (ad->class != AME_DTYPE_CLASS_INTEGER) {
                continue;
            }
            g_assert_cmpint(riscv_ame_int_supports(AME_OP_ABS, dd, ad, NULL),
                            ==, riscv_ame_capabilities_has(
                                caps, AME_OP_ABS, AME_FORM_BASE,
                                d, a, AME_DTYPE_ID_INVALID));
            for (unsigned b = 0; b < AME_DTYPE_COUNT; b++) {
                const AmeDTypeDesc *bd = &riscv_ame_dtypes[b];

                if (bd->class != AME_DTYPE_CLASS_INTEGER) {
                    continue;
                }
                for (unsigned op = 0; op < ARRAY_SIZE(ops); op++) {
                    g_assert_cmpint(riscv_ame_int_supports(ops[op], dd, ad, bd),
                                    ==, riscv_ame_capabilities_has(
                                        caps, ops[op], AME_FORM_BASE,
                                        d, a, b));
                }
            }
        }
    }
}

static void test_numeric_capability_primitives(void)
{
    static const AmeOperation ops[] = {
        AME_OP_ABSDIFF, AME_OP_ADD, AME_OP_SUB, AME_OP_HDIFF, AME_OP_MEAN,
        AME_OP_MUL, AME_OP_NMUL, AME_OP_MULACC, AME_OP_MULNACC,
        AME_OP_MULADD, AME_OP_MULSUB, AME_OP_CMPGE, AME_OP_CMPLT,
        AME_OP_LDEXP, AME_OP_RDEXP, AME_OP_LDEXPACC, AME_OP_RDEXPACC,
        AME_OP_LOG2SUB, AME_OP_SUBLOG2,
    };
    const AmeCapabilities *caps = riscv_ame_maximum_capabilities();

    for (unsigned d = 0; d < AME_DTYPE_COUNT; d++) {
        for (unsigned a = 0; a < AME_DTYPE_COUNT; a++) {
            for (unsigned b = 0; b < AME_DTYPE_COUNT; b++) {
                const AmeDTypeDesc *dd = &riscv_ame_dtypes[d];
                const AmeDTypeDesc *ad = &riscv_ame_dtypes[a];
                const AmeDTypeDesc *bd = &riscv_ame_dtypes[b];

                for (unsigned i = 0; i < G_N_ELEMENTS(ops); i++) {
                    AmeImplClass impl;

                    if (!riscv_ame_capabilities_has(
                            caps, ops[i], AME_FORM_BASE, d, a, b)) {
                        continue;
                    }
                    impl = riscv_ame_exec_impl(ops[i], dd, ad, bd);
                    g_assert_cmpint(impl, !=, AME_IMPL_UNSUPPORTED);
                    if (riscv_ame_impl_is_fp(impl)) {
                        g_assert_true(riscv_ame_fp_supports(ops[i], dd, ad,
                                                            bd));
                    } else {
                        g_assert_true(riscv_ame_int_supports(ops[i], dd, ad,
                                                             bd));
                    }
                }
            }
        }
    }
}

static void test_integer_float_tensor(void)
{
    RISCVAMELayout layout = test_layout_geometry(2, 64);
    const AmeDTypeDesc *u128 = dtype(0x00000080);
    const AmeDTypeDesc *fp64 = dtype(0x2c300140);
    const AmeValue large = { .lo = 1, .hi = UINT64_C(0x8000000000000000) };
    const AmeValue minus_power = { .lo = UINT64_C(0xc7e0000000000000) };
    const AmeValue one = { .lo = UINT64_C(0x3ff0000000000000) };
    g_autofree uint8_t *mregs = g_malloc0(layout.mregs_bytes);
    g_autofree uint8_t *result = g_malloc0(layout.mregs_bytes);
    AmeMView dst, a, b, staged;
    AmeExecFlags flags;

    g_assert_true(riscv_ame_m_view_init(&layout, &a, 4, u128, 1));
    g_assert_true(riscv_ame_m_view_init(&layout, &b, 8, fp64, 1));
    for (unsigned d = 0; d < 2; d++) {
        const AmeDTypeDesc *dd = d ? fp64 : u128;
        AmeImplClass impl = riscv_ame_exec_impl(AME_OP_ADD, dd, u128, fp64);

        g_assert_cmpint(impl, ==, AME_IMPL_FP_BINARY);
        g_assert_true(riscv_ame_m_view_init(&layout, &dst, 0, dd, 1));
        staged = staged_view(&dst);
        for (unsigned k = 0; k < a.total_elements; k++) {
            riscv_ame_m_write(&layout, mregs, &a, k, &large);
            riscv_ame_m_write(&layout, mregs, &b, k, &minus_power);
        }
        g_assert_true(riscv_ame_exec_m_binary(
            impl, AME_OP_ADD, &layout, mregs, &dst, &a, &b,
            result, layout.mregs_bytes, &flags));
        for (unsigned k = 0; k < dst.total_elements; k++) {
            AmeValue v = riscv_ame_m_read(&layout, result, &staged, k);

            g_assert_cmphex(v.lo, ==, d ? one.lo : 1);
            g_assert_cmphex(v.hi, ==, 0);
        }
        g_assert_cmphex(flags.fflags, ==, 0);
        g_assert_false(flags.saturated);
    }
    for (unsigned k = 0; k < dst.total_elements; k++) {
        riscv_ame_m_write(&layout, mregs, &dst, k, &minus_power);
        riscv_ame_m_write(&layout, mregs, &b, k, &one);
    }
    g_assert_true(riscv_ame_exec_m_ternary(
        AME_IMPL_FP_TERNARY, AME_OP_MULACC, &layout, mregs, &dst, &a, &b,
        result, layout.mregs_bytes, &flags));
    g_assert_cmphex(riscv_ame_m_read(&layout, result, &staged, 0).lo,
                    ==, one.lo);
    g_assert_cmphex(flags.fflags, ==, 0);

    /* Int8's integer union fields must never select the FP16 fast path. */
    const AmeDTypeDesc *i8 = dtype(0x40000008);
    const AmeDTypeDesc *fp16 = dtype(0x14300110);
    const AmeDTypeDesc *types[] = { fp16, i8, fp16 };
    uint32_t n_sq;

    g_assert_true(riscv_ame_n_sq(&layout, types, G_N_ELEMENTS(types), &n_sq));
    g_assert_true(riscv_ame_m_view_init(&layout, &dst, 0, fp16, n_sq));
    g_assert_true(riscv_ame_m_view_init(&layout, &a, 4, i8, n_sq));
    g_assert_true(riscv_ame_m_view_init(&layout, &b, 8, fp16, n_sq));
    for (unsigned k = 0; k < a.total_elements; k++) {
        riscv_ame_m_write(&layout, mregs, &a, k, &(AmeValue) { .lo = 3 });
        riscv_ame_m_write(&layout, mregs, &b, k,
                          &(AmeValue) { .lo = 0x3c00 });
    }
    g_assert_true(riscv_ame_exec_m_binary(
        AME_IMPL_FP_BINARY, AME_OP_ADD, &layout, mregs, &dst, &a, &b,
        result, layout.mregs_bytes, &flags));
    staged = staged_view(&dst);
    for (unsigned k = 0; k < dst.total_elements; k++) {
        g_assert_cmphex(riscv_ame_m_read(&layout, result, &staged, k).lo, ==,
                        0x4400);
    }
    g_assert_cmphex(flags.fflags, ==, 0);
}

static void test_integer_half_exhaustive_small(void)
{
    static const AmeOperation ops[] = {
        AME_OP_HDIFF, AME_OP_MEAN,
    };
    unsigned int signs, saturating, rounding;

    for (signs = 0; signs < 8; signs++) {
        bool is_signed = signs & 1, a_signed = signs & 2, b_signed = signs & 4;
        const AmeDTypeDesc *a = dtype(4 | ((uint32_t)a_signed << 30));
        const AmeDTypeDesc *b = dtype(4 | ((uint32_t)b_signed << 30));

        for (saturating = 0; saturating < 2; saturating++) {
            for (rounding = 0; rounding < AME_INT_ROUND_MAX; rounding++) {
                uint32_t encoding = 4 | (is_signed << 30) |
                                    (saturating << 29) | (rounding << 27);
                const AmeDTypeDesc *type = dtype(encoding);
                unsigned int left_raw, right_raw;

                for (left_raw = 0; left_raw < 16; left_raw++) {
                    int left = int4_value(left_raw, a_signed);

                    for (right_raw = 0; right_raw < 16; right_raw++) {
                        int right = int4_value(right_raw, b_signed);
                        size_t op_index;

                        for (op_index = 0; op_index < ARRAY_SIZE(ops);
                             op_index++) {
                            int exact = ops[op_index] == AME_OP_HDIFF ?
                                        right - left : left + right;
                            int rounded = round_half_reference(
                                exact, rounding);
                            bool expected_sat;
                            unsigned int expected = int4_finish(
                                rounded, is_signed, saturating,
                                &expected_sat);
                            AmeExecResult result = exec_binary(
                                AME_IMPL_INT_BINARY, ops[op_index], type,
                                a, (AmeValue) { .lo = left_raw }, b,
                                (AmeValue) { .lo = right_raw });

                            assert_value(&result, expected, 0,
                                         expected_sat);
                        }
                    }
                }
            }
        }
    }
}

typedef struct AmeInt4ScaleReference {
    uint8_t raw;
    bool saturated;
} AmeInt4ScaleReference;

static AmeInt4ScaleReference int4_scale_reference(
    AmeOperation op, bool is_signed, bool data_signed, bool saturating,
    AmeIntRoundMode rounding, uint8_t data_raw, bool exponent_signed,
    uint8_t exponent_raw, uint8_t current_raw)
{
    bool accumulate = op == AME_OP_LDEXPACC ||
                      op == AME_OP_RDEXPACC;
    bool reverse = op == AME_OP_RDEXP || op == AME_OP_RDEXPACC;
    int64_t data = int4_value(data_raw, data_signed);
    int64_t current = accumulate ?
                      int4_value(current_raw, is_signed) : 0;
    int exponent = int4_value(exponent_raw, exponent_signed);
    int64_t rounded;
    bool result_saturated;
    AmeInt4ScaleReference result;

    if (reverse) {
        exponent = -exponent;
    }
    if (exponent >= 0) {
        rounded = data * (INT64_C(1) << exponent) + current;
    } else {
        int shift = -exponent;
        int64_t divisor = INT64_C(1) << shift;
        int64_t floor = data / divisor;
        int64_t remainder;
        bool increment = false;

        if (data < 0 && data % divisor != 0) {
            floor--;
        }
        remainder = data - floor * divisor;
        floor += current;
        switch (rounding) {
        case AME_INT_RNU:
            increment = remainder * 2 >= divisor;
            break;
        case AME_INT_RNE:
            increment = remainder * 2 > divisor ||
                        (remainder * 2 == divisor && (floor & 1));
            break;
        case AME_INT_RDN:
            break;
        case AME_INT_ROD:
            if (remainder != 0) {
                floor |= 1;
            }
            break;
        case AME_INT_ROUND_MAX:
        default:
            g_assert_not_reached();
        }
        rounded = floor + increment;
    }

    result.raw = int4_finish(rounded, is_signed, saturating,
                             &result_saturated);
    result.saturated = result_saturated;
    return result;
}

static void test_integer_scale_exhaustive_small(void)
{
    static const AmeOperation ops[] = {
        AME_OP_LDEXP, AME_OP_RDEXP,
        AME_OP_LDEXPACC, AME_OP_RDEXPACC,
    };
    unsigned int signs, saturating, rounding, exponent_signed;

    for (signs = 0; signs < 4; signs++) {
        bool is_signed = signs & 1, data_signed = signs & 2;
        const AmeDTypeDesc *source_dtype = dtype(
            4 | ((uint32_t)data_signed << 30));

        for (saturating = 0; saturating < 2; saturating++) {
            for (rounding = 0; rounding < AME_INT_ROUND_MAX; rounding++) {
                uint32_t encoding = 4 | (is_signed << 30) |
                                    (saturating << 29) | (rounding << 27);
                const AmeDTypeDesc *data_dtype = dtype(encoding);

                for (exponent_signed = 0; exponent_signed < 2;
                     exponent_signed++) {
                    const AmeDTypeDesc *exponent_dtype = dtype(
                        4 | (exponent_signed << 30));
                    size_t op_index;

                    for (op_index = 0; op_index < ARRAY_SIZE(ops);
                         op_index++) {
                        bool accumulate = op_index >= 2;
                        unsigned int data_raw, exponent_raw;

                        for (data_raw = 0; data_raw < 16; data_raw++) {
                            for (exponent_raw = 0; exponent_raw < 16;
                                 exponent_raw++) {
                                unsigned int current_limit = accumulate ?
                                                             16 : 1;
                                unsigned int current_raw;

                                for (current_raw = 0;
                                     current_raw < current_limit;
                                     current_raw++) {
                                    AmeInt4ScaleReference expected =
                                        int4_scale_reference(
                                            ops[op_index], is_signed,
                                            data_signed, saturating,
                                            rounding, data_raw,
                                            exponent_signed, exponent_raw,
                                            current_raw);
                                    AmeExecResult actual;

                                    if (accumulate) {
                                        actual = exec_ternary(
                                            AME_IMPL_INT_TERNARY,
                                            ops[op_index], data_dtype,
                                            source_dtype,
                                            (AmeValue) { .lo = data_raw },
                                            exponent_dtype,
                                            (AmeValue) {
                                                .lo = exponent_raw,
                                            },
                                            (AmeValue) {
                                                .lo = current_raw,
                                            });
                                    } else {
                                        actual = exec_binary(
                                            AME_IMPL_INT_BINARY,
                                            ops[op_index], data_dtype,
                                            source_dtype,
                                            (AmeValue) { .lo = data_raw },
                                            exponent_dtype,
                                            (AmeValue) {
                                                .lo = exponent_raw,
                                            });
                                    }
                                    assert_value(&actual, expected.raw, 0,
                                                 expected.saturated);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

static void test_integer_scale_boundaries(void)
{
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    const AmeDTypeDesc *uint8 = dtype(UINT32_C(0x00000008));
    const AmeDTypeDesc *uint8_sat = dtype(UINT32_C(0x20000008));
    const AmeDTypeDesc *int8_sat = dtype(UINT32_C(0x60000008));
    const AmeDTypeDesc *int8_rod = dtype(UINT32_C(0x58000008));
    const AmeDTypeDesc *int128 = dtype(UINT32_C(0x40000080));
    AmeValue data = { .lo = 5 };
    AmeValue huge_positive = { .hi = 1 };
    AmeValue huge_negative = { .hi = UINT64_MAX };
    AmeExecResult result;

    result = exec_binary(AME_IMPL_INT_BINARY, AME_OP_LDEXP,
                         int8, int8, data, int128, huge_positive);
    assert_value(&result, 0, 0, false);
    result = exec_binary(AME_IMPL_INT_BINARY, AME_OP_LDEXP,
                         int8_sat, int8_sat, data,
                         int128, huge_positive);
    assert_value(&result, INT8_MAX, 0, true);
    result = exec_binary(AME_IMPL_INT_BINARY, AME_OP_LDEXP,
                         int8_rod, int8_rod, data,
                         int128, huge_negative);
    assert_value(&result, 1, 0, false);

    /* ROD applies after the old destination joins the exact expression. */
    result = exec_ternary(AME_IMPL_INT_TERNARY, AME_OP_RDEXPACC,
                          int8_rod, int8_rod,
                          (AmeValue) { .lo = 3 }, int8,
                          (AmeValue) { .lo = 1 },
                          (AmeValue) { .lo = 10 });
    assert_value(&result, 11, 0, false);
    result = exec_ternary(AME_IMPL_INT_TERNARY, AME_OP_RDEXPACC,
                          int8_rod, int8_rod,
                          (AmeValue) { .lo = 3 }, int8,
                          (AmeValue) { .lo = 1 },
                          (AmeValue) { .lo = 11 });
    assert_value(&result, 13, 0, false);

    /* A negative huge left shift clamps to zero for an unsigned result. */
    data = (AmeValue) { .lo = UINT8_MAX };
    result = exec_binary(AME_IMPL_INT_BINARY, AME_OP_LDEXP,
                         uint8_sat, int8, data, int128, huge_positive);
    assert_value(&result, 0, 0, true);
    result = exec_binary(AME_IMPL_INT_BINARY, AME_OP_RDEXP,
                         uint8_sat, int8, data, int128, huge_negative);
    assert_value(&result, 0, 0, true);
    result = exec_ternary(AME_IMPL_INT_TERNARY, AME_OP_LDEXPACC,
                          uint8_sat, int8, data, int128, huge_positive,
                          (AmeValue) { .lo = UINT8_MAX });
    assert_value(&result, 0, 0, true);
    result = exec_ternary(AME_IMPL_INT_TERNARY, AME_OP_RDEXPACC,
                          uint8, int8, data, int128, huge_negative,
                          (AmeValue) { .lo = UINT8_MAX });
    assert_value(&result, UINT8_MAX, 0, false);
}

static void test_integer_fused_exhaustive_small(void)
{
    static const AmeOperation ops[] = {
        AME_OP_MULACC, AME_OP_MULNACC,
        AME_OP_MULADD, AME_OP_MULSUB,
    };
    unsigned int signs, saturating;

    for (signs = 0; signs < 8; signs++) {
        bool is_signed = signs & 1, a_signed = signs & 2, b_signed = signs & 4;
        const AmeDTypeDesc *a = dtype(4 | ((uint32_t)a_signed << 30));
        const AmeDTypeDesc *b = dtype(4 | ((uint32_t)b_signed << 30));

        for (saturating = 0; saturating < 2; saturating++) {
            uint32_t encoding = 4 | (is_signed << 30) |
                                (saturating << 29);
            const AmeDTypeDesc *type = dtype(encoding);
            unsigned int src1_raw, src2_raw, current_raw;

            for (src1_raw = 0; src1_raw < 16; src1_raw++) {
                int src1 = int4_value(src1_raw, a_signed);

                for (src2_raw = 0; src2_raw < 16; src2_raw++) {
                    int src2 = int4_value(src2_raw, b_signed);

                    for (current_raw = 0; current_raw < 16;
                         current_raw++) {
                        int current = int4_value(current_raw, is_signed);
                        int exact[] = {
                            current + src1 * src2,
                            current - src1 * src2,
                            src1 + src2 * current,
                            src1 - src2 * current,
                        };
                        size_t op_index;

                        for (op_index = 0; op_index < ARRAY_SIZE(ops);
                             op_index++) {
                            bool expected_sat;
                            unsigned int expected = int4_finish(
                                exact[op_index], is_signed, saturating,
                                &expected_sat);
                            AmeExecResult actual = exec_ternary(
                                AME_IMPL_INT_TERNARY, ops[op_index], type,
                                a, (AmeValue) { .lo = src1_raw }, b,
                                (AmeValue) { .lo = src2_raw },
                                (AmeValue) { .lo = current_raw });

                            assert_value(&actual, expected, 0,
                                         expected_sat);
                        }
                    }
                }
            }
        }
    }
}

static void test_integer_ternary_dispatch(void)
{
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    const AmeDTypeDesc *uint8 = dtype(UINT32_C(0x00000008));
    const AmeDTypeDesc *int8_sat = dtype(UINT32_C(0x60000008));
    const AmeDTypeDesc *uint128 = dtype(UINT32_C(0x00000080));
    const AmeDTypeDesc *uint128_sat = dtype(UINT32_C(0x20000080));
    const AmeDTypeDesc *fp32 = dtype(UINT32_C(0x20300120));
    AmeExecResult result;

    result = exec_ternary(AME_IMPL_SELECT, AME_OP_MUXGE,
                          int8, int8, (AmeValue) { .lo = 0xfd },
                          int8, (AmeValue) { .lo = 4 },
                          (AmeValue) { .lo = 10 });
    assert_value(&result, 10, 0, false);
    result = exec_ternary(AME_IMPL_SELECT, AME_OP_MUXLT,
                          int8, int8, (AmeValue) { .lo = 0xfd },
                          int8, (AmeValue) { .lo = 4 },
                          (AmeValue) { .lo = 10 });
    assert_value(&result, 4, 0, false);
    result = exec_ternary(AME_IMPL_SELECT, AME_OP_MUXGE,
                          int8, uint8, (AmeValue) { .lo = UINT8_MAX },
                          int8, (AmeValue) { .lo = 4 },
                          (AmeValue) { .lo = 10 });
    assert_value(&result, 4, 0, false);
    result = exec_ternary(AME_IMPL_SELECT, AME_OP_MUXGE,
                          int8, fp32,
                          (AmeValue) { .lo = UINT32_C(0x80000000) },
                          int8, (AmeValue) { .lo = 4 },
                          (AmeValue) { .lo = 10 });
    assert_value(&result, 4, 0, false);
    result = exec_ternary(AME_IMPL_SELECT, AME_OP_MUXLT,
                          int8, fp32,
                          (AmeValue) { .lo = UINT32_C(0x7fc00000) },
                          int8, (AmeValue) { .lo = 4 },
                          (AmeValue) { .lo = 10 });
    assert_value(&result, 10, 0, false);

    result = exec_ternary(AME_IMPL_INT_TERNARY, AME_OP_MULACC,
                          int8_sat, int8_sat,
                          (AmeValue) { .lo = 20 }, int8_sat,
                          (AmeValue) { .lo = 2 },
                          (AmeValue) { .lo = 100 });
    assert_value(&result, INT8_MAX, 0, true);

    result = exec_ternary(
        AME_IMPL_INT_TERNARY, AME_OP_MULACC, uint128,
        uint128, (AmeValue) { .lo = UINT64_MAX, .hi = UINT64_MAX },
        uint128, (AmeValue) { .lo = 2 },
        (AmeValue) { .lo = 1 });
    assert_value(&result, UINT64_MAX, UINT64_MAX, false);
    result = exec_ternary(
        AME_IMPL_INT_TERNARY, AME_OP_MULACC, uint128_sat,
        uint128_sat, (AmeValue) { .lo = UINT64_MAX, .hi = UINT64_MAX },
        uint128_sat, (AmeValue) { .lo = 2 },
        (AmeValue) { .lo = 1 });
    assert_value(&result, UINT64_MAX, UINT64_MAX, true);

    g_assert_false(riscv_ame_exec_ternary(
        AME_IMPL_INT_BINARY, AME_OP_MULACC, int8, int8,
        &(AmeValue) { .lo = 1 }, int8, &(AmeValue) { .lo = 2 },
        &(AmeValue) { .lo = 3 }, &result, NULL));
}

static void init_binary_views(const RISCVAMELayout *layout,
                              AmeMView *destination, uint32_t dest_base,
                              const AmeDTypeDesc *dest_dtype,
                              AmeMView *source1, uint32_t src1_base,
                              const AmeDTypeDesc *src1_dtype,
                              AmeMView *source2, uint32_t src2_base,
                              const AmeDTypeDesc *src2_dtype)
{
    const AmeDTypeDesc *dtypes[] = {
        dest_dtype, src1_dtype, src2_dtype,
    };
    uint32_t n_sq;

    g_assert_true(riscv_ame_n_sq(layout, dtypes, ARRAY_SIZE(dtypes),
                                &n_sq));
    g_assert_true(riscv_ame_m_view_init(layout, destination, dest_base,
                                       dest_dtype, n_sq));
    g_assert_true(riscv_ame_m_view_init(layout, source1, src1_base,
                                       src1_dtype, n_sq));
    g_assert_true(riscv_ame_m_view_init(layout, source2, src2_base,
                                       src2_dtype, n_sq));
}

static AmeValue equal_width_test_value(unsigned bits, size_t index)
{
    uint64_t sign = BIT_ULL(bits - 1);
    const uint64_t values[] = {
        0, 1, sign, sign - 1, UINT64_MAX, UINT64_MAX - 1,
        UINT64_C(0x5555555555555555), UINT64_C(0xaaaaaaaaaaaaaaaa),
    };

    return (AmeValue) { .lo = values[index % ARRAY_SIZE(values)] };
}

static void assert_equal_width_binary(const AmeDTypeDesc *d,
                                      const AmeDTypeDesc *a,
                                      const AmeDTypeDesc *b)
{
    static const AmeOperation ops[] = {
        AME_OP_ADD, AME_OP_SUB, AME_OP_MUL, AME_OP_NMUL,
    };
    RISCVAMELayout layout = test_layout_geometry(2, 16);
    size_t capacity = layout.mregs_bytes + 16;
    g_autofree uint8_t *mregs = g_malloc(layout.mregs_bytes);
    g_autofree uint8_t *original = g_malloc(layout.mregs_bytes);
    g_autofree uint8_t *result = g_malloc(capacity);
    g_autofree uint8_t *expected = g_malloc(capacity);

    for (unsigned overlap = 0; overlap < 2; overlap++) {
        AmeMView dst, first, second, staged;
        AmeExecFlags rejected_flags;
        size_t bytes;

        init_binary_views(&layout, &dst, overlap ? 0 : 16, d,
                          &first, 0, a, &second, 8, b);
        staged = staged_view(&dst);
        bytes = riscv_ame_m_view_bytes(&layout, &dst);
        for (unsigned batch = 0; batch < 2; batch++) {
            memset(mregs, 0xa5, layout.mregs_bytes);
            for (size_t e = 0; e < dst.total_elements; e++) {
                AmeValue left = equal_width_test_value(d->bits, e + 4 * batch);
                AmeValue right = equal_width_test_value(d->bits,
                                                        3 * e + 4 * batch + 1);

                riscv_ame_m_write(&layout, mregs, &first, e, &left);
                riscv_ame_m_write(&layout, mregs, &second, e, &right);
            }
            memcpy(original, mregs, layout.mregs_bytes);
            for (unsigned form = 0; form < 3; form++) {
                AmeValue scalar_a = equal_width_test_value(d->bits, 4 * batch);
                AmeValue scalar_b = equal_width_test_value(d->bits,
                                                           4 * batch + 2);

                for (unsigned op = 0; op < ARRAY_SIZE(ops); op++) {
                    AmeExecFlags flags = {
                        .fflags = UINT8_MAX, .saturated = true,
                    };
                    AmeExecFlags reference = { 0 };
                    bool valid;

                    memset(result, 0xa5, capacity);
                    memset(expected, 0xa5, capacity);
                    memset(expected, 0, bytes);
                    for (size_t e = 0; e < dst.total_elements; e++) {
                        AmeValue left = form == 2 ? scalar_a :
                            riscv_ame_m_read(&layout, mregs, &first, e);
                        AmeValue right = form == 1 ? scalar_b :
                            riscv_ame_m_read(&layout, mregs, &second, e);
                        AmeExecResult executed;

                        g_assert_true(riscv_ame_exec_binary(
                            AME_IMPL_INT_BINARY, ops[op], d, a, &left,
                            b, &right, &executed, NULL));
                        riscv_ame_m_write(&layout, expected, &staged, e,
                                          &executed.value);
                        reference.fflags |= executed.fflags;
                        reference.saturated |= executed.saturated;
                    }
                    if (form == 0) {
                        valid = riscv_ame_exec_m_binary(
                            AME_IMPL_INT_BINARY, ops[op], &layout, mregs,
                            &dst, &first, &second, result, capacity, &flags);
                    } else {
                        valid = riscv_ame_exec_m_binary_scalar(
                            AME_IMPL_INT_BINARY, ops[op], &layout, mregs,
                            &dst, form == 2 ? &second : &first,
                            form == 2 ? a : b,
                            form == 2 ? &scalar_a : &scalar_b, form == 2,
                            result, capacity, &flags);
                    }
                    g_assert_true(valid);
                    g_assert_cmpmem(result, capacity, expected, capacity);
                    g_assert_cmpmem(mregs, layout.mregs_bytes,
                                    original, layout.mregs_bytes);
                    g_assert_cmphex(flags.fflags, ==, reference.fflags);
                    g_assert_cmpint(flags.saturated, ==, reference.saturated);
                }
            }
        }

        /* Equal width does not authorize a different-datatype bitwise tuple. */
        memset(result, 0xa5, capacity);
        memset(expected, 0xa5, capacity);
        rejected_flags = (AmeExecFlags) { .fflags = 7, .saturated = true };

        g_assert_false(riscv_ame_exec_m_binary(
            AME_IMPL_INT_BINARY, AME_OP_AND, &layout, mregs,
            &dst, &first, &second, result, capacity, &rejected_flags));
        g_assert_cmpmem(result, capacity, expected, capacity);
        g_assert_cmphex(rejected_flags.fflags, ==, 7);
        g_assert_true(rejected_flags.saturated);
    }
}

static void test_equal_width_integer_binary(void)
{
    for (unsigned bits = 4; bits <= 64; bits *= 2) {
        for (unsigned rm = 0; rm < AME_INT_ROUND_MAX; rm++) {
            for (unsigned d_sign = 0; d_sign < 2; d_sign++) {
                const AmeDTypeDesc *d = dtype(bits | (rm << 27) |
                                              (d_sign << 30));

                for (unsigned signs = 0; signs < 4; signs++) {
                    for (unsigned sat = 0; sat < 4; sat++) {
                        const AmeDTypeDesc *a = dtype(
                            bits | (((rm + 1) % AME_INT_ROUND_MAX) << 27) |
                            ((signs & 1) << 30) | ((sat & 1) << 29));
                        const AmeDTypeDesc *b = dtype(
                            bits | (((rm + 2) % AME_INT_ROUND_MAX) << 27) |
                            ((signs >> 1) << 30) | ((sat >> 1) << 29));

                        assert_equal_width_binary(d, a, b);
                    }
                }
            }
        }
    }
}

static AmeValue elementwise_test_value(const AmeDTypeDesc *desc, size_t index)
{
    static const uint64_t fp16[] = { 0x3e00, 0xbc00, 0x7c01, 0x7bff };
    static const uint64_t fp32[] = {
        0x3fc00000, 0xbf800000, 0x7f800001, 0x7f7fffff,
    };
    static const int64_t integers[] = { 127, -129, 1000, -17 };

    index %= ARRAY_SIZE(integers);
    if (riscv_ame_dtype_is_fp(desc)) {
        switch (desc->fp.format) {
        case AME_FP_FORMAT_FP16:
            return (AmeValue) { .lo = fp16[index] };
        case AME_FP_FORMAT_FP32:
            return (AmeValue) { .lo = fp32[index] };
        default:
            g_assert_not_reached();
        }
    }
    return (AmeValue) {
        .lo = integers[index],
        .hi = integers[index] < 0 ? UINT64_MAX : 0,
    };
}

static void test_elementwise_loop_staging(void)
{
    static const struct {
        unsigned sources;
        AmeImplClass impl;
        AmeOperation op;
        uint32_t d, a, b;
    } cases[] = {
        { 1, AME_IMPL_INT_CONVERT, AME_OP_CONV,
          0x60000004, 0x00000010 },
        { 1, AME_IMPL_FP_CONVERT, AME_OP_CONV,
          0x60000008, 0x20300120 },
        { 1, AME_IMPL_INT_UNARY, AME_OP_ABS,
          0x60000008, 0x40000010 },
        { 1, AME_IMPL_FP_UNARY, AME_OP_SQRT,
          0x14300110, 0x20300120 },
        { 2, AME_IMPL_INT_BINARY, AME_OP_ADD,
          0x60000008, 0x40000010, 0x00000008 },
        { 2, AME_IMPL_INT_BINARY, AME_OP_MUL,
          0x60000080, 0x00000080, 0x40000040 },
        { 2, AME_IMPL_FP_BINARY, AME_OP_SUB,
          0x14300110, 0x20300120, 0x14300110 },
        { 2, AME_IMPL_SELECT, AME_OP_SELGE,
          0x40000010, 0x40000008, 0x40000010 },
        { 2, AME_IMPL_SELECT, AME_OP_SELGE,
          0x40000010, 0x20300120, 0x40000010 },
        { 2, AME_IMPL_SELECT, AME_OP_SELGE,
          0x20300120, 0x40000008, 0x20300120 },
        { 3, AME_IMPL_INT_TERNARY, AME_OP_MULACC,
          0x60000008, 0x40000010, 0x00000008 },
        { 3, AME_IMPL_FP_TERNARY, AME_OP_MULACC,
          0x14300110, 0x20300120, 0x14300110 },
        { 3, AME_IMPL_FP_TERNARY, AME_OP_LDEXPACC,
          0x14300110, 0x20300120, 0x40000008 },
        { 3, AME_IMPL_SELECT, AME_OP_MUXGE,
          0x40000010, 0x40000008, 0x40000010 },
        { 3, AME_IMPL_SELECT, AME_OP_MUXGE,
          0x40000010, 0x20300120, 0x40000010 },
        { 3, AME_IMPL_SELECT, AME_OP_MUXGE,
          0x20300120, 0x40000008, 0x20300120 },
    };
    RISCVAMELayout layout = test_layout_geometry(2, 16);
    size_t capacity = layout.mregs_bytes + 16;
    g_autofree uint8_t *mregs = g_malloc(layout.mregs_bytes);
    g_autofree uint8_t *original = g_malloc(layout.mregs_bytes);
    g_autofree uint8_t *result = g_malloc(capacity);
    g_autofree uint8_t *expected = g_malloc(capacity);

    for (size_t c = 0; c < ARRAY_SIZE(cases); c++) {
        const AmeDTypeDesc *d = dtype(cases[c].d);
        const AmeDTypeDesc *a = dtype(cases[c].a);
        const AmeDTypeDesc *b = a;
        unsigned forms = 1;

        if (cases[c].sources > 1) {
            b = dtype(cases[c].b);
            forms = 3;
        }

        for (unsigned overlap = 0; overlap < 2; overlap++) {
            AmeMView destination, source1, source2, staged;
            size_t bytes;

            memset(mregs, 0xa5, layout.mregs_bytes);
            init_binary_views(&layout, &destination, overlap ? 0 : 16, d,
                              &source1, 0, a, &source2, 8, b);
            staged = staged_view(&destination);
            bytes = riscv_ame_m_view_bytes(&layout, &destination);
            for (size_t e = 0; e < destination.total_elements; e++) {
                AmeValue left = elementwise_test_value(a, e);
                AmeValue right = elementwise_test_value(b, e + 1);
                AmeValue current = elementwise_test_value(d, e + 2);

                riscv_ame_m_write(&layout, mregs, &source1, e, &left);
                riscv_ame_m_write(&layout, mregs, &source2, e, &right);
                riscv_ame_m_write(&layout, mregs, &destination, e, &current);
            }
            memcpy(original, mregs, layout.mregs_bytes);

            /* Tensor inputs, then a scalar in either arithmetic position. */
            for (unsigned form = 0; form < forms; form++) {
                bool scalar_first = form == 2;
                const AmeMView *source = scalar_first ? &source2 : &source1;
                const AmeDTypeDesc *scalar_dtype = scalar_first ? a : b;
                AmeValue scalar = riscv_ame_m_read(
                    &layout, mregs, scalar_first ? &source1 : &source2, 1);
                AmeExecFlags flags = { .fflags = UINT8_MAX, .saturated = true };
                AmeExecFlags reference = { 0 };
                float_status storage;
                float_status *status = NULL;
                bool valid;

                if (riscv_ame_impl_is_fp(cases[c].impl)) {
                    g_assert_true(riscv_ame_fp_status_prepare(&storage, d));
                    status = &storage;
                } else if (cases[c].impl == AME_IMPL_SELECT &&
                           riscv_ame_dtype_is_fp(a)) {
                    g_assert_true(riscv_ame_fp_status_prepare(&storage, a));
                    status = &storage;
                }
                memset(result, 0xa5, capacity);
                memset(expected, 0xa5, capacity);
                memset(expected, 0, bytes);
                for (size_t e = 0; e < destination.total_elements; e++) {
                    AmeValue left = form == 2 ? scalar :
                        riscv_ame_m_read(&layout, mregs, &source1, e);
                    AmeValue right = form == 1 ? scalar :
                        riscv_ame_m_read(&layout, mregs, &source2, e);
                    AmeValue current = riscv_ame_m_read(
                        &layout, mregs, &destination, e);
                    AmeExecResult executed;

                    if (cases[c].op == AME_OP_CONV) {
                        valid = riscv_ame_exec_convert(
                            cases[c].impl, d, a, &left, &executed, status);
                    } else if (cases[c].sources == 1) {
                        valid = riscv_ame_exec_unary(
                            cases[c].impl, cases[c].op, d, a, &left,
                            &executed, status);
                    } else if (cases[c].sources == 2) {
                        valid = riscv_ame_exec_binary(
                            cases[c].impl, cases[c].op, d, a, &left,
                            b, &right, &executed, status);
                    } else {
                        valid = riscv_ame_exec_ternary(
                            cases[c].impl, cases[c].op, d, a, &left,
                            b, &right, &current, &executed, status);
                    }
                    g_assert_true(valid);
                    riscv_ame_m_write(&layout, expected, &staged, e,
                                      &executed.value);
                    reference.fflags |= executed.fflags;
                    reference.saturated |= executed.saturated;
                }

                if (cases[c].op == AME_OP_CONV) {
                    valid = riscv_ame_exec_m_convert(
                        cases[c].impl, &layout, mregs, &destination, &source1,
                        result, capacity, &flags);
                } else if (cases[c].sources == 1) {
                    valid = riscv_ame_exec_m_unary(
                        cases[c].impl, cases[c].op, &layout, mregs,
                        &destination, &source1, result, capacity, &flags);
                } else if (cases[c].sources == 2 && form == 0) {
                    valid = riscv_ame_exec_m_binary(
                        cases[c].impl, cases[c].op, &layout, mregs,
                        &destination, &source1, &source2,
                        result, capacity, &flags);
                } else if (cases[c].sources == 2) {
                    valid = riscv_ame_exec_m_binary_scalar(
                        cases[c].impl, cases[c].op, &layout, mregs,
                        &destination, source, scalar_dtype, &scalar,
                        scalar_first, result, capacity, &flags);
                } else if (form == 0) {
                    valid = riscv_ame_exec_m_ternary(
                        cases[c].impl, cases[c].op, &layout, mregs,
                        &destination, &source1, &source2,
                        result, capacity, &flags);
                } else {
                    valid = riscv_ame_exec_m_ternary_scalar(
                        cases[c].impl, cases[c].op, &layout, mregs,
                        &destination, source, scalar_dtype, &scalar,
                        scalar_first, result, capacity, &flags);
                }
                g_assert_true(valid);
                g_assert_cmpmem(result, capacity, expected, capacity);
                g_assert_cmpmem(mregs, layout.mregs_bytes,
                                original, layout.mregs_bytes);
                g_assert_cmphex(flags.fflags, ==, reference.fflags);
                g_assert_cmpint(flags.saturated, ==, reference.saturated);
            }
        }
    }
}

static void test_integer_narrowing_staging(void)
{
    RISCVAMELayout layout = test_layout();
    const AmeDTypeDesc *int4_sat = dtype(0x60000004);
    const AmeDTypeDesc *uint16 = dtype(0x00000010);
    const AmeDTypeDesc *int16 = dtype(0x40000010);
    g_autofree uint8_t *mregs = g_malloc0(layout.mregs_bytes);
    g_autofree uint8_t *result = g_malloc0(layout.mregs_bytes);

    for (unsigned overlap = 0; overlap < 2; overlap++) {
        AmeMView destination, source1, source2, staged;
        AmeExecFlags flags = { .fflags = UINT8_MAX, .saturated = true };

        memset(mregs, 0xa5, layout.mregs_bytes);
        memset(result, 0xa5, layout.mregs_bytes);
        init_binary_views(&layout, &destination, overlap ? 1 : 4, int4_sat,
                          &source1, 0, uint16,
                          &source2, 8, int16);
        g_assert_cmpuint(destination.pack_factor, ==, 2);
        g_assert_cmpuint(destination.n_sq, ==, 2);
        g_assert_cmpuint(source1.span, ==, 4);
        for (size_t e = 0; e < source1.total_elements; e++) {
            AmeValue left = { .lo = 1000 + e };
            AmeValue right = { .lo = (uint16_t)(-995 - (int)e) };

            riscv_ame_m_write(&layout, mregs, &source1, e, &left);
            riscv_ame_m_write(&layout, mregs, &source2, e, &right);
        }
        g_assert_true(riscv_ame_exec_m_binary(
            AME_IMPL_INT_BINARY, AME_OP_ADD, &layout, mregs,
            &destination, &source1, &source2, result, layout.mregs_bytes,
            &flags));
        g_assert_false(flags.saturated);
        g_assert_cmphex(flags.fflags, ==, 0);
        staged = staged_view(&destination);
        for (size_t e = 0; e < destination.total_elements; e++) {
            AmeValue actual = riscv_ame_m_read(&layout, result, &staged, e);

            g_assert_cmphex(actual.lo, ==, 5);
            g_assert_cmphex(actual.hi, ==, 0);
        }
        /* D fills one packed register; do not write beyond its backing. */
        g_assert_cmphex(result[riscv_ame_m_view_bytes(&layout, &destination)],
                        ==, 0xa5);
    }
}

static void test_integer_tensor_and_scalar_staging(void)
{
    RISCVAMELayout layout = test_layout();
    const AmeDTypeDesc *int4 = dtype(UINT32_C(0x40000004));
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    const AmeDTypeDesc *int16 = dtype(UINT32_C(0x40000010));
    g_autofree uint8_t *mregs = g_malloc0(layout.mregs_bytes);
    g_autofree uint8_t *result = g_malloc0(layout.mregs_bytes);
    AmeMView destination, source1, source2, staged;
    AmeValue scalar = { .lo = 10 };
    AmeExecFlags flags = { .saturated = true };
    size_t element;

    /* Source1 overlaps the wide destination; every read sees old state. */
    init_binary_views(&layout, &destination, 0, int16,
                      &source1, 1, int8, &source2, 4, int4);
    for (element = 0; element < destination.total_elements; element++) {
        AmeValue left = { .lo = element + 1 };
        AmeValue right = { .lo = 2 };

        riscv_ame_m_write(&layout, mregs, &source1, element, &left);
        riscv_ame_m_write(&layout, mregs, &source2, element, &right);
    }
    g_assert_true(riscv_ame_exec_m_binary(
        AME_IMPL_INT_BINARY, AME_OP_ADD, &layout, mregs,
        &destination, &source1, &source2, result, layout.mregs_bytes,
        &flags));
    g_assert_false(flags.saturated);
    g_assert_cmphex(flags.fflags, ==, 0);
    staged = staged_view(&destination);
    for (element = 0; element < staged.total_elements; element++) {
        AmeValue value = riscv_ame_m_read(&layout, result, &staged,
                                          element);

        g_assert_cmphex(value.lo, ==, element + 3);
    }

    init_views(&layout, &destination, 8, int8,
               &source1, 10, int8);
    for (element = 0; element < source1.total_elements; element++) {
        AmeValue value = { .lo = element + 1 };

        riscv_ame_m_write(&layout, mregs, &source1, element, &value);
    }
    g_assert_true(riscv_ame_exec_m_binary_scalar(
        AME_IMPL_INT_BINARY, AME_OP_SUB, &layout, mregs,
        &destination, &source1, source1.dtype, &scalar, false, result,
        layout.mregs_bytes, &flags));
    staged = staged_view(&destination);
    for (element = 0; element < staged.total_elements; element++) {
        AmeValue value = riscv_ame_m_read(&layout, result, &staged,
                                          element);

        g_assert_cmphex(value.lo, ==, 9 - element);
    }
}

static void test_integer_mixed_tensor(void)
{
    RISCVAMELayout layout = test_layout();
    const AmeDTypeDesc *s4 = dtype(0x40000004);
    const AmeDTypeDesc *u8 = dtype(0x00000008);
    const AmeDTypeDesc *s8 = dtype(0x40000008);
    const AmeDTypeDesc *u16 = dtype(0x00000010);
    const AmeDTypeDesc *s16 = dtype(0x40000010);
    g_autofree uint8_t *mregs = g_malloc0(layout.mregs_bytes);
    g_autofree uint8_t *result = g_malloc0(layout.mregs_bytes);
    AmeMView d, a, b, staged;
    AmeExecFlags flags = { 0 };
    AmeExecResult converted;

    /* Packed negative A and unsigned B form two squares, widening into D. */
    init_binary_views(&layout, &d, 0, s16, &a, 4, s4, &b, 6, u8);
    for (size_t i = 0; i < d.total_elements; i++) {
        riscv_ame_m_write(&layout, mregs, &a, i, &(AmeValue) { .lo = 15 });
        riscv_ame_m_write(&layout, mregs, &b, i, &(AmeValue) { .lo = 255 });
    }
    g_assert_true(riscv_ame_exec_m_binary(
        AME_IMPL_INT_BINARY, AME_OP_ADD, &layout, mregs,
        &d, &a, &b, result, layout.mregs_bytes, &flags));
    g_assert_false(flags.saturated);
    staged = staged_view(&d);
    for (size_t i = 0; i < d.total_elements; i++) {
        g_assert_cmphex(riscv_ame_m_read(&layout, result, &staged, i).lo,
                        ==, 254);
    }

    /* Uint16(255) converts to Int8(-1) before the unsigned destination add. */
    init_views(&layout, &d, 0, u16, &a, 4, s8);
    for (size_t i = 0; i < a.total_elements; i++) {
        riscv_ame_m_write(&layout, mregs, &a, i, &(AmeValue) { .lo = 2 });
    }
    /* The TCG helper performs this conversion before entering the loop. */
    g_assert_true(riscv_ame_exec_convert(
        AME_IMPL_INT_CONVERT, s8, u16, &(AmeValue) { .lo = 255 },
        &converted, NULL));
    g_assert_false(converted.saturated);
    g_assert_true(riscv_ame_exec_m_binary_scalar(
        AME_IMPL_INT_BINARY, AME_OP_ADD, &layout, mregs, &d, &a,
        s8, &converted.value, false, result,
        layout.mregs_bytes, &flags));
    g_assert_false(flags.saturated);
    staged = staged_view(&d);
    for (size_t i = 0; i < d.total_elements; i++) {
        g_assert_cmphex(riscv_ame_m_read(&layout, result, &staged, i).lo,
                        ==, 1);
    }
}

static void test_integer_ternary_staging(void)
{
    RISCVAMELayout layout = test_layout();
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    const AmeDTypeDesc *int16 = dtype(UINT32_C(0x40000010));
    const AmeDTypeDesc *int128 = dtype(UINT32_C(0x40000080));
    g_autofree uint8_t *mregs = g_malloc0(layout.mregs_bytes);
    g_autofree uint8_t *result = g_malloc0(layout.mregs_bytes);
    g_autofree AmeValue *expected = NULL;
    AmeMView destination, source1, source2, staged;
    AmeExecFlags flags = { .saturated = true };
    AmeValue exponent = { .lo = 1 };
    size_t element;

    init_binary_views(&layout, &destination, 0, int16,
                      &source1, 1, int8, &source2, 4, int8);
    for (element = 0; element < destination.total_elements; element++) {
        AmeValue current = { .lo = element + 10 };
        AmeValue right = { .lo = 2 };

        riscv_ame_m_write(&layout, mregs, &destination, element,
                          &current);
        riscv_ame_m_write(&layout, mregs, &source2, element, &right);
    }
    expected = g_new(AmeValue, destination.total_elements);
    for (element = 0; element < destination.total_elements; element++) {
        AmeValue current = riscv_ame_m_read(
            &layout, mregs, &destination, element);
        AmeValue left = riscv_ame_m_read(&layout, mregs, &source1,
                                         element);
        AmeValue right = riscv_ame_m_read(&layout, mregs, &source2,
                                          element);
        AmeExecResult executed = exec_ternary(
            AME_IMPL_INT_TERNARY, AME_OP_MULACC, int16,
            int8, left, int8, right, current);

        expected[element] = executed.value;
    }
    g_assert_true(riscv_ame_exec_m_ternary(
        AME_IMPL_INT_TERNARY, AME_OP_MULACC, &layout, mregs,
        &destination, &source1, &source2, result, layout.mregs_bytes,
        &flags));
    g_assert_false(flags.saturated);
    staged = staged_view(&destination);
    for (element = 0; element < staged.total_elements; element++) {
        AmeValue actual = riscv_ame_m_read(&layout, result, &staged,
                                           element);

        g_assert_cmphex(actual.lo, ==, expected[element].lo);
        g_assert_cmphex(actual.hi, ==, expected[element].hi);
    }

    init_views(&layout, &destination, 8, int8,
               &source1, 10, int8);
    for (element = 0; element < destination.total_elements; element++) {
        AmeValue current = { .lo = 10 };
        AmeValue source = { .lo = element + 1 };

        riscv_ame_m_write(&layout, mregs, &destination, element,
                          &current);
        riscv_ame_m_write(&layout, mregs, &source1, element, &source);
    }
    g_assert_true(riscv_ame_exec_m_ternary_scalar(
        AME_IMPL_INT_TERNARY, AME_OP_LDEXPACC, &layout, mregs,
        &destination, &source1, int128, &exponent, false,
        result, layout.mregs_bytes, &flags));
    g_assert_false(flags.saturated);
    staged = staged_view(&destination);
    for (element = 0; element < staged.total_elements; element++) {
        AmeValue actual = riscv_ame_m_read(&layout, result, &staged,
                                           element);

        g_assert_cmphex(actual.lo, ==, 10 + 2 * (element + 1));
    }
}

static void test_ieee_tensor_conversion_elementwise(void)
{
    RISCVAMELayout layout = test_layout_geometry(2, 32);
    const AmeDTypeDesc *int64 = dtype(UINT32_C(0x40000040));
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    const AmeDTypeDesc *fp16 = dtype(UINT32_C(0x14300110));
    const AmeDTypeDesc *bf16 = dtype(UINT32_C(0x20300110));
    const AmeDTypeDesc *fp32 = dtype(UINT32_C(0x20300120));
    g_autofree uint8_t *mregs = g_malloc0(layout.mregs_bytes);
    g_autofree uint8_t *result = g_malloc(layout.mregs_bytes);
    AmeMView destination, source1, source2, staged;
    AmeExecFlags flags = { .fflags = UINT8_MAX, .saturated = true };
    size_t element;

    init_views(&layout, &destination, 0, fp32, &source1, 4, int64);
    for (element = 0; element < source1.total_elements; element++) {
        AmeValue value = { .lo = element == 0 ? UINT64_C(16777217) : 1 };

        riscv_ame_m_write(&layout, mregs, &source1, element, &value);
    }
    g_assert_true(riscv_ame_exec_m_convert(
        AME_IMPL_FP_CONVERT, &layout, mregs, &destination, &source1,
        result, layout.mregs_bytes, &flags));
    g_assert_cmphex(flags.fflags, ==, AME_FFLAG_NX);
    g_assert_false(flags.saturated);
    staged = staged_view(&destination);
    for (element = 0; element < staged.total_elements; element++) {
        AmeValue actual = riscv_ame_m_read(&layout, result, &staged,
                                           element);

        g_assert_cmphex(actual.lo, ==,
                        element == 0 ? UINT32_C(0x4b800000) :
                                       UINT32_C(0x3f800000));
    }

    memset(mregs, 0, layout.mregs_bytes);
    init_binary_views(&layout, &destination, 0, fp32,
                      &source1, 4, fp16, &source2, 5, fp16);
    for (element = 0; element < destination.total_elements; element++) {
        AmeValue left = {
            .lo = element == 1 ? UINT16_C(0x7d00) :
                                 UINT16_C(0x3e00),
        };
        AmeValue right = { .lo = UINT16_C(0x4000) };

        riscv_ame_m_write(&layout, mregs, &source1, element, &left);
        riscv_ame_m_write(&layout, mregs, &source2, element, &right);
    }
    flags = (AmeExecFlags) { 0 };
    g_assert_true(riscv_ame_exec_m_binary(
        AME_IMPL_FP_BINARY, AME_OP_ADD, &layout, mregs,
        &destination, &source1, &source2, result, layout.mregs_bytes,
        &flags));
    g_assert_cmphex(flags.fflags, ==, AME_FFLAG_NV);
    staged = staged_view(&destination);
    for (element = 0; element < staged.total_elements; element++) {
        AmeValue actual = riscv_ame_m_read(&layout, result, &staged,
                                           element);

        g_assert_cmphex(actual.lo, ==,
                        element == 1 ? UINT32_C(0x7fc00000) :
                                       UINT32_C(0x40600000));
    }

    memset(mregs, 0, layout.mregs_bytes);
    init_views(&layout, &destination, 0, fp32, &source1, 4, fp32);
    for (element = 0; element < source1.total_elements; element++) {
        AmeValue value = {
            .lo = element == 0 ? 0 : UINT32_C(0x3f800000),
        };

        riscv_ame_m_write(&layout, mregs, &source1, element, &value);
    }
    g_assert_true(riscv_ame_exec_m_unary(
        AME_IMPL_FP_UNARY, AME_OP_REC, &layout, mregs,
        &destination, &source1, result, layout.mregs_bytes, &flags));
    g_assert_cmphex(flags.fflags, ==, AME_FFLAG_DZ);
    staged = staged_view(&destination);
    g_assert_cmphex(riscv_ame_m_read(&layout, result, &staged, 0).lo, ==,
                    UINT32_C(0x7f800000));

    init_binary_views(&layout, &destination, 0, bf16,
                      &source1, 4, fp16, &source2, 5, fp16);
    memset(mregs, 0, layout.mregs_bytes);
    for (element = 0; element < destination.total_elements; element++) {
        AmeValue left = { .lo = UINT16_C(0x3e00) };
        AmeValue right = { .lo = UINT16_C(0x4000) };

        riscv_ame_m_write(&layout, mregs, &source1, element, &left);
        riscv_ame_m_write(&layout, mregs, &source2, element, &right);
    }
    flags = (AmeExecFlags) { 0 };
    g_assert_true(riscv_ame_exec_m_binary(
        AME_IMPL_FP_BINARY, AME_OP_ADD, &layout, mregs,
        &destination, &source1, &source2, result, layout.mregs_bytes,
        &flags));
    g_assert_cmphex(flags.fflags, ==, 0);
    staged = staged_view(&destination);
    for (element = 0; element < staged.total_elements; element++) {
        g_assert_cmphex(riscv_ame_m_read(&layout, result, &staged,
                                         element).lo, ==,
                        UINT16_C(0x4060));
    }

    init_binary_views(&layout, &destination, 0, fp32,
                      &source1, 8, int8, &source2, 12, fp32);
    memset(mregs, 0, layout.mregs_bytes);
    for (element = 0; element < destination.total_elements; element++) {
        AmeValue predicate = { .lo = element & 1 ? UINT8_C(0x80) : 0 };
        AmeValue selected = { .lo = fp32_bits((float)element + 1) };

        riscv_ame_m_write(&layout, mregs, &source1, element, &predicate);
        riscv_ame_m_write(&layout, mregs, &source2, element, &selected);
    }
    g_assert_true(riscv_ame_exec_m_binary(
        AME_IMPL_SELECT, AME_OP_SEL, &layout, mregs,
        &destination, &source1, &source2, result, layout.mregs_bytes,
        &flags));
    staged = staged_view(&destination);
    for (element = 0; element < staged.total_elements; element++) {
        AmeValue actual = riscv_ame_m_read(&layout, result, &staged,
                                           element);

        g_assert_cmphex(actual.lo, ==,
                        element & 1 ? fp32_bits((float)element + 1) : 0);
    }
}

static void test_ieee_scalar_fused_scale(void)
{
    RISCVAMELayout layout = test_layout_geometry(2, 32);
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    const AmeDTypeDesc *fp32 = dtype(UINT32_C(0x20300120));
    g_autofree uint8_t *mregs = g_malloc0(layout.mregs_bytes);
    g_autofree uint8_t *result = g_malloc0(layout.mregs_bytes);
    AmeMView destination, source, staged;
    AmeValue two = { .lo = UINT32_C(0x40000000) };
    AmeValue exponent = { .lo = 1 };
    AmeExecFlags flags = { 0 };
    size_t element;

    init_views(&layout, &destination, 0, fp32, &source, 4, fp32);
    for (element = 0; element < destination.total_elements; element++) {
        riscv_ame_m_write(&layout, mregs, &destination, element,
                          &(AmeValue) { .lo = UINT32_C(0x40400000) });
        riscv_ame_m_write(&layout, mregs, &source, element,
                          &(AmeValue) { .lo = UINT32_C(0x41200000) });
    }

    g_assert_true(riscv_ame_exec_m_binary_scalar(
        AME_IMPL_FP_BINARY, AME_OP_ADD, &layout, mregs,
        &destination, &source, fp32, &two, false, result,
        layout.mregs_bytes, &flags));
    staged = staged_view(&destination);
    for (element = 0; element < staged.total_elements; element++) {
        g_assert_cmphex(riscv_ame_m_read(
                            &layout, result, &staged, element).lo, ==,
                        UINT32_C(0x41400000));
    }

    g_assert_true(riscv_ame_exec_m_ternary_scalar(
        AME_IMPL_FP_TERNARY, AME_OP_MULACC, &layout, mregs,
        &destination, &source, fp32, &two, true, result,
        layout.mregs_bytes, &flags));
    for (element = 0; element < staged.total_elements; element++) {
        g_assert_cmphex(riscv_ame_m_read(
                            &layout, result, &staged, element).lo, ==,
                        UINT32_C(0x41b80000));
    }

    g_assert_true(riscv_ame_exec_m_binary_scalar(
        AME_IMPL_FP_SCALE, AME_OP_LDEXP, &layout, mregs,
        &destination, &source, int8, &exponent, false, result,
        layout.mregs_bytes, &flags));
    for (element = 0; element < staged.total_elements; element++) {
        g_assert_cmphex(riscv_ame_m_read(
                            &layout, result, &staged, element).lo, ==,
                        UINT32_C(0x41a00000));
    }

    g_assert_true(riscv_ame_exec_m_ternary_scalar(
        AME_IMPL_FP_TERNARY, AME_OP_LDEXPACC, &layout, mregs,
        &destination, &source, int8, &exponent, false, result,
        layout.mregs_bytes, &flags));
    for (element = 0; element < staged.total_elements; element++) {
        g_assert_cmphex(riscv_ame_m_read(
                            &layout, result, &staged, element).lo, ==,
                        UINT32_C(0x41b80000));
    }

    for (element = 0; element < source.total_elements; element++) {
        riscv_ame_m_write(&layout, mregs, &source, element,
                          &(AmeValue) { .lo = UINT32_C(0x7f7fffff) });
    }
    g_assert_true(riscv_ame_exec_m_binary_scalar(
        AME_IMPL_FP_BINARY, AME_OP_MUL, &layout, mregs,
        &destination, &source, fp32, &two, false, result,
        layout.mregs_bytes, &flags));
    g_assert_cmphex(flags.fflags, ==, AME_FFLAG_OF | AME_FFLAG_NX);
}

static void test_impl_selection(void)
{
    static const struct {
        AmeOperation op;
        uint32_t dest;
        uint32_t source;
        AmeImplClass impl;
    } cases[] = {
        { AME_OP_ZERO, 0x10100108, 0, AME_IMPL_ZERO },
        { AME_OP_COLID, 0x40000008, 0, AME_IMPL_INT_INDEX },
        { AME_OP_ROWZIP, 0x14300110, 0x14300110, AME_IMPL_COPY },
        { AME_OP_SEL, 0x14300110, 0x14300108, AME_IMPL_SELECT },
        { AME_OP_SRA, 0x40000080, 0x40000080, AME_IMPL_INT_SHIFT },
        { AME_OP_CONV, 0x40000080, 0x40000008, AME_IMPL_INT_CONVERT },
        { AME_OP_CONV, 0x14300110, 0x40000080, AME_IMPL_FP_CONVERT },
        { AME_OP_CONV, 0x10100108, 0x14300110, AME_IMPL_FP_CONVERT },
        { AME_OP_CONV, 0x40000080, 0x2c300140, AME_IMPL_FP_CONVERT },
        { AME_OP_CMPGE, 0x40000080, 0x40000008, AME_IMPL_INT_COMPARE },
        { AME_OP_CMPGE, 0x40000080, 0x10100108, AME_IMPL_FP_COMPARE },
        { AME_OP_CMPLT, 0x40000008, 0x14300110, AME_IMPL_FP_COMPARE },
        { AME_OP_ABS, 0x40000080, 0x40000008, AME_IMPL_INT_UNARY },
        { AME_OP_ABS, 0x14300110, 0x10100108, AME_IMPL_FP_UNARY },
        { AME_OP_SIN, 0x10100108, 0x10100108, AME_IMPL_FP_UNARY },
        { AME_OP_ADD, 0x40000080, 0x40000008, AME_IMPL_INT_BINARY },
        { AME_OP_ADD, 0x10100108, 0x14300110, AME_IMPL_FP_BINARY },
        { AME_OP_LDEXP, 0x40000080, 0x40000008, AME_IMPL_INT_BINARY },
        { AME_OP_LDEXP, 0x14300110, 0x10100108, AME_IMPL_FP_SCALE },
        { AME_OP_MULACC, 0x40000080, 0x40000008, AME_IMPL_INT_TERNARY },
        { AME_OP_MULACC, 0x14300110, 0x10100108, AME_IMPL_FP_TERNARY },
        { AME_OP_LDEXPACC, 0x10100108, 0x14300110, AME_IMPL_FP_TERNARY },
        { AME_OP_PREFIXADD, 0x40000080, 0x40000008, AME_IMPL_INT_REDUCE },
        { AME_OP_REDUCEADD, 0x14300110, 0x10100108, AME_IMPL_FP_REDUCE },
        { AME_OP_ROWSCATADD, 0x40000080, 0x40000008, AME_IMPL_INT_SCATTER },
        { AME_OP_COLSCATMAX, 0x14300110, 0x10100108, AME_IMPL_FP_SCATTER },
        { AME_OP_MMATACC, 0x40000080, 0x40000008, AME_IMPL_INT_MATMUL },
        { AME_OP_MMBTACCNEG, 0x10100108, 0x14300110, AME_IMPL_FP_MATMUL },
    };

    for (size_t i = 0; i < ARRAY_SIZE(cases); i++) {
        const AmeDTypeDesc *source = NULL;

        if (cases[i].source) {
            source = dtype(cases[i].source);
        }
        g_assert_cmpuint(riscv_ame_exec_impl(cases[i].op, dtype(cases[i].dest),
                                            source, NULL),
                         ==, cases[i].impl);
    }
}

static void test_tanh_lane_order(void)
{
    static const struct {
        uint32_t encoding;
        uint64_t one;
    } formats[] = {
        { 0x14300110, UINT16_C(0x3c00) },
        { 0x20300110, UINT16_C(0x3f80) },
        { 0x20300120, UINT32_C(0x3f800000) },
        { 0x2c300140, UINT64_C(0x3ff0000000000000) },
        { 0x10100108, UINT8_C(0x38) },
        { 0x14300108, UINT8_C(0x3c) },
    };

    for (size_t f = 0; f < ARRAY_SIZE(formats); f++) {
        for (AmeFPRoundMode mode = AME_FP_RNE; mode <= AME_FP_RNO; mode++) {
            const AmeDTypeDesc *fp = dtype(formats[f].encoding | (mode << 22));
            RISCVAMELayout layout = test_layout_geometry(2, fp->bits);
            g_autofree uint8_t *mregs = g_malloc0(layout.mregs_bytes);
            g_autofree uint8_t *result = g_malloc0(layout.mreg_bytes);
            AmeMView destination, source, staged;
            AmeExecFlags flags;

            if (fp->class == AME_DTYPE_CLASS_OCP && mode == AME_FP_RNO) {
                continue;
            }
            init_views(&layout, &destination, 0, fp, &source, 1, fp);
            staged = staged_view(&destination);
            for (size_t one = 0; one < source.total_elements; one++) {
                for (size_t i = 0; i < source.total_elements; i++) {
                    AmeValue value = {
                        .lo = i == one ? formats[f].one :
                              (i & 1) ? BIT_ULL(fp->bits - 1) : 0,
                    };

                    riscv_ame_m_write(&layout, mregs, &source, i, &value);
                }
                g_assert_true(riscv_ame_exec_m_unary(
                    AME_IMPL_FP_UNARY, AME_OP_TANH, &layout, mregs,
                    &destination, &source, result, layout.mreg_bytes, &flags));
                g_assert_cmphex(flags.fflags, ==, AME_FFLAG_NX);
                for (size_t i = 0; i < source.total_elements; i++) {
                    if (i != one) {
                        g_assert_cmphex(riscv_ame_m_read(
                            &layout, result, &staged, i).lo, ==,
                            (i & 1) ? BIT_ULL(fp->bits - 1) : 0);
                    }
                }
            }
        }
    }
}

static void test_nonstandard_impl_selection(void)
{
    const AmeDTypeDesc *fp32 = dtype(0x20300120);
    AmeDTypeDesc unknown = *fp32;

    /* These descriptors only exercise rejection; no custom format is added. */
    for (unsigned cls = AME_DTYPE_CLASS_CUSTOM;
         cls <= AME_DTYPE_CLASS_MAX; cls++) {
        unknown.class = cls;
        for (AmeOperation op = 0; op < AME_OP_COUNT; op++) {
            g_assert_cmpuint(riscv_ame_exec_impl(op, &unknown, fp32, fp32),
                             ==, AME_IMPL_UNSUPPORTED);
            g_assert_cmpuint(riscv_ame_exec_impl(op, fp32, &unknown, fp32),
                             ==, AME_IMPL_UNSUPPORTED);
            g_assert_cmpuint(riscv_ame_exec_impl(op, fp32, fp32, &unknown),
                             ==, AME_IMPL_UNSUPPORTED);
        }
    }
    g_assert_cmpuint(riscv_ame_exec_impl(AME_OP_ADD, NULL, fp32, fp32),
                     ==, AME_IMPL_UNSUPPORTED);
    g_assert_cmpuint(riscv_ame_exec_impl(AME_OP_COUNT, fp32, fp32, fp32),
                     ==, AME_IMPL_UNSUPPORTED);
}

static void test_ocp_tensor_execution(void)
{
    RISCVAMELayout layout = test_layout_geometry(2, 8);
    const AmeDTypeDesc *e4m3 = dtype(UINT32_C(0x10100108));
    const AmeDTypeDesc *e4m3_rno = dtype(UINT32_C(0x11500108));
    const AmeDTypeDesc *e5m2 = dtype(UINT32_C(0x14300108));
    const AmeDTypeDesc *fp16 = dtype(UINT32_C(0x14300110));
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    g_autofree uint8_t *mregs = g_malloc0(layout.mregs_bytes);
    g_autofree uint8_t *result = g_malloc(layout.mregs_bytes);
    AmeMView destination, source1, source2, staged;
    AmeExecFlags flags = { 0 };
    size_t element;

    init_views(&layout, &destination, 0, e5m2, &source1, 1, e4m3);
    for (element = 0; element < source1.total_elements; element++) {
        riscv_ame_m_write(&layout, mregs, &source1, element,
                          &(AmeValue) { .lo = UINT8_C(0x38) });
    }
    g_assert_true(riscv_ame_exec_m_convert(
        AME_IMPL_FP_CONVERT, &layout, mregs, &destination, &source1,
        result, layout.mregs_bytes, &flags));
    staged = staged_view(&destination);
    for (element = 0; element < staged.total_elements; element++) {
        g_assert_cmphex(riscv_ame_m_read(
                            &layout, result, &staged, element).lo, ==,
                        UINT8_C(0x3c));
    }

    memset(mregs, 0, layout.mregs_bytes);
    init_binary_views(&layout, &destination, 0, e4m3,
                      &source1, 1, e4m3, &source2, 2, e4m3);
    for (element = 0; element < destination.total_elements; element++) {
        riscv_ame_m_write(&layout, mregs, &source1, element,
                          &(AmeValue) {
                              .lo = element == 0 ? UINT8_C(0x7e) :
                                                   UINT8_C(0x38),
                          });
        riscv_ame_m_write(&layout, mregs, &source2, element,
                          &(AmeValue) {
                              .lo = element == 0 ? UINT8_C(0x7e) :
                                                   UINT8_C(0x40),
                          });
    }
    flags = (AmeExecFlags) { 0 };
    g_assert_true(riscv_ame_exec_m_binary(
        AME_IMPL_FP_BINARY, AME_OP_ADD, &layout, mregs,
        &destination, &source1, &source2, result, layout.mregs_bytes,
        &flags));
    g_assert_cmphex(flags.fflags, ==, AME_FFLAG_OF | AME_FFLAG_NX);
    staged = staged_view(&destination);
    g_assert_cmphex(riscv_ame_m_read(
                        &layout, result, &staged, 0).lo, ==,
                    UINT8_C(0x7e));
    g_assert_cmphex(riscv_ame_m_read(
                        &layout, result, &staged, 1).lo, ==,
                    UINT8_C(0x44));

    memset(mregs, 0, layout.mregs_bytes);
    init_binary_views(&layout, &destination, 0, fp16,
                      &source1, 1, e4m3, &source2, 2, e5m2);
    for (element = 0; element < destination.total_elements; element++) {
        riscv_ame_m_write(&layout, mregs, &source1, element,
                          &(AmeValue) { .lo = UINT8_C(0x38) });
        riscv_ame_m_write(&layout, mregs, &source2, element,
                          &(AmeValue) { .lo = UINT8_C(0x40) });
    }
    flags = (AmeExecFlags) { 0 };
    g_assert_true(riscv_ame_exec_m_binary(
        AME_IMPL_FP_BINARY, AME_OP_ADD, &layout, mregs,
        &destination, &source1, &source2, result, layout.mregs_bytes,
        &flags));
    staged = staged_view(&destination);
    for (element = 0; element < staged.total_elements; element++) {
        g_assert_cmphex(riscv_ame_m_read(
                            &layout, result, &staged, element).lo, ==,
                        UINT16_C(0x4200));
    }

    init_binary_views(&layout, &destination, 0, int8,
                      &source1, 1, e4m3, &source2, 2, e4m3);
    flags = (AmeExecFlags) { 0 };
    g_assert_true(riscv_ame_exec_m_binary(
        AME_IMPL_FP_COMPARE, AME_OP_CMPLT, &layout, mregs,
        &destination, &source1, &source2, result, layout.mregs_bytes,
        &flags));
    staged = staged_view(&destination);
    for (element = 0; element < staged.total_elements; element++) {
        g_assert_cmphex(riscv_ame_m_read(
                            &layout, result, &staged, element).lo, ==,
                        UINT8_MAX);
    }

    init_views(&layout, &destination, 0, e4m3, &source1, 1, e4m3);
    g_assert_true(riscv_ame_exec_m_binary_scalar(
        AME_IMPL_FP_SCALE, AME_OP_LDEXP, &layout, mregs,
        &destination, &source1, int8, &(AmeValue) { .lo = 1 }, false,
        result, layout.mregs_bytes, &flags));
    staged = staged_view(&destination);
    for (element = 0; element < staged.total_elements; element++) {
        g_assert_cmphex(riscv_ame_m_read(
                            &layout, result, &staged, element).lo, ==,
                        UINT8_C(0x40));
    }

    init_binary_views(&layout, &destination, 0, e4m3_rno,
                      &source1, 1, e4m3_rno,
                      &source2, 2, e4m3_rno);
    memset(result, 0xa5, layout.mregs_bytes);
    flags = (AmeExecFlags) { .fflags = 7, .saturated = true };
    g_assert_false(riscv_ame_capabilities_has(
        riscv_ame_maximum_capabilities(), AME_OP_ADD, AME_FORM_BASE,
        e4m3_rno->id, e4m3_rno->id, e4m3_rno->id));
    g_assert_false(riscv_ame_exec_m_binary(
        AME_IMPL_UNSUPPORTED, AME_OP_ADD, &layout, mregs,
        &destination, &source1, &source2, result, layout.mregs_bytes,
        &flags));
    g_assert_cmphex(flags.fflags, ==, 7);
    g_assert_true(flags.saturated);
    for (element = 0; element < layout.mregs_bytes; element++) {
        g_assert_cmphex(result[element], ==, UINT8_C(0xa5));
    }
}

static void assert_bitwise_bytes(const RISCVAMELayout *layout,
                                 const AmeDTypeDesc *type, uint32_t n_sq,
                                 uint32_t second_base)
{
    static const AmeOperation ops[] = {
        AME_OP_AND, AME_OP_ANDNOT, AME_OP_OR, AME_OP_ORNOT, AME_OP_XOR,
    };
    size_t capacity = layout->mregs_bytes + 2;
    g_autofree uint8_t *storage = g_malloc(capacity);
    g_autofree uint8_t *original = g_malloc(capacity);
    g_autofree uint8_t *result = g_malloc(capacity);
    g_autofree uint8_t *expected = g_malloc(capacity);
    AmeMView first, second;
    const uint8_t *a, *b;
    size_t bytes;

    g_assert_true(riscv_ame_m_view_init(layout, &first, 0, type, n_sq));
    g_assert_true(riscv_ame_m_view_init(
        layout, &second, second_base, type, n_sq));
    bytes = riscv_ame_m_view_bytes(layout, &first);
    for (size_t i = 0; i < capacity; i++) {
        storage[i] = i * 73 + 19;
    }
    memcpy(original, storage, capacity);
    a = storage + 1;
    b = a + second.base * layout->mreg_bytes;

    for (size_t op = 0; op < ARRAY_SIZE(ops); op++) {
        AmeExecFlags flags = { .fflags = 31, .saturated = true };

        memset(result, 0xa5, capacity);
        memset(expected, 0xa5, capacity);
        for (size_t i = 0; i < bytes; i++) {
            uint8_t value;

            switch (ops[op]) {
            case AME_OP_AND:
                value = a[i] & b[i];
                break;
            case AME_OP_ANDNOT:
                value = a[i] & ~b[i];
                break;
            case AME_OP_OR:
                value = a[i] | b[i];
                break;
            case AME_OP_ORNOT:
                value = a[i] | ~b[i];
                break;
            case AME_OP_XOR:
                value = a[i] ^ b[i];
                break;
            default:
                g_assert_not_reached();
            }
            expected[i + 1] = value;
        }
        g_assert_true(riscv_ame_exec_m_binary(
            AME_IMPL_INT_BINARY, ops[op], layout, storage + 1,
            &first, &first, &second, result + 1, capacity - 2, &flags));
        g_assert_cmpmem(result, capacity, expected, capacity);
        g_assert_cmpmem(storage, capacity, original, capacity);
        g_assert_cmphex(flags.fflags, ==, 0);
        g_assert_false(flags.saturated);
    }
}

static void test_homogeneous_binary_kernels(void)
{
    static const AmeOperation ops[] = {
        AME_OP_ADD, AME_OP_SUB, AME_OP_MUL, AME_OP_NMUL,
        AME_OP_AND, AME_OP_ANDNOT, AME_OP_OR, AME_OP_ORNOT,
        AME_OP_XOR,
    };
    RISCVAMELayout layout = test_layout_geometry(2, 64);
    g_autofree uint8_t *mregs = g_malloc0(layout.mregs_bytes);
    g_autofree uint8_t *result = g_malloc(layout.mregs_bytes);

    for (unsigned bits = 4; bits <= 64; bits *= 2) {
        uint64_t mask = UINT64_MAX >> (64 - bits);

        for (unsigned sign = 0; sign < 2; sign++) {
            const AmeDTypeDesc *d = dtype(bits | (sign << 30));
            AmeMView dst, a, b, staged;

            init_binary_views(&layout, &dst, 0, d, &a, 1, d, &b, 2, d);
            staged = staged_view(&dst);
            for (size_t k = 0; k < a.total_elements; k++) {
                AmeValue left = { .lo = (mask - k) & mask };
                AmeValue right = { .lo = (k * 37 + 3) & mask };

                riscv_ame_m_write(&layout, mregs, &a, k, &left);
                riscv_ame_m_write(&layout, mregs, &b, k, &right);
            }
            for (size_t i = 0; i < ARRAY_SIZE(ops); i++) {
                AmeExecFlags flags = { .fflags = 31, .saturated = true };

                memset(result, 0xa5, layout.mregs_bytes);
                g_assert_true(riscv_ame_exec_m_binary(
                    AME_IMPL_INT_BINARY, ops[i], &layout, mregs,
                    &dst, &a, &b, result, layout.mregs_bytes, &flags));
                for (size_t k = 0; k < dst.total_elements; k++) {
                    uint64_t x = (mask - k) & mask;
                    uint64_t y = (k * 37 + 3) & mask;
                    uint64_t expected;

                    switch (ops[i]) {
                    case AME_OP_ADD:
                        expected = x + y;
                        break;
                    case AME_OP_SUB:
                        expected = y - x;
                        break;
                    case AME_OP_MUL:
                        expected = x * y;
                        break;
                    case AME_OP_NMUL:
                        expected = -(x * y);
                        break;
                    case AME_OP_AND:
                        expected = x & y;
                        break;
                    case AME_OP_ANDNOT:
                        expected = x & ~y;
                        break;
                    case AME_OP_OR:
                        expected = x | y;
                        break;
                    case AME_OP_ORNOT:
                        expected = x | ~y;
                        break;
                    case AME_OP_XOR:
                        expected = x ^ y;
                        break;
                    default:
                        g_assert_not_reached();
                    }
                    g_assert_cmphex(riscv_ame_m_read(
                        &layout, result, &staged, k).lo, ==, expected & mask);
                }
                g_assert_cmpuint(flags.fflags, ==, 0);
                g_assert_false(flags.saturated);
            }
        }
    }

    layout = test_layout_geometry(2, 16);
    for (unsigned bits = 4; bits <= 128; bits *= 2) {
        unsigned pack = MAX(1, layout.u / bits);
        unsigned group = MAX(1, bits / layout.u);

        for (unsigned rm = 0; rm < AME_INT_ROUND_MAX; rm++) {
            for (unsigned variant = 0; variant < 4; variant++) {
                const AmeDTypeDesc *type = dtype(
                    bits | (rm << 27) | ((variant & 1) << 29) |
                    ((variant >> 1) << 30));

                /* Disjoint, identical and partially overlapping sources. */
                for (unsigned overlap = 0; overlap < 3; overlap++) {
                    unsigned second = overlap == 0 ? 2 * group :
                                      overlap == 1 ? 0 : group;

                    assert_bitwise_bytes(&layout, type, 2 * pack, second);
                }
            }
        }
    }

    /* Exact 1-, 2- and 4-byte spans exercise the scalar tails. */
    for (unsigned u = 8; u <= 32; u *= 2) {
        layout = test_layout_geometry(1, u);
        assert_bitwise_bytes(&layout, dtype(4), u / 4, 31);
    }

    /* A full maximum-sized file, with both sources sharing its last byte. */
    layout = test_layout_geometry(RV_AME_N_MAX, RV_AME_UDSZ_MAX);
    assert_bitwise_bytes(&layout, dtype(8),
                          layout.u / 8 * layout.num_mregs, 0);
}

static void test_homogeneous_fp_binary_kernels(void)
{
    /* 1.5 and 2.25: sum 3.75, second-first 0.75, product +/-3.375. */
    static const struct {
        uint32_t dtype;
        uint64_t a, b, expected[4];
        uint64_t one, half_ulp, sign, snan, qnan;
    } cases[] = {
        { 0x14300110, 0x3e00, 0x4080, { 0x4380, 0x3a00, 0x42c0, 0xc2c0 },
          0x3c00, 0x1000, 0x8000, 0x7c01, 0x7e00 },
        { 0x20300110, 0x3fc0, 0x4010, { 0x4070, 0x3f40, 0x4058, 0xc058 },
          0x3f80, 0x3b80, 0x8000, 0x7f81, 0x7fc0 },
        { 0x20300120, 0x3fc00000, 0x40100000,
          { 0x40700000, 0x3f400000, 0x40580000, 0xc0580000 },
          0x3f800000, 0x33800000, 0x80000000, 0x7f800001, 0x7fc00000 },
        { 0x2c300140, UINT64_C(0x3ff8000000000000),
          UINT64_C(0x4002000000000000),
          { UINT64_C(0x400e000000000000), UINT64_C(0x3fe8000000000000),
            UINT64_C(0x400b000000000000), UINT64_C(0xc00b000000000000) },
          UINT64_C(0x3ff0000000000000), UINT64_C(0x3ca0000000000000),
          UINT64_C(0x8000000000000000), UINT64_C(0x7ff0000000000001),
          UINT64_C(0x7ff8000000000000) },
    };
    static const AmeOperation ops[] = {
        AME_OP_ADD, AME_OP_SUB, AME_OP_MUL, AME_OP_NMUL,
    };
    RISCVAMELayout layout = test_layout_geometry(2, 64);
    g_autofree uint8_t *mregs = g_malloc0(layout.mregs_bytes);
    g_autofree uint8_t *result = g_malloc(layout.mregs_bytes);
    g_autofree uint8_t *expected_result = g_malloc(layout.mregs_bytes);
    g_autofree uint8_t *original = g_malloc(layout.mregs_bytes);

    for (size_t c = 0; c < ARRAY_SIZE(cases); c++) {
        for (unsigned r = 0; r < 6; r++) {
            const AmeDTypeDesc *d = dtype(cases[c].dtype | (r << 22));
            AmeMView dst, a, b, staged;
            AmeExecFlags flags;
            const uint64_t left[] = {
                cases[c].one, cases[c].sign, cases[c].snan, 0,
            };
            const uint64_t right[] = {
                cases[c].half_ulp, 0, cases[c].one, 0,
            };
            uint64_t expected[] = {
                cases[c].one + (r == 3 || r == 4 || r == 5),
                0, cases[c].qnan, 0,
            };

            if (r == 2) {
                expected[1] = cases[c].sign;
            }
            init_binary_views(&layout, &dst, 0, d, &a, 1, d, &b, 2, d);
            staged = staged_view(&dst);
            for (size_t k = 0; k < a.total_elements; k++) {
                riscv_ame_m_write(&layout, mregs, &a, k,
                                  &(AmeValue) { .lo = cases[c].a });
                riscv_ame_m_write(&layout, mregs, &b, k,
                                  &(AmeValue) { .lo = cases[c].b });
            }
            for (size_t i = 0; i < ARRAY_SIZE(ops); i++) {
                flags = (AmeExecFlags) { .fflags = 31, .saturated = true };

                g_assert_true(riscv_ame_exec_m_binary(
                    AME_IMPL_FP_BINARY, ops[i], &layout, mregs,
                    &dst, &a, &b, result, layout.mregs_bytes, &flags));
                for (size_t k = 0; k < dst.total_elements; k++) {
                    g_assert_cmphex(riscv_ame_m_read(
                        &layout, result, &staged, k).lo, ==,
                        cases[c].expected[i]);
                }
                g_assert_cmpuint(flags.fflags, ==, 0);
                g_assert_false(flags.saturated);
            }
            /* Inexact ties, signed zero, sNaN, and cross-lane sticky flags. */
            for (size_t k = 0; k < a.total_elements; k++) {
                riscv_ame_m_write(&layout, mregs, &a, k,
                                  &(AmeValue) { .lo = left[k % 4] });
                riscv_ame_m_write(&layout, mregs, &b, k,
                                  &(AmeValue) { .lo = right[k % 4] });
            }
            g_assert_true(riscv_ame_exec_m_binary(
                AME_IMPL_FP_BINARY, AME_OP_ADD, &layout, mregs,
                &dst, &a, &b, result, layout.mregs_bytes, &flags));
            for (size_t k = 0; k < dst.total_elements; k++) {
                g_assert_cmphex(riscv_ame_m_read(
                    &layout, result, &staged, k).lo, ==, expected[k % 4]);
            }
            g_assert_cmphex(flags.fflags, ==, AME_FFLAG_NV | AME_FFLAG_NX);
            g_assert_false(flags.saturated);

            /*
             * Different source rounding fields do not change the destination
             * rounding mode when all three formats match.
             */
            const AmeDTypeDesc *source1_dtype = dtype(
                cases[c].dtype | (((r + 1) % 6) << 22));
            const AmeDTypeDesc *source2_dtype = dtype(
                cases[c].dtype | (((r + 2) % 6) << 22));
            uint64_t infinity = cases[c].snan & ~UINT64_C(1);
            const uint64_t values[] = {
                0, cases[c].sign, cases[c].one,
                cases[c].one | cases[c].sign,
                infinity, infinity | cases[c].sign,
                infinity - 1, (infinity - 1) | cases[c].sign,
                1, 1 | cases[c].sign,
                cases[c].snan, cases[c].qnan, cases[c].half_ulp,
            };

            for (unsigned overlap = 0; overlap < 2; overlap++) {
                size_t bytes;

                init_binary_views(&layout, &dst, overlap ? 0 : 8, d,
                                  &a, 0, source1_dtype, &b, 4, source2_dtype);
                staged = staged_view(&dst);
                bytes = riscv_ame_m_view_bytes(&layout, &dst);
                for (size_t x = 0; x < ARRAY_SIZE(values); x++) {
                    for (size_t y = 0; y < ARRAY_SIZE(values); y++) {
                        for (size_t e = 0; e < dst.total_elements; e++) {
                            AmeValue left_value = {
                                .lo = values[(x + e) % ARRAY_SIZE(values)],
                            };
                            AmeValue right_value = {
                                .lo = values[(y + 3 * e) % ARRAY_SIZE(values)],
                            };

                            riscv_ame_m_write(&layout, mregs, &a, e,
                                              &left_value);
                            riscv_ame_m_write(&layout, mregs, &b, e,
                                              &right_value);
                        }
                        memcpy(original, mregs, layout.mregs_bytes);
                        for (size_t op = 0; op < ARRAY_SIZE(ops); op++) {
                            float_status status;
                            AmeExecFlags reference = { 0 };
                            g_assert_true(riscv_ame_fp_status_prepare(
                                &status, d));
                            memset(result, 0xa5, layout.mregs_bytes);
                            memset(expected_result, 0xa5, layout.mregs_bytes);
                            memset(expected_result, 0, bytes);
                            for (size_t e = 0; e < dst.total_elements; e++) {
                                AmeValue left_value = riscv_ame_m_read(
                                    &layout, mregs, &a, e);
                                AmeValue right_value = riscv_ame_m_read(
                                    &layout, mregs, &b, e);
                                AmeExecResult executed;

                                riscv_ame_exec_fp_binary(
                                    ops[op], d, source1_dtype, &left_value,
                                    source2_dtype, &right_value,
                                    &executed, &status);
                                riscv_ame_m_write(&layout, expected_result,
                                                  &staged, e, &executed.value);
                                reference.fflags |= executed.fflags;
                                reference.saturated |= executed.saturated;
                            }
                            flags = (AmeExecFlags) {
                                .fflags = UINT8_MAX, .saturated = true,
                            };
                            g_assert_true(riscv_ame_exec_m_binary(
                                AME_IMPL_FP_BINARY, ops[op], &layout, mregs,
                                &dst, &a, &b, result, layout.mregs_bytes,
                                &flags));
                            g_assert_cmpmem(result, layout.mregs_bytes,
                                            expected_result,
                                            layout.mregs_bytes);
                            g_assert_cmpmem(mregs, layout.mregs_bytes,
                                            original, layout.mregs_bytes);
                            g_assert_cmphex(flags.fflags, ==, reference.fflags);
                            g_assert_cmpint(flags.saturated, ==,
                                            reference.saturated);
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
    g_test_add_func("/riscv/ame/exec/impl-selection", test_impl_selection);
    g_test_add_func("/riscv/ame/exec/nonstandard-impl-selection",
                    test_nonstandard_impl_selection);
    g_test_add_func("/riscv/ame/exec/homogeneous-integer-kernels",
                    test_homogeneous_binary_kernels);
    g_test_add_func("/riscv/ame/exec/homogeneous-fp-kernels",
                    test_homogeneous_fp_binary_kernels);
    g_test_add_func("/riscv/ame/exec/elementwise-loop-staging",
                    test_elementwise_loop_staging);

    g_test_add_func("/riscv/ame/exec/convert-scalar-dispatch",
                    test_convert_scalar_dispatch);
    g_test_add_func("/riscv/ame/exec/checked-rejects",
                    test_checked_execution_rejects);
    g_test_add_func("/riscv/ame/exec/elementwise-rejects",
                    test_elementwise_rejects_invalid_requests);
    g_test_add_func("/riscv/ame/exec/convert-packed-to-unit",
                    test_convert_packed_to_unit);
    g_test_add_func("/riscv/ame/exec/convert-unit-to-packed-saturation",
                    test_convert_unit_to_packed_saturation);
    g_test_add_func("/riscv/ame/exec/convert-overlapping-views",
                    test_convert_overlapping_views);
    g_test_add_func("/riscv/ame/exec/convert-int128-saturation",
                    test_convert_int128_saturation);
    g_test_add_func("/riscv/ame/exec/convert-fp-to-packed-int",
                    test_convert_fp_to_packed_int);
    g_test_add_func("/riscv/ame/exec/convert-invalid-executor",
                    test_convert_rejects_invalid_executor);
    g_test_add_func("/riscv/ame/exec/convert-same-width-integer-bits",
                    test_convert_same_width_integer_bits);
    g_test_add_func("/riscv/ame/exec/integer-arithmetic-dispatch",
                    test_integer_arithmetic_dispatch);
    g_test_add_func("/riscv/ame/exec/integer-bitwise-compare-select",
                    test_integer_bitwise_compare_select);
    g_test_add_func("/riscv/ame/exec/integer-shift-dispatch",
                    test_integer_shift_dispatch);
    g_test_add_func("/riscv/ame/exec/integer-elementwise-exhaustive-small",
                    test_integer_elementwise_exhaustive_small);
    g_test_add_func("/riscv/ame/exec/integer-mixed-exhaustive-small",
                    test_integer_mixed_exhaustive_small);
    g_test_add_func("/riscv/ame/exec/integer-capability-primitives",
                    test_integer_capability_primitives);
    g_test_add_func("/riscv/ame/exec/numeric-capability-primitives",
                    test_numeric_capability_primitives);
    g_test_add_func("/riscv/ame/exec/integer-float-tensor",
                    test_integer_float_tensor);
    g_test_add_func("/riscv/ame/exec/integer-half-exhaustive-small",
                    test_integer_half_exhaustive_small);
    g_test_add_func("/riscv/ame/exec/integer-scale-exhaustive-small",
                    test_integer_scale_exhaustive_small);
    g_test_add_func("/riscv/ame/exec/integer-scale-boundaries",
                    test_integer_scale_boundaries);
    g_test_add_func("/riscv/ame/exec/integer-fused-exhaustive-small",
                    test_integer_fused_exhaustive_small);
    g_test_add_func("/riscv/ame/exec/integer-ternary-dispatch",
                    test_integer_ternary_dispatch);
    g_test_add_func("/riscv/ame/exec/integer-tensor-scalar-staging",
                    test_integer_tensor_and_scalar_staging);
    g_test_add_func("/riscv/ame/exec/integer-equal-width-binary",
                    test_equal_width_integer_binary);
    g_test_add_func("/riscv/ame/exec/integer-narrowing-staging",
                    test_integer_narrowing_staging);
    g_test_add_func("/riscv/ame/exec/integer-mixed-tensor",
                    test_integer_mixed_tensor);
    g_test_add_func("/riscv/ame/exec/integer-ternary-staging",
                    test_integer_ternary_staging);
    g_test_add_func("/riscv/ame/exec/ieee-conversion-elementwise",
                    test_ieee_tensor_conversion_elementwise);
    g_test_add_func("/riscv/ame/exec/tanh-lane-order",
                    test_tanh_lane_order);
    g_test_add_func("/riscv/ame/exec/ieee-scalar-fused-scale",
                    test_ieee_scalar_fused_scale);
    g_test_add_func("/riscv/ame/exec/ocp-tensor",
                    test_ocp_tensor_execution);

    return g_test_run();
}
