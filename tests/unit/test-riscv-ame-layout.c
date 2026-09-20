/*
 * RISC-V Attached Matrix Extension tensor layout tests
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qapi/error.h"
#include "qemu/bitops.h"
#include "qemu/bswap.h"
#include "qemu/units.h"
#include "target/riscv/ame/layout.h"

static const AmeCapabilities *maximum_capabilities(void)
{
    return riscv_ame_maximum_capabilities();
}

static const AmeDTypeDesc *dtype(uint32_t encoding)
{
    const AmeDTypeDesc *desc = riscv_ame_dtype_by_encoding(encoding);

    g_assert_nonnull(desc);
    return desc;
}

static void assert_bad_layout(uint64_t n, uint64_t u, uint32_t num_mregs,
                              uint32_t num_accregs)
{
    RISCVAMELayout layout = { .n = UINT32_MAX };
    Error *err = NULL;

    g_assert_false(riscv_ame_compute_layout(
        n, u, num_mregs, num_accregs, maximum_capabilities(),
        &layout, &err));
    g_assert_nonnull(err);
    g_assert_cmpuint(layout.n, ==, UINT32_MAX);
    error_free(err);
}

static void test_layout_geometry(void)
{
    static const uint32_t acc_counts[] = { 1, 2, 4, 8, 16 };
    RISCVAMELayout layout, equal;

    g_assert_true(riscv_ame_compute_layout(
        128, 64, 32, 16, maximum_capabilities(),
        &layout, &error_abort));
    g_assert_cmpuint(layout.n, ==, 128);
    g_assert_cmpuint(layout.u, ==, 64);
    g_assert_cmpuint(layout.num_mregs, ==, 32);
    g_assert_cmpuint(layout.num_accregs, ==, 16);
    g_assert_cmpuint(layout.mreg_bytes, ==, 128 * KiB);
    g_assert_cmpuint(layout.mregs_bytes, ==, 4 * MiB);
    g_assert_cmpuint(layout.acc_bytes, ==, 256 * KiB);
    g_assert_cmpuint(layout.accs_bytes, ==, 4 * MiB);

    g_assert_true(riscv_ame_compute_layout(
        4, 8, 16, 1, maximum_capabilities(),
        &equal, &error_abort));
    g_assert_cmpuint(equal.mreg_bytes, ==, 16);
    g_assert_cmpuint(equal.mregs_bytes, ==, 256);
    g_assert_cmpuint(equal.acc_bytes, ==, 256);
    g_assert_cmpuint(equal.accs_bytes, ==, 256);
    g_assert_false(riscv_ame_layout_equal(&layout, &equal));

    layout = equal;
    g_assert_true(riscv_ame_layout_equal(&layout, &equal));
    g_assert_true(riscv_ame_layout_equal(&layout, &layout));
    g_assert_false(riscv_ame_layout_equal(&layout, NULL));

    for (size_t i = 0; i < ARRAY_SIZE(acc_counts); i++) {
        g_assert_true(riscv_ame_compute_layout(
            4, 8, 16, acc_counts[i], maximum_capabilities(),
            &layout, &error_abort));
        g_assert_cmpuint(layout.num_accregs, ==, acc_counts[i]);
        g_assert_cmpuint(layout.accs_bytes, ==,
                         acc_counts[i] * layout.acc_bytes);
    }
}

static void test_layout_rejects_invalid(void)
{
    AmeDTypeSet allowed = { 0 };
    AmeCapabilities int4_only = { 0 };
    AmeCapabilities empty = { 0 };
    RISCVAMELayout layout = { 0 };
    Error *err = NULL;
    size_t i;

    assert_bad_layout(0, 64, 32, 4);
    assert_bad_layout(3, 64, 32, 4);
    assert_bad_layout(256, 64, 32, 4);
    assert_bad_layout(UINT64_C(1) << 63, 64, 32, 4);
    assert_bad_layout((UINT64_C(1) << 32) + 4, 64, 32, 4);
    assert_bad_layout(128, 0, 32, 4);
    assert_bad_layout(128, 3, 32, 4);
    assert_bad_layout(128, 128, 32, 4);
    assert_bad_layout(128, (UINT64_C(1) << 32) + 8, 32, 4);
    assert_bad_layout(128, 64, 15, 4);
    assert_bad_layout(128, 64, 31, 4);
    assert_bad_layout(128, 64, UINT32_MAX, 4);
    assert_bad_layout(128, 64, 32, 0);
    assert_bad_layout(128, 64, 32, 3);
    assert_bad_layout(1, 1, 16, 1);

    g_assert_false(riscv_ame_compute_layout(
        4, 8, 16, 1, NULL, &layout, &err));
    g_assert_nonnull(err);
    error_free(err);
    err = NULL;
    g_assert_false(riscv_ame_compute_layout(
        4, 8, 16, 1, &empty, &layout, &err));
    g_assert_nonnull(err);
    error_free(err);
    err = NULL;
    g_assert_false(riscv_ame_compute_layout(
        4, 8, 16, 1, maximum_capabilities(), NULL, &err));
    g_assert_nonnull(err);
    error_free(err);

    for (i = 0; i < AME_DTYPE_COUNT; i++) {
        const AmeDTypeDesc *desc = &riscv_ame_dtypes[i];

        if (desc->class == AME_DTYPE_CLASS_INTEGER && desc->bits == 4) {
            riscv_ame_dtype_set_add(&allowed, i);
        }
    }
    g_assert_true(riscv_ame_capabilities_init(
        &int4_only, 0x40000004, &allowed, NULL, 0, &error_abort));
    g_assert_true(riscv_ame_compute_layout(
        1, 8, 16, 1, &int4_only, &layout, &error_abort));
    g_assert_cmpuint(layout.acc_bytes, ==, 1);
    riscv_ame_capabilities_clear(&int4_only);
}

static void test_subbyte_accumulator(void)
{
    const AmeDTypeDesc *int4 = dtype(0x40000004);
    AmeDTypeSet allowed = { 0 };
    AmeCapabilities capabilities = { 0 };
    RISCVAMELayout layout;
    AmeAccView first, second;
    AmeValue value = { .lo = 0xa };
    uint8_t backing[2] = { 0xc5, 0xd6 };

    riscv_ame_dtype_set_add(&allowed, int4->id);
    g_assert_true(riscv_ame_capabilities_init(
        &capabilities, int4->encoding, &allowed, NULL, 0, &error_abort));
    g_assert_true(riscv_ame_compute_layout(
        1, 8, 16, 2, &capabilities, &layout, &error_abort));
    g_assert_cmpuint(layout.mreg_bytes, ==, 1);
    g_assert_cmpuint(layout.acc_bytes, ==, 1);
    g_assert_cmpuint(layout.accs_bytes, ==, sizeof(backing));
    g_assert_true(riscv_ame_acc_view_init(&layout, &first, 0, int4));
    g_assert_true(riscv_ame_acc_view_init(&layout, &second, 1, int4));
    g_assert_cmpuint(first.elements_per_square, ==, 1);

    riscv_ame_acc_write(&layout, backing, &first, 0, &value);
    g_assert_cmphex(backing[0], ==, 0xca);
    g_assert_cmphex(backing[1], ==, 0xd6);
    value.lo = 0xb;
    riscv_ame_acc_write(&layout, backing, &second, 0, &value);
    g_assert_cmphex(backing[0], ==, 0xca);
    g_assert_cmphex(backing[1], ==, 0xdb);
    value = riscv_ame_acc_read(&layout, backing, &first, 0);
    g_assert_cmphex(value.lo, ==, 0xa);
    value = riscv_ame_acc_read(&layout, backing, &second, 0);
    g_assert_cmphex(value.lo, ==, 0xb);
    riscv_ame_capabilities_clear(&capabilities);
}

static void test_fp_only_layout(void)
{
    static const uint32_t encodings[] = { 0x14300110, 0x20300120 };
    AmeDTypeSet dtypes = { 0 };
    AmeCapabilities capabilities = { 0 };
    RISCVAMELayout layout;

    g_assert_true(riscv_ame_dtype_set_from_encodings(
        &dtypes, encodings, G_N_ELEMENTS(encodings), &error_abort));
    g_assert_true(riscv_ame_capabilities_init(
        &capabilities, 0, &dtypes, NULL, 0, &error_abort));
    g_assert_true(riscv_ame_compute_layout(
        4, 8, 16, 1, &capabilities, &layout, &error_abort));
    g_assert_cmpuint(layout.mreg_bytes, ==, 16);
    g_assert_cmpuint(layout.mregs_bytes, ==, 256);
    /* Acc storage follows the largest enabled datatype, here FP32. */
    g_assert_cmpuint(layout.acc_bytes, ==, 64);
    g_assert_cmpuint(layout.accs_bytes, ==, 64);
    riscv_ame_capabilities_clear(&capabilities);
}

static void test_dtype_shapes_and_views(void)
{
    const AmeDTypeDesc *int4 = dtype(UINT32_C(0x40000004));
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    const AmeDTypeDesc *int16 = dtype(UINT32_C(0x40000010));
    const AmeDTypeDesc *int128 = dtype(UINT32_C(0x40000080));
    const AmeDTypeDesc *mixed[] = { int4, int16 };
    const AmeDTypeDesc invalid = { .bits = 6 };
    RISCVAMELayout layout;
    AmeDTypeShape shape = { 0 };
    AmeMView view = { 0 };
    uint32_t n_sq = 0;

    g_assert_true(riscv_ame_compute_layout(
        4, 8, 16, 1, maximum_capabilities(),
        &layout, &error_abort));

    g_assert_true(riscv_ame_dtype_shape(&layout, int4, &shape));
    g_assert_cmpuint(shape.pack_factor, ==, 2);
    g_assert_cmpuint(shape.group_size, ==, 1);
    g_assert_true(riscv_ame_dtype_shape(&layout, int8, &shape));
    g_assert_cmpuint(shape.pack_factor, ==, 1);
    g_assert_cmpuint(shape.group_size, ==, 1);
    g_assert_true(riscv_ame_dtype_shape(&layout, int16, &shape));
    g_assert_cmpuint(shape.pack_factor, ==, 1);
    g_assert_cmpuint(shape.group_size, ==, 2);
    g_assert_true(riscv_ame_dtype_shape(&layout, int128, &shape));
    g_assert_cmpuint(shape.group_size, ==, 16);
    g_assert_false(riscv_ame_dtype_shape(&layout, &invalid, &shape));

    g_assert_true(riscv_ame_n_sq(&layout, mixed, ARRAY_SIZE(mixed), &n_sq));
    g_assert_cmpuint(n_sq, ==, 2);
    g_assert_false(riscv_ame_n_sq(&layout, NULL, 0, &n_sq));

    g_assert_true(riscv_ame_m_view_init(&layout, &view, 3, int4, 2));
    g_assert_cmpuint(view.base, ==, 3);
    g_assert_cmpuint(view.pack_factor, ==, 2);
    g_assert_cmpuint(view.group_size, ==, 1);
    g_assert_cmpuint(view.n_sq, ==, 2);
    g_assert_cmpuint(view.span, ==, 1);
    g_assert_cmpuint(view.elements_per_square, ==, 16);
    g_assert_cmpuint(view.total_elements, ==, 32);
    g_assert_cmpuint(riscv_ame_m_view_bytes(&layout, &view), ==, 16);

    g_assert_true(riscv_ame_m_view_init(&layout, &view, 2, int8, 2));
    g_assert_cmpuint(view.span, ==, 2);
    g_assert_true(riscv_ame_m_view_init(&layout, &view, 4, int16, 2));
    g_assert_cmpuint(view.span, ==, 4);
    g_assert_true(riscv_ame_m_view_init(&layout, &view, 0, int128, 1));
    g_assert_cmpuint(view.span, ==, 16);

    g_assert_false(riscv_ame_m_view_init(&layout, &view, 1, int16, 1));
    g_assert_false(riscv_ame_m_view_init(&layout, &view, 14, int16, 2));
    g_assert_false(riscv_ame_m_view_init(&layout, &view, 0, int4, 1));
    g_assert_false(riscv_ame_m_view_init(&layout, &view, 0, int8, 3));
    g_assert_false(riscv_ame_m_view_init(&layout, &view, 0, &invalid, 1));
}

static void test_layout_size_bounds(void)
{
    const AmeDTypeDesc *int4 = dtype(UINT32_C(0x40000004));
    const AmeDTypeDesc *int128 = dtype(UINT32_C(0x40000080));
    RISCVAMELayout layout, invalid;
    AmeMView mview;
    AmeAccView accview;
    AmeMemPlan plan;
    AmeValue value = { .lo = 0xa }, read;
    g_autofree uint8_t *backing = NULL;
    uint32_t squares, elements;
    size_t bytes;

    g_assert_true(riscv_ame_compute_layout(
        RV_AME_N_MAX, RV_AME_UDSZ_MAX, RV_AME_M_REGS_MAX,
        RV_AME_ACC_REGS_MAX, maximum_capabilities(), &layout, &error_abort));
    backing = g_malloc0(MAX(layout.mregs_bytes, layout.accs_bytes));
    squares = layout.num_mregs * layout.u / int4->bits;
    g_assert_true(riscv_ame_m_view_init(&layout, &mview, 0, int4, squares));
    g_assert_cmpuint(mview.span, ==, layout.num_mregs);
    g_assert_cmpuint(mview.total_elements, ==,
                     (uint64_t)layout.n * layout.n * squares);
    elements = mview.total_elements;
    riscv_ame_m_write(&layout, backing, &mview, elements - 1, &value);
    read = riscv_ame_m_read(&layout, backing, &mview, elements - 1);
    g_assert_cmphex(read.lo, ==, value.lo);
    g_assert_cmphex(backing[layout.mregs_bytes - 1], ==, 0xa0);
    g_assert_cmphex(backing[0], ==, 0);

    invalid = layout;
    invalid.n = UINT32_C(1) << 16;
    g_assert_false(riscv_ame_m_view_init(&invalid, &mview, 0, int4, squares));
    invalid.n = UINT32_C(1) << 15;
    g_assert_false(riscv_ame_m_view_init(&invalid, &mview, 0, int4, squares));
    g_assert_cmpuint(mview.total_elements, ==, elements);

    g_assert_true(riscv_ame_acc_view_init(
        &layout, &accview, layout.num_accregs - 1, int128));
    g_assert_cmpuint(accview.elements_per_square, ==,
                     (uint64_t)layout.n * layout.n);
    value = (AmeValue) {
        .lo = UINT64_C(0x0123456789abcdef),
        .hi = UINT64_C(0xfedcba9876543210),
    };
    riscv_ame_acc_write(&layout, backing, &accview,
                        accview.elements_per_square - 1, &value);
    read = riscv_ame_acc_read(&layout, backing, &accview,
                             accview.elements_per_square - 1);
    g_assert_cmphex(read.lo, ==, value.lo);
    g_assert_cmphex(read.hi, ==, value.hi);
    g_assert_cmphex(ldq_le_p(backing + layout.accs_bytes - sizeof(value.hi)),
                    ==, value.hi);
    invalid = layout;
    invalid.acc_bytes = SIZE_MAX / CHAR_BIT;
    invalid.n = UINT32_C(1) << 16;
    g_assert_false(riscv_ame_acc_view_init(&invalid, &accview, 0, int128));
    invalid.n = UINT32_C(1) << 14;
    g_assert_false(riscv_ame_acc_view_init(&invalid, &accview, 0, int128));
    g_assert_cmpuint(accview.elements_per_square, ==,
                     (uint64_t)layout.n * layout.n);

    g_assert_true(riscv_ame_mem_plan_init(
        &layout, int128, AME_MEM_ROW_MAJOR, &plan));
    g_assert_cmpuint(plan.total_bytes, ==, layout.acc_bytes);
    bytes = plan.total_bytes;
    invalid = layout;
    invalid.n = UINT32_C(1) << 28;
    g_assert_false(riscv_ame_mem_plan_init(
        &invalid, int4, AME_MEM_TRANSPOSED_STRIDED, &plan));
    invalid.n = UINT32_MAX;
    invalid.u = UINT32_MAX - 3;
    g_assert_false(riscv_ame_mem_plan_init(
        &invalid, int4, AME_MEM_ROW_MAJOR, &plan));
    g_assert_cmpuint(plan.total_bytes, ==, bytes);
}

static void test_m_physical_mapping(void)
{
    const AmeDTypeDesc *int4 = dtype(UINT32_C(0x40000004));
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    const AmeDTypeDesc *int16 = dtype(UINT32_C(0x40000010));
    const AmeDTypeDesc *int32 = dtype(UINT32_C(0x40000020));
    const AmeDTypeDesc *int128 = dtype(UINT32_C(0x40000080));
    RISCVAMELayout layout;
    AmeMView view;
    AmeValue value;
    uint8_t backing[256];

    g_assert_true(riscv_ame_compute_layout(
        4, 8, 16, 1, maximum_capabilities(),
        &layout, &error_abort));
    memset(backing, 0xcc, sizeof(backing));

    g_assert_true(riscv_ame_m_view_init(&layout, &view, 3, int4, 2));
    value = (AmeValue) { .lo = 0xa5, .hi = UINT64_MAX };
    riscv_ame_m_write(&layout, backing, &view, 0, &value);
    value.lo = 0xb;
    riscv_ame_m_write(&layout, backing, &view, 1, &value);
    value.lo = 0xd;
    riscv_ame_m_write(&layout, backing, &view, 16, &value);
    g_assert_cmphex(backing[3 * 16], ==, 0xb5);
    g_assert_cmphex(backing[3 * 16 + 8], ==, 0xcd);
    g_assert_cmphex(backing[3 * 16 - 1], ==, 0xcc);
    g_assert_cmphex(backing[3 * 16 + 16], ==, 0xcc);
    g_assert_cmphex(riscv_ame_m_read(&layout, backing, &view, 0).lo, ==, 5);
    g_assert_cmphex(riscv_ame_m_read(&layout, backing, &view, 16).lo, ==,
                    0xd);

    memset(backing, 0, sizeof(backing));
    g_assert_true(riscv_ame_m_view_init(&layout, &view, 2, int8, 2));
    value = (AmeValue) { .lo = 0x5a };
    riscv_ame_m_write(&layout, backing, &view, 16, &value);
    g_assert_cmphex(backing[3 * 16], ==, 0x5a);

    g_assert_true(riscv_ame_m_view_init(&layout, &view, 4, int16, 2));
    value.lo = 0x1234;
    riscv_ame_m_write(&layout, backing, &view, 16, &value);
    g_assert_cmphex(lduw_le_p(backing + 6 * 16), ==, 0x1234);

    g_assert_true(riscv_ame_m_view_init(&layout, &view, 8, int32, 1));
    value.lo = UINT32_C(0x80000001);
    riscv_ame_m_write(&layout, backing, &view, 0, &value);
    value = riscv_ame_m_read(&layout, backing, &view, 0);
    g_assert_cmphex(value.lo, ==, UINT32_C(0x80000001));

    memset(backing, 0, sizeof(backing));
    g_assert_true(riscv_ame_m_view_init(&layout, &view, 0, int128, 1));
    value = (AmeValue) {
        .lo = UINT64_C(0x0123456789abcdef),
        .hi = UINT64_C(0xfedcba9876543210),
    };
    riscv_ame_m_write(&layout, backing, &view, 15, &value);
    value = riscv_ame_m_read(&layout, backing, &view, 15);
    g_assert_cmphex(value.lo, ==, UINT64_C(0x0123456789abcdef));
    g_assert_cmphex(value.hi, ==, UINT64_C(0xfedcba9876543210));
    g_assert_cmphex(ldq_le_p(backing + 15 * 16), ==,
                    UINT64_C(0x0123456789abcdef));
}

static void test_acc_physical_mapping(void)
{
    const AmeDTypeDesc *int4 = dtype(UINT32_C(0x40000004));
    const AmeDTypeDesc *int32 = dtype(UINT32_C(0x40000020));
    const AmeDTypeDesc *int128 = dtype(UINT32_C(0x40000080));
    RISCVAMELayout layout;
    AmeAccView view;
    AmeValue value;
    uint8_t backing[1024];

    g_assert_true(riscv_ame_compute_layout(
        4, 8, 16, 4, maximum_capabilities(),
        &layout, &error_abort));
    memset(backing, 0xcc, sizeof(backing));

    g_assert_true(riscv_ame_acc_view_init(&layout, &view, 2, int4));
    g_assert_cmpuint(view.index, ==, 2);
    g_assert_cmpuint(view.elements_per_square, ==, 16);
    value = (AmeValue) { .lo = 0xa };
    riscv_ame_acc_write(&layout, backing, &view, 1, &value);
    value.lo = 0xb;
    riscv_ame_acc_write(&layout, backing, &view, 2, &value);
    g_assert_cmphex(backing[2 * 256], ==, 0xac);
    g_assert_cmphex(backing[2 * 256 + 1], ==, 0xcb);
    g_assert_cmphex(riscv_ame_acc_read(&layout, backing, &view, 1).lo, ==,
                    0xa);
    g_assert_cmphex(backing[2 * 256 - 1], ==, 0xcc);

    g_assert_true(riscv_ame_acc_view_init(&layout, &view, 1, int32));
    value.lo = UINT32_C(0x80000001);
    riscv_ame_acc_write(&layout, backing, &view, 0, &value);
    value = riscv_ame_acc_read(&layout, backing, &view, 0);
    g_assert_cmphex(value.lo, ==, UINT32_C(0x80000001));

    g_assert_true(riscv_ame_acc_view_init(&layout, &view, 3, int128));
    value = (AmeValue) {
        .lo = UINT64_C(0x1122334455667788),
        .hi = UINT64_C(0x99aabbccddeeff00),
    };
    riscv_ame_acc_write(&layout, backing, &view, 15, &value);
    value = riscv_ame_acc_read(&layout, backing, &view, 15);
    g_assert_cmphex(value.lo, ==, UINT64_C(0x1122334455667788));
    g_assert_cmphex(value.hi, ==, UINT64_C(0x99aabbccddeeff00));
    g_assert_false(riscv_ame_acc_view_init(&layout, &view, 4, int128));
}

static void test_memory_plans(void)
{
    const AmeDTypeDesc *int4 = dtype(UINT32_C(0x40000004));
    const AmeDTypeDesc *int128 = dtype(UINT32_C(0x40000080));
    RISCVAMELayout layout, maximum, packed;
    AmeMemPlan plan;

    g_assert_true(riscv_ame_compute_layout(
        4, 8, 16, 1, maximum_capabilities(),
        &layout, &error_abort));

    g_assert_true(riscv_ame_mem_plan_init(
        &layout, NULL, AME_MEM_WHOLE_REGISTER, &plan));
    g_assert_cmpuint(plan.num_segments, ==, 1);
    g_assert_cmpuint(plan.segment_bits, ==, 128);
    g_assert_cmpuint(plan.segment_bytes, ==, 16);
    g_assert_cmpuint(plan.total_bytes, ==, 16);
    g_assert_cmpuint(plan.pack_factor, ==, 1);
    g_assert_cmpuint(plan.element_bits, ==, 8);
    g_assert_cmpuint(plan.alignment, ==, 1);

    g_assert_true(riscv_ame_mem_plan_init(
        &layout, int4, AME_MEM_ROW_MAJOR, &plan));
    g_assert_cmpuint(plan.pack_factor, ==, 2);
    g_assert_cmpuint(plan.element_bits, ==, 4);
    g_assert_cmpuint(plan.num_segments, ==, 1);
    g_assert_cmpuint(plan.segment_bits, ==, 128);
    g_assert_cmpuint(plan.segment_bytes, ==, 16);
    g_assert_cmpuint(plan.total_bytes, ==, 16);
    g_assert_cmpuint(plan.alignment, ==, 1);

    g_assert_true(riscv_ame_mem_plan_init(
        &layout, int4, AME_MEM_STRIDED, &plan));
    g_assert_cmpuint(plan.num_segments, ==, 4);
    g_assert_cmpuint(plan.segment_bits, ==, 32);
    g_assert_cmpuint(plan.segment_bytes, ==, 4);
    g_assert_cmpuint(plan.total_bytes, ==, 16);
    g_assert_cmpuint(plan.alignment, ==, 1);

    g_assert_true(riscv_ame_mem_plan_init(
        &layout, int4, AME_MEM_TRANSPOSED_STRIDED, &plan));
    g_assert_cmpuint(plan.num_segments, ==, 8);
    g_assert_cmpuint(plan.segment_bits, ==, 16);
    g_assert_cmpuint(plan.segment_bytes, ==, 2);
    g_assert_cmpuint(plan.total_bytes, ==, 16);

    g_assert_true(riscv_ame_mem_plan_init(
        &layout, int128, AME_MEM_COLUMN_MAJOR, &plan));
    g_assert_cmpuint(plan.pack_factor, ==, 1);
    g_assert_cmpuint(plan.segment_bits, ==, 2048);
    g_assert_cmpuint(plan.segment_bytes, ==, 256);
    g_assert_cmpuint(plan.alignment, ==, 16);

    g_assert_true(riscv_ame_mem_plan_init(
        &layout, int128, AME_MEM_STRIDED, &plan));
    g_assert_cmpuint(plan.num_segments, ==, 4);
    g_assert_cmpuint(plan.segment_bits, ==, 512);
    g_assert_cmpuint(plan.segment_bytes, ==, 64);
    g_assert_cmpuint(plan.total_bytes, ==, 256);
    g_assert_cmpuint(plan.alignment, ==, 16);

    g_assert_false(riscv_ame_mem_plan_init(
        &layout, NULL, AME_MEM_ROW_MAJOR, &plan));
    g_assert_false(riscv_ame_mem_plan_init(
        &layout, int4, AME_MEM_FORMAT_MAX, &plan));

    g_assert_true(riscv_ame_compute_layout(
        128, 64, 32, 4, maximum_capabilities(),
        &maximum, &error_abort));
    g_assert_true(riscv_ame_mem_plan_init(
        &maximum, NULL, AME_MEM_WHOLE_REGISTER, &plan));
    g_assert_cmpuint(plan.alignment, ==, 8);
    g_assert_true(riscv_ame_mem_plan_init(
        &maximum, int128, AME_MEM_ROW_MAJOR, &plan));
    g_assert_cmpuint(plan.alignment, ==, 16);

    g_assert_true(riscv_ame_compute_layout(
        1, 64, 16, 1, maximum_capabilities(),
        &packed, &error_abort));
    g_assert_true(riscv_ame_mem_plan_init(
        &packed, int4, AME_MEM_TRANSPOSED_STRIDED, &plan));
    g_assert_cmpuint(plan.pack_factor, ==, 16);
    g_assert_cmpuint(plan.num_segments, ==, 16);
    g_assert_cmpuint(plan.segment_bits, ==, 4);
    g_assert_cmpuint(plan.segment_bytes, ==, 1);
    g_assert_cmpuint(plan.total_bytes, ==, 16);

    g_assert_cmphex(
        riscv_ame_xlen_madd(UINT32_C(0xfffffff0), 2, 16, 32), ==, 16);
    g_assert_cmphex(
        riscv_ame_xlen_madd(16, 2, UINT32_C(0xfffffff8), 32), ==, 0);
    g_assert_cmphex(
        riscv_ame_xlen_madd(UINT64_MAX - 3, 2, 4, 64), ==, 4);
}

static void test_memory_round_trip(void)
{
    const AmeDTypeDesc *int4 = dtype(UINT32_C(0x40000004));
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    const AmeDTypeDesc *int128 = dtype(UINT32_C(0x40000080));
    RISCVAMELayout layout, tiny_layout, wide_layout;
    AmeMView source, destination, tiny_source, tiny_destination;
    AmeMView wide_source, wide_destination;
    AmeMemPlan plan;
    AmeValue value;
    uint8_t backing[256] = { 0 };
    uint8_t imported[256] = { 0 };
    uint8_t memory[256] = { 0 };
    uint8_t wide_backing[512] = { 0 };
    uint8_t wide_imported[512] = { 0 };
    static const uint8_t column_major[16] = {
        0, 4, 8, 12, 1, 5, 9, 13,
        2, 6, 10, 14, 3, 7, 11, 15,
    };
    size_t i;

    g_assert_true(riscv_ame_compute_layout(
        4, 8, 16, 1, maximum_capabilities(),
        &layout, &error_abort));
    g_assert_true(riscv_ame_m_view_init(&layout, &source, 2, int8, 1));
    g_assert_true(riscv_ame_m_view_init(&layout, &destination, 5, int8, 1));
    for (i = 0; i < source.total_elements; i++) {
        value = (AmeValue) { .lo = i };
        riscv_ame_m_write(&layout, backing, &source, i, &value);
    }

    g_assert_true(riscv_ame_mem_plan_init(
        &layout, int8, AME_MEM_ROW_MAJOR, &plan));
    g_assert_true(riscv_ame_m_export(
        &layout, &source, backing, &plan, memory));
    for (i = 0; i < ARRAY_SIZE(column_major); i++) {
        g_assert_cmpuint(memory[i], ==, i);
    }
    g_assert_true(riscv_ame_m_import(
        &layout, &destination, imported, &plan, memory));
    for (i = 0; i < destination.total_elements; i++) {
        g_assert_cmpuint(riscv_ame_m_read(
                             &layout, imported, &destination, i).lo, ==, i);
    }

    g_assert_true(riscv_ame_mem_plan_init(
        &layout, int8, AME_MEM_COLUMN_MAJOR, &plan));
    g_assert_true(riscv_ame_m_export(
        &layout, &source, backing, &plan, memory));
    g_assert_cmpmem(memory, 16, column_major, sizeof(column_major));

    g_assert_true(riscv_ame_mem_plan_init(
        &layout, int8, AME_MEM_STRIDED, &plan));
    g_assert_true(riscv_ame_m_export(
        &layout, &source, backing, &plan, memory));
    for (i = 0; i < ARRAY_SIZE(column_major); i++) {
        g_assert_cmpuint(memory[i], ==, i);
    }

    g_assert_true(riscv_ame_mem_plan_init(
        &layout, int8, AME_MEM_TRANSPOSED_STRIDED, &plan));
    g_assert_true(riscv_ame_m_export(
        &layout, &source, backing, &plan, memory));
    g_assert_cmpmem(memory, 16, column_major, sizeof(column_major));

    g_assert_true(riscv_ame_compute_layout(
        1, 8, 16, 1, maximum_capabilities(),
        &tiny_layout, &error_abort));
    g_assert_true(riscv_ame_m_view_init(
        &tiny_layout, &tiny_source, 5, int4, 2));
    g_assert_true(riscv_ame_m_view_init(
        &tiny_layout, &tiny_destination, 9, int4, 2));
    value = (AmeValue) { .lo = 0xa };
    riscv_ame_m_write(&tiny_layout, backing, &tiny_source, 0, &value);
    value.lo = 0xb;
    riscv_ame_m_write(&tiny_layout, backing, &tiny_source, 1, &value);
    g_assert_true(riscv_ame_mem_plan_init(
        &tiny_layout, int4, AME_MEM_TRANSPOSED_STRIDED, &plan));
    memset(memory, 0xff, sizeof(memory));
    g_assert_true(riscv_ame_m_export(
        &tiny_layout, &tiny_source, backing, &plan, memory));
    g_assert_cmphex(memory[0], ==, 0x0a);
    g_assert_cmphex(memory[1], ==, 0x0b);
    memory[0] |= 0xf0;
    memory[1] |= 0xf0;
    g_assert_true(riscv_ame_m_import(
        &tiny_layout, &tiny_destination, imported, &plan, memory));
    g_assert_cmphex(imported[9], ==, 0xba);

    g_assert_true(riscv_ame_compute_layout(
        2, 64, 16, 1, maximum_capabilities(),
        &wide_layout, &error_abort));
    g_assert_true(riscv_ame_m_view_init(
        &wide_layout, &wide_source, 2, int128, 1));
    g_assert_true(riscv_ame_m_view_init(
        &wide_layout, &wide_destination, 4, int128, 1));
    for (i = 0; i < wide_source.total_elements; i++) {
        value = (AmeValue) {
            .lo = UINT64_C(0x1111111111111111) + i,
            .hi = UINT64_C(0xeeeeeeeeeeeeeeee) - i,
        };
        riscv_ame_m_write(&wide_layout, wide_backing,
                          &wide_source, i, &value);
    }
    g_assert_true(riscv_ame_mem_plan_init(
        &wide_layout, int128, AME_MEM_ROW_MAJOR, &plan));
    g_assert_true(riscv_ame_m_export(
        &wide_layout, &wide_source, wide_backing, &plan, memory));
    g_assert_true(riscv_ame_m_import(
        &wide_layout, &wide_destination, wide_imported, &plan, memory));
    for (i = 0; i < wide_destination.total_elements; i++) {
        value = riscv_ame_m_read(&wide_layout, wide_imported,
                                 &wide_destination, i);
        g_assert_cmphex(value.lo, ==,
                        UINT64_C(0x1111111111111111) + i);
        g_assert_cmphex(value.hi, ==,
                        UINT64_C(0xeeeeeeeeeeeeeeee) - i);
    }

    g_assert_true(riscv_ame_mem_plan_init(
        &tiny_layout, int4, AME_MEM_TRANSPOSED_STRIDED, &plan));
    plan.segment_bytes++;
    memory[0] = 0x5a;
    g_assert_false(riscv_ame_m_export(
        &tiny_layout, &tiny_source, backing, &plan, memory));
    g_assert_cmphex(memory[0], ==, 0x5a);
}

static void test_row_major_memory_copy(void)
{
    static const struct {
        unsigned n;
        unsigned u;
    } geometries[] = { { 1, 8 }, { 2, 16 }, { 16, 64 }, { 128, 64 } };

    for (size_t g = 0; g < ARRAY_SIZE(geometries); g++) {
        RISCVAMELayout layout;

        g_assert_true(riscv_ame_compute_layout(
            geometries[g].n, geometries[g].u, 32, 1,
            maximum_capabilities(), &layout, &error_abort));
        for (unsigned bits = 4; bits <= 128; bits *= 2) {
            const AmeDTypeDesc *desc = dtype(0x40000000 | bits);
            AmeDTypeShape shape;
            AmeMView view;
            AmeMemPlan plan, bad_plan;
            g_autofree uint8_t *backing = g_malloc(layout.mregs_bytes + 2);
            g_autofree uint8_t *expected = g_malloc(layout.mregs_bytes + 2);
            g_autofree uint8_t *memory = NULL;
            size_t offset;

            g_assert_true(riscv_ame_dtype_shape(&layout, desc, &shape));
            g_assert_true(riscv_ame_m_view_init(
                &layout, &view, layout.num_mregs - shape.group_size,
                desc, shape.pack_factor));
            g_assert_true(riscv_ame_mem_plan_init(
                &layout, desc, AME_MEM_ROW_MAJOR, &plan));
            g_assert_cmpuint(plan.total_bytes, ==,
                             riscv_ame_m_view_bytes(&layout, &view));
            offset = 1 + view.base * layout.mreg_bytes;
            memory = g_malloc(plan.total_bytes + 2);
            for (size_t i = 0; i < layout.mregs_bytes + 2; i++) {
                backing[i] = i * 73 + 19;
            }
            memcpy(expected, backing, layout.mregs_bytes + 2);
            memset(memory, 0xa5, plan.total_bytes + 2);

            /* Both buffers are byte-unaligned; the last M group is selected. */
            g_assert_true(riscv_ame_m_export(
                &layout, &view, backing + 1, &plan, memory + 1));
            g_assert_cmpmem(memory + 1, plan.total_bytes,
                            backing + offset, plan.total_bytes);
            g_assert_cmpmem(backing, layout.mregs_bytes + 2,
                            expected, layout.mregs_bytes + 2);
            g_assert_cmphex(memory[0], ==, 0xa5);
            g_assert_cmphex(memory[plan.total_bytes + 1], ==, 0xa5);

            for (size_t i = 0; i < plan.total_bytes; i++) {
                memory[i + 1] = i * 29 + 0x5a;
            }
            memcpy(expected + offset, memory + 1, plan.total_bytes);
            g_assert_true(riscv_ame_m_import(
                &layout, &view, backing + 1, &plan, memory + 1));
            g_assert_cmpmem(backing, layout.mregs_bytes + 2,
                            expected, layout.mregs_bytes + 2);

            /* A malformed plan must be rejected before the bulk-copy path. */
            bad_plan = plan;
            bad_plan.segment_bits++;
            memset(memory, 0xa5, plan.total_bytes + 2);
            g_assert_false(riscv_ame_m_import(
                &layout, &view, backing + 1, &bad_plan, memory + 1));
            g_assert_cmpmem(backing, layout.mregs_bytes + 2,
                            expected, layout.mregs_bytes + 2);
            g_assert_false(riscv_ame_m_export(
                &layout, &view, backing + 1, &bad_plan, memory + 1));
            for (size_t i = 0; i < plan.total_bytes + 2; i++) {
                g_assert_cmphex(memory[i], ==, 0xa5);
            }
        }
    }
}

static void test_raw_register_move_overlap(void)
{
    RISCVAMELayout layout;

    g_assert_true(riscv_ame_compute_layout(
        2, 16, 32, 1, maximum_capabilities(), &layout, &error_abort));
    for (unsigned bits = 4; bits <= 128; bits *= 2) {
        const AmeDTypeDesc *desc = dtype(0x40000000 | bits);
        AmeDTypeShape shape;
        g_autofree uint8_t *mregs = g_malloc(layout.mregs_bytes);
        g_autofree uint8_t *expected = g_malloc(layout.mregs_bytes);

        g_assert_true(riscv_ame_dtype_shape(&layout, desc, &shape));
        for (unsigned direction = 0; direction < 3; direction++) {
            AmeMView source, destination;
            uint32_t source_base = direction == 1 ? shape.group_size : 0;
            uint32_t dest_base = direction == 0 ? shape.group_size : 0;
            g_autofree uint8_t *snapshot = NULL;
            size_t bytes;

            g_assert_true(riscv_ame_m_view_init(
                &layout, &source, source_base, desc,
                MAX(shape.pack_factor, 2)));
            g_assert_true(riscv_ame_m_view_init(
                &layout, &destination, dest_base, desc,
                source.n_sq));
            bytes = riscv_ame_m_view_bytes(&layout, &source);
            for (size_t i = 0; i < layout.mregs_bytes; i++) {
                mregs[i] = i * 73 + bits;
            }
            memcpy(expected, mregs, layout.mregs_bytes);
            snapshot = g_memdup2(mregs + source_base * layout.mreg_bytes,
                                  bytes);
            memcpy(expected + dest_base * layout.mreg_bytes, snapshot, bytes);

            riscv_ame_move_m_to_m(&layout, mregs, &destination, &source);
            g_assert_cmpmem(mregs, layout.mregs_bytes,
                            expected, layout.mregs_bytes);
        }
    }
}

static void test_staged_register_moves(void)
{
    static const struct {
        uint32_t encoding;
        uint32_t source_base;
        uint32_t destination_base;
        uint32_t acc_base;
        uint32_t n_sq;
    } cases[] = {
        { 0x40000004, 0, 1, 1, 2 },
        { 0x40000008, 2, 3, 0, 1 },
        { 0x40000010, 4, 6, 3, 1 },
    };
    RISCVAMELayout layout;
    g_autofree uint8_t *mregs = NULL;
    g_autofree uint8_t *accs = NULL;
    g_autofree uint8_t *scratch = NULL;
    size_t scratch_bytes;

    g_assert_true(riscv_ame_compute_layout(
        4, 8, 16, 4, maximum_capabilities(),
        &layout, &error_abort));
    mregs = g_malloc0(layout.mregs_bytes);
    accs = g_malloc0(layout.accs_bytes);
    scratch_bytes = MAX(layout.mregs_bytes, layout.accs_bytes);
    scratch = g_malloc0(scratch_bytes);

    for (size_t test = 0; test < ARRAY_SIZE(cases); test++) {
        const AmeDTypeDesc *desc = dtype(cases[test].encoding);
        AmeMView source, destination;
        size_t payload_bytes = layout.n * layout.n * desc->bits / CHAR_BIT;
        uint64_t mask = MAKE_64BIT_MASK(0, desc->bits);
        size_t element, byte;
        uint32_t square;

        g_assert_true(riscv_ame_m_view_init(
            &layout, &source, cases[test].source_base, desc,
            cases[test].n_sq));
        g_assert_true(riscv_ame_m_view_init(
            &layout, &destination, cases[test].destination_base, desc,
            cases[test].n_sq));
        for (element = 0; element < source.total_elements; element++) {
            AmeValue value = {
                .lo = (UINT64_C(0x31) * (test + 1) + element) & mask,
            };

            riscv_ame_m_write(&layout, mregs, &source, element, &value);
        }

        riscv_ame_move_m_to_m(&layout, mregs, &destination, &source);
        for (element = 0; element < source.total_elements; element++) {
            AmeValue expected = riscv_ame_m_read(
                &layout, mregs, &source, element);
            AmeValue actual = riscv_ame_m_read(
                &layout, mregs, &destination, element);

            g_assert_cmphex(actual.lo, ==, expected.lo);
            g_assert_cmphex(actual.hi, ==, expected.hi);
        }

        riscv_ame_move_m_to_m(&layout, mregs, &source, &source);

        memset(accs, 0xcc, layout.accs_bytes);
        riscv_ame_move_m_to_acc(
            &layout, accs, cases[test].acc_base, cases[test].n_sq,
            mregs, &source, scratch, scratch_bytes);
        for (square = 0; square < cases[test].n_sq; square++) {
            AmeAccView acc;

            g_assert_true(riscv_ame_acc_view_init(
                &layout, &acc, cases[test].acc_base + square, desc));
            for (element = 0; element < acc.elements_per_square; element++) {
                AmeValue expected = riscv_ame_m_read(
                    &layout, mregs, &source,
                    square * acc.elements_per_square + element);
                AmeValue actual = riscv_ame_acc_read(
                    &layout, accs, &acc, element);

                g_assert_cmphex(actual.lo, ==, expected.lo);
                g_assert_cmphex(actual.hi, ==, expected.hi);
            }
            for (byte = payload_bytes; byte < layout.acc_bytes; byte++) {
                g_assert_cmphex(
                    accs[(cases[test].acc_base + square) *
                         layout.acc_bytes + byte], ==, 0xcc);
            }
        }
        for (square = 0; square < layout.num_accregs; square++) {
            if (square >= cases[test].acc_base &&
                square < cases[test].acc_base + cases[test].n_sq) {
                continue;
            }
            for (byte = 0; byte < layout.acc_bytes; byte++) {
                g_assert_cmphex(accs[square * layout.acc_bytes + byte], ==,
                                0xcc);
            }
        }

        memset(mregs + destination.base * layout.mreg_bytes, 0,
               riscv_ame_m_view_bytes(&layout, &destination));
        riscv_ame_move_acc_to_m(
            &layout, mregs, &destination, accs, cases[test].acc_base,
            cases[test].n_sq, scratch, scratch_bytes);
        for (element = 0; element < source.total_elements; element++) {
            AmeValue expected = riscv_ame_m_read(
                &layout, mregs, &source, element);
            AmeValue actual = riscv_ame_m_read(
                &layout, mregs, &destination, element);

            g_assert_cmphex(actual.lo, ==, expected.lo);
            g_assert_cmphex(actual.hi, ==, expected.hi);
        }
    }
}

/* Physical square packing, independent of the implementation's view offset. */
static size_t reference_m_offset(uint64_t n, uint64_t u, unsigned int bits,
                                 unsigned int base, size_t element)
{
    size_t square_elements = n * n;
    size_t square = element / square_elements;
    size_t local = element % square_elements;

    if (bits < u) {
        size_t packed = u / bits;
        size_t reg = base + square / packed;
        size_t pos = (square % packed) * square_elements + local;

        return reg * square_elements * u + pos * bits;
    }
    return (base + square * (bits / u)) * square_elements * u + local * bits;
}

static void reference_copy_bits(uint8_t *destination, size_t dest_bit,
                                const uint8_t *source, size_t source_bit,
                                size_t bits)
{
    for (size_t i = 0; i < bits; i++) {
        unsigned int value = (source[(source_bit + i) / 8] >>
                              ((source_bit + i) % 8)) & 1;
        size_t bit = dest_bit + i;

        destination[bit / 8] = (destination[bit / 8] & ~(1u << (bit % 8))) |
                               (value << (bit % 8));
    }
}

static void check_staged_square_moves(const RISCVAMELayout *layout,
                                      const AmeMView *view, bool alias)
{
    size_t m_bytes = alias ? MAX(layout->mregs_bytes, layout->accs_bytes) :
                            layout->mregs_bytes;
    size_t scratch_bytes = MAX(riscv_ame_m_view_bytes(layout, view),
                               view->n_sq * layout->acc_bytes);
    uint32_t acc_base = view->n_sq == layout->num_accregs ? 0 : 1;
    g_autofree uint8_t *m_storage = g_malloc(m_bytes + 2);
    g_autofree uint8_t *acc_storage = g_malloc(layout->accs_bytes + 2);
    g_autofree uint8_t *scratch_storage = g_malloc(scratch_bytes + 2);
    uint8_t *mregs = m_storage + 1;
    uint8_t *accs = alias ? mregs : acc_storage + 1;
    uint8_t *scratch = scratch_storage + 1;

    for (unsigned int to_acc = 0; to_acc < 2; to_acc++) {
        g_autofree uint8_t *m_original = NULL;
        g_autofree uint8_t *acc_original = NULL;
        g_autofree uint8_t *m_expected = NULL;
        g_autofree uint8_t *acc_expected_storage = NULL;
        uint8_t *acc_expected;
        size_t used_bytes = to_acc ? view->n_sq * layout->acc_bytes :
                                    riscv_ame_m_view_bytes(layout, view);

        for (size_t i = 0; i < m_bytes; i++) {
            mregs[i] = (i * 73 + (i >> 3) + 0x39) ^ (i >> 7);
        }
        if (!alias) {
            for (size_t i = 0; i < layout->accs_bytes; i++) {
                accs[i] = (i * 139 + (i >> 5) + 0xa6) ^ (i >> 9);
            }
        }
        m_storage[0] = acc_storage[0] = 0x5a;
        mregs[m_bytes] = acc_storage[layout->accs_bytes + 1] = 0xa5;
        memset(scratch_storage, 0xcd, scratch_bytes + 2);
        m_original = g_memdup2(mregs, m_bytes);
        acc_original = g_memdup2(accs, layout->accs_bytes);
        m_expected = g_memdup2(m_original, m_bytes);
        if (alias) {
            acc_expected = m_expected;
        } else {
            acc_expected_storage = g_memdup2(acc_original, layout->accs_bytes);
            acc_expected = acc_expected_storage;
        }

        for (size_t square = 0; square < view->n_sq; square++) {
            for (size_t element = 0; element < layout->n * layout->n;
                 element++) {
                size_t m_bit = reference_m_offset(
                    layout->n, layout->u, view->dtype->bits, view->base,
                    square * layout->n * layout->n + element);
                size_t acc_bit = (acc_base + square) * layout->acc_bytes * 8 +
                                 element * view->dtype->bits;

                if (to_acc) {
                    reference_copy_bits(acc_expected, acc_bit, m_original,
                                         m_bit, view->dtype->bits);
                } else {
                    reference_copy_bits(m_expected, m_bit, acc_original,
                                         acc_bit, view->dtype->bits);
                }
            }
        }
        if (to_acc) {
            riscv_ame_move_m_to_acc(layout, accs, acc_base, view->n_sq,
                                   mregs, view, scratch, used_bytes);
        } else {
            riscv_ame_move_acc_to_m(layout, mregs, view, accs, acc_base,
                                   view->n_sq, scratch, used_bytes);
        }
        g_assert_cmpmem(mregs, m_bytes, m_expected, m_bytes);
        g_assert_cmpmem(accs, layout->accs_bytes,
                        acc_expected, layout->accs_bytes);
        g_assert_cmphex(m_storage[0], ==, 0x5a);
        g_assert_cmphex(mregs[m_bytes], ==, 0xa5);
        g_assert_cmphex(acc_storage[0], ==, 0x5a);
        g_assert_cmphex(acc_storage[layout->accs_bytes + 1], ==, 0xa5);
        g_assert_cmphex(scratch_storage[0], ==, 0xcd);
        for (size_t i = used_bytes; i <= scratch_bytes; i++) {
            g_assert_cmphex(scratch[i], ==, 0xcd);
        }
    }
}

static void test_staged_register_moves_bit_reference(void)
{
    static const struct {
        uint32_t n;
        uint32_t u;
    } geometries[] = { { 1, 8 }, { 2, 2 }, { 4, 8 }, { 8, 16 }, { 8, 64 } };
    size_t cases = 0;

    for (size_t i = 0; i < ARRAY_SIZE(geometries); i++) {
        RISCVAMELayout layout;

        g_assert_true(riscv_ame_compute_layout(
            geometries[i].n, geometries[i].u, 32, 16,
            maximum_capabilities(), &layout, &error_abort));
        for (unsigned int bits = 4; bits <= 128; bits *= 2) {
            const AmeDTypeDesc *desc = dtype(0x40000000 | bits);
            uint32_t group = MAX(1, bits / layout.u);
            uint32_t pack = MAX(1, layout.u / bits);

            for (uint32_t count = pack; count <= 16; count *= 2) {
                AmeMView view;

                if (!riscv_ame_m_view_init(&layout, &view, group, desc,
                                           count)) {
                    continue;
                }
                check_staged_square_moves(&layout, &view, false);
                check_staged_square_moves(&layout, &view, true);
                cases += 4;
            }
        }
    }
    g_test_message("%zu M/Acc moves checked against independent bit layout",
                    cases);
    g_assert_cmpuint(cases, >, 0);
}

static void check_m_access(const RISCVAMELayout *layout, const AmeMView *view,
                           uint8_t *backing, const uint8_t *original,
                           size_t element)
{
    unsigned int bits = view->dtype->bits;
    size_t offset = reference_m_offset(layout->n, layout->u, bits,
                                       view->base, element);
    size_t byte = offset / 8;
    size_t bytes = MAX(1, bits / 8);
    AmeValue expected = { 0 }, replacement, actual;

    for (unsigned int bit = 0; bit < bits; bit++) {
        uint64_t value = (original[(offset + bit) / 8] >>
                          ((offset + bit) % 8)) & 1;

        if (bit < 64) {
            expected.lo |= value << bit;
        } else {
            expected.hi |= value << (bit - 64);
        }
    }
    actual = riscv_ame_m_read(layout, backing, view, element);
    g_assert_cmphex(actual.lo, ==, expected.lo);
    g_assert_cmphex(actual.hi, ==, expected.hi);
    replacement = (AmeValue) { ~expected.lo, ~expected.hi };
    riscv_ame_m_write(layout, backing, view, element, &replacement);
    for (size_t bit = byte * 8; bit < (byte + bytes) * 8; bit++) {
        unsigned int want = (original[bit / 8] >> (bit % 8)) & 1;
        unsigned int got = (backing[bit / 8] >> (bit % 8)) & 1;

        if (bit >= offset && bit < offset + bits) {
            want ^= 1;
        }
        g_assert_cmpuint(got, ==, want);
    }
    /* Only the selected element is restored; stray writes remain detectable. */
    memcpy(backing + byte, original + byte, bytes);
}

static void test_m_bit_reference(void)
{
    size_t valid_views = 0, accesses = 0;

    for (uint64_t n = 1; n <= 128; n *= 2) {
        for (uint64_t u = 1; u <= 64; u *= 2) {
            for (uint32_t nm = 16; nm <= 32; nm *= 2) {
                RISCVAMELayout layout;
                g_autofree uint8_t *backing_storage = NULL;
                g_autofree uint8_t *original = NULL;
                uint8_t *backing;

                if (n * n * u % 8 != 0) {
                    continue;
                }
                g_assert_true(riscv_ame_compute_layout(
                    n, u, nm, 1, maximum_capabilities(),
                    &layout, &error_abort));
                backing_storage = g_malloc(layout.mregs_bytes + 2);
                backing = backing_storage + 1;
                backing_storage[0] = 0x5a;
                backing[layout.mregs_bytes] = 0xa5;
                original = g_malloc(layout.mregs_bytes);
                for (size_t i = 0; i < layout.mregs_bytes; i++) {
                    original[i] = (i * UINT64_C(0x123456789abcdef) >> 29) ^ i;
                }
                memcpy(backing, original, layout.mregs_bytes);
                for (unsigned int bits = 4; bits <= 128; bits *= 2) {
                    const AmeDTypeDesc *desc = dtype(0x40000000 | bits);
                    unsigned int pack = MAX(1, u / bits);
                    unsigned int group = MAX(1, bits / u);

                    for (unsigned int nsq = 1; nsq <= 16; nsq *= 2) {
                        for (unsigned int base = 0; base < nm; base++) {
                            unsigned int span = nsq * group / pack;
                            bool valid = nsq % pack == 0 && base % group == 0 &&
                                         span <= nm - base;
                            AmeMView view;

                            g_assert_cmpint(riscv_ame_m_view_init(
                                &layout, &view, base, desc, nsq), ==, valid);
                            if (!valid) {
                                continue;
                            }
                            valid_views++;
                            for (size_t q = 0; q < nsq; q++) {
                                size_t step = n <= 8 ? 1 : n * n - 1;

                                for (size_t e = 0; e < n * n; e += step) {
                                    check_m_access(&layout, &view, backing,
                                                    original, q * n * n + e);
                                    accesses++;
                                }
                            }
                        }
                    }
                }
                g_assert_cmpmem(backing, layout.mregs_bytes,
                                original, layout.mregs_bytes);
                g_assert_cmphex(backing_storage[0], ==, 0x5a);
                g_assert_cmphex(backing[layout.mregs_bytes], ==, 0xa5);
            }
        }
    }
    g_test_message("%zu valid views, %zu bit-reference accesses",
                    valid_views, accesses);
    g_assert_cmpuint(valid_views, >, 0);
    g_assert_cmpuint(accesses, >, valid_views);
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);

    g_test_add_func("/riscv/ame/layout/geometry", test_layout_geometry);
    g_test_add_func("/riscv/ame/layout/invalid", test_layout_rejects_invalid);
    g_test_add_func("/riscv/ame/layout/subbyte-accumulator",
                    test_subbyte_accumulator);
    g_test_add_func("/riscv/ame/layout/fp-only", test_fp_only_layout);
    g_test_add_func("/riscv/ame/layout/views", test_dtype_shapes_and_views);
    g_test_add_func("/riscv/ame/layout/size-bounds", test_layout_size_bounds);
    g_test_add_func("/riscv/ame/layout/m-mapping", test_m_physical_mapping);
    g_test_add_func("/riscv/ame/layout/m-bit-reference", test_m_bit_reference);
    g_test_add_func("/riscv/ame/layout/acc-mapping",
                    test_acc_physical_mapping);
    g_test_add_func("/riscv/ame/layout/memory-plans", test_memory_plans);
    g_test_add_func("/riscv/ame/layout/memory-round-trip",
                    test_memory_round_trip);
    g_test_add_func("/riscv/ame/layout/row-major-memory-copy",
                    test_row_major_memory_copy);
    g_test_add_func("/riscv/ame/layout/raw-register-move-overlap",
                    test_raw_register_move_overlap);
    g_test_add_func("/riscv/ame/layout/staged-register-moves",
                    test_staged_register_moves);
    g_test_add_func("/riscv/ame/layout/staged-register-moves-bit-reference",
                    test_staged_register_moves_bit_reference);

    return g_test_run();
}
