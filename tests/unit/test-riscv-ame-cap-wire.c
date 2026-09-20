/*
 * Ztt capability manifest tests
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/bswap.h"
#include "target/riscv/ame/cap-wire.h"

#define I8  UINT32_C(0x40000008)
#define I16 UINT32_C(0x40000010)
#define I32 UINT32_C(0x40000020)
#define I64 UINT32_C(0x40000040)

static uint16_t id(uint32_t encoding)
{
    const AmeDTypeDesc *dtype = riscv_ame_dtype_by_encoding(encoding);

    g_assert_nonnull(dtype);
    return dtype->id;
}

static AmeDTypeSet types(const uint32_t *encodings, size_t count)
{
    AmeDTypeSet result = { 0 };

    g_assert_true(riscv_ame_dtype_set_from_encodings(
        &result, encodings, count, &error_abort));
    return result;
}

static AmeDTypeTuple tuple(uint32_t d, uint32_t a, uint32_t b)
{
    return (AmeDTypeTuple) { .dtype_ids = { id(d), id(a), id(b) } };
}

static void test_exact_manifest(void)
{
    /* Independently encoded G02 fixture: two correlated ADD/integer tuples. */
    static const uint8_t expected[] = {
        0x41, 0x4d, 0x45, 0x43, 0, 0, 0, 1, 0x40, 0, 0, 0x40,
        0, 4, 0x40, 0, 0, 8, 0x40, 0, 0, 0x10,
        0x40, 0, 0, 0x20, 0x40, 0, 0, 0x40,
        0, 1, 0, 2, 0, 1, 2, 0, 0, 0, 0, 2,
        0x40, 0, 0, 0x20, 0x40, 0, 0, 8, 0x40, 0, 0, 8,
        0x40, 0, 0, 0x40, 0x40, 0, 0, 0x10, 0x40, 0, 0, 0x10,
    };
    static const uint32_t encodings[] = { I64, I16, I8, I32 };
    AmeDTypeSet public = types(encodings, G_N_ELEMENTS(encodings));
    AmeDTypeTuple entries[] = { tuple(I64, I16, I16), tuple(I32, I8, I8),
                                tuple(I32, I8, I8) };
    AmeCapRequest request = {
        .key = { AME_OP_ADD, AME_FORM_BASE, AME_CAP_CATEGORY_INTEGER },
        .kind = AME_TUPLE_SET_EXACT,
        .exact = { entries, G_N_ELEMENTS(entries) },
    };
    AmeCapabilities set = { 0 };
    g_autoptr(GByteArray) manifest = NULL;

    g_assert_true(riscv_ame_capabilities_init(
        &set, I64, &public, &request, 1, &error_abort));
    manifest = riscv_ame_capabilities_manifest(&set, &error_abort);
    g_assert_cmpmem(manifest->data, manifest->len, expected, sizeof(expected));
    riscv_ame_capabilities_clear(&set);
}

static void test_rule_manifest(void)
{
    static const uint8_t expected[] = {
        0x41, 0x4d, 0x45, 0x43, 0, 0, 0, 1, 0x40, 0, 0, 0x40,
        0, 4, 0x40, 0, 0, 8, 0x40, 0, 0, 0x10,
        0x40, 0, 0, 0x20, 0x40, 0, 0, 0x40,
        0, 1, 0, 2, 0, 1, 1, 0,
        0, 2, 0x40, 0, 0, 0x20, 0x40, 0, 0, 0x40,
        0, 2, 0x40, 0, 0, 8, 0x40, 0, 0, 0x10,
        0, 2, 0x40, 0, 0, 8, 0x40, 0, 0, 0x10,
    };
    static const uint32_t encodings[] = { I64, I16, I8, I32 };
    static const uint32_t destinations[] = { I32, I64 };
    static const uint32_t sources[] = { I8, I16 };
    AmeDTypeSet public = types(encodings, G_N_ELEMENTS(encodings));
    AmeCapRequest request = {
        .key = { AME_OP_ADD, AME_FORM_BASE, AME_CAP_CATEGORY_INTEGER },
        .kind = AME_TUPLE_SET_RULE,
        .operand_types = { types(destinations, 2), types(sources, 2),
                   types(sources, 2) },
    };
    AmeDTypeTuple all[8];
    AmeCapabilities rule = { 0 }, exact = { 0 };
    g_autoptr(GByteArray) rule_manifest = NULL;
    g_autoptr(GByteArray) exact_manifest = NULL;
    unsigned n = 0;

    g_assert_true(riscv_ame_capabilities_init(
        &rule, I64, &public, &request, 1, &error_abort));
    for (unsigned d = 0; d < 2; d++) {
        for (unsigned a = 0; a < 2; a++) {
            for (unsigned b = 0; b < 2; b++) {
                all[n++] = tuple(destinations[d], sources[a], sources[b]);
            }
        }
    }
    request.kind = AME_TUPLE_SET_EXACT;
    request.exact.tuples = all;
    request.exact.count = n;
    g_assert_true(riscv_ame_capabilities_init(
        &exact, I64, &public, &request, 1, &error_abort));
    rule_manifest = riscv_ame_capabilities_manifest(&rule, &error_abort);
    exact_manifest = riscv_ame_capabilities_manifest(&exact, &error_abort);
    g_assert_cmpmem(rule_manifest->data, rule_manifest->len,
                    expected, sizeof(expected));
    g_assert_cmpmem(exact_manifest->data, exact_manifest->len,
                    expected, sizeof(expected));
    riscv_ame_capabilities_clear(&rule);
    riscv_ame_capabilities_clear(&exact);
}

static void test_empty_manifest(void)
{
    static const uint8_t expected[] = {
        0x41, 0x4d, 0x45, 0x43, 0, 0, 0, 1, 0x40, 0, 0, 8,
        0, 1, 0x40, 0, 0, 8, 0, 0,
    };
    static const uint32_t encodings[] = { I8 };
    AmeDTypeSet public = types(encodings, 1);
    AmeCapabilities set = { 0 };
    g_autoptr(GByteArray) manifest = NULL;

    g_assert_true(riscv_ame_capabilities_init(
        &set, I8, &public, NULL, 0, &error_abort));
    manifest = riscv_ame_capabilities_manifest(&set, &error_abort);
    g_assert_cmpmem(manifest->data, manifest->len, expected, sizeof(expected));
    riscv_ame_capabilities_clear(&set);
}

static void test_encoding_order(void)
{
    static const uint32_t encodings[] = { I8, 0x00000080 };
    AmeDTypeSet public = types(encodings, 2);
    AmeDTypeTuple entries[] = {
        { { id(I8), id(0x00000080), AME_DTYPE_ID_INVALID } },
        { { id(0x00000080), id(I8), AME_DTYPE_ID_INVALID } },
    };
    AmeCapRequest request = {
        .key = { AME_OP_CONV, AME_FORM_BASE, AME_CAP_CATEGORY_INTEGER },
        .kind = AME_TUPLE_SET_EXACT,
        .exact = { entries, 2 },
    };
    AmeCapabilities set = { 0 };
    g_autoptr(GByteArray) manifest = NULL;

    /* Dense ID order and encoding order deliberately disagree. */
    g_assert_cmpuint(id(I8), <, id(0x00000080));
    g_assert_true(riscv_ame_capabilities_init(
        &set, 0x00000080, &public, &request, 1, &error_abort));
    manifest = riscv_ame_capabilities_manifest(&set, &error_abort);
    g_assert_cmpuint(manifest->len, ==, 58);
    g_assert_cmphex(ldl_be_p(manifest->data + 14), ==, 0x00000080);
    g_assert_cmphex(ldl_be_p(manifest->data + 18), ==, I8);
    /* CONV has wire ID 11, independently of the internal operation enum. */
    g_assert_cmpuint(lduw_be_p(manifest->data + 24), ==, 11);
    g_assert_cmphex(ldl_be_p(manifest->data + 34), ==, 0x00000080);
    g_assert_cmphex(ldl_be_p(manifest->data + 38), ==, I8);
    g_assert_cmphex(ldl_be_p(manifest->data + 42), ==, 0); /* Absent B. */
    g_assert_cmphex(ldl_be_p(manifest->data + 46), ==, I8);
    g_assert_cmphex(ldl_be_p(manifest->data + 50), ==, 0x00000080);
    g_assert_cmphex(ldl_be_p(manifest->data + 54), ==, 0);
    riscv_ame_capabilities_clear(&set);
}

static void test_fp_only_manifest(void)
{
    static const uint8_t expected[] = {
        0x41, 0x4d, 0x45, 0x43, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 2, 0x14, 0x30, 1, 0x10, 0x20, 0x30, 1, 0x20, 0, 0,
    };
    static const uint32_t encodings[] = { 0x20300120, 0x14300110 };
    AmeDTypeSet public = types(encodings, G_N_ELEMENTS(encodings));
    AmeCapabilities set = { 0 }, copy = { 0 };
    g_autoptr(GByteArray) manifest = NULL;
    g_autoptr(GByteArray) copy_manifest = NULL;

    g_assert_true(riscv_ame_capabilities_init(
        &set, 0, &public, NULL, 0, &error_abort));
    riscv_ame_capabilities_copy(&copy, &set);
    manifest = riscv_ame_capabilities_manifest(&set, &error_abort);
    copy_manifest = riscv_ame_capabilities_manifest(&copy, &error_abort);
    g_assert_cmpmem(manifest->data, manifest->len, expected, sizeof(expected));
    g_assert_cmpmem(copy_manifest->data, copy_manifest->len,
                    expected, sizeof(expected));
    riscv_ame_capabilities_clear(&copy);
    riscv_ame_capabilities_clear(&set);
}

static void test_maximum_structure(void)
{
    const AmeCapabilities *max = riscv_ame_maximum_capabilities();
    g_autoptr(GByteArray) manifest =
        riscv_ame_capabilities_manifest(max, &error_abort);
    unsigned offset = 0, count, previous = 0;
    bool operations[80] = { false };

    g_assert_cmphex(ldl_be_p(manifest->data), ==, 0x414d4543);
    g_assert_cmpuint(ldl_be_p(manifest->data + 4), ==, 1);
    g_assert_cmphex(ldl_be_p(manifest->data + 8), ==, 0x40000080);
    g_assert_cmpuint(lduw_be_p(manifest->data + 12), ==, 130);
    offset = 14;
    for (unsigned i = 0; i < 130; i++, offset += 4) {
        unsigned encoding = ldl_be_p(manifest->data + offset);

        g_assert_cmpuint(encoding, >, previous);
        previous = encoding;
    }
    count = lduw_be_p(manifest->data + offset);
    g_assert_cmpuint(count, ==, max->nr_entries);
    offset += 2;
    previous = 0;
    for (unsigned i = 0; i < count; i++) {
        unsigned op, form, category, key;

        g_assert_cmpuint(offset + 6, <=, manifest->len);
        op = lduw_be_p(manifest->data + offset);
        form = manifest->data[offset + 2];
        category = manifest->data[offset + 3];
        g_assert_cmpuint(op, <, G_N_ELEMENTS(operations));
        g_assert_cmpuint(form, <, 3);
        g_assert_cmpuint(category, <, 7);
        g_assert_cmpuint(manifest->data[offset + 4], ==, 1); /* RULE */
        g_assert_cmpuint(manifest->data[offset + 5], ==, 0);
        operations[op] = true;
        key = (op << 16) | (form << 8) | category;
        g_assert_true(i == 0 || key > previous);
        previous = key;
        offset += 6;
        for (unsigned pos = 0; pos < 3; pos++) {
            unsigned nr, prior_encoding = 0;

            g_assert_cmpuint(offset + 2, <=, manifest->len);
            nr = lduw_be_p(manifest->data + offset);
            g_assert_cmpuint(nr, <=, 130);
            offset += 2;
            g_assert_cmpuint(offset + 4 * nr, <=, manifest->len);
            for (unsigned t = 0; t < nr; t++, offset += 4) {
                uint32_t encoding = ldl_be_p(manifest->data + offset);

                g_assert_cmpuint(encoding, >, prior_encoding);
                prior_encoding = encoding;
            }
        }
    }
    for (unsigned i = 0; i < G_N_ELEMENTS(operations); i++) {
        g_assert_true(operations[i]);
    }
    g_assert_cmpuint(offset, ==, manifest->len);
    g_test_message("maximum: %u entries, %u manifest bytes", count, offset);
}

static void test_fp_to_int_manifest(void)
{
    static const uint8_t expected[] = {
        0x41, 0x4d, 0x45, 0x43, 0, 0, 0, 1, 0x40, 0, 0, 8,
        0, 2, 0x14, 0x30, 1, 0x10, 0x40, 0, 0, 8,
        0, 1, 0, 11, 0, 4, 1, 0,
        0, 1, 0x40, 0, 0, 8, 0, 1, 0x14, 0x30, 1, 0x10, 0, 0,
    };
    static const uint32_t encodings[] = { I8, 0x14300110 };
    AmeDTypeSet public = types(encodings, G_N_ELEMENTS(encodings));
    AmeCapRequest request = {
        .key = { AME_OP_CONV, AME_FORM_BASE, AME_CAP_CATEGORY_FP_TO_INT },
        .kind = AME_TUPLE_SET_RULE,
        .operand_types = { types(&encodings[0], 1), types(&encodings[1], 1) },
    };
    AmeCapabilities set = { 0 };
    g_autoptr(GByteArray) manifest = NULL;

    g_assert_true(riscv_ame_capabilities_init(
        &set, I8, &public, &request, 1, &error_abort));
    manifest = riscv_ame_capabilities_manifest(&set, &error_abort);
    g_assert_cmpmem(manifest->data, manifest->len, expected, sizeof(expected));
    riscv_ame_capabilities_clear(&set);
}

static void test_mixed_arithmetic_manifest(void)
{
    static const uint32_t encodings[] = { I8, 0x20300120 };
    static const uint8_t expected[] = {
        0x41, 0x4d, 0x45, 0x43, 0, 0, 0, 1, 0x40, 0, 0, 8,
        0, 2, 0x20, 0x30, 1, 0x20, 0x40, 0, 0, 8, 0, 2,
        0, 2, 0, 5, 1, 0,
        0, 1, 0x20, 0x30, 1, 0x20,
        0, 1, 0x40, 0, 0, 8,
        0, 1, 0x20, 0x30, 1, 0x20,
        0, 2, 0, 6, 1, 0,
        0, 1, 0x40, 0, 0, 8,
        0, 1, 0x20, 0x30, 1, 0x20,
        0, 1, 0x40, 0, 0, 8,
    };
    AmeDTypeSet public = types(encodings, G_N_ELEMENTS(encodings));
    AmeDTypeTuple members[] = { tuple(encodings[1], I8, encodings[1]),
                                tuple(I8, encodings[1], I8) };
    AmeCapRequest requests[] = {
        { .key = { AME_OP_ADD, AME_FORM_BASE,
                   AME_CAP_CATEGORY_INT_FP_SOURCES },
          .kind = AME_TUPLE_SET_EXACT, .exact = { &members[0], 1 } },
        { .key = { AME_OP_ADD, AME_FORM_BASE,
                   AME_CAP_CATEGORY_FP_INT_SOURCES },
          .kind = AME_TUPLE_SET_EXACT, .exact = { &members[1], 1 } },
    };
    AmeCapabilities set = { 0 };
    g_autoptr(GByteArray) manifest = NULL;

    g_assert_true(riscv_ame_capabilities_init(
        &set, I8, &public, requests, G_N_ELEMENTS(requests), &error_abort));
    manifest = riscv_ame_capabilities_manifest(&set, &error_abort);
    g_assert_cmpmem(manifest->data, manifest->len, expected, sizeof(expected));
    riscv_ame_capabilities_clear(&set);
}

static void test_bounds(void)
{
    AmeCapEntry entry = { .tuples.kind = AME_TUPLE_SET_EXACT,
                         .tuples.nr_tuples = SIZE_MAX };
    AmeCapabilities set = { .entries = &entry, .nr_entries = 1 };
    Error *err = NULL;

    g_assert_null(riscv_ame_capabilities_manifest(&set, &err));
    g_assert_nonnull(err);
    error_free(err);
    err = NULL;
    set.nr_entries = UINT16_MAX + 1;
    g_assert_null(riscv_ame_capabilities_manifest(&set, &err));
    g_assert_nonnull(err);
    error_free(err);
    err = NULL;
    set.nr_entries = 1;
    entry.tuples.kind = AME_TUPLE_SET_EMPTY;
    g_assert_null(riscv_ame_capabilities_manifest(&set, &err));
    g_assert_nonnull(err);
    error_free(err);
    err = NULL;
    entry.tuples.kind = AME_TUPLE_SET_RULE;
    entry.key.op = AME_OP_COUNT;
    g_assert_null(riscv_ame_capabilities_manifest(&set, &err));
    g_assert_nonnull(err);
    error_free(err);
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);
    g_test_add_func("/riscv/ame/cap-wire/exact", test_exact_manifest);
    g_test_add_func("/riscv/ame/cap-wire/rule", test_rule_manifest);
    g_test_add_func("/riscv/ame/cap-wire/empty", test_empty_manifest);
    g_test_add_func("/riscv/ame/cap-wire/fp-only", test_fp_only_manifest);
    g_test_add_func("/riscv/ame/cap-wire/encoding-order", test_encoding_order);
    g_test_add_func("/riscv/ame/cap-wire/maximum", test_maximum_structure);
    g_test_add_func("/riscv/ame/cap-wire/fp-to-int", test_fp_to_int_manifest);
    g_test_add_func("/riscv/ame/cap-wire/mixed-arithmetic",
                    test_mixed_arithmetic_manifest);
    g_test_add_func("/riscv/ame/cap-wire/bounds", test_bounds);
    return g_test_run();
}
