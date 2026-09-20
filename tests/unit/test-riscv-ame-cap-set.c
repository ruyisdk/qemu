/*
 * Ztt v0.6 positive capability sets
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "target/riscv/ame/cap-set.h"

#define I8  UINT32_C(0x40000008)
#define I16 UINT32_C(0x40000010)
#define I32 UINT32_C(0x40000020)
#define I64 UINT32_C(0x40000040)
#define U8  UINT32_C(0x00000008)
#define H   UINT32_C(0x14300110)
#define HZ  UINT32_C(0x14700110)
#define F   UINT32_C(0x20300120)
#define E4  UINT32_C(0x10100108)
#define E5  UINT32_C(0x14300108)
#define NONE AME_DTYPE_ID_INVALID

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
    return (AmeDTypeTuple) {
        .dtype_ids = { id(d), a ? id(a) : NONE, b ? id(b) : NONE },
    };
}

static bool supports(const AmeCapabilities *set, AmeOperation op,
                     AmeCapForm form, uint32_t d, uint32_t a, uint32_t b)
{
    AmeDTypeTuple t = tuple(d, a, b);

    return riscv_ame_capabilities_has(set, op, form,
                                     t.dtype_ids[0], t.dtype_ids[1],
                                     t.dtype_ids[2]);
}

static void test_maximum(void)
{
    const AmeCapabilities *max = riscv_ame_maximum_capabilities();
    bool seen[AME_OP_COUNT] = { false };
    unsigned data_forms = 0, control_forms = 0, settable = 0;

    g_assert_cmphex(max->max_int_dtype, ==, RV_AME_MAX_INT_DTYPE);
    for (unsigned i = 0; i < AME_DTYPE_COUNT; i++) {
        settable += riscv_ame_dtype_set_has(&max->dtypes, i);
    }
    g_assert_cmpuint(settable, ==, 130);
    for (size_t i = 0; i < max->nr_entries; i++) {
        const AmeCapEntry *entry = &max->entries[i];

        g_assert_cmpuint(entry->tuples.kind, ==, AME_TUPLE_SET_RULE);
        g_assert_cmpuint(entry->tuples.count, >, 0);
        g_assert_true(riscv_ame_capabilities_entry(max, entry->key) == entry);
        seen[entry->key.op] = true;
    }
    for (unsigned op = 0; op < AME_OP_COUNT; op++) {
        g_assert_true(seen[op]);
        data_forms += max->by_op[op][AME_FORM_DATA_SCALAR].count != 0;
        control_forms += max->by_op[op][AME_FORM_CONTROL_SCALAR].count != 0;
    }
    g_assert_cmpuint(data_forms, ==, 22);
    g_assert_cmpuint(control_forms, ==, 5);
    g_assert_true(supports(max, AME_OP_ADD, AME_FORM_BASE, I32, I8, I16));
    g_assert_true(supports(max, AME_OP_ADD, AME_FORM_BASE, I8, I16, I16));
    g_assert_true(supports(max, AME_OP_ADD, AME_FORM_BASE, I32, I8, U8));
    g_assert_true(supports(max, AME_OP_ADD, AME_FORM_BASE, F, H, E4));
    g_assert_false(supports(max, AME_OP_MAX, AME_FORM_BASE, H, H, HZ));
    g_assert_true(supports(max, AME_OP_SIN, AME_FORM_BASE, H, HZ, 0));
    g_assert_true(supports(max, AME_OP_SIN, AME_FORM_BASE, E4, E4, 0));
    g_assert_true(supports(max, AME_OP_SIN, AME_FORM_BASE, E5, E5, 0));
    g_assert_true(supports(max, AME_OP_SIN, AME_FORM_BASE, F, F, 0));
    g_assert_true(supports(max, AME_OP_SIN, AME_FORM_BASE, E4, E5, 0));
    g_assert_true(supports(max, AME_OP_COLZIP, AME_FORM_BASE, I16, I16, 0));
    g_assert_false(supports(max, AME_OP_COLZIP, AME_FORM_BASE,
                            I16, I16, I16));
    g_assert_false(riscv_ame_capabilities_has(max, AME_OP_COUNT,
                                             AME_FORM_BASE, 0, 0, 0));
    g_assert_false(riscv_ame_capabilities_has(max, AME_OP_ADD,
                                             AME_FORM_MAX, 0, 0, 0));
}

static void test_scalar(void)
{
    static const uint32_t encodings[] = { I8, I16, U8, H, HZ, F };
    const AmeCapabilities *max = riscv_ame_maximum_capabilities();

    /* Independent v0.6 scalar conversion followed by (D,T,T) arithmetic. */
    for (unsigned d = 0; d < G_N_ELEMENTS(encodings); d++) {
        for (unsigned t = 0; t < G_N_ELEMENTS(encodings); t++) {
            for (unsigned s = 0; s < G_N_ELEMENTS(encodings); s++) {
                uint32_t de = encodings[d], te = encodings[t];
                uint32_t se = encodings[s];
                bool di = !(de & 0x100);

                g_assert_cmpint(supports(max, AME_OP_ADD,
                                         AME_FORM_DATA_SCALAR, de, te, se),
                                ==, true);
                g_assert_cmpint(supports(max, AME_OP_MAX,
                                         AME_FORM_DATA_SCALAR, de, te, se),
                                ==, de == te);
                g_assert_cmpint(supports(max, AME_OP_CMPGE,
                                         AME_FORM_DATA_SCALAR, de, te, se),
                                ==, di);
            }
        }
    }
    g_assert_true(supports(max, AME_OP_LDEXP, AME_FORM_CONTROL_SCALAR,
                           F, H, 0));
    g_assert_false(supports(max, AME_OP_LDEXP, AME_FORM_CONTROL_SCALAR,
                            F, H, I8));
    g_assert_false(supports(max, AME_OP_RDEXP, AME_FORM_CONTROL_SCALAR,
                            F, H, 0));
    g_assert_true(supports(max, AME_OP_SLL, AME_FORM_CONTROL_SCALAR,
                           I8, I8, 0));
    g_assert_false(supports(max, AME_OP_SRA, AME_FORM_CONTROL_SCALAR,
                            U8, U8, 0));
}

static void test_integer_narrowing(void)
{
    static const uint32_t enabled[] = { I8, I16, I32 };
    static const uint32_t sources[] = { I16, I32 };
    AmeDTypeSet public = types(enabled, G_N_ELEMENTS(enabled));
    AmeCapabilities rule = { 0 }, exact = { 0 };
    AmeDTypeTuple pairs[] = { tuple(I8, I16, I16), tuple(I8, I32, I32) };
    AmeCapRequest request = {
        .key = { AME_OP_ADD, AME_FORM_BASE, AME_CAP_CATEGORY_INTEGER },
        .kind = AME_TUPLE_SET_RULE,
    };

    riscv_ame_dtype_set_add(&request.operand_types[AME_TUPLE_D], id(I8));
    request.operand_types[AME_TUPLE_A] = types(sources, G_N_ELEMENTS(sources));
    request.operand_types[AME_TUPLE_B] = request.operand_types[AME_TUPLE_A];
    g_assert_true(riscv_ame_capabilities_init(
        &rule, I32, &public, &request, 1, &error_abort));
    g_assert_cmpuint(rule.entries[0].tuples.count, ==, 4);
    for (unsigned a = 0; a < G_N_ELEMENTS(sources); a++) {
        for (unsigned b = 0; b < G_N_ELEMENTS(sources); b++) {
            g_assert_true(supports(&rule, AME_OP_ADD, AME_FORM_BASE,
                                   I8, sources[a], sources[b]));
        }
    }

    request.kind = AME_TUPLE_SET_EXACT;
    request.exact.tuples = pairs;
    request.exact.count = G_N_ELEMENTS(pairs);
    g_assert_true(riscv_ame_capabilities_init(
        &exact, I32, &public, &request, 1, &error_abort));
    g_assert_cmpuint(exact.entries[0].tuples.kind, ==, AME_TUPLE_SET_EXACT);
    g_assert_true(supports(&exact, AME_OP_ADD, AME_FORM_BASE,
                           I8, I32, I32));
    /* QEMU's expanded maximum does not grant omitted backend tuples. */
    g_assert_false(supports(&exact, AME_OP_ADD, AME_FORM_BASE,
                            I8, I16, I32));
    g_assert_false(supports(&exact, AME_OP_ABS, AME_FORM_BASE,
                            I8, I16, 0));
    g_assert_true(riscv_ame_capabilities_for_cpu(
        AME_CAP_POLICY_EXPLICIT, &exact, &error_abort) == &exact);
    riscv_ame_capabilities_clear(&rule);
    riscv_ame_capabilities_clear(&exact);
}

static void test_fp_to_int_allowlist(void)
{
    static const uint32_t encodings[] = { I8, I16, H, F };
    AmeDTypeSet public = types(encodings, G_N_ELEMENTS(encodings));
    AmeDTypeTuple pair = tuple(I8, H, 0);
    AmeCapRequest request = {
        .key = { AME_OP_CONV, AME_FORM_BASE, AME_CAP_CATEGORY_FP_TO_INT },
        .kind = AME_TUPLE_SET_EXACT,
        .exact = { &pair, 1 },
    };
    AmeCapabilities set = { 0 };

    g_assert_true(riscv_ame_capabilities_init(
        &set, I16, &public, &request, 1, &error_abort));
    g_assert_true(supports(&set, AME_OP_CONV, AME_FORM_BASE, I8, H, 0));
    g_assert_false(supports(&set, AME_OP_CONV, AME_FORM_BASE, I16, H, 0));
    g_assert_false(supports(&set, AME_OP_CONV, AME_FORM_BASE, I8, F, 0));
    g_assert_false(supports(&set, AME_OP_CONV, AME_FORM_BASE, H, I8, 0));
    riscv_ame_capabilities_clear(&set);

    /*
     * A scalar instruction's conversion and arithmetic do not enable
     * standalone mconv.ew or madd.ew instructions.
     */
    pair = tuple(I8, I8, H);
    request.key = (AmeCapKey) {
        AME_OP_ADD, AME_FORM_DATA_SCALAR, AME_CAP_CATEGORY_INTEGER,
    };
    g_assert_true(riscv_ame_capabilities_init(
        &set, I16, &public, &request, 1, &error_abort));
    g_assert_true(supports(&set, AME_OP_ADD, AME_FORM_DATA_SCALAR,
                           I8, I8, H));
    g_assert_false(supports(&set, AME_OP_ADD, AME_FORM_DATA_SCALAR,
                            I8, I8, F));
    g_assert_false(supports(&set, AME_OP_CONV, AME_FORM_BASE, I8, H, 0));
    g_assert_false(supports(&set, AME_OP_ADD, AME_FORM_BASE, I8, I8, I8));
    riscv_ame_capabilities_clear(&set);
}

static void test_exact_and_identity(void)
{
    static const uint32_t enabled[] = { I8, I16, I32, I64, U8 };
    static const uint32_t destinations[] = { I32, I64 };
    static const uint32_t sources[] = { I8, I16 };
    AmeCapabilities exact = { 0 }, rule = { 0 }, clone = { 0 };
    AmeCapabilities complete = { 0 };
    AmeDTypeTuple sparse[] = { tuple(I64, I16, I16), tuple(I32, I8, I8),
                              tuple(I32, I8, I8) };
    AmeDTypeTuple all[8];
    AmeDTypeSet public = types(enabled, G_N_ELEMENTS(enabled));
    AmeCapRequest request = {
        .key = { AME_OP_ADD, AME_FORM_BASE, AME_CAP_CATEGORY_INTEGER },
        .kind = AME_TUPLE_SET_EXACT,
        .exact = { sparse, G_N_ELEMENTS(sparse) },
    };
    unsigned n = 0;

    g_assert_true(riscv_ame_capabilities_init(
        &exact, I64, &public, &request, 1, &error_abort));
    g_assert_cmpuint(exact.entries[0].tuples.kind, ==, AME_TUPLE_SET_EXACT);
    g_assert_cmpuint(exact.entries[0].tuples.count, ==, 2);
    /* Input memory is borrowed only during construction. */
    sparse[0] = tuple(I8, I8, I8);
    g_assert_true(supports(&exact, AME_OP_ADD, AME_FORM_BASE,
                           I64, I16, I16));
    g_assert_false(supports(&exact, AME_OP_ADD, AME_FORM_BASE,
                            I64, I8, I16));
    request.kind = AME_TUPLE_SET_RULE;
    request.operand_types[0] = types(destinations, G_N_ELEMENTS(destinations));
    request.operand_types[1] = types(sources, G_N_ELEMENTS(sources));
    request.operand_types[2] = request.operand_types[1];
    g_assert_true(riscv_ame_capabilities_init(
        &rule, I64, &public, &request, 1, &error_abort));
    g_assert_false(riscv_ame_capabilities_equal(&exact, &rule));
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
        &complete, I64, &public, &request, 1, &error_abort));
    g_assert_true(riscv_ame_capabilities_equal(&complete, &rule));
    riscv_ame_capabilities_copy(&clone, &exact);
    g_assert_true(riscv_ame_capabilities_equal(&clone, &exact));
    riscv_ame_capabilities_clear(&exact);
    g_assert_true(supports(&clone, AME_OP_ADD, AME_FORM_BASE,
                           I32, I8, I8));
    riscv_ame_capabilities_clear(&clone);
    riscv_ame_capabilities_clear(&complete);
    riscv_ame_capabilities_clear(&rule);
}

static void test_scalar_only_backend(void)
{
    static const uint32_t enabled[] = { I8, I16, I32, U8 };
    AmeDTypeSet public = types(enabled, G_N_ELEMENTS(enabled));
    AmeDTypeTuple only = tuple(I32, I16, I8);
    AmeCapRequest request = {
        .key = { AME_OP_ADD, AME_FORM_DATA_SCALAR, AME_CAP_CATEGORY_INTEGER },
        .kind = AME_TUPLE_SET_EXACT,
        .exact = { &only, 1 },
    };
    AmeCapabilities backend = { 0 };

    g_assert_true(riscv_ame_capabilities_init(
        &backend, I32, &public, &request, 1, &error_abort));
    g_assert_true(supports(&backend, AME_OP_ADD, AME_FORM_DATA_SCALAR,
                           I32, I16, I8));
    g_assert_false(supports(&backend, AME_OP_ADD, AME_FORM_DATA_SCALAR,
                            I32, I16, U8));
    /* Implementation dependencies do not authorize guest instructions. */
    g_assert_false(supports(&backend, AME_OP_ADD, AME_FORM_BASE,
                            I32, I16, I16));
    g_assert_false(supports(&backend, AME_OP_CONV, AME_FORM_BASE,
                            I16, I8, 0));
    riscv_ame_capabilities_clear(&backend);
}

static void test_mixed_exact_allowlist(void)
{
    const uint32_t encodings[] = { I8, U8 };
    AmeDTypeSet public = types(encodings, G_N_ELEMENTS(encodings));
    AmeDTypeTuple members[] = { tuple(I8, I8, I8), tuple(U8, U8, U8) };
    AmeCapRequest request = {
        .key = { AME_OP_ADD, AME_FORM_BASE, AME_CAP_CATEGORY_INTEGER },
        .kind = AME_TUPLE_SET_EXACT,
        .exact = { members, G_N_ELEMENTS(members) },
    };
    AmeCapabilities backend = { 0 };
    const AmeCapabilities *maximum = riscv_ame_maximum_capabilities();

    g_assert_true(riscv_ame_capabilities_init(
        &backend, I8, &public, &request, 1, &error_abort));
    /* Both types are exposed, but only the two listed tuples are allowed. */
    g_assert_cmpuint(backend.entries[0].tuples.kind, ==, AME_TUPLE_SET_EXACT);
    for (unsigned d = 0; d < G_N_ELEMENTS(encodings); d++) {
        for (unsigned a = 0; a < G_N_ELEMENTS(encodings); a++) {
            for (unsigned b = 0; b < G_N_ELEMENTS(encodings); b++) {
                g_assert_true(supports(maximum, AME_OP_ADD, AME_FORM_BASE,
                                       encodings[d], encodings[a],
                                       encodings[b]));
                g_assert_cmpint(supports(&backend, AME_OP_ADD,
                                         AME_FORM_BASE, encodings[d],
                                         encodings[a], encodings[b]),
                                ==, d == a && d == b);
            }
        }
    }
    riscv_ame_capabilities_clear(&backend);
}

static void test_integer_float_allowlist(void)
{
    const uint32_t encodings[] = { I8, F };
    AmeDTypeSet public = types(encodings, G_N_ELEMENTS(encodings));

    for (unsigned order = 0; order < 2; order++) {
        AmeCapabilities rule = { 0 }, exact = { 0 };
        AmeDTypeTuple tuples[] = {
            tuple(I8, order ? F : I8, order ? I8 : F),
            tuple(F, order ? F : I8, order ? I8 : F),
        };
        AmeCapRequest request = {
            .key = { AME_OP_ADD, AME_FORM_BASE,
                     order ? AME_CAP_CATEGORY_FP_INT_SOURCES :
                             AME_CAP_CATEGORY_INT_FP_SOURCES },
            .kind = AME_TUPLE_SET_RULE,
            .operand_types = { public, types(&encodings[order], 1),
                               types(&encodings[!order], 1) },
        };
        Error *err = NULL;

        g_assert_true(riscv_ame_capabilities_init(
            &rule, I8, &public, &request, 1, &error_abort));
        request.kind = AME_TUPLE_SET_EXACT;
        request.exact.tuples = tuples;
        request.exact.count = G_N_ELEMENTS(tuples);
        g_assert_true(riscv_ame_capabilities_init(
            &exact, I8, &public, &request, 1, &error_abort));
        g_assert_true(riscv_ame_capabilities_equal(&rule, &exact));
        for (unsigned d = 0; d < 2; d++) {
            for (unsigned a = 0; a < 2; a++) {
                for (unsigned b = 0; b < 2; b++) {
                    g_assert_cmpint(supports(
                        &rule, AME_OP_ADD, AME_FORM_BASE,
                        encodings[d], encodings[a], encodings[b]), ==,
                        a == order && b != order);
                }
            }
        }
        /* A mixed-source key cannot describe the converted scalar tuple. */
        request.key.form = AME_FORM_DATA_SCALAR;
        AmeCapabilities invalid = { 0 };

        g_assert_false(riscv_ame_capabilities_init(
            &invalid, I8, &public, &request, 1, &err));
        g_assert_nonnull(err);
        error_free(err);
        riscv_ame_capabilities_clear(&rule);
        riscv_ame_capabilities_clear(&exact);
    }
}

static void test_fp_unary_allowlist(void)
{
    const uint32_t encodings[] = { H, F, E4 };
    AmeDTypeSet public = types(encodings, G_N_ELEMENTS(encodings));
    AmeDTypeTuple members[] = { tuple(H, F, 0), tuple(F, H, 0),
                               tuple(E4, E4, 0) };
    AmeCapRequest request = {
        .key = { AME_OP_SQRT, AME_FORM_BASE, AME_CAP_CATEGORY_FP },
        .kind = AME_TUPLE_SET_EXACT,
        .exact = { members, G_N_ELEMENTS(members) },
    };
    AmeCapabilities backend = { 0 }, product = { 0 };
    const AmeCapabilities *maximum = riscv_ame_maximum_capabilities();

    g_assert_true(riscv_ame_capabilities_init(
        &backend, 0, &public, &request, 1, &error_abort));
    request.kind = AME_TUPLE_SET_RULE;
    request.operand_types[AME_TUPLE_D] = public;
    request.operand_types[AME_TUPLE_A] = public;
    g_assert_true(riscv_ame_capabilities_init(
        &product, 0, &public, &request, 1, &error_abort));
    g_assert_false(riscv_ame_capabilities_equal(&backend, &product));
    for (unsigned d = 0; d < G_N_ELEMENTS(encodings); d++) {
        for (unsigned a = 0; a < G_N_ELEMENTS(encodings); a++) {
            g_assert_true(supports(maximum, AME_OP_SQRT, AME_FORM_BASE,
                                   encodings[d], encodings[a], 0));
            g_assert_true(supports(&product, AME_OP_SQRT, AME_FORM_BASE,
                                   encodings[d], encodings[a], 0));
            g_assert_cmpint(supports(&backend, AME_OP_SQRT,
                                     AME_FORM_BASE, encodings[d],
                                     encodings[a], 0),
                            ==, (d == 0 && a == 1) ||
                                (d == 1 && a == 0) || (d == 2 && a == 2));
        }
    }
    riscv_ame_capabilities_clear(&backend);
    riscv_ame_capabilities_clear(&product);
}

static void expect_invalid(uint32_t max_int, const AmeDTypeSet *public,
                            const AmeCapRequest *requests, size_t count)
{
    AmeCapabilities result = { 0 };
    Error *err = NULL;

    g_assert_false(riscv_ame_capabilities_init(
        &result, max_int, public, requests, count, &err));
    g_assert_nonnull(err);
    error_free(err);
    g_assert_null(result.entries);
    g_assert_cmpuint(result.nr_entries, ==, 0);
    g_assert_cmpuint(result.max_int_dtype, ==, 0);
    for (unsigned w = 0; w < AME_DTYPE_WORD_COUNT; w++) {
        g_assert_cmphex(result.dtypes.words[w], ==, 0);
    }
}

static void test_invalid(void)
{
    static const uint32_t enabled[] = { I8, I16, I32, H };
    static const uint32_t invalid_encodings[] = { 0, 0x80000008, 0x40000009 };
    static const uint32_t duplicate[] = { I8, I8 };
    AmeDTypeSet public = types(enabled, G_N_ELEMENTS(enabled));
    AmeDTypeSet bad;
    AmeDTypeTuple exact[] = { tuple(I32, I8, I8), tuple(I16, I16, I16) };
    AmeCapRequest requests[2] = {
        { .key = { AME_OP_ADD, AME_FORM_BASE, AME_CAP_CATEGORY_INTEGER },
          .kind = AME_TUPLE_SET_EXACT, .exact = { exact, 2 } },
    };
    Error *err = NULL;

    for (unsigned i = 0; i < G_N_ELEMENTS(invalid_encodings); i++) {
        bad = public;
        g_assert_false(riscv_ame_dtype_set_from_encodings(
            &bad, &invalid_encodings[i], 1, &err));
        g_assert_nonnull(err);
        error_free(err);
        err = NULL;
        g_assert_cmpmem(bad.words, sizeof(bad.words),
                        public.words, sizeof(public.words));
    }
    g_assert_false(riscv_ame_dtype_set_from_encodings(
        &bad, duplicate, G_N_ELEMENTS(duplicate), &err));
    error_free(err);

    expect_invalid(I8, &public, NULL, 0);  /* Not the greatest integer width. */
    expect_invalid(H, &public, NULL, 0);
    expect_invalid(I64, &public, NULL, 0); /* Max datatype is not settable. */
    expect_invalid(I32, NULL, NULL, 0);
    expect_invalid(I32, &public, NULL, 1);
    expect_invalid(I32, &public, requests, SIZE_MAX);
    bad = public;
    bad.words[AME_DTYPE_WORD_COUNT - 1] |= UINT64_C(1) << 63;
    expect_invalid(I32, &bad, NULL, 0);
    bad = public;
    g_assert_true(riscv_ame_dtype_set_add(&bad, id(0x11500108))); /* E4 RNO. */
    expect_invalid(I32, &bad, NULL, 0);

    requests[1] = requests[0];
    expect_invalid(I32, &public, requests, 2); /* Duplicate key. */
    requests[1].key.op = AME_OP_AND;
    requests[1].kind = AME_TUPLE_SET_EMPTY;
    expect_invalid(I32, &public, requests, 2); /* After one owned allocation. */
    requests[1].kind = AME_TUPLE_SET_RULE;
    memset(requests[1].operand_types, 0, sizeof(requests[1].operand_types));
    riscv_ame_dtype_set_add(&requests[1].operand_types[AME_TUPLE_D], id(I8));
    riscv_ame_dtype_set_add(&requests[1].operand_types[AME_TUPLE_A], id(I16));
    riscv_ame_dtype_set_add(&requests[1].operand_types[AME_TUPLE_B], id(I16));
    expect_invalid(I32, &public, requests, 2); /* Unreachable second entry. */
    requests[1] = requests[0];
    requests[1].key.op = AME_OP_COUNT;
    expect_invalid(I32, &public, requests, 2);
    requests[1].key.op = AME_OP_ADD;
    requests[1].key.form = AME_FORM_MAX;
    expect_invalid(I32, &public, requests, 2);
    requests[1].key.form = AME_FORM_BASE;
    requests[1].key.category = AME_CAP_CATEGORY_GENERIC;
    expect_invalid(I32, &public, requests, 2);
    requests[0].key.form = AME_FORM_CONTROL_SCALAR;
    expect_invalid(I32, &public, requests, 1); /* No ADD control form. */
    requests[0].key.form = AME_FORM_DATA_SCALAR;
    exact[0] = tuple(I32, H, I8); /* Integer destination, floating tensor. */
    expect_invalid(I32, &public, requests, 1);
    requests[0].exact.count = SIZE_MAX;
    expect_invalid(I32, &public, requests, 1);
    requests[0].exact.count = 1;
    requests[0].exact.tuples = NULL;
    expect_invalid(I32, &public, requests, 1);
    requests[0].kind = AME_TUPLE_SET_RULE;
    memset(requests[0].operand_types, 0, sizeof(requests[0].operand_types));
    /* Known FP type in the integer category. */
    requests[0].operand_types[0] = public;
    expect_invalid(I32, &public, requests, 1);
    memset(requests[0].operand_types, 0, sizeof(requests[0].operand_types));
    /* Not public. */
    riscv_ame_dtype_set_add(&requests[0].operand_types[0], id(I64));
    expect_invalid(I32, &public, requests, 1);
}

static void test_empty_and_public_identity(void)
{
    static const uint32_t enabled[] = { I8, I16 };
    static const uint32_t with_fp[] = { I8, I16, H };
    AmeDTypeSet public = types(enabled, G_N_ELEMENTS(enabled));
    AmeDTypeSet more = types(with_fp, G_N_ELEMENTS(with_fp));
    AmeCapabilities absent = { 0 }, empty = { 0 }, unused = { 0 };
    AmeCapRequest request = {
        .key = { AME_OP_ADD, AME_FORM_BASE, AME_CAP_CATEGORY_INTEGER },
        .kind = AME_TUPLE_SET_RULE,
    };

    /* An explicitly empty request is still equivalent to an absent entry. */
    g_assert_true(riscv_ame_capabilities_init(
        &absent, I16, &public, NULL, 0, &error_abort));
    g_assert_true(riscv_ame_capabilities_init(
        &empty, I16, &public, &request, 1, &error_abort));
    g_assert_true(riscv_ame_capabilities_equal(&absent, &empty));
    g_assert_cmpuint(empty.nr_entries, ==, 0);
    g_assert_true(riscv_ame_capabilities_init(
        &unused, I16, &more, NULL, 0, &error_abort));
    g_assert_false(riscv_ame_capabilities_equal(&absent, &unused));
    riscv_ame_capabilities_clear(&absent);
    riscv_ame_capabilities_clear(&empty);
    riscv_ame_capabilities_clear(&unused);
}

static void test_unreachable_types(void)
{
    static const uint32_t enabled[] = { I8, I16, I32, H, F };
    static const struct {
        AmeOperation op;
        AmeCapCategory category;
        uint32_t base;
        uint32_t unreachable[AME_TUPLE_SIZE];
    } cases[] = {
        { AME_OP_AND, AME_CAP_CATEGORY_INTEGER, I16, { I8, I32, I32 } },
        { AME_OP_MAX, AME_CAP_CATEGORY_FP, H, { F, F, F } },
    };
    AmeDTypeSet public = types(enabled, G_N_ELEMENTS(enabled));
    AmeCapRequest request = {
        .key = { AME_OP_ADD, AME_FORM_BASE, AME_CAP_CATEGORY_INTEGER },
        .kind = AME_TUPLE_SET_RULE,
    };
    AmeCapabilities empty = { 0 };

    for (unsigned c = 0; c < G_N_ELEMENTS(cases); c++) {
        for (unsigned pos = 0; pos < AME_TUPLE_SIZE; pos++) {
            AmeCapabilities result = { 0 };
            Error *err = NULL;
            g_autofree char *context = NULL;

            request.key.op = cases[c].op;
            request.key.category = cases[c].category;
            memset(request.operand_types, 0, sizeof(request.operand_types));
            for (unsigned i = 0; i < AME_TUPLE_SIZE; i++) {
                riscv_ame_dtype_set_add(&request.operand_types[i],
                                        id(cases[c].base));
            }
            riscv_ame_dtype_set_add(&request.operand_types[pos],
                                    id(cases[c].unreachable[pos]));
            g_assert_false(riscv_ame_capabilities_init(
                &result, I32, &public, &request, 1, &err));
            g_assert_nonnull(err);
            context = g_strdup_printf(
                "op=%u form=%u category=%u operand %c datatype 0x%08x",
                request.key.op, request.key.form, request.key.category,
                "DAB"[pos], cases[c].unreachable[pos]);
            g_assert_nonnull(strstr(error_get_pretty(err), context));
            g_assert_nonnull(strstr(error_get_pretty(err),
                                   "does not occur in any supported tuple"));
            error_free(err);
            g_assert_null(result.entries);
            g_assert_cmpuint(result.nr_entries, ==, 0);
            g_assert_cmpuint(result.max_int_dtype, ==, 0);
            g_assert_true(riscv_ame_dtype_set_empty(&result.dtypes));
        }
    }

    request.key.op = AME_OP_AND;
    request.key.category = AME_CAP_CATEGORY_INTEGER;
    memset(request.operand_types, 0, sizeof(request.operand_types));
    riscv_ame_dtype_set_add(&request.operand_types[AME_TUPLE_D], id(I8));
    riscv_ame_dtype_set_add(&request.operand_types[AME_TUPLE_A], id(I16));
    riscv_ame_dtype_set_add(&request.operand_types[AME_TUPLE_B], id(I16));
    /* Nonempty candidates with no legal combination must not disappear. */
    expect_invalid(I32, &public, &request, 1);

    /* An empty exact list requests no tuples, rather than unreachable types. */
    request.kind = AME_TUPLE_SET_EXACT;
    request.exact.tuples = NULL;
    request.exact.count = 0;
    g_assert_true(riscv_ame_capabilities_init(
        &empty, I32, &public, &request, 1, &error_abort));
    g_assert_cmpuint(empty.nr_entries, ==, 0);
    riscv_ame_capabilities_clear(&empty);
}

static void test_fp_only(void)
{
    static const uint32_t enabled[] = { H, F };
    static const AmeOperation exponents[] = {
        AME_OP_LDEXP, AME_OP_LDEXPACC,
    };
    AmeDTypeSet public = types(enabled, G_N_ELEMENTS(enabled));
    AmeCapabilities set = { 0 }, clone = { 0 };
    AmeCapRequest requests[] = {
        { .key = { AME_OP_ADD, AME_FORM_BASE, AME_CAP_CATEGORY_FP },
          .kind = AME_TUPLE_SET_RULE,
          .operand_types = { public, public, public } },
        { .key = { AME_OP_ADD, AME_FORM_DATA_SCALAR, AME_CAP_CATEGORY_FP },
          .kind = AME_TUPLE_SET_RULE,
          .operand_types = { public, public, public } },
    };

    g_assert_true(riscv_ame_capabilities_init(
        &set, 0, &public, requests, G_N_ELEMENTS(requests), &error_abort));
    g_assert_cmphex(set.max_int_dtype, ==, 0);
    g_assert_true(riscv_ame_capabilities_for_cpu(
        AME_CAP_POLICY_EXPLICIT, &set, &error_abort) == &set);
    g_assert_true(supports(&set, AME_OP_ADD, AME_FORM_BASE, F, H, F));
    g_assert_true(supports(&set, AME_OP_ADD, AME_FORM_DATA_SCALAR, F, H, F));
    g_assert_false(supports(&set, AME_OP_ADD, AME_FORM_BASE, F, H, I8));
    /* Internal scalar conversion does not grant a CONV guest operation. */
    g_assert_false(supports(&set, AME_OP_CONV, AME_FORM_BASE, H, F, 0));
    riscv_ame_capabilities_copy(&clone, &set);
    g_assert_true(riscv_ame_capabilities_equal(&clone, &set));

    for (unsigned i = 0; i < G_N_ELEMENTS(exponents); i++) {
        AmeCapRequest request = {
            .key = { exponents[i], AME_FORM_CONTROL_SCALAR,
                     AME_CAP_CATEGORY_FP },
            .kind = AME_TUPLE_SET_RULE, .operand_types = { public, public },
        };

        g_assert_false(supports(&set, exponents[i], AME_FORM_CONTROL_SCALAR,
                                F, H, 0));
        expect_invalid(0, &public, &request, 1);
    }
    riscv_ame_capabilities_clear(&clone);
    riscv_ame_capabilities_clear(&set);
}

static void test_max_int_domain(void)
{
    static const uint32_t encodings[] = {
        I8, U8, I16, UINT32_C(0x00000010), H, F,
    };
    static const uint32_t candidates[] = {
        0, I8, U8, I16, UINT32_C(0x00000010), I32, H,
        UINT32_C(0x40000009), UINT32_C(0x80000020),
    };
    unsigned accepted = 0, rejected = 0;

    for (unsigned mask = 0; mask < (1u << G_N_ELEMENTS(encodings)); mask++) {
        AmeDTypeSet public = { 0 };
        unsigned max_bits = 0;

        for (unsigned i = 0; i < G_N_ELEMENTS(encodings); i++) {
            if (mask & (1u << i)) {
                riscv_ame_dtype_set_add(&public, id(encodings[i]));
                /* Read the architectural class and width, not the table. */
                if (!(encodings[i] & 0x100)) {
                    max_bits = MAX(max_bits, encodings[i] & 0xff);
                }
            }
        }
        for (unsigned i = 0; i < G_N_ELEMENTS(candidates); i++) {
            uint32_t candidate = candidates[i];
            AmeCapabilities set = { 0 };
            bool expected = mask && !max_bits && !candidate;

            for (unsigned j = 0; j < G_N_ELEMENTS(encodings); j++) {
                expected |= (mask & (1u << j)) &&
                            candidate == encodings[j] &&
                            !(candidate & 0x100) &&
                            (candidate & 0xff) == max_bits;
            }
            if (!expected) {
                expect_invalid(candidate, &public, NULL, 0);
                rejected++;
                continue;
            }
            g_assert_true(riscv_ame_capabilities_init(
                &set, candidate, &public, NULL, 0, &error_abort));
            g_assert_true(riscv_ame_capabilities_for_cpu(
                AME_CAP_POLICY_EXPLICIT, &set, &error_abort) == &set);
            riscv_ame_capabilities_clear(&set);
            accepted++;
        }
    }
    g_assert_cmpuint(accepted, >, 0);
    g_assert_cmpuint(rejected, >, 0);
    g_assert_cmpuint(accepted + rejected, ==, 576);
    g_test_message("MAX_INT_DTYPE domain: %u accepted, %u rejected",
                    accepted, rejected);
}

/* Signatures transcribed from v0.6, independent of the production table. */
static unsigned base_operand_count(AmeOperation op)
{
    switch (op) {
    case AME_OP_ZERO:
    case AME_OP_COLID:
    case AME_OP_ROWID:
        return 1;
    case AME_OP_ABS:
    case AME_OP_COLBCAST:
    case AME_OP_COLUNZIP:
    case AME_OP_COLZIP:
    case AME_OP_CONV:
    case AME_OP_COS:
    case AME_OP_EXP2:
    case AME_OP_FRINTM:
    case AME_OP_FRINTN:
    case AME_OP_FRINTP:
    case AME_OP_FRINTZ:
    case AME_OP_LOG2:
    case AME_OP_PREFIXADD:
    case AME_OP_PREFIXADDCOL:
    case AME_OP_PREFIXMAX:
    case AME_OP_PREFIXMAXCOL:
    case AME_OP_REC:
    case AME_OP_REDUCEADD:
    case AME_OP_REDUCEADDCOL:
    case AME_OP_REDUCEMAX:
    case AME_OP_REDUCEMAXCOL:
    case AME_OP_REDUCEMIN:
    case AME_OP_REDUCEMINCOL:
    case AME_OP_ROWBCAST:
    case AME_OP_ROWSHIFT:
    case AME_OP_ROWUNZIP:
    case AME_OP_ROWZIP:
    case AME_OP_RSQRT:
    case AME_OP_SHIFT:
    case AME_OP_SIN:
    case AME_OP_SQRT:
    case AME_OP_TANH:
        return 2;
    default:
        return 3;
    }
}

/* Expected standard tuple domains, using architectural encodings. */
static bool base_oracle(AmeOperation op, uint32_t d, uint32_t a, uint32_t b)
{
    bool di = !(d & 0x100), ai = !(a & 0x100), bi = !(b & 0x100);

    /* OCP RNO is recognized but is not a settable datatype. */
    if (((d & 0xff) == 8 && !di && ((d >> 22) & 15) == 5) ||
        ((a & 0xff) == 8 && !ai && ((a >> 22) & 15) == 5) ||
        ((b & 0xff) == 8 && !bi && ((b >> 22) & 15) == 5)) {
        return false;
    }
    switch (op) {
    case AME_OP_ZERO:
        return true;
    case AME_OP_COLID:
    case AME_OP_ROWID:
        return di;
    case AME_OP_COLBCAST:
    case AME_OP_ROWBCAST:
    case AME_OP_COLZIP:
    case AME_OP_COLUNZIP:
    case AME_OP_ROWZIP:
    case AME_OP_ROWUNZIP:
    case AME_OP_ROWSHIFT:
    case AME_OP_SHIFT:
        return d == a;
    case AME_OP_GATHER:
    case AME_OP_GATHERROW:
        return d == a && bi;
    case AME_OP_SEL:
    case AME_OP_SELGE:
    case AME_OP_MUXGE:
    case AME_OP_MUXLT:
        return d == b;
    case AME_OP_CONV:
        return true;
    case AME_OP_AND:
    case AME_OP_ANDNOT:
    case AME_OP_OR:
    case AME_OP_ORNOT:
    case AME_OP_XOR:
        return di && d == a && d == b;
    case AME_OP_MAX:
    case AME_OP_MIN:
        return d == a && d == b;
    case AME_OP_CMPGE:
    case AME_OP_CMPLT:
        return di;
    case AME_OP_ABS:
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
        return true;
    case AME_OP_LDEXP:
    case AME_OP_LDEXPACC:
    case AME_OP_RDEXP:
    case AME_OP_RDEXPACC:
    case AME_OP_COLSCATADD:
    case AME_OP_COLSCATMAX:
    case AME_OP_ROWSCATADD:
    case AME_OP_ROWSCATMAX:
        return bi;
    case AME_OP_SLL:
    case AME_OP_SRL:
        return di && ai && bi;
    case AME_OP_SRA:
        return di && ai && (a & 0x40000000) && bi;
    case AME_OP_EXP2:
    case AME_OP_FRINTM:
    case AME_OP_FRINTN:
    case AME_OP_FRINTP:
    case AME_OP_FRINTZ:
    case AME_OP_LOG2:
    case AME_OP_REC:
    case AME_OP_RSQRT:
    case AME_OP_SQRT:
    case AME_OP_TANH:
    case AME_OP_SIN:
    case AME_OP_COS:
        return !di && !ai;
    case AME_OP_LOG2SUB:
    case AME_OP_SUBLOG2:
        return !di && !ai;
    case AME_OP_ABSDIFF:
    case AME_OP_ADD:
    case AME_OP_HDIFF:
    case AME_OP_MEAN:
    case AME_OP_MUL:
    case AME_OP_NMUL:
    case AME_OP_SUB:
    case AME_OP_MULACC:
    case AME_OP_MULADD:
    case AME_OP_MULNACC:
    case AME_OP_MULSUB:
    case AME_OP_MMACC:
    case AME_OP_MMACCNEG:
    case AME_OP_MMATACC:
    case AME_OP_MMATACCNEG:
    case AME_OP_MMBTACC:
    case AME_OP_MMBTACCNEG:
        return true;
    default:
        g_assert_not_reached();
    }
}

static void test_base_form(void)
{
    static const uint32_t encodings[] = {
        0x40000004, I8, I16, I32, I64, 0x40000080, U8, 0x00000080,
        0x68000008, H, HZ, 0x20300110, F, 0x2c300140, E4, E5, 0x11500108,
    };
    const AmeCapabilities *max = riscv_ame_maximum_capabilities();
    unsigned checks = 0;

    for (unsigned op = 0; op < AME_OP_COUNT; op++) {
        unsigned operands = base_operand_count(op);
        unsigned na = operands >= 2 ? G_N_ELEMENTS(encodings) : 1;
        unsigned nb = operands >= 3 ? G_N_ELEMENTS(encodings) : 1;

        for (unsigned d = 0; d < G_N_ELEMENTS(encodings); d++) {
            for (unsigned a = 0; a < na; a++) {
                for (unsigned b = 0; b < nb; b++) {
                    uint32_t de = encodings[d];
                    uint32_t ae = operands >= 2 ? encodings[a] : 0;
                    uint32_t be = operands >= 3 ? encodings[b] : 0;
                    bool expected = base_oracle(op, de, ae, be);
                    bool actual = supports(max, op, AME_FORM_BASE,
                                            de, ae, be);

                    if (actual != expected) {
                        g_error("op=%u D=%08x A=%08x B=%08x got=%d expected=%d",
                                op, de, ae, be, actual, expected);
                    }
                    checks++;
                }
            }
        }
    }
    g_test_message("independent base-form encoding checks: %u", checks);
}

static void test_explicit_rule_reachability(void)
{
    static const uint32_t enc[] = { I8, I16, I32, U8 };
    static const AmeCapKey keys[] = {
        { AME_OP_CMPGE, AME_FORM_BASE, AME_CAP_CATEGORY_INTEGER }, /* ANY */
        { AME_OP_AND, AME_FORM_BASE, AME_CAP_CATEGORY_INTEGER },   /* SAME */
        { AME_OP_ADD, AME_FORM_BASE, AME_CAP_CATEGORY_INTEGER },   /* ANY */
        { AME_OP_GATHER, AME_FORM_BASE, AME_CAP_CATEGORY_GENERIC }, /* DA */
        { AME_OP_SEL, AME_FORM_BASE, AME_CAP_CATEGORY_GENERIC },   /* DB */
    };
    AmeDTypeSet public = types(enc, G_N_ELEMENTS(enc));

    for (unsigned k = 0; k < G_N_ELEMENTS(keys); k++) {
        unsigned accepted = 0, rejected = 0;

        for (unsigned masks = 0; masks < 4096; masks++) {
            AmeCapRequest request = {
                .key = keys[k], .kind = AME_TUPLE_SET_RULE,
            };
            AmeCapabilities result = { 0 };
            unsigned projected = 0, members = 0;

            for (unsigned pos = 0; pos < AME_TUPLE_SIZE; pos++) {
                for (unsigned i = 0; i < G_N_ELEMENTS(enc); i++) {
                    if (masks & BIT(pos * 4 + i)) {
                        riscv_ame_dtype_set_add(&request.operand_types[pos],
                                                id(enc[i]));
                    }
                }
            }
            /* Enumerate tuples without using the production projection. */
            for (unsigned d = 0; d < G_N_ELEMENTS(enc); d++) {
                for (unsigned a = 0; a < G_N_ELEMENTS(enc); a++) {
                    for (unsigned b = 0; b < G_N_ELEMENTS(enc); b++) {
                        if ((masks & BIT(d)) && (masks & BIT(4 + a)) &&
                            (masks & BIT(8 + b)) &&
                            base_oracle(keys[k].op, enc[d], enc[a], enc[b])) {
                            projected |= BIT(d) | BIT(4 + a) | BIT(8 + b);
                            members++;
                        }
                    }
                }
            }
            if (projected != masks) {
                expect_invalid(I32, &public, &request, 1);
                rejected++;
                continue;
            }
            g_assert_true(riscv_ame_capabilities_init(
                &result, I32, &public, &request, 1, &error_abort));
            g_assert_cmpuint(result.nr_entries, ==, members != 0);
            if (members) {
                g_assert_cmpuint(result.entries[0].tuples.count, ==, members);
            }
            riscv_ame_capabilities_clear(&result);
            accepted++;
        }
        g_assert_cmpuint(accepted, >, 1);
        g_assert_cmpuint(rejected, >, 0);
        g_assert_cmpuint(accepted + rejected, ==, 4096);
        g_test_message("explicit rule op=%u: %u accepted, %u rejected",
                       keys[k].op, accepted, rejected);
    }
}

static void test_scalar_matrix(void)
{
    static const uint32_t encodings[] = {
        0x40000004, I8, I16, I32, I64, 0x40000080, U8, 0x00000080,
        0x68000008, H, HZ, 0x20300110, F, 0x2c300140, E4, E5, 0x11500108,
    };
    static const AmeOperation data_ops[] = {
        AME_OP_ABSDIFF, AME_OP_ADD, AME_OP_HDIFF, AME_OP_LOG2SUB,
        AME_OP_AND, AME_OP_ANDNOT, AME_OP_CMPGE, AME_OP_CMPLT,
        AME_OP_MAX, AME_OP_MEAN, AME_OP_MIN, AME_OP_MUL,
        AME_OP_NMUL, AME_OP_OR, AME_OP_ORNOT, AME_OP_SUB,
        AME_OP_SUBLOG2, AME_OP_XOR, AME_OP_MULACC, AME_OP_MULNACC,
        AME_OP_MULADD, AME_OP_MULSUB,
    };
    static const AmeOperation control_ops[] = {
        AME_OP_LDEXP, AME_OP_LDEXPACC,
        AME_OP_SLL, AME_OP_SRL, AME_OP_SRA,
    };
    const AmeCapabilities *max = riscv_ame_maximum_capabilities();
    unsigned checks = 0;

    for (unsigned op = 0; op < AME_OP_COUNT; op++) {
        bool data = false, control = false;

        for (unsigned i = 0; i < G_N_ELEMENTS(data_ops); i++) {
            data |= op == data_ops[i];
        }
        for (unsigned i = 0; i < G_N_ELEMENTS(control_ops); i++) {
            control |= op == control_ops[i];
        }
        for (unsigned d = 0; d < G_N_ELEMENTS(encodings); d++) {
            for (unsigned t = 0; t < G_N_ELEMENTS(encodings); t++) {
                uint32_t de = encodings[d], te = encodings[t];
                bool expected = control &&
                    base_oracle(op, de, te, RV_AME_MAX_INT_DTYPE);

                g_assert_cmpint(supports(max, op, AME_FORM_CONTROL_SCALAR,
                                         de, te, 0), ==, expected);
                checks++;
                for (unsigned s = 0; s < G_N_ELEMENTS(encodings); s++) {
                    uint32_t se = encodings[s];

                    expected = data &&
                               base_oracle(AME_OP_CONV, te, se, 0) &&
                               base_oracle(op, de, te, te);
                    g_assert_cmpint(supports(max, op, AME_FORM_DATA_SCALAR,
                                             de, te, se), ==, expected);
                    checks++;
                }
            }
        }
    }
    g_test_message("independent scalar encoding checks: %u", checks);
}

static void test_cpu_policy(void)
{
    static const uint32_t enabled[] = { I8 };
    AmeDTypeSet public = types(enabled, 1);
    AmeCapabilities fixed = { 0 }, empty = { 0 };
    Error *err = NULL;

    g_assert_true(riscv_ame_capabilities_init(
        &fixed, I8, &public, NULL, 0, &error_abort));
    g_assert_true(riscv_ame_capabilities_for_cpu(
        AME_CAP_POLICY_FOLLOW_MAX, NULL, &error_abort) ==
        riscv_ame_maximum_capabilities());
    g_assert_true(riscv_ame_capabilities_for_cpu(
        AME_CAP_POLICY_EXPLICIT, &fixed, &error_abort) == &fixed);
    g_assert_null(riscv_ame_capabilities_for_cpu(
        AME_CAP_POLICY_UNSET, NULL, &err));
    g_assert_nonnull(err);
    error_free(err);
    err = NULL;
    g_assert_null(riscv_ame_capabilities_for_cpu(
        AME_CAP_POLICY_EXPLICIT, NULL, &err));
    g_assert_nonnull(err);
    error_free(err);
    err = NULL;
    g_assert_null(riscv_ame_capabilities_for_cpu(
        AME_CAP_POLICY_EXPLICIT, &empty, &err));
    g_assert_nonnull(err);
    error_free(err);
    err = NULL;
    g_assert_null(riscv_ame_capabilities_for_cpu(
        AME_CAP_POLICY_FOLLOW_MAX, &fixed, &err));
    g_assert_nonnull(err);
    error_free(err);
    riscv_ame_capabilities_clear(&fixed);
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);
    g_test_add_func("/riscv/ame/cap-set/integer-float-allowlist",
                    test_integer_float_allowlist);
    g_test_add_func("/riscv/ame/cap-set/maximum", test_maximum);
    g_test_add_func("/riscv/ame/cap-set/scalar", test_scalar);
    g_test_add_func("/riscv/ame/cap-set/integer-narrowing",
                    test_integer_narrowing);
    g_test_add_func("/riscv/ame/cap-set/fp-to-int-allowlist",
                    test_fp_to_int_allowlist);
    g_test_add_func("/riscv/ame/cap-set/exact-identity",
                    test_exact_and_identity);
    g_test_add_func("/riscv/ame/cap-set/scalar-only", test_scalar_only_backend);
    g_test_add_func("/riscv/ame/cap-set/invalid", test_invalid);
    g_test_add_func("/riscv/ame/cap-set/empty-public-identity",
                    test_empty_and_public_identity);
    g_test_add_func("/riscv/ame/cap-set/unreachable-types",
                    test_unreachable_types);
    g_test_add_func("/riscv/ame/cap-set/explicit-rule-reachability",
                    test_explicit_rule_reachability);
    g_test_add_func("/riscv/ame/cap-set/fp-only", test_fp_only);
    g_test_add_func("/riscv/ame/cap-set/max-int-domain", test_max_int_domain);
    g_test_add_func("/riscv/ame/cap-set/mixed-exact-allowlist",
                    test_mixed_exact_allowlist);
    g_test_add_func("/riscv/ame/cap-set/fp-unary-allowlist",
                    test_fp_unary_allowlist);
    g_test_add_func("/riscv/ame/cap-set/base-form", test_base_form);
    g_test_add_func("/riscv/ame/cap-set/scalar-matrix", test_scalar_matrix);
    g_test_add_func("/riscv/ame/cap-set/cpu-policy", test_cpu_policy);
    return g_test_run();
}
