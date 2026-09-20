/*
 * RISC-V AME tuple-set normalization tests
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/bitops.h"
#include "target/riscv/ame/tuple-set.h"

static uint16_t dtype(uint32_t encoding)
{
    const AmeDTypeDesc *desc = riscv_ame_dtype_by_encoding(encoding);

    g_assert_nonnull(desc);
    return desc->id;
}

static AmeDTypeTuple tuple(uint32_t d, uint32_t a, uint32_t b)
{
    return (AmeDTypeTuple) {
        .dtype_ids = {
            d ? dtype(d) : AME_DTYPE_ID_INVALID,
            a ? dtype(a) : AME_DTYPE_ID_INVALID,
            b ? dtype(b) : AME_DTYPE_ID_INVALID,
        },
    };
}

static AmeDTypeSet type_set(const uint32_t *encodings, size_t count,
                            unsigned mask)
{
    AmeDTypeSet set = { 0 };

    for (size_t i = 0; i < count; i++) {
        if (mask & BIT(i)) {
            g_assert_true(riscv_ame_dtype_set_add(&set, dtype(encodings[i])));
        }
    }
    return set;
}

static void test_bitmap(void)
{
    for (unsigned id = 0; id < AME_DTYPE_COUNT; id++) {
        AmeDTypeSet set = { 0 };

        g_assert_true(riscv_ame_dtype_set_empty(&set));
        g_assert_true(riscv_ame_dtype_set_add(&set, id));
        /* Success reports a valid ID, not whether it was newly inserted. */
        g_assert_true(riscv_ame_dtype_set_add(&set, id));
        g_assert_false(riscv_ame_dtype_set_add(&set, AME_DTYPE_COUNT));
        g_assert_false(riscv_ame_dtype_set_has(&set, AME_DTYPE_COUNT));
        g_assert_false(riscv_ame_dtype_set_empty(&set));
        for (unsigned w = 0; w < AME_DTYPE_WORD_COUNT; w++) {
            uint64_t want = w == id / 64 ? UINT64_C(1) << (id % 64) : 0;

            g_assert_cmphex(set.words[w], ==, want);
        }
        for (unsigned other = 0; other < AME_DTYPE_COUNT; other++) {
            g_assert_cmpint(riscv_ame_dtype_set_has(&set, other), ==,
                            id == other);
        }
    }
}

/* Independent relations over the encoding fields, not production helpers. */
static bool expected(AmeTupleRelation relation, const uint32_t value[3],
                      unsigned operands)
{
    uint32_t d = value[0], a = value[1], b = value[2];

    switch (relation) {
    case AME_TUPLE_ANY:
        return true;
    case AME_TUPLE_SAME:
        return (!(operands & BIT(AME_TUPLE_A)) || d == a) &&
               (!(operands & BIT(AME_TUPLE_B)) || d == b);
    case AME_TUPLE_SAME_DA:
        return d == a;
    case AME_TUPLE_SAME_DB:
        return d == b;
    default:
        g_assert_not_reached();
    }
}

static void test_rule_projection(void)
{
    static const uint32_t enc[] = {
        0x40000008, 0x40000010, 0x40000020, 0x00000010,
    };
    unsigned tested = 0;

    for (AmeTupleRelation relation = 0; relation < AME_TUPLE_RELATION_MAX;
         relation++) {
        AmeTupleRule rule = {
            .relation = relation,
            .operand_mask = AME_TUPLE_ALL_MASK,
        };
        AmeDTypeSet allowed[3];

        for (unsigned i = 0; i < 3; i++) {
            allowed[i] = type_set(enc, 4, 15);
        }
        for (unsigned masks = 0; masks < 4096; masks++) {
            AmeDTypeSet inputs[3], projection[3] = { 0 };
            AmeTupleSet set = { 0 }, repeated = { 0 }, exact = { 0 };
            AmeDTypeTuple members[64];
            size_t count = 0;

            for (unsigned i = 0; i < 3; i++) {
                inputs[i] = type_set(enc, 4, (masks >> (i * 4)) & 15);
            }
            g_assert_true(riscv_ame_tuple_set_rule(
                &set, rule, allowed, inputs, &error_abort));
            for (unsigned d = 0; d < 4; d++) {
                for (unsigned a = 0; a < 4; a++) {
                    for (unsigned b = 0; b < 4; b++) {
                        uint32_t values[] = { enc[d], enc[a], enc[b] };
                        AmeDTypeTuple member = tuple(enc[d], enc[a], enc[b]);
                        bool accept = (masks & BIT(d)) &&
                                      (masks & BIT(a + 4)) &&
                                      (masks & BIT(b + 8)) &&
                                      expected(relation, values,
                                               AME_TUPLE_ALL_MASK);

                        g_assert_cmpint(riscv_ame_tuple_set_has(&set, member),
                                        ==, accept);
                        if (accept) {
                            members[count++] = member;
                            for (unsigned i = 0; i < 3; i++) {
                                riscv_ame_dtype_set_add(&projection[i],
                                                        member.dtype_ids[i]);
                            }
                        }
                    }
                }
            }
            g_assert_cmpuint(set.count, ==, count);
            g_assert_cmpmem(set.operand_types, sizeof(set.operand_types),
                            projection, sizeof(projection));
            g_assert_true(riscv_ame_tuple_set_rule(
                &repeated, rule, allowed, set.operand_types, &error_abort));
            g_assert_true(riscv_ame_tuple_set_equal(&set, &repeated));
            g_assert_true(riscv_ame_tuple_set_exact(
                &exact, rule, allowed, members, count, &error_abort));
            g_assert_true(riscv_ame_tuple_set_equal(&set, &exact));
            g_assert_true(riscv_ame_tuple_set_subset(&set, &exact));
            g_assert_true(riscv_ame_tuple_set_subset(&exact, &set));
            riscv_ame_tuple_set_clear(&set);
            riscv_ame_tuple_set_clear(&repeated);
            riscv_ame_tuple_set_clear(&exact);
            tested++;
        }
    }
    g_assert_cmpuint(tested, ==, 4096 * AME_TUPLE_RELATION_MAX);
}

static void test_correlated_allowlist(void)
{
    static const uint32_t enc[] = {
        0x40000008, 0x40000010, 0x40000020, 0x40000040,
    };
    AmeTupleRule rule = {
        .relation = AME_TUPLE_ANY,
        .operand_mask = AME_TUPLE_ALL_MASK,
    };
    AmeDTypeTuple first = tuple(enc[2], enc[0], enc[0]);
    AmeDTypeTuple second = tuple(enc[3], enc[1], enc[1]);
    AmeDTypeTuple input[] = { second, first, second };
    AmeDTypeSet allowed[3];
    AmeTupleSet set = { 0 }, broad = { 0 }, single = { 0 };

    for (unsigned i = 0; i < 3; i++) {
        allowed[i] = type_set(enc, 4, 15);
    }
    g_assert_true(riscv_ame_tuple_set_exact(
        &set, rule, allowed, input, ARRAY_SIZE(input), &error_abort));
    g_assert_cmpuint(set.kind, ==, AME_TUPLE_SET_EXACT);
    g_assert_cmpuint(set.nr_tuples, ==, 2);
    g_assert_true(riscv_ame_tuple_set_has(&set, first));
    g_assert_true(riscv_ame_tuple_set_has(&set, second));
    g_assert_false(riscv_ame_tuple_set_has(
        &set, tuple(enc[2], enc[0], enc[1])));
    g_assert_true(riscv_ame_tuple_set_rule(
        &broad, rule, allowed, set.operand_types, &error_abort));
    g_assert_cmpuint(broad.count, ==, 8);
    g_assert_true(riscv_ame_tuple_set_subset(&set, &broad));
    g_assert_false(riscv_ame_tuple_set_subset(&broad, &set));
    g_assert_true(riscv_ame_tuple_set_exact(
        &single, rule, allowed, &first, 1, &error_abort));
    g_assert_cmpuint(single.kind, ==, AME_TUPLE_SET_RULE);
    g_assert_true(riscv_ame_tuple_set_subset(&single, &set));
    g_assert_false(riscv_ame_tuple_set_subset(&set, &single));
    riscv_ame_tuple_set_clear(&set);
    riscv_ame_tuple_set_clear(&broad);
    riscv_ame_tuple_set_clear(&single);
}

static void test_lower_arity(void)
{
    static const uint32_t enc[] = { 0x40000008, 0x40000010 };

    for (unsigned arity = 1; arity <= 2; arity++) {
        AmeTupleRule rule = {
            .relation = AME_TUPLE_ANY,
            .operand_mask = (1 << arity) - 1,
        };
        AmeDTypeSet allowed[3] = { 0 };
        AmeTupleSet set = { 0 };
        AmeDTypeTuple member = tuple(enc[0], arity == 2 ? enc[1] : 0, 0);

        for (unsigned i = 0; i < arity; i++) {
            allowed[i] = type_set(enc, 2, 3);
        }
        g_assert_true(riscv_ame_tuple_set_rule(
            &set, rule, allowed, allowed, &error_abort));
        g_assert_cmpuint(set.count, ==, 1 << arity);
        g_assert_true(riscv_ame_tuple_set_has(&set, member));
        member.dtype_ids[arity] = dtype(enc[0]);
        g_assert_false(riscv_ame_tuple_set_has(&set, member));
        riscv_ame_tuple_set_clear(&set);
    }
}

static void test_source_relations(void)
{
    static const uint32_t enc[] = { 0x40000008, 0x40000010, 0x00000080 };
    AmeTupleRule rule = {
        .relation = AME_TUPLE_SAME_DA,
        .operand_mask = AME_TUPLE_ALL_MASK,
    };
    AmeDTypeSet allowed[3] = {
        type_set(enc, 3, 2), type_set(enc, 3, 2), type_set(enc, 3, 4),
    };
    AmeTupleSet set = { 0 };

    /* D and A must match; B independently selects an allowed datatype. */
    g_assert_true(riscv_ame_tuple_set_rule(
        &set, rule, allowed, allowed, &error_abort));
    g_assert_cmpuint(set.count, ==, 1);
    g_assert_true(riscv_ame_tuple_set_has(
        &set, tuple(enc[1], enc[1], enc[2])));
    riscv_ame_tuple_set_clear(&set);
    allowed[AME_TUPLE_A] = type_set(enc, 3, 1);
    g_assert_true(riscv_ame_tuple_set_rule(
        &set, rule, allowed, allowed, &error_abort));
    g_assert_cmpuint(set.kind, ==, AME_TUPLE_SET_EMPTY);
    riscv_ame_tuple_set_clear(&set);

    /* D and B must match; A independently selects an allowed datatype. */
    allowed[AME_TUPLE_A] = type_set(enc, 3, 4);
    allowed[AME_TUPLE_B] = type_set(enc, 3, 2);
    rule.relation = AME_TUPLE_SAME_DB;
    g_assert_true(riscv_ame_tuple_set_rule(
        &set, rule, allowed, allowed, &error_abort));
    g_assert_cmpuint(set.count, ==, 1);
    g_assert_true(riscv_ame_tuple_set_has(
        &set, tuple(enc[1], enc[2], enc[1])));
    riscv_ame_tuple_set_clear(&set);
}

static void test_sparse_subset_matrix(void)
{
    static const uint32_t enc[] = { 0x40000008, 0x40000010 };
    AmeTupleRule rule = {
        .relation = AME_TUPLE_ANY,
        .operand_mask = AME_TUPLE_ALL_MASK,
    };
    AmeDTypeSet allowed[3];
    AmeTupleSet sets[256] = { 0 };

    for (unsigned pos = 0; pos < 3; pos++) {
        allowed[pos] = type_set(enc, 2, 3);
    }
    for (unsigned mask = 0; mask < 256; mask++) {
        AmeDTypeTuple members[8];
        size_t nr_members = 0;

        /* Reverse construction order must not affect identity. */
        for (int i = 7; i >= 0; i--) {
            if (mask & BIT(i)) {
                members[nr_members++] = tuple(enc[(i >> 2) & 1],
                                              enc[(i >> 1) & 1], enc[i & 1]);
            }
        }
        g_assert_true(riscv_ame_tuple_set_exact(
            &sets[mask], rule, allowed, members, nr_members, &error_abort));
        g_assert_cmpuint(sets[mask].count, ==, ctpop32(mask));
    }
    for (unsigned left = 0; left < 256; left++) {
        for (unsigned right = 0; right < 256; right++) {
            g_assert_cmpint(riscv_ame_tuple_set_subset(
                                &sets[left], &sets[right]),
                            ==, (left & ~right) == 0);
            g_assert_cmpint(riscv_ame_tuple_set_equal(
                                &sets[left], &sets[right]),
                            ==, left == right);
        }
    }
    for (unsigned i = 0; i < 256; i++) {
        riscv_ame_tuple_set_clear(&sets[i]);
    }
}

static void test_fp_identity(void)
{
    static const uint32_t enc[] = { 0x14300110, 0x14700110, 0x20300110 };
    AmeTupleRule rule = {
        .relation = AME_TUPLE_SAME,
        .operand_mask = BIT(AME_TUPLE_D) | BIT(AME_TUPLE_A),
    };
    AmeDTypeSet allowed[3] = {
        type_set(enc, 3, 7), type_set(enc, 3, 7), { .words = { 0 } },
    };
    AmeTupleSet set = { 0 };

    g_assert_true(riscv_ame_tuple_set_rule(
        &set, rule, allowed, allowed, &error_abort));
    g_assert_cmpuint(set.count, ==, 3);
    g_assert_true(riscv_ame_tuple_set_has(&set, tuple(enc[0], enc[0], 0)));
    g_assert_false(riscv_ame_tuple_set_has(&set, tuple(enc[0], enc[1], 0)));
    g_assert_false(riscv_ame_tuple_set_has(&set, tuple(enc[0], enc[2], 0)));
    riscv_ame_tuple_set_clear(&set);
}

static void test_invalid_inputs(void)
{
    static const uint32_t enc[] = { 0x40000008, 0x40000010 };
    AmeTupleRule rule = {
        .relation = AME_TUPLE_SAME,
        .operand_mask = BIT(AME_TUPLE_D) | BIT(AME_TUPLE_A),
    };
    AmeDTypeSet allowed[3] = { 0 }, inputs[3];
    AmeTupleSet set = { 0 };
    AmeDTypeTuple member = tuple(enc[0], enc[0], enc[0]);
    Error *err = NULL;

    allowed[0] = allowed[1] = type_set(enc, 2, 1);
    memcpy(inputs, allowed, sizeof(inputs));
    inputs[0] = type_set(enc, 2, 3);
    g_assert_false(riscv_ame_tuple_set_rule(
        &set, rule, allowed, inputs, &err));
    g_assert_nonnull(err);
    g_clear_pointer(&err, error_free);
    g_assert_false(riscv_ame_tuple_set_exact(
        &set, rule, allowed, &member, 1, &err));
    g_assert_nonnull(err);
    g_clear_pointer(&err, error_free);
    member.dtype_ids[2] = AME_DTYPE_ID_INVALID;
    member.dtype_ids[1] = dtype(enc[1]);
    allowed[1] = type_set(enc, 2, 3);
    g_assert_false(riscv_ame_tuple_set_exact(
        &set, rule, allowed, &member, 1, &err));
    g_assert_nonnull(err);
    g_clear_pointer(&err, error_free);
    g_assert_cmpuint(set.kind, ==, AME_TUPLE_SET_EMPTY);
    g_assert_null(set.exact_tuples);
    g_assert_false(riscv_ame_dtype_set_add(&inputs[0], AME_DTYPE_ID_INVALID));
    g_assert_false(riscv_ame_dtype_set_has(&inputs[0], AME_DTYPE_ID_INVALID));

    memcpy(inputs, allowed, sizeof(inputs));
    inputs[0].words[AME_DTYPE_WORD_COUNT - 1] |= BIT_ULL(63);
    g_assert_false(riscv_ame_tuple_set_rule(
        &set, rule, allowed, inputs, &err));
    g_assert_nonnull(err);
    g_clear_pointer(&err, error_free);
    memcpy(inputs, allowed, sizeof(inputs));
    inputs[2] = allowed[0];
    g_assert_false(riscv_ame_tuple_set_rule(
        &set, rule, allowed, inputs, &err));
    g_assert_nonnull(err);
    g_clear_pointer(&err, error_free);
    rule.operand_mask = BIT(AME_TUPLE_B); /* Missing D and A. */
    g_assert_false(riscv_ame_tuple_set_rule(
        &set, rule, allowed, inputs, &err));
    g_assert_nonnull(err);
    g_clear_pointer(&err, error_free);
    rule.operand_mask = BIT(AME_TUPLE_D) | BIT(AME_TUPLE_A);
    rule.relation = AME_TUPLE_RELATION_MAX;
    g_assert_false(riscv_ame_tuple_set_exact(
        &set, rule, allowed, NULL, 0, &err));
    g_assert_nonnull(err);
    g_clear_pointer(&err, error_free);
    rule.relation = AME_TUPLE_SAME;
    g_assert_false(riscv_ame_tuple_set_exact(
        &set, rule, allowed, NULL, 1, &err));
    g_assert_nonnull(err);
    g_clear_pointer(&err, error_free);
    g_assert_false(riscv_ame_tuple_set_exact(
        &set, rule, allowed, &member, SIZE_MAX, &err));
    g_assert_nonnull(err);
    g_clear_pointer(&err, error_free);
    g_assert_cmpuint(set.kind, ==, AME_TUPLE_SET_EMPTY);
    g_assert_cmpuint(set.count, ==, 0);
    g_assert_cmpuint(set.nr_tuples, ==, 0);
    g_assert_cmpuint(set.rule.operand_mask, ==, 0);
    g_assert_null(set.exact_tuples);
    for (unsigned i = 0; i < AME_TUPLE_SIZE; i++) {
        for (unsigned w = 0; w < AME_DTYPE_WORD_COUNT; w++) {
            g_assert_cmphex(set.operand_types[i].words[w], ==, 0);
        }
    }
}

static void test_full_domain_projection(void)
{
    AmeDTypeSet allowed[AME_TUPLE_SIZE] = { 0 };
    AmeDTypeSet types[AME_TUPLE_SIZE];
    AmeTupleSet set = { 0 };
    AmeTupleRule rule = {
        .relation = AME_TUPLE_ANY,
        .operand_mask = AME_TUPLE_ALL_MASK,
    };
    uint64_t full_count = AME_DTYPE_COUNT;
    unsigned common_count = 0;

    for (unsigned id = 0; id < AME_DTYPE_COUNT; id++) {
        for (unsigned i = 0; i < AME_TUPLE_SIZE; i++) {
            riscv_ame_dtype_set_add(&allowed[i], id);
        }
    }
    memcpy(types, allowed, sizeof(types));
    full_count = full_count * full_count * full_count;
    g_assert_true(riscv_ame_tuple_set_rule(
        &set, rule, allowed, types, &error_abort));
    g_assert_cmpuint(set.count, ==, full_count);
    g_assert_cmpmem(set.operand_types, sizeof(set.operand_types),
                    types, sizeof(types));
    riscv_ame_tuple_set_clear(&set);

    types[1] = (AmeDTypeSet) { 0 };
    g_assert_true(riscv_ame_tuple_set_rule(
        &set, rule, allowed, types, &error_abort));
    g_assert_cmpuint(set.kind, ==, AME_TUPLE_SET_EMPTY);
    g_assert_cmpuint(set.count, ==, 0);
    riscv_ame_tuple_set_clear(&set);

    rule.relation = AME_TUPLE_SAME;
    types[2] = (AmeDTypeSet) { 0 };
    for (unsigned id = 0; id < AME_DTYPE_COUNT; id++) {
        if (id & 1) {
            riscv_ame_dtype_set_add(&types[1], id);
        }
        if (id >= 64) {
            riscv_ame_dtype_set_add(&types[2], id);
            common_count += id & 1;
        }
    }
    g_assert_true(riscv_ame_tuple_set_rule(
        &set, rule, allowed, types, &error_abort));
    g_assert_cmpuint(set.count, ==, common_count);
    for (unsigned id = 0; id < AME_DTYPE_COUNT; id++) {
        for (unsigned i = 0; i < AME_TUPLE_SIZE; i++) {
            g_assert_cmpint(riscv_ame_dtype_set_has(&set.operand_types[i], id),
                            ==, (id & 1) && id >= 64);
        }
    }
    riscv_ame_tuple_set_clear(&set);
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);
    g_test_add_func("/riscv/ame/tuple-set/bitmap", test_bitmap);
    g_test_add_func("/riscv/ame/tuple-set/full-domain-projection",
                    test_full_domain_projection);
    g_test_add_func("/riscv/ame/tuple-set/rule-projection",
                    test_rule_projection);
    g_test_add_func("/riscv/ame/tuple-set/correlated-allowlist",
                    test_correlated_allowlist);
    g_test_add_func("/riscv/ame/tuple-set/lower-arity", test_lower_arity);
    g_test_add_func("/riscv/ame/tuple-set/source-relations",
                    test_source_relations);
    g_test_add_func("/riscv/ame/tuple-set/sparse-subset-matrix",
                    test_sparse_subset_matrix);
    g_test_add_func("/riscv/ame/tuple-set/fp-identity", test_fp_identity);
    g_test_add_func("/riscv/ame/tuple-set/invalid-inputs", test_invalid_inputs);
    return g_test_run();
}
