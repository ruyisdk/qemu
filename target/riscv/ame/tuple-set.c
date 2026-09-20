/*
 * RISC-V Attached Matrix Extension datatype tuple sets
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/bitops.h"
#include "target/riscv/ame/tuple-set.h"

bool riscv_ame_dtype_set_add(AmeDTypeSet *set, uint16_t id)
{
    if (id >= AME_DTYPE_COUNT) {
        return false;
    }
    set->words[id / 64] |= BIT_ULL(id % 64);
    return true;
}

bool riscv_ame_dtype_set_has(const AmeDTypeSet *set, uint16_t id)
{
    return id < AME_DTYPE_COUNT &&
           (set->words[id / 64] & BIT_ULL(id % 64));
}

bool riscv_ame_dtype_set_empty(const AmeDTypeSet *set)
{
    for (unsigned i = 0; i < AME_DTYPE_WORD_COUNT; i++) {
        if (set->words[i]) {
            return false;
        }
    }
    return true;
}

static bool dtype_set_subset(const AmeDTypeSet *left,
                              const AmeDTypeSet *right)
{
    for (unsigned i = 0; i < AME_DTYPE_WORD_COUNT; i++) {
        if (left->words[i] & ~right->words[i]) {
            return false;
        }
    }
    return true;
}

static unsigned dtype_set_count(const AmeDTypeSet *set)
{
    unsigned count = 0;

    for (unsigned i = 0; i < AME_DTYPE_WORD_COUNT; i++) {
        count += ctpop64(set->words[i]);
    }
    return count;
}

static bool tuple_rule_valid(AmeTupleRule rule, Error **errp)
{
    if ((unsigned)rule.relation >= AME_TUPLE_RELATION_MAX ||
        (rule.operand_mask != BIT(AME_TUPLE_D) &&
         rule.operand_mask != (BIT(AME_TUPLE_D) | BIT(AME_TUPLE_A)) &&
         rule.operand_mask != AME_TUPLE_ALL_MASK) ||
        (rule.relation == AME_TUPLE_SAME_DA &&
         !(rule.operand_mask & BIT(AME_TUPLE_A))) ||
        (rule.relation == AME_TUPLE_SAME_DB &&
         rule.operand_mask != AME_TUPLE_ALL_MASK)) {
        error_setg(errp, "Ztt tuple set has an invalid operation signature");
        return false;
    }
    return true;
}

static bool tuple_rule_equal(AmeTupleRule left, AmeTupleRule right)
{
    return left.relation == right.relation &&
           left.operand_mask == right.operand_mask;
}

/* All admitted relations are two independent edges from D. */
static bool tuple_pair_matches(AmeTupleRule rule, uint16_t pivot,
                                unsigned position, uint16_t id)
{
    g_assert(pivot < AME_DTYPE_COUNT && id < AME_DTYPE_COUNT);
    switch (rule.relation) {
    case AME_TUPLE_ANY:
        return true;
    case AME_TUPLE_SAME:
        return pivot == id;
    case AME_TUPLE_SAME_DA:
        return position != AME_TUPLE_A || pivot == id;
    case AME_TUPLE_SAME_DB:
        return position != AME_TUPLE_B || pivot == id;
    default:
        g_assert_not_reached();
    }
}

static bool tuple_rule_matches(AmeTupleRule rule, AmeDTypeTuple tuple)
{
    unsigned pivot = AME_TUPLE_D;

    for (unsigned i = 0; i < AME_TUPLE_SIZE; i++) {
        if (i != pivot && (rule.operand_mask & BIT(i)) &&
            !tuple_pair_matches(rule, tuple.dtype_ids[pivot], i,
                                tuple.dtype_ids[i])) {
            return false;
        }
    }
    return true;
}

static bool tuple_in_types(const AmeDTypeSet types[AME_TUPLE_SIZE],
                            AmeTupleRule rule, AmeDTypeTuple tuple)
{
    for (unsigned i = 0; i < AME_TUPLE_SIZE; i++) {
        if (rule.operand_mask & BIT(i)) {
            if (!riscv_ame_dtype_set_has(&types[i], tuple.dtype_ids[i])) {
                return false;
            }
        } else if (tuple.dtype_ids[i] != AME_DTYPE_ID_INVALID) {
            return false;
        }
    }
    return true;
}

/* Count and project a rule in O(K^2), without building its tuple product. */
static uint64_t tuple_rule_project(AmeTupleRule rule,
                                    const AmeDTypeSet types[AME_TUPLE_SIZE],
                                    AmeDTypeSet result[AME_TUPLE_SIZE])
{
    unsigned pivot = AME_TUPLE_D;
    uint64_t count = 0;

    memset(result, 0, sizeof(*result) * AME_TUPLE_SIZE);
    if (rule.relation == AME_TUPLE_ANY) {
        count = 1;
        for (unsigned i = 0; i < AME_TUPLE_SIZE; i++) {
            if (rule.operand_mask & BIT(i)) {
                count *= dtype_set_count(&types[i]);
            }
        }
        if (count) {
            memcpy(result, types, sizeof(*result) * AME_TUPLE_SIZE);
        }
        return count;
    }
    if (rule.relation == AME_TUPLE_SAME) {
        AmeDTypeSet common = types[AME_TUPLE_D];

        for (unsigned i = AME_TUPLE_A; i < AME_TUPLE_SIZE; i++) {
            if (rule.operand_mask & BIT(i)) {
                for (unsigned w = 0; w < AME_DTYPE_WORD_COUNT; w++) {
                    common.words[w] &= types[i].words[w];
                }
            }
        }
        for (unsigned i = 0; i < AME_TUPLE_SIZE; i++) {
            if (rule.operand_mask & BIT(i)) {
                result[i] = common;
            }
        }
        return dtype_set_count(&common);
    }
    for (unsigned id = 0; id < AME_DTYPE_COUNT; id++) {
        AmeDTypeSet matches[AME_TUPLE_SIZE] = { 0 };
        uint64_t combinations = 1;

        if (!riscv_ame_dtype_set_has(&types[pivot], id)) {
            continue;
        }
        for (unsigned i = AME_TUPLE_A; i < AME_TUPLE_SIZE; i++) {
            unsigned compatible = 0;

            if (!(rule.operand_mask & BIT(i))) {
                continue;
            }
            for (unsigned other = 0; other < AME_DTYPE_COUNT; other++) {
                if (riscv_ame_dtype_set_has(&types[i], other) &&
                    tuple_pair_matches(rule, id, i, other)) {
                    riscv_ame_dtype_set_add(&matches[i], other);
                    compatible++;
                }
            }
            combinations *= compatible;
        }
        if (!combinations) {
            continue;
        }
        count += combinations;
        riscv_ame_dtype_set_add(&matches[pivot], id);
        for (unsigned i = 0; i < AME_TUPLE_SIZE; i++) {
            for (unsigned w = 0; w < AME_DTYPE_WORD_COUNT; w++) {
                result[i].words[w] |= matches[i].words[w];
            }
        }
    }
    return count;
}

static bool tuple_types_valid(AmeTupleRule rule,
                               const AmeDTypeSet allowed[AME_TUPLE_SIZE],
                               const AmeDTypeSet types[AME_TUPLE_SIZE],
                               Error **errp)
{
    for (unsigned i = 0; i < AME_TUPLE_SIZE; i++) {
        for (unsigned id = 0; id < AME_DTYPE_WORD_COUNT * 64; id++) {
            if ((types[i].words[id / 64] & BIT_ULL(id % 64)) &&
                (!(rule.operand_mask & BIT(i)) ||
                 !riscv_ame_dtype_set_has(&allowed[i], id))) {
                error_setg(errp,
                           "Ztt tuple position %u has an invalid datatype %u",
                           i, id);
                return false;
            }
        }
    }
    return true;
}

bool riscv_ame_tuple_set_rule(AmeTupleSet *out, AmeTupleRule rule,
                              const AmeDTypeSet allowed[AME_TUPLE_SIZE],
                              const AmeDTypeSet types[AME_TUPLE_SIZE],
                              Error **errp)
{
    AmeTupleSet result = { .rule = rule };

    g_assert(out->kind == AME_TUPLE_SET_EMPTY && !out->exact_tuples);
    if (!tuple_rule_valid(rule, errp) ||
        !tuple_types_valid(rule, allowed, types, errp)) {
        return false;
    }
    result.count = tuple_rule_project(rule, types, result.operand_types);
    if (result.count) {
        result.kind = AME_TUPLE_SET_RULE;
        *out = result;
    }
    return true;
}

static int tuple_compare(const void *va, const void *vb)
{
    const AmeDTypeTuple *a = va;
    const AmeDTypeTuple *b = vb;

    for (unsigned i = 0; i < AME_TUPLE_SIZE; i++) {
        if (a->dtype_ids[i] != b->dtype_ids[i]) {
            return a->dtype_ids[i] < b->dtype_ids[i] ? -1 : 1;
        }
    }
    return 0;
}

bool riscv_ame_tuple_set_exact(AmeTupleSet *out, AmeTupleRule rule,
                               const AmeDTypeSet allowed[AME_TUPLE_SIZE],
                               const AmeDTypeTuple *tuples, size_t nr_tuples,
                               Error **errp)
{
    AmeTupleSet result = { .rule = rule };
    AmeDTypeSet projection[AME_TUPLE_SIZE];
    uint64_t rule_tuple_count;

    g_assert(out->kind == AME_TUPLE_SET_EMPTY && !out->exact_tuples);
    if (!tuple_rule_valid(rule, errp)) {
        return false;
    }
    if (nr_tuples > SIZE_MAX / sizeof(*tuples) || (nr_tuples && !tuples)) {
        error_setg(errp, "Ztt exact tuple list has an invalid size");
        return false;
    }
    for (size_t i = 0; i < nr_tuples; i++) {
        if (!tuple_in_types(allowed, rule, tuples[i]) ||
            !tuple_rule_matches(rule, tuples[i])) {
            error_setg(errp, "Ztt exact tuple %zu is not supported", i);
            return false;
        }
    }
    if (!nr_tuples) {
        return true;
    }
    result.exact_tuples = g_memdup2(tuples, nr_tuples * sizeof(*tuples));
    qsort(result.exact_tuples, nr_tuples, sizeof(*tuples), tuple_compare);
    for (size_t i = 0; i < nr_tuples; i++) {
        AmeDTypeTuple current = result.exact_tuples[i];

        if (result.nr_tuples &&
            !tuple_compare(&result.exact_tuples[result.nr_tuples - 1],
                           &current)) {
            continue;
        }
        result.exact_tuples[result.nr_tuples++] = current;
        for (unsigned pos = 0; pos < AME_TUPLE_SIZE; pos++) {
            if (rule.operand_mask & BIT(pos)) {
                riscv_ame_dtype_set_add(&result.operand_types[pos],
                                        current.dtype_ids[pos]);
            }
        }
    }
    rule_tuple_count = tuple_rule_project(rule, result.operand_types,
                                          projection);
    result.count = result.nr_tuples;
    if (rule_tuple_count == result.count) {
        result.kind = AME_TUPLE_SET_RULE;
        g_clear_pointer(&result.exact_tuples, g_free);
        result.nr_tuples = 0;
    } else {
        result.kind = AME_TUPLE_SET_EXACT;
    }
    *out = result;
    return true;
}

void riscv_ame_tuple_set_clear(AmeTupleSet *set)
{
    g_free(set->exact_tuples);
    memset(set, 0, sizeof(*set));
}

bool riscv_ame_tuple_set_has(const AmeTupleSet *set, AmeDTypeTuple tuple)
{
    if (set->kind == AME_TUPLE_SET_EMPTY ||
        !tuple_in_types(set->operand_types, set->rule, tuple)) {
        return false;
    }
    if (set->kind == AME_TUPLE_SET_EXACT) {
        return bsearch(&tuple, set->exact_tuples, set->nr_tuples,
                       sizeof(tuple), tuple_compare) != NULL;
    }
    return tuple_rule_matches(set->rule, tuple);
}

bool riscv_ame_tuple_set_equal(const AmeTupleSet *left,
                               const AmeTupleSet *right)
{
    if (left->kind != right->kind || left->count != right->count) {
        return false;
    }
    if (left->kind == AME_TUPLE_SET_EMPTY) {
        return true;
    }
    if (!tuple_rule_equal(left->rule, right->rule)) {
        return false;
    }
    for (unsigned i = 0; i < AME_TUPLE_SIZE; i++) {
        if (!dtype_set_subset(&left->operand_types[i],
                              &right->operand_types[i]) ||
            !dtype_set_subset(&right->operand_types[i],
                              &left->operand_types[i])) {
            return false;
        }
    }
    for (size_t i = 0; i < left->nr_tuples; i++) {
        if (tuple_compare(&left->exact_tuples[i], &right->exact_tuples[i])) {
            return false;
        }
    }
    return true;
}

bool riscv_ame_tuple_set_subset(const AmeTupleSet *left,
                                const AmeTupleSet *right)
{
    if (left->kind == AME_TUPLE_SET_EMPTY) {
        return true;
    }
    if (left->count > right->count ||
        !tuple_rule_equal(left->rule, right->rule)) {
        return false;
    }
    if (left->kind == AME_TUPLE_SET_EXACT) {
        for (size_t i = 0; i < left->nr_tuples; i++) {
            if (!riscv_ame_tuple_set_has(right, left->exact_tuples[i])) {
                return false;
            }
        }
        return true;
    }
    if (right->kind == AME_TUPLE_SET_RULE) {
        for (unsigned i = 0; i < AME_TUPLE_SIZE; i++) {
            if (!dtype_set_subset(&left->operand_types[i],
                                  &right->operand_types[i])) {
                return false;
            }
        }
        return true;
    } else {
        uint64_t included = 0;

        for (size_t i = 0; i < right->nr_tuples; i++) {
            included += riscv_ame_tuple_set_has(left, right->exact_tuples[i]);
        }
        return included == left->count;
    }
}
