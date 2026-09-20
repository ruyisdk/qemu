/*
 * RISC-V Attached Matrix Extension capability wire representation
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/bswap.h"
#include "target/riscv/ame/cap-wire.h"

/* Wire v1 numbers do not depend on enum declaration order. */
static const uint16_t operation_ids[] = {
    [AME_OP_ABS] = 0,
    [AME_OP_ABSDIFF] = 1,
    [AME_OP_ADD] = 2,
    [AME_OP_AND] = 3,
    [AME_OP_ANDNOT] = 4,
    [AME_OP_CMPGE] = 5,
    [AME_OP_CMPLT] = 6,
    [AME_OP_COLBCAST] = 7,
    [AME_OP_COLID] = 8,
    [AME_OP_COLUNZIP] = 9,
    [AME_OP_COLZIP] = 10,
    [AME_OP_CONV] = 11,
    [AME_OP_COS] = 12,
    [AME_OP_EXP2] = 13,
    [AME_OP_FRINTM] = 14,
    [AME_OP_FRINTN] = 15,
    [AME_OP_FRINTP] = 16,
    [AME_OP_FRINTZ] = 17,
    [AME_OP_GATHER] = 18,
    [AME_OP_GATHERROW] = 19,
    [AME_OP_HDIFF] = 20,
    [AME_OP_LDEXP] = 21,
    [AME_OP_LDEXPACC] = 22,
    [AME_OP_LOG2] = 23,
    [AME_OP_LOG2SUB] = 24,
    [AME_OP_MAX] = 25,
    [AME_OP_MEAN] = 26,
    [AME_OP_MIN] = 27,
    [AME_OP_MMACC] = 28,
    [AME_OP_MMACCNEG] = 29,
    [AME_OP_MMATACC] = 30,
    [AME_OP_MMATACCNEG] = 31,
    [AME_OP_MMBTACC] = 32,
    [AME_OP_MMBTACCNEG] = 33,
    [AME_OP_MUL] = 34,
    [AME_OP_MULACC] = 35,
    [AME_OP_MULADD] = 36,
    [AME_OP_MULNACC] = 37,
    [AME_OP_MULSUB] = 38,
    [AME_OP_MUXGE] = 39,
    [AME_OP_MUXLT] = 40,
    [AME_OP_NMUL] = 41,
    [AME_OP_OR] = 42,
    [AME_OP_ORNOT] = 43,
    [AME_OP_PREFIXADD] = 44,
    [AME_OP_PREFIXADDCOL] = 45,
    [AME_OP_PREFIXMAX] = 46,
    [AME_OP_PREFIXMAXCOL] = 47,
    [AME_OP_RDEXP] = 48,
    [AME_OP_RDEXPACC] = 49,
    [AME_OP_REC] = 50,
    [AME_OP_REDUCEADD] = 51,
    [AME_OP_REDUCEADDCOL] = 52,
    [AME_OP_REDUCEMAX] = 53,
    [AME_OP_REDUCEMAXCOL] = 54,
    [AME_OP_REDUCEMIN] = 55,
    [AME_OP_REDUCEMINCOL] = 56,
    [AME_OP_ROWBCAST] = 57,
    [AME_OP_ROWID] = 58,
    [AME_OP_ROWSHIFT] = 59,
    [AME_OP_ROWUNZIP] = 60,
    [AME_OP_ROWZIP] = 61,
    [AME_OP_RSQRT] = 62,
    [AME_OP_COLSCATADD] = 63,
    [AME_OP_COLSCATMAX] = 64,
    [AME_OP_ROWSCATADD] = 65,
    [AME_OP_ROWSCATMAX] = 66,
    [AME_OP_SEL] = 67,
    [AME_OP_SELGE] = 68,
    [AME_OP_SHIFT] = 69,
    [AME_OP_SIN] = 70,
    [AME_OP_SLL] = 71,
    [AME_OP_SQRT] = 72,
    [AME_OP_SRA] = 73,
    [AME_OP_SRL] = 74,
    [AME_OP_SUB] = 75,
    [AME_OP_SUBLOG2] = 76,
    [AME_OP_TANH] = 77,
    [AME_OP_XOR] = 78,
    [AME_OP_ZERO] = 79,
};
static const uint8_t form_ids[] = {
    [AME_FORM_BASE] = 0,
    [AME_FORM_DATA_SCALAR] = 1,
    [AME_FORM_CONTROL_SCALAR] = 2,
};
static const uint8_t category_ids[] = {
    [AME_CAP_CATEGORY_GENERIC] = 0,
    [AME_CAP_CATEGORY_INTEGER] = 1,
    [AME_CAP_CATEGORY_FP] = 2,
    [AME_CAP_CATEGORY_INT_TO_FP] = 3,
    [AME_CAP_CATEGORY_FP_TO_INT] = 4,
    [AME_CAP_CATEGORY_INT_FP_SOURCES] = 5,
    [AME_CAP_CATEGORY_FP_INT_SOURCES] = 6,
};
G_STATIC_ASSERT(G_N_ELEMENTS(operation_ids) == AME_OP_COUNT);
G_STATIC_ASSERT(G_N_ELEMENTS(form_ids) == AME_FORM_MAX);
G_STATIC_ASSERT(G_N_ELEMENTS(category_ids) == AME_CAP_CATEGORY_MAX);

typedef struct AmeWireEntry {
    const AmeCapEntry *entry;
    uint16_t op;
    uint8_t form;
    uint8_t category;
} AmeWireEntry;

typedef struct AmeWireTuple {
    uint32_t encoding[AME_TUPLE_SIZE];
} AmeWireTuple;

static int compare_u32(const void *va, const void *vb)
{
    const uint32_t *a = va, *b = vb;

    return *a < *b ? -1 : *a != *b;
}

static int compare_entry(const void *va, const void *vb)
{
    const AmeWireEntry *a = va, *b = vb;

    if (a->op != b->op) {
        return a->op < b->op ? -1 : 1;
    }
    if (a->form != b->form) {
        return a->form < b->form ? -1 : 1;
    }
    return a->category < b->category ? -1 : a->category != b->category;
}

static int compare_tuple(const void *va, const void *vb)
{
    const AmeWireTuple *a = va, *b = vb;

    for (unsigned i = 0; i < AME_TUPLE_SIZE; i++) {
        if (a->encoding[i] != b->encoding[i]) {
            return a->encoding[i] < b->encoding[i] ? -1 : 1;
        }
    }
    return 0;
}

static unsigned dtype_encodings(const AmeDTypeSet *types,
                                  uint32_t encodings[AME_DTYPE_COUNT])
{
    unsigned count = 0;

    for (unsigned id = 0; id < AME_DTYPE_COUNT; id++) {
        if (riscv_ame_dtype_set_has(types, id)) {
            encodings[count++] = riscv_ame_dtypes[id].encoding;
        }
    }
    qsort(encodings, count, sizeof(*encodings), compare_u32);
    return count;
}

static void put_u16(GByteArray *output, uint16_t value)
{
    uint8_t bytes[sizeof(value)];

    stw_be_p(bytes, value);
    g_byte_array_append(output, bytes, sizeof(bytes));
}

static void put_u32(GByteArray *output, uint32_t value)
{
    uint8_t bytes[sizeof(value)];

    stl_be_p(bytes, value);
    g_byte_array_append(output, bytes, sizeof(bytes));
}

static void put_dtypes(GByteArray *output, const AmeDTypeSet *types)
{
    uint32_t encodings[AME_DTYPE_COUNT];
    unsigned count = dtype_encodings(types, encodings);

    put_u16(output, count);
    for (unsigned i = 0; i < count; i++) {
        put_u32(output, encodings[i]);
    }
}

static void put_exact(GByteArray *output, const AmeTupleSet *set)
{
    g_autofree AmeWireTuple *tuples = g_new(AmeWireTuple, set->nr_tuples);

    for (size_t i = 0; i < set->nr_tuples; i++) {
        for (unsigned pos = 0; pos < AME_TUPLE_SIZE; pos++) {
            uint16_t id = set->exact_tuples[i].dtype_ids[pos];

            tuples[i].encoding[pos] = id == AME_DTYPE_ID_INVALID ? 0 :
                                     riscv_ame_dtypes[id].encoding;
        }
    }
    if (set->nr_tuples > 1) {
        qsort(tuples, set->nr_tuples, sizeof(*tuples), compare_tuple);
    }
    put_u32(output, set->nr_tuples);
    for (size_t i = 0; i < set->nr_tuples; i++) {
        for (unsigned pos = 0; pos < AME_TUPLE_SIZE; pos++) {
            put_u32(output, tuples[i].encoding[pos]);
        }
    }
}

/* Bound allocation using the local, already constructed capability set. */
static bool manifest_size(const AmeCapabilities *set, unsigned *size,
                            Error **errp)
{
    uint32_t encodings[AME_DTYPE_COUNT];
    uint64_t total = 16 + 4 * dtype_encodings(&set->dtypes, encodings);

    if (set->nr_entries > UINT16_MAX) {
        goto too_large;
    }
    for (size_t i = 0; i < set->nr_entries; i++) {
        const AmeTupleSet *tuples = &set->entries[i].tuples;

        total += 6;
        switch (tuples->kind) {
        case AME_TUPLE_SET_RULE:
            for (unsigned pos = 0; pos < AME_TUPLE_SIZE; pos++) {
                unsigned count = dtype_encodings(&tuples->operand_types[pos],
                                                   encodings);

                total += 2 + 4 * count;
            }
            break;
        case AME_TUPLE_SET_EXACT:
            if (tuples->nr_tuples > (UINT_MAX - 4) / 12) {
                goto too_large;
            }
            total += 4 + 12 * tuples->nr_tuples;
            break;
        default:
            error_setg(errp,
                       "Ztt manifest requires normalized capability entries");
            return false;
        }
        if (total > UINT_MAX) {
            goto too_large;
        }
    }
    *size = total;
    return true;

too_large:
    error_setg(errp, "Ztt capability manifest exceeds its wire size limits");
    return false;
}

GByteArray *riscv_ame_capabilities_manifest(const AmeCapabilities *set,
                                          Error **errp)
{
    g_autofree AmeWireEntry *entries = NULL;
    g_autoptr(GByteArray) output = NULL;
    unsigned size;

    if (!manifest_size(set, &size, errp)) {
        return NULL;
    }
    entries = g_new(AmeWireEntry, set->nr_entries);
    for (size_t i = 0; i < set->nr_entries; i++) {
        AmeCapKey key = set->entries[i].key;

        if ((unsigned)key.op >= AME_OP_COUNT ||
            (unsigned)key.form >= AME_FORM_MAX ||
            (unsigned)key.category >= AME_CAP_CATEGORY_MAX) {
            error_setg(errp, "Ztt capability manifest has an unknown key");
            return NULL;
        }
        entries[i] = (AmeWireEntry) {
            .entry = &set->entries[i],
            .op = operation_ids[key.op],
            .form = form_ids[key.form],
            .category = category_ids[key.category],
        };
    }
    if (set->nr_entries > 1) {
        qsort(entries, set->nr_entries, sizeof(*entries), compare_entry);
    }
    output = g_byte_array_sized_new(size);
    put_u32(output, 0x414d4543); /* AMEC */
    put_u32(output, 1);
    put_u32(output, set->max_int_dtype);
    put_dtypes(output, &set->dtypes);
    put_u16(output, set->nr_entries);
    for (size_t i = 0; i < set->nr_entries; i++) {
        const AmeWireEntry *entry = &entries[i];
        const AmeTupleSet *tuples = &entry->entry->tuples;
        uint8_t fields[] = { entry->form, entry->category,
                            tuples->kind == AME_TUPLE_SET_RULE ? 1 : 2, 0 };

        put_u16(output, entry->op);
        g_byte_array_append(output, fields, sizeof(fields));
        if (tuples->kind == AME_TUPLE_SET_RULE) {
            for (unsigned pos = 0; pos < AME_TUPLE_SIZE; pos++) {
                put_dtypes(output, &tuples->operand_types[pos]);
            }
        } else {
            put_exact(output, tuples);
        }
    }
    g_assert(output->len == size);
    return g_steal_pointer(&output);
}
