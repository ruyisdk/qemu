/*
 * RISC-V Attached Matrix Extension positive capability sets
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/bitops.h"
#include "target/riscv/ame/cap-set.h"

/* Datatype groups used to construct the maximum capability set. */
typedef enum AmeTypeDomain {
    TYPE_NONE,
    TYPE_INT,
    TYPE_SIGNED,
    TYPE_FP,
    TYPE_STANDARD,
} AmeTypeDomain;

typedef struct AmeCapSchema {
    AmeCapCategory category;
    AmeTupleRelation relation;
    /* D/A/B domains, expanded into descriptor-ID sets during construction. */
    AmeTypeDomain operand_domains[AME_TUPLE_SIZE];
} AmeCapSchema;

#define NO_SCALAR_FORMS     0
#define DATA_SCALAR_MASK    BIT(AME_FORM_DATA_SCALAR)
#define CONTROL_SCALAR_MASK BIT(AME_FORM_CONTROL_SCALAR)

static const AmeCapSchema zero[] = {
    { AME_CAP_CATEGORY_GENERIC, AME_TUPLE_ANY,
      { TYPE_STANDARD, TYPE_NONE, TYPE_NONE } },
};
static const AmeCapSchema index_dest[] = {
    { AME_CAP_CATEGORY_INTEGER, AME_TUPLE_ANY,
      { TYPE_INT, TYPE_NONE, TYPE_NONE } },
};
static const AmeCapSchema copy[] = {
    { AME_CAP_CATEGORY_GENERIC, AME_TUPLE_SAME,
      { TYPE_STANDARD, TYPE_STANDARD, TYPE_NONE } },
};
static const AmeCapSchema gather[] = {
    { AME_CAP_CATEGORY_GENERIC, AME_TUPLE_SAME_DA,
      { TYPE_STANDARD, TYPE_STANDARD, TYPE_INT } },
};
static const AmeCapSchema select_value[] = {
    { AME_CAP_CATEGORY_GENERIC, AME_TUPLE_SAME_DB,
      { TYPE_STANDARD, TYPE_STANDARD, TYPE_STANDARD } },
};
static const AmeCapSchema convert[] = {
    { AME_CAP_CATEGORY_INTEGER, AME_TUPLE_ANY,
      { TYPE_INT, TYPE_INT, TYPE_NONE } },
    { AME_CAP_CATEGORY_FP, AME_TUPLE_ANY,
      { TYPE_FP, TYPE_FP, TYPE_NONE } },
    { AME_CAP_CATEGORY_INT_TO_FP, AME_TUPLE_ANY,
      { TYPE_FP, TYPE_INT, TYPE_NONE } },
    { AME_CAP_CATEGORY_FP_TO_INT, AME_TUPLE_ANY,
      { TYPE_INT, TYPE_FP, TYPE_NONE } },
};
static const AmeCapSchema bitwise[] = {
    { AME_CAP_CATEGORY_INTEGER, AME_TUPLE_SAME,
      { TYPE_INT, TYPE_INT, TYPE_INT } },
};
static const AmeCapSchema numeric_same[] = {
    { AME_CAP_CATEGORY_INTEGER, AME_TUPLE_SAME,
      { TYPE_INT, TYPE_INT, TYPE_INT } },
    { AME_CAP_CATEGORY_FP, AME_TUPLE_SAME,
      { TYPE_FP, TYPE_FP, TYPE_FP } },
};
static const AmeCapSchema compare[] = {
    { AME_CAP_CATEGORY_INTEGER, AME_TUPLE_ANY,
      { TYPE_INT, TYPE_INT, TYPE_INT } },
    { AME_CAP_CATEGORY_FP, AME_TUPLE_ANY,
      { TYPE_INT, TYPE_FP, TYPE_FP } },
    { AME_CAP_CATEGORY_INT_FP_SOURCES, AME_TUPLE_ANY,
      { TYPE_INT, TYPE_INT, TYPE_FP } },
    { AME_CAP_CATEGORY_FP_INT_SOURCES, AME_TUPLE_ANY,
      { TYPE_INT, TYPE_FP, TYPE_INT } },
};
static const AmeCapSchema numeric_unary[] = {
    { AME_CAP_CATEGORY_INTEGER, AME_TUPLE_ANY,
      { TYPE_STANDARD, TYPE_INT, TYPE_NONE } },
    { AME_CAP_CATEGORY_FP, AME_TUPLE_ANY,
      { TYPE_STANDARD, TYPE_FP, TYPE_NONE } },
};
static const AmeCapSchema numeric_binary[] = {
    { AME_CAP_CATEGORY_INTEGER, AME_TUPLE_ANY,
      { TYPE_STANDARD, TYPE_INT, TYPE_INT } },
    { AME_CAP_CATEGORY_FP, AME_TUPLE_ANY,
      { TYPE_STANDARD, TYPE_FP, TYPE_FP } },
    { AME_CAP_CATEGORY_INT_FP_SOURCES, AME_TUPLE_ANY,
      { TYPE_STANDARD, TYPE_INT, TYPE_FP } },
    { AME_CAP_CATEGORY_FP_INT_SOURCES, AME_TUPLE_ANY,
      { TYPE_STANDARD, TYPE_FP, TYPE_INT } },
};
static const AmeCapSchema fp_binary[] = {
    { AME_CAP_CATEGORY_FP, AME_TUPLE_ANY,
      { TYPE_FP, TYPE_FP, TYPE_FP } },
    { AME_CAP_CATEGORY_FP_INT_SOURCES, AME_TUPLE_ANY,
      { TYPE_FP, TYPE_FP, TYPE_INT } },
};
static const AmeCapSchema scale_index[] = {
    { AME_CAP_CATEGORY_INTEGER, AME_TUPLE_ANY,
      { TYPE_STANDARD, TYPE_INT, TYPE_INT } },
    { AME_CAP_CATEGORY_FP, AME_TUPLE_ANY,
      { TYPE_STANDARD, TYPE_FP, TYPE_INT } },
};
static const AmeCapSchema shift[] = {
    { AME_CAP_CATEGORY_INTEGER, AME_TUPLE_ANY,
      { TYPE_INT, TYPE_INT, TYPE_INT } },
};
static const AmeCapSchema signed_shift[] = {
    { AME_CAP_CATEGORY_INTEGER, AME_TUPLE_ANY,
      { TYPE_INT, TYPE_SIGNED, TYPE_INT } },
};
static const AmeCapSchema fp_unary[] = {
    { AME_CAP_CATEGORY_FP, AME_TUPLE_ANY,
      { TYPE_FP, TYPE_FP, TYPE_NONE } },
};

/*
 * Every operation has a base form; scalar forms are listed explicitly.
 * Add custom tuples as explicit categories and schemas here, alongside
 * their executor. Standard domains must not implicitly admit custom types.
 */
static const struct {
    const AmeCapSchema *schemas;
    size_t nr_schemas;
    uint8_t scalar_forms;
} op_schemas[AME_OP_COUNT] = {
#define OP(op_, schema_, forms_) \
    [AME_OP_ ## op_] = { schema_, G_N_ELEMENTS(schema_), forms_ }
    OP(ABS,          numeric_unary,  NO_SCALAR_FORMS),
    OP(ABSDIFF,      numeric_binary, DATA_SCALAR_MASK),
    OP(ADD,          numeric_binary, DATA_SCALAR_MASK),
    OP(AND,          bitwise,        DATA_SCALAR_MASK),
    OP(ANDNOT,       bitwise,        DATA_SCALAR_MASK),
    OP(CMPGE,        compare,        DATA_SCALAR_MASK),
    OP(CMPLT,        compare,        DATA_SCALAR_MASK),
    OP(COLBCAST,     copy,           NO_SCALAR_FORMS),
    OP(COLID,        index_dest,     NO_SCALAR_FORMS),
    OP(COLUNZIP,     copy,           NO_SCALAR_FORMS),
    OP(COLZIP,       copy,           NO_SCALAR_FORMS),
    OP(CONV,         convert,        NO_SCALAR_FORMS),
    OP(COS,          fp_unary,       NO_SCALAR_FORMS),
    OP(EXP2,         fp_unary,       NO_SCALAR_FORMS),
    OP(FRINTM,       fp_unary,       NO_SCALAR_FORMS),
    OP(FRINTN,       fp_unary,       NO_SCALAR_FORMS),
    OP(FRINTP,       fp_unary,       NO_SCALAR_FORMS),
    OP(FRINTZ,       fp_unary,       NO_SCALAR_FORMS),
    OP(GATHER,       gather,         NO_SCALAR_FORMS),
    OP(GATHERROW,    gather,         NO_SCALAR_FORMS),
    OP(HDIFF,        numeric_binary, DATA_SCALAR_MASK),
    OP(LDEXP,        scale_index,    CONTROL_SCALAR_MASK),
    OP(LDEXPACC,     scale_index,    CONTROL_SCALAR_MASK),
    OP(LOG2,         fp_unary,       NO_SCALAR_FORMS),
    OP(LOG2SUB,      fp_binary,      DATA_SCALAR_MASK),
    OP(MAX,          numeric_same,   DATA_SCALAR_MASK),
    OP(MEAN,         numeric_binary, DATA_SCALAR_MASK),
    OP(MIN,          numeric_same,   DATA_SCALAR_MASK),
    OP(MMACC,        numeric_binary, NO_SCALAR_FORMS),
    OP(MMACCNEG,     numeric_binary, NO_SCALAR_FORMS),
    OP(MMATACC,      numeric_binary, NO_SCALAR_FORMS),
    OP(MMATACCNEG,   numeric_binary, NO_SCALAR_FORMS),
    OP(MMBTACC,      numeric_binary, NO_SCALAR_FORMS),
    OP(MMBTACCNEG,   numeric_binary, NO_SCALAR_FORMS),
    OP(MUL,          numeric_binary, DATA_SCALAR_MASK),
    OP(MULACC,       numeric_binary, DATA_SCALAR_MASK),
    OP(MULADD,       numeric_binary, DATA_SCALAR_MASK),
    OP(MULNACC,      numeric_binary, DATA_SCALAR_MASK),
    OP(MULSUB,       numeric_binary, DATA_SCALAR_MASK),
    OP(MUXGE,        select_value,   NO_SCALAR_FORMS),
    OP(MUXLT,        select_value,   NO_SCALAR_FORMS),
    OP(NMUL,         numeric_binary, DATA_SCALAR_MASK),
    OP(OR,           bitwise,        DATA_SCALAR_MASK),
    OP(ORNOT,        bitwise,        DATA_SCALAR_MASK),
    OP(PREFIXADD,    numeric_unary,  NO_SCALAR_FORMS),
    OP(PREFIXADDCOL, numeric_unary,  NO_SCALAR_FORMS),
    OP(PREFIXMAX,    numeric_unary,  NO_SCALAR_FORMS),
    OP(PREFIXMAXCOL, numeric_unary,  NO_SCALAR_FORMS),
    OP(RDEXP,        scale_index,    NO_SCALAR_FORMS),
    OP(RDEXPACC,     scale_index,    NO_SCALAR_FORMS),
    OP(REC,          fp_unary,       NO_SCALAR_FORMS),
    OP(REDUCEADD,    numeric_unary,  NO_SCALAR_FORMS),
    OP(REDUCEADDCOL, numeric_unary,  NO_SCALAR_FORMS),
    OP(REDUCEMAX,    numeric_unary,  NO_SCALAR_FORMS),
    OP(REDUCEMAXCOL, numeric_unary,  NO_SCALAR_FORMS),
    OP(REDUCEMIN,    numeric_unary,  NO_SCALAR_FORMS),
    OP(REDUCEMINCOL, numeric_unary,  NO_SCALAR_FORMS),
    OP(ROWBCAST,     copy,           NO_SCALAR_FORMS),
    OP(ROWID,        index_dest,     NO_SCALAR_FORMS),
    OP(ROWSHIFT,     copy,           NO_SCALAR_FORMS),
    OP(ROWUNZIP,     copy,           NO_SCALAR_FORMS),
    OP(ROWZIP,       copy,           NO_SCALAR_FORMS),
    OP(RSQRT,        fp_unary,       NO_SCALAR_FORMS),
    OP(COLSCATADD,   scale_index,    NO_SCALAR_FORMS),
    OP(COLSCATMAX,   scale_index,    NO_SCALAR_FORMS),
    OP(ROWSCATADD,   scale_index,    NO_SCALAR_FORMS),
    OP(ROWSCATMAX,   scale_index,    NO_SCALAR_FORMS),
    OP(SEL,          select_value,   NO_SCALAR_FORMS),
    OP(SELGE,        select_value,   NO_SCALAR_FORMS),
    OP(SHIFT,        copy,           NO_SCALAR_FORMS),
    OP(SIN,          fp_unary,       NO_SCALAR_FORMS),
    OP(SLL,          shift,          CONTROL_SCALAR_MASK),
    OP(SQRT,         fp_unary,       NO_SCALAR_FORMS),
    OP(SRA,          signed_shift,   CONTROL_SCALAR_MASK),
    OP(SRL,          shift,          CONTROL_SCALAR_MASK),
    OP(SUB,          numeric_binary, DATA_SCALAR_MASK),
    OP(SUBLOG2,      fp_binary,      DATA_SCALAR_MASK),
    OP(TANH,         fp_unary,       NO_SCALAR_FORMS),
    OP(XOR,          bitwise,        DATA_SCALAR_MASK),
    OP(ZERO,         zero,           NO_SCALAR_FORMS),
#undef OP
};

static bool type_in_domain(uint16_t id, AmeTypeDomain domain)
{
    const AmeDTypeDesc *dtype = &riscv_ame_dtypes[id];
    bool integer = dtype->class == AME_DTYPE_CLASS_INTEGER;

    switch (domain) {
    case TYPE_NONE:
        return false;
    case TYPE_INT:
        return integer;
    case TYPE_SIGNED:
        return integer && dtype->integer.is_signed;
    case TYPE_FP:
        return riscv_ame_dtype_is_fp(dtype);
    case TYPE_STANDARD:
        return riscv_ame_dtype_is_standard(dtype);
    default:
        g_assert_not_reached();
    }
}

static bool key_valid(AmeCapKey key)
{
    return (unsigned)key.op < AME_OP_COUNT &&
           (unsigned)key.form < AME_FORM_MAX &&
           (unsigned)key.category < AME_CAP_CATEGORY_MAX;
}

static bool entry_schema(AmeCapKey key, const AmeDTypeSet *public,
                         AmeTupleRule *rule,
                         AmeDTypeSet allowed[AME_TUPLE_SIZE])
{
    const AmeCapSchema *schema = NULL;
    AmeTypeDomain domains[AME_TUPLE_SIZE];

    if (!key_valid(key) ||
        (key.form != AME_FORM_BASE &&
         !(op_schemas[key.op].scalar_forms & BIT(key.form)))) {
        return false;
    }
    if (key.form == AME_FORM_DATA_SCALAR &&
        (key.category == AME_CAP_CATEGORY_INT_FP_SOURCES ||
         key.category == AME_CAP_CATEGORY_FP_INT_SOURCES)) {
        /* Both arithmetic sources have T's datatype after scalar ingress. */
        return false;
    }
    for (size_t i = 0; i < op_schemas[key.op].nr_schemas; i++) {
        if (op_schemas[key.op].schemas[i].category == key.category) {
            schema = &op_schemas[key.op].schemas[i];
            break;
        }
    }
    if (!schema) {
        return false;
    }
    memcpy(domains, schema->operand_domains, sizeof(domains));
    *rule = (AmeTupleRule) { .relation = schema->relation };
    if (key.form == AME_FORM_DATA_SCALAR) {
        /* S is converted to T before executing the base tuple (D,T,T). */
        g_assert(domains[AME_TUPLE_A] == TYPE_INT ||
                 domains[AME_TUPLE_A] == TYPE_FP);
        domains[AME_TUPLE_B] = TYPE_STANDARD;
        if (rule->relation == AME_TUPLE_SAME) {
            rule->relation = AME_TUPLE_SAME_DA;
        }
    } else if (key.form == AME_FORM_CONTROL_SCALAR) {
        domains[AME_TUPLE_B] = TYPE_NONE;
    }
    memset(allowed, 0, sizeof(*allowed) * AME_TUPLE_SIZE);
    for (unsigned pos = 0; pos < AME_TUPLE_SIZE; pos++) {
        if (domains[pos] != TYPE_NONE) {
            rule->operand_mask |= BIT(pos);
        }
        for (unsigned id = 0; id < AME_DTYPE_COUNT; id++) {
            if (riscv_ame_dtype_set_has(public, id) &&
                type_in_domain(id, domains[pos])) {
                riscv_ame_dtype_set_add(&allowed[pos], id);
            }
        }
    }
    return true;
}

static int key_compare(AmeCapKey a, AmeCapKey b)
{
    if (a.op != b.op) {
        return a.op < b.op ? -1 : 1;
    }
    if (a.form != b.form) {
        return a.form < b.form ? -1 : 1;
    }
    return a.category < b.category ? -1 : a.category != b.category;
}

static int entry_compare(const void *va, const void *vb)
{
    const AmeCapEntry *a = va, *b = vb;

    return key_compare(a->key, b->key);
}

static void entries_index(AmeCapabilities *set)
{
    if (set->nr_entries > 1) {
        qsort(set->entries, set->nr_entries,
              sizeof(*set->entries), entry_compare);
    }
    for (size_t i = 0; i < set->nr_entries; i++) {
        AmeCapKey key = set->entries[i].key;

        if (!set->by_op[key.op][key.form].count) {
            set->by_op[key.op][key.form].first = i;
        }
        set->by_op[key.op][key.form].count++;
    }
}

const AmeCapEntry *riscv_ame_capabilities_entry(const AmeCapabilities *set,
                                              AmeCapKey key)
{
    unsigned first, count;

    if (!key_valid(key)) {
        return NULL;
    }
    first = set->by_op[key.op][key.form].first;
    count = set->by_op[key.op][key.form].count;
    for (unsigned i = first; i < first + count; i++) {
        if (set->entries[i].key.category == key.category) {
            return &set->entries[i];
        }
    }
    return NULL;
}

bool riscv_ame_capabilities_has(const AmeCapabilities *set, AmeOperation op,
                                AmeCapForm form, uint16_t dest,
                                uint16_t source1, uint16_t source2)
{
    unsigned first, count;
    AmeDTypeTuple tuple = { .dtype_ids = { dest, source1, source2 } };

    if ((unsigned)op >= AME_OP_COUNT || (unsigned)form >= AME_FORM_MAX) {
        return false;
    }
    first = set->by_op[op][form].first;
    count = set->by_op[op][form].count;
    for (unsigned i = first; i < first + count; i++) {
        if (riscv_ame_tuple_set_has(&set->entries[i].tuples, tuple)) {
            return true;
        }
    }
    return false;
}

/* Check scalar conversion and arithmetic dependencies in the maximum set. */
static bool scalar_pair_supported(const AmeCapabilities *maximum,
                                   AmeOperation op, uint16_t d, uint16_t t,
                                   uint16_t s)
{
    return riscv_ame_capabilities_has(maximum, AME_OP_CONV, AME_FORM_BASE,
                                      t, s, AME_DTYPE_ID_INVALID) &&
           riscv_ame_capabilities_has(maximum, op, AME_FORM_BASE, d, t, t);
}

static bool scalar_dependencies_valid(const AmeCapabilities *set,
                                       const AmeCapabilities *maximum,
                                       Error **errp)
{
    for (size_t i = 0; i < set->nr_entries; i++) {
        const AmeCapEntry *entry = &set->entries[i];
        const AmeTupleSet *tuples = &entry->tuples;
        uint16_t first_s = 0;

        if (entry->key.form == AME_FORM_CONTROL_SCALAR &&
            (entry->key.op == AME_OP_LDEXP ||
             entry->key.op == AME_OP_LDEXPACC) && !set->max_int_dtype) {
            error_setg(errp, "Ztt scalar exponent tuple requires "
                       "AME_MAX_INT_DTYPE");
            return false;
        }
        if (entry->key.form != AME_FORM_DATA_SCALAR) {
            continue;
        }
        if (tuples->kind == AME_TUPLE_SET_EXACT) {
            for (size_t n = 0; n < tuples->nr_tuples; n++) {
                const uint16_t *t = tuples->exact_tuples[n].dtype_ids;

                if (!scalar_pair_supported(maximum, entry->key.op,
                                             t[0], t[1], t[2])) {
                    goto unsupported;
                }
            }
            continue;
        }
        while (!riscv_ame_dtype_set_has(&tuples->operand_types[2], first_s)) {
            first_s++;
        }
        for (unsigned t = 0; t < AME_DTYPE_COUNT; t++) {
            if (!riscv_ame_dtype_set_has(&tuples->operand_types[1], t)) {
                continue;
            }
            for (unsigned s = 0; s < AME_DTYPE_COUNT; s++) {
                if (riscv_ame_dtype_set_has(&tuples->operand_types[2], s) &&
                    !riscv_ame_capabilities_has(
                        maximum, AME_OP_CONV, AME_FORM_BASE, t, s,
                        AME_DTYPE_ID_INVALID)) {
                    goto unsupported;
                }
            }
            for (unsigned d = 0; d < AME_DTYPE_COUNT; d++) {
                AmeDTypeTuple original = { .dtype_ids = { d, t, first_s } };

                if (riscv_ame_tuple_set_has(tuples, original) &&
                    !riscv_ame_capabilities_has(maximum, entry->key.op,
                                                AME_FORM_BASE, d, t, t)) {
                    goto unsupported;
                }
            }
        }
    }
    return true;

unsupported:
    error_setg(errp, "Ztt scalar tuple lacks a conversion or arithmetic path");
    return false;
}

const AmeCapabilities *riscv_ame_maximum_capabilities(void)
{
    static AmeCapabilities maximum;
    static gsize initialized;

    if (g_once_init_enter(&initialized)) {
        maximum.max_int_dtype = RV_AME_MAX_INT_DTYPE;
        maximum.entries = g_new0(AmeCapEntry,
                                 AME_OP_COUNT * AME_FORM_MAX *
                                 AME_CAP_CATEGORY_MAX);
        for (unsigned id = 0; id < AME_STANDARD_DTYPE_COUNT; id++) {
            const AmeDTypeDesc *dtype = &riscv_ame_dtypes[id];

            if (dtype->class != AME_DTYPE_CLASS_OCP ||
                dtype->fp.rounding != AME_FP_RNO) {
                riscv_ame_dtype_set_add(&maximum.dtypes, id);
            }
        }
        for (unsigned op = 0; op < AME_OP_COUNT; op++) {
            g_assert(op_schemas[op].nr_schemas);
            for (unsigned form = 0; form < AME_FORM_MAX; form++) {
                for (unsigned category = 0; category < AME_CAP_CATEGORY_MAX;
                     category++) {
                    AmeCapKey key = { op, form, category };
                    AmeTupleRule rule;
                    AmeDTypeSet allowed[AME_TUPLE_SIZE];
                    AmeCapEntry *entry;

                    if (!entry_schema(key, &maximum.dtypes, &rule, allowed)) {
                        continue;
                    }
                    entry = &maximum.entries[maximum.nr_entries++];
                    entry->key = key;
                    riscv_ame_tuple_set_rule(&entry->tuples, rule, allowed,
                                             allowed, &error_abort);
                    g_assert(entry->tuples.kind == AME_TUPLE_SET_RULE);
                }
            }
        }
        maximum.entries = g_renew(AmeCapEntry, maximum.entries,
                                    maximum.nr_entries);
        entries_index(&maximum);
        scalar_dependencies_valid(&maximum, &maximum, &error_abort);
        g_once_init_leave(&initialized, 1);
    }
    return &maximum;
}

const AmeCapabilities *riscv_ame_capabilities_for_cpu(
    AmeCapabilityPolicy policy, const AmeCapabilities *requirements,
    Error **errp)
{
    switch (policy) {
    case AME_CAP_POLICY_FOLLOW_MAX:
        if (requirements) {
            error_setg(errp,
                       "Ztt generic CPU cannot have explicit capabilities");
            return NULL;
        }
        return riscv_ame_maximum_capabilities();
    case AME_CAP_POLICY_EXPLICIT:
        /* The immutable requirement is built by capabilities_init(). */
        if (!requirements ||
            riscv_ame_dtype_set_empty(&requirements->dtypes)) {
            error_setg(errp, "Ztt fixed CPU requires an initialized allowlist");
            return NULL;
        }
        return requirements;
    default:
        error_setg(errp, "Ztt requires an explicit CPU capability policy");
        return NULL;
    }
}

bool riscv_ame_dtype_set_from_encodings(AmeDTypeSet *out,
                                       const uint32_t *encodings, size_t count,
                                       Error **errp)
{
    AmeDTypeSet result = { 0 };

    if (count > AME_DTYPE_COUNT || (count && !encodings)) {
        error_setg(errp, "Ztt datatype allowlist has an invalid size");
        return false;
    }
    for (size_t i = 0; i < count; i++) {
        const AmeDTypeDesc *dtype = riscv_ame_dtype_by_encoding(encodings[i]);

        if (!dtype || riscv_ame_dtype_set_has(&result, dtype->id)) {
            error_setg(errp, "Ztt datatype allowlist has invalid entry %zu", i);
            return false;
        }
        riscv_ame_dtype_set_add(&result, dtype->id);
    }
    *out = result;
    return true;
}

static bool public_dtypes_valid(uint32_t max_int_dtype,
                                 const AmeDTypeSet *dtypes,
                                 const AmeCapabilities *maximum, Error **errp)
{
    const AmeDTypeDesc *max_int = riscv_ame_dtype_by_encoding(max_int_dtype);
    unsigned max_bits = 0;

    if (!dtypes || riscv_ame_dtype_set_empty(dtypes)) {
        error_setg(errp, "Ztt capability set must enable a datatype");
        return false;
    }
    for (unsigned w = 0; w < AME_DTYPE_WORD_COUNT; w++) {
        if (dtypes->words[w] & ~maximum->dtypes.words[w]) {
            error_setg(errp,
                       "Ztt capability set enables unsupported datatypes");
            return false;
        }
    }
    for (unsigned id = 0; id < AME_DTYPE_COUNT; id++) {
        if (riscv_ame_dtype_set_has(dtypes, id) &&
            riscv_ame_dtypes[id].class == AME_DTYPE_CLASS_INTEGER) {
            max_bits = MAX(max_bits, riscv_ame_dtypes[id].bits);
        }
    }
    if (!max_bits && !max_int_dtype) {
        return true;
    }
    if (!max_int || max_int->class != AME_DTYPE_CLASS_INTEGER ||
        !riscv_ame_dtype_set_has(dtypes, max_int->id)) {
        error_setg(errp, "Ztt capability set has invalid AME_MAX_INT_DTYPE");
        return false;
    }
    if (max_int->bits != max_bits) {
        error_setg(errp, "Ztt AME_MAX_INT_DTYPE must have the greatest allowed "
                   "integer width");
        return false;
    }
    return true;
}

/* Explicit requests must not silently lose types during normalization. */
static bool rule_request_reachable(const AmeCapRequest *request,
                                   const AmeTupleSet *set, Error **errp)
{
    static const char positions[AME_TUPLE_SIZE] = {
        [AME_TUPLE_D] = 'D', [AME_TUPLE_A] = 'A', [AME_TUPLE_B] = 'B',
    };

    for (unsigned pos = 0; pos < AME_TUPLE_SIZE; pos++) {
        for (unsigned id = 0; id < AME_DTYPE_COUNT; id++) {
            if (riscv_ame_dtype_set_has(&request->operand_types[pos], id) &&
                !riscv_ame_dtype_set_has(&set->operand_types[pos], id)) {
                error_setg(errp, "Ztt capability op=%u form=%u category=%u "
                           "operand %c datatype 0x%08x does not occur in any "
                           "supported tuple", request->key.op,
                           request->key.form, request->key.category,
                           positions[pos], riscv_ame_dtypes[id].encoding);
                return false;
            }
        }
    }
    return true;
}

bool riscv_ame_capabilities_init(AmeCapabilities *out, uint32_t max_int_dtype,
                                const AmeDTypeSet *dtypes,
                                const AmeCapRequest *requests, size_t count,
                                Error **errp)
{
    const AmeCapabilities *maximum = riscv_ame_maximum_capabilities();
    AmeCapabilities result = { .max_int_dtype = max_int_dtype };
    bool seen[AME_OP_COUNT][AME_FORM_MAX][AME_CAP_CATEGORY_MAX] = { 0 };

    g_assert(!out->entries && !out->nr_entries && !out->max_int_dtype &&
             riscv_ame_dtype_set_empty(&out->dtypes));
    if (count > AME_OP_COUNT * AME_FORM_MAX * AME_CAP_CATEGORY_MAX ||
        (count && !requests)) {
        error_setg(errp, "Ztt capability entry list has an invalid size");
        return false;
    }
    if (!public_dtypes_valid(max_int_dtype, dtypes, maximum, errp)) {
        return false;
    }
    result.dtypes = *dtypes;
    result.entries = g_new0(AmeCapEntry, count);
    for (size_t i = 0; i < count; i++) {
        const AmeCapRequest *request = &requests[i];
        AmeCapKey key = request->key;
        AmeTupleRule rule;
        AmeDTypeSet allowed[AME_TUPLE_SIZE];
        AmeCapEntry *entry = &result.entries[result.nr_entries];
        const AmeCapEntry *implemented;
        bool valid;

        if (!entry_schema(key, dtypes, &rule, allowed)) {
            error_setg(errp, "Ztt capability entry %zu has an unknown key", i);
            goto fail;
        }
        if (seen[key.op][key.form][key.category]) {
            error_setg(errp, "Ztt capability entry %zu duplicates a key", i);
            goto fail;
        }
        seen[key.op][key.form][key.category] = true;
        entry->key = key;
        switch (request->kind) {
        case AME_TUPLE_SET_RULE:
            valid = riscv_ame_tuple_set_rule(&entry->tuples, rule, allowed,
                                              request->operand_types, errp);
            if (valid) {
                valid = rule_request_reachable(request, &entry->tuples, errp);
            }
            break;
        case AME_TUPLE_SET_EXACT:
            valid = riscv_ame_tuple_set_exact(&entry->tuples, rule, allowed,
                                               request->exact.tuples,
                                               request->exact.count, errp);
            break;
        default:
            error_setg(errp, "Ztt capability entry %zu has an invalid kind", i);
            goto fail;
        }
        if (!valid) {
            goto fail;
        }
        if (entry->tuples.kind == AME_TUPLE_SET_EMPTY) {
            continue;
        }
        /* Count owned storage before any following check can fail. */
        result.nr_entries++;
        implemented = riscv_ame_capabilities_entry(maximum, key);
        if (!implemented ||
            !riscv_ame_tuple_set_subset(&entry->tuples, &implemented->tuples)) {
            error_setg(errp, "Ztt capability entry %zu exceeds QEMU support",
                       i);
            goto fail;
        }
    }
    entries_index(&result);
    if (!scalar_dependencies_valid(&result, maximum, errp)) {
        goto fail;
    }
    *out = result;
    return true;

fail:
    riscv_ame_capabilities_clear(&result);
    return false;
}

void riscv_ame_capabilities_copy(AmeCapabilities *out,
                                 const AmeCapabilities *source)
{
    g_assert(!out->entries && !out->nr_entries && !out->max_int_dtype &&
             riscv_ame_dtype_set_empty(&out->dtypes));
    *out = *source;
    out->entries = g_memdup2(source->entries,
                              source->nr_entries * sizeof(*source->entries));
    for (size_t i = 0; i < out->nr_entries; i++) {
        AmeTupleSet *tuples = &out->entries[i].tuples;

        tuples->exact_tuples =
            g_memdup2(tuples->exact_tuples,
                      tuples->nr_tuples * sizeof(*tuples->exact_tuples));
    }
}

void riscv_ame_capabilities_clear(AmeCapabilities *set)
{
    for (size_t i = 0; i < set->nr_entries; i++) {
        riscv_ame_tuple_set_clear(&set->entries[i].tuples);
    }
    g_free(set->entries);
    memset(set, 0, sizeof(*set));
}

bool riscv_ame_capabilities_equal(const AmeCapabilities *left,
                                  const AmeCapabilities *right)
{
    if (left->max_int_dtype != right->max_int_dtype ||
        left->nr_entries != right->nr_entries ||
        memcmp(left->dtypes.words, right->dtypes.words,
               sizeof(left->dtypes.words))) {
        return false;
    }
    for (size_t i = 0; i < left->nr_entries; i++) {
        if (key_compare(left->entries[i].key, right->entries[i].key) ||
            !riscv_ame_tuple_set_equal(&left->entries[i].tuples,
                                       &right->entries[i].tuples)) {
            return false;
        }
    }
    return true;
}
