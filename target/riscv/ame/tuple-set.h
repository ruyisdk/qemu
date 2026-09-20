/*
 * RISC-V Attached Matrix Extension datatype tuple sets
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef TARGET_RISCV_AME_TUPLE_SET_H
#define TARGET_RISCV_AME_TUPLE_SET_H

#include "qapi/error.h"
#include "qemu/bitops.h"
#include "target/riscv/ame/datatype.h"

/*
 * Datatype tuple positions: D is the destination (M or Acc), and A/B are the
 * sources used by the operation form. Use these constants to index operand
 * arrays, and BIT(AME_TUPLE_*) to select operands in a mask.
 */
enum {
    AME_TUPLE_D,
    AME_TUPLE_A,
    AME_TUPLE_B,
    AME_TUPLE_SIZE,
};

#define AME_TUPLE_ALL_MASK (BIT(AME_TUPLE_SIZE) - 1)

/*
 * Datatypes allowed for one operand, for example Int8 and Int16 for source A.
 * The capability layer also uses this set for a backend's accepted datatypes.
 * Each member is a complete datatype, including rounding mode and, for
 * integers, signedness and saturation behavior.
 *
 * Bit id represents the datatype in riscv_ame_dtypes[id]. Look up a 32-bit
 * AME datatype encoding with riscv_ame_dtype_by_encoding() and use the result's
 * id to add or test that datatype in the set. A zero-initialized set is empty.
 */
typedef struct AmeDTypeSet {
    uint64_t words[AME_DTYPE_WORD_COUNT];
} AmeDTypeSet;

/*
 * One concrete combination of destination and source datatypes (D, A, B),
 * for example (Int32, Int8, Uint16) for an integer add operation.
 * Each value is the datatype's id in riscv_ame_dtypes[]. If the operation
 * does not use A or B, that position contains AME_DTYPE_ID_INVALID.
 * A query checks whether this combination belongs to the operation's
 * supported datatype tuples.
 */
typedef struct AmeDTypeTuple {
    uint16_t dtype_ids[AME_TUPLE_SIZE];
} AmeDTypeTuple;

typedef enum AmeTupleRelation {
    /* Any combination of the per-operand allowed datatypes. */
    AME_TUPLE_ANY,
    /* All used operands have the same complete datatype. */
    AME_TUPLE_SAME,
    /* D and A have the same complete datatype; B is independently selected. */
    AME_TUPLE_SAME_DA,
    /* D and B have the same complete datatype; A is independently selected. */
    AME_TUPLE_SAME_DB,
    AME_TUPLE_RELATION_MAX,
} AmeTupleRelation;

/*
 * Datatype constraints for an operation form, selected by the capability
 * layer. A backend selects its supported datatypes or exact tuples while
 * keeping these constraints.
 */
typedef struct AmeTupleRule {
    AmeTupleRelation relation;
    /* Used operands: D, D+A or D+A+B, selected with BIT(AME_TUPLE_*). */
    uint8_t operand_mask;
} AmeTupleRule;

typedef enum AmeTupleSetKind {
    /* No supported datatype tuples. All fields of AmeTupleSet are zero. */
    AME_TUPLE_SET_EMPTY,
    /* Accept combinations from operand_types[] that satisfy rule. */
    AME_TUPLE_SET_RULE,
    /* Accept only the complete combinations listed in exact_tuples[]. */
    AME_TUPLE_SET_EXACT,
} AmeTupleSetKind;

/*
 * Supported datatype tuples. The capability layer associates this set with
 * one operation form and datatype category, such as integer or floating-point.
 * RULE stores per-operand datatypes and their relation; EXACT stores a list
 * of complete combinations. Both describe which tuples the operation accepts.
 *
 * Initialize to zero, construct with the functions below, then keep the
 * fields read-only. Release storage with riscv_ame_tuple_set_clear(). An EXACT
 * set owns its tuple array; copying the struct does not duplicate that array.
 */
typedef struct AmeTupleSet {
    AmeTupleSetKind kind;
    AmeTupleRule rule;
    /*
     * Datatypes that occur at D/A/B in at least one supported tuple.
     * Unused operands have empty sets. For RULE, also check rule; for EXACT,
     * also check the complete combination in exact_tuples[].
     */
    AmeDTypeSet operand_types[AME_TUPLE_SIZE];
    /* EXACT only: tuples sorted by D/A/B id, with duplicates removed. */
    AmeDTypeTuple *exact_tuples;
    /* Length of exact_tuples; other kinds have a NULL array and zero length. */
    size_t nr_tuples;
    /* Total supported combinations, also counted when stored as RULE. */
    uint64_t count;
} AmeTupleSet;

/**
 * riscv_ame_dtype_set_add: Add a datatype to a type set
 * @set: Zero-initialized or populated datatype set to update
 * @id: Index of the datatype in riscv_ame_dtypes[]
 *
 * An out-of-range @id leaves @set unchanged. The caller chooses which
 * datatypes are allowed; this function only updates the bitmap.
 *
 * Returns: true if @id is below AME_DTYPE_COUNT, even if already present;
 * otherwise false.
 */
bool riscv_ame_dtype_set_add(AmeDTypeSet *set, uint16_t id);

/**
 * riscv_ame_dtype_set_has: Check whether a datatype belongs to a type set
 * @set: Datatype set to check
 * @id: Index of the datatype in riscv_ame_dtypes[]
 *
 * Returns: true if @id belongs to @set; false if absent or out of range.
 */
bool riscv_ame_dtype_set_has(const AmeDTypeSet *set, uint16_t id);

/**
 * riscv_ame_dtype_set_empty: Check whether a datatype set is empty
 * @set: Datatype set to check
 *
 * Returns: true if all bitmap words are zero; otherwise false.
 */
bool riscv_ame_dtype_set_empty(const AmeDTypeSet *set);

/**
 * riscv_ame_tuple_set_rule: Build supported tuples from per-operand datatypes
 * @out: Result; must be zero-initialized or previously cleared
 * @rule: Datatype constraints fixed for the operation form
 * @allowed: Datatypes permitted for each D/A/B operand by the capability layer
 * @types: Candidate datatypes for each D/A/B operand
 * @errp: Error destination
 *
 * Each @types set must be contained in its operand's @allowed set. Unused
 * operands must have empty @types sets. A tuple is accepted when each used
 * operand has a datatype in its @types set and the combination satisfies
 * @rule. The input arrays are not modified or retained.
 *
 * Keep only datatypes that participate in an accepted tuple. For example, a
 * source Int32 is omitted from @out if D allows only Int16 and @rule requires
 * D and A to have the same datatype. Omitting these candidates is not an
 * error; the input @types sets remain unchanged.
 *
 * Store a nonempty result as RULE. No accepted tuples produces EMPTY.
 * Invalid input leaves @out unchanged.
 *
 * Returns: true on success, including an EMPTY result; false on invalid
 * input, setting @errp.
 */
bool riscv_ame_tuple_set_rule(AmeTupleSet *out, AmeTupleRule rule,
                              const AmeDTypeSet allowed[AME_TUPLE_SIZE],
                              const AmeDTypeSet types[AME_TUPLE_SIZE],
                              Error **errp);

/**
 * riscv_ame_tuple_set_exact: Build supported tuples from a complete tuple list
 * @out: Result; must be zero-initialized or previously cleared
 * @rule: Datatype constraints fixed for the operation form
 * @allowed: Datatypes permitted for each D/A/B operand by the capability layer
 * @tuples: Complete combinations to support; may be NULL if @nr_tuples is zero
 * @nr_tuples: Number of entries in @tuples
 * @errp: Error destination
 *
 * Each datatype must belong to its operand's @allowed set and each tuple
 * must satisfy @rule. Unused operands must contain AME_DTYPE_ID_INVALID.
 * An invalid tuple fails the entire call and leaves @out unchanged.
 *
 * Copy, sort and remove duplicates from @tuples. An empty list produces
 * EMPTY. Use RULE if the per-operand datatype sets plus @rule accept exactly
 * the listed combinations; otherwise use EXACT and let @out own the copied
 * list. No combination absent from the input becomes supported. The input
 * arrays remain unchanged and owned by the caller.
 *
 * Returns: true on success; false on invalid input, setting @errp.
 */
bool riscv_ame_tuple_set_exact(AmeTupleSet *out, AmeTupleRule rule,
                               const AmeDTypeSet allowed[AME_TUPLE_SIZE],
                               const AmeDTypeTuple *tuples, size_t nr_tuples,
                               Error **errp);

/**
 * riscv_ame_tuple_set_clear: Release a tuple set and reset it to EMPTY
 * @set: Zero-initialized or previously constructed tuple set
 *
 * Free the tuple array, if present, and zero all fields. Safe to call again
 * on a cleared set.
 */
void riscv_ame_tuple_set_clear(AmeTupleSet *set);

/**
 * riscv_ame_tuple_set_has: Check whether a datatype combination is supported
 * @set: Tuple set for the operation to check
 * @tuple: Destination/source datatype combination to check
 *
 * RULE checks each operand's datatype and the rule. EXACT requires the
 * complete combination in the tuple list. Unused operands in @tuple must
 * contain AME_DTYPE_ID_INVALID.
 *
 * Returns: true if @set accepts @tuple; false for EMPTY, an unsupported
 * combination, an out-of-range id or a datatype in an unused position.
 */
bool riscv_ame_tuple_set_has(const AmeTupleSet *set, AmeDTypeTuple tuple);

/**
 * riscv_ame_tuple_set_equal: Compare supported datatype combinations
 * @left: First tuple set to compare
 * @right: Second tuple set to compare
 *
 * Both sets must come from the constructors above and describe the same
 * operation form and datatype category. Nonempty sets must use identical
 * rules; different rules compare unequal.
 *
 * EMPTY equals only EMPTY. RULE compares per-operand types and the rule;
 * EXACT also compares the complete tuple lists. The constructors store a
 * list as RULE whenever it accepts exactly the same tuples, so equal
 * nonempty sets have the same kind.
 *
 * Returns: true if both sets accept exactly the same datatype combinations;
 * otherwise false.
 */
bool riscv_ame_tuple_set_equal(const AmeTupleSet *left,
                               const AmeTupleSet *right);

/**
 * riscv_ame_tuple_set_subset: Check whether every left tuple is in right
 * @left: Tuple set whose supported combinations must all be present in @right
 * @right: Tuple set to check for those combinations
 *
 * Both sets must come from the constructors above and describe the same
 * operation form and datatype category. Nonempty sets must use identical
 * rules; different rules do not establish a subset relationship.
 *
 * Works with RULE and EXACT in either order. Equality is allowed. EMPTY is
 * a subset of every set; a nonempty set is not a subset of EMPTY.
 *
 * Returns: true if every tuple accepted by @left is also accepted by @right;
 * otherwise false.
 */
bool riscv_ame_tuple_set_subset(const AmeTupleSet *left,
                                const AmeTupleSet *right);

#endif /* TARGET_RISCV_AME_TUPLE_SET_H */
