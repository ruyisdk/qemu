/*
 * RISC-V Attached Matrix Extension positive capability sets
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef TARGET_RISCV_AME_CAP_SET_H
#define TARGET_RISCV_AME_CAP_SET_H

#include "target/riscv/ame/ame.h"
#include "target/riscv/ame/tuple-set.h"

/* CPU construction policy; the configured pool owns runtime capabilities. */
typedef enum AmeCapabilityPolicy {
    AME_CAP_POLICY_UNSET,      /* No policy chosen; enabling AME is rejected. */
    AME_CAP_POLICY_FOLLOW_MAX, /* Use QEMU's maximum to configure a new pool. */
    AME_CAP_POLICY_EXPLICIT,   /* Require the CPU model's complete allowlist. */
} AmeCapabilityPolicy;

/* Selects how a capability tuple describes an operation's operands. */
typedef enum AmeCapForm {
    /*
     * D/A/B: destination and the operation's used source datatypes.
     * Includes register operations and explicitly typed scalar conversions.
     */
    AME_FORM_BASE,
    /*
     * D/A/B: destination, matrix source T and scalar source S (amestype).
     * Convert S to T before executing the base operation with (D,T,T).
     */
    AME_FORM_DATA_SCALAR,
    /*
     * D/A: destination and matrix source datatypes; B is unused.
     * Interpret the control scalar as defined by the instruction.
     */
    AME_FORM_CONTROL_SCALAR,
    AME_FORM_MAX,
} AmeCapForm;

/* Datatype category within an operation/form, with fixed datatype rules. */
typedef enum AmeCapCategory {
    /* Structural, raw-bit or conditional selection. */
    AME_CAP_CATEGORY_GENERIC,
    /* Integer data sources, or integer-to-integer conversion. */
    AME_CAP_CATEGORY_INTEGER,
    /* Floating data sources, or floating-to-floating conversion. */
    AME_CAP_CATEGORY_FP,
    /* Integer-to-floating conversion. */
    AME_CAP_CATEGORY_INT_TO_FP,
    /* Floating-to-integer conversion. */
    AME_CAP_CATEGORY_FP_TO_INT,
    /* Integer A and floating B data sources. */
    AME_CAP_CATEGORY_INT_FP_SOURCES,
    /* Floating A and integer B data sources. */
    AME_CAP_CATEGORY_FP_INT_SOURCES,
    AME_CAP_CATEGORY_MAX,
} AmeCapCategory;

/* Key for one operation, operand form and datatype category. */
typedef struct AmeCapKey {
    AmeOperation op;
    AmeCapForm form;
    AmeCapCategory category;
} AmeCapKey;

/* Construction input using the key's fixed datatype relation. */
typedef struct AmeCapRequest {
    AmeCapKey key;
    AmeTupleSetKind kind;
    union {
        /*
         * RULE input: requested datatypes for each D/A/B operand.
         * Every listed ID must occur in at least one supported tuple at that
         * position; normalization must not silently remove requested types.
         */
        AmeDTypeSet operand_types[AME_TUPLE_SIZE];
        /* EXACT input: borrowed list of complete supported combinations. */
        struct {
            const AmeDTypeTuple *tuples;
            size_t count;
        } exact;
    };
} AmeCapRequest;

typedef struct AmeCapEntry {
    AmeCapKey key;
    /* Supported (D, A, B) combinations for this operation/form/category. */
    AmeTupleSet tuples;
} AmeCapEntry;

/*
 * Accepted datatypes and operation/datatype tuples for one backend model.
 * dtypes controls type-setting and data access; entries controls operations.
 *
 * Initialize to zero, construct or copy, then keep read-only. Owns entries
 * and their tuple storage; release with riscv_ame_capabilities_clear(). QEMU's
 * maximum uses the same representation but has process lifetime.
 */
typedef struct AmeCapabilities {
    /* Zero means no integer datatype, so no fixed-type scalar exponents. */
    uint32_t max_int_dtype;
    AmeDTypeSet dtypes;
    /* Sorted by operation, form and category. */
    AmeCapEntry *entries;
    size_t nr_entries;
    /* Contiguous entries[first .. first + count) for each operation/form. */
    struct {
        uint16_t first;
        uint16_t count;
    } by_op[AME_OP_COUNT][AME_FORM_MAX];
} AmeCapabilities;

/**
 * riscv_ame_maximum_capabilities: Get QEMU's maximum capability set
 *
 * Construct once on first use. The returned set must not be changed or freed.
 *
 * Returns: the immutable capability set with process lifetime.
 */
const AmeCapabilities *riscv_ame_maximum_capabilities(void);

/**
 * riscv_ame_capabilities_for_cpu: Select initial CPU requirements
 * @policy: FOLLOW_MAX or EXPLICIT; UNSET is invalid
 * @requirements: Constructed EXPLICIT allowlist; NULL for FOLLOW_MAX
 * @errp: Error destination
 *
 * Returns: a borrowed pointer to QEMU's maximum or @requirements; NULL on
 * invalid input, setting @errp.
 */
const AmeCapabilities *riscv_ame_capabilities_for_cpu(
    AmeCapabilityPolicy policy, const AmeCapabilities *requirements,
    Error **errp);

/**
 * riscv_ame_dtype_set_from_encodings: Build a datatype set from AME encodings
 * @out: Datatype set to replace on success
 * @encodings: Encoding array; may be NULL if @count is zero
 * @count: Number of encodings
 * @errp: Error destination
 *
 * Reject unrecognized or duplicate encodings. An empty input produces an
 * empty set; failure leaves @out unchanged. The input array is not retained.
 *
 * Returns: true on success; false on invalid input, setting @errp.
 */
bool riscv_ame_dtype_set_from_encodings(AmeDTypeSet *out,
                                       const uint32_t *encodings, size_t count,
                                       Error **errp);

/**
 * riscv_ame_capabilities_init: Build an explicit backend capability allowlist
 * @out: Result; must be zero-initialized or previously cleared
 * @max_int_dtype: AME_MAX_INT_DTYPE encoding; zero only with no integer types
 * @dtypes: Nonempty set of accepted datatypes within QEMU's supported set
 * @requests: Entries to enable; may be NULL if @count is zero
 * @count: Number of requests; each operation/form/category key must be unique
 * @errp: Error destination
 *
 * Build an explicit allowlist within QEMU's maximum capability set.
 * For a RULE request, each requested operand datatype must occur in at least
 * one supported tuple at that position. Reject the request if tuple
 * construction omits any requested datatype; do not silently install a
 * smaller allowlist.
 * All-empty RULE inputs and empty EXACT lists add no entry. Global @dtypes
 * need not participate in an entry. A nonzero @max_int_dtype must belong to
 * @dtypes and have the greatest enabled integer width. Inputs are copied.
 * Failure leaves @out unchanged.
 *
 * Returns: true on success; false on invalid input, setting @errp.
 */
bool riscv_ame_capabilities_init(AmeCapabilities *out, uint32_t max_int_dtype,
                                const AmeDTypeSet *dtypes,
                                const AmeCapRequest *requests, size_t count,
                                Error **errp);
/**
 * riscv_ame_capabilities_copy: Deep-copy a constructed capability set
 * @out: Result; must be zero-initialized or previously cleared
 * @source: Constructed set to copy
 *
 * The result owns independent entry and tuple arrays. Release it with
 * riscv_ame_capabilities_clear().
 */
void riscv_ame_capabilities_copy(AmeCapabilities *out,
                                 const AmeCapabilities *source);

/**
 * riscv_ame_capabilities_clear: Free capability storage and zero the set
 * @set: Zero-initialized or constructed set; must not be QEMU's shared maximum
 *
 * Safe to call again on a cleared set.
 */
void riscv_ame_capabilities_clear(AmeCapabilities *set);

/**
 * riscv_ame_capabilities_has: Check one operation/datatype combination
 * @set: Constructed capability set
 * @op: Operation to check
 * @form: Base, data-scalar or control-scalar operand form
 * @dest: D datatype index in riscv_ame_dtypes[]
 * @source1: A datatype index, or AME_DTYPE_ID_INVALID if unused
 * @source2: B datatype index, or AME_DTYPE_ID_INVALID if unused
 *
 * Check the complete tuple against all categories for @op and @form.
 *
 * Returns: true if any entry accepts the tuple; otherwise false.
 */
bool riscv_ame_capabilities_has(const AmeCapabilities *set, AmeOperation op,
                                AmeCapForm form, uint16_t dest,
                                uint16_t source1, uint16_t source2);
/**
 * riscv_ame_capabilities_equal: Compare accepted datatypes and operation tuples
 * @left: First constructed capability set
 * @right: Second constructed capability set
 *
 * Compare AME_MAX_INT_DTYPE, accepted datatypes, entry keys and tuple sets.
 * Storage addresses and derived lookup indexes do not affect equality.
 *
 * Returns: true if these capabilities match; otherwise false.
 */
bool riscv_ame_capabilities_equal(const AmeCapabilities *left,
                                  const AmeCapabilities *right);

/**
 * riscv_ame_capabilities_entry: Find the entry for one operation/form/category
 * @set: Constructed capability set
 * @key: Entry identity to look up
 *
 * Returns: a borrowed entry pointer valid until @set is cleared; NULL for
 * an invalid or absent key.
 */
const AmeCapEntry *riscv_ame_capabilities_entry(const AmeCapabilities *set,
                                              AmeCapKey key);

#endif /* TARGET_RISCV_AME_CAP_SET_H */
