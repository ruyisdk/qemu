/*
 * RISC-V Attached Matrix Extension (AME) backend state
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef TARGET_RISCV_AME_BACKEND_H
#define TARGET_RISCV_AME_BACKEND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "qapi/error.h"
#include "qemu/queue.h"
#include "qemu/thread.h"
#include "qemu/timer.h"
#include "qom/object.h"
#include "target/riscv/ame/layout.h"

/* Host ownership state, separate from the architectural MS state. */
typedef enum RISCVAMEBackendSlotState {
    RISCV_AME_BACKEND_FREE,
    RISCV_AME_BACKEND_INITIALIZING,
    RISCV_AME_BACKEND_OWNED,
} RISCVAMEBackendSlotState;

typedef struct RISCVAMEState RISCVAMEState;

/* Configuration came from the machine or the first CPU's defaults. */
typedef enum RISCVAMEPoolConfigSource {
    RISCV_AME_POOL_AUTO,
    RISCV_AME_POOL_EXPLICIT,
} RISCVAMEPoolConfigSource;

/*
 * Matrix registers, datatype registers and amestatus for one physical AME
 * backend. Data pointers refer to this slot's slices of pool-owned buffers.
 * The current hart alone accesses register data. The pool lock protects
 * ownership changes; owner is published atomically for instruction checks.
 */
typedef struct RISCVAMEBackendSlot {
    uint8_t *mregs;
    uint32_t md[RV_AME_M_REGS_MAX]; /* M datatype register encodings. */
    /* Cached indexes in riscv_ame_dtypes[], or AME_DTYPE_ID_INVALID. */
    uint16_t md_dtype_id[RV_AME_M_REGS_MAX];

    uint8_t *accs;
    uint32_t ad[RV_AME_ACC_REGS_MAX]; /* Acc datatype register encodings. */
    uint16_t ad_dtype_id[RV_AME_ACC_REGS_MAX];

    /* amestype encoding and its cached datatype index. */
    uint32_t scalar_dtype;
    uint16_t scalar_dtype_id;
    uint32_t status; /* amestatus value, including UN. */

    /* Non-architectural staging, exclusive to the current slot owner. */
    uint8_t *scratch;
    size_t scratch_bytes;

    RISCVAMEBackendSlotState state;
    Object *initializer; /* Borrowed hart reserving an INITIALIZING slot. */
    Object *owner;       /* Borrowed owning hart; NULL when not owned. */
    RISCVAMEState *bound_state; /* Owning hart's binding bookkeeping. */
} RISCVAMEBackendSlot;

typedef QTAILQ_HEAD(RISCVAMEWaiterList, RISCVAMEState) RISCVAMEWaiterList;

/*
 * Pool of interchangeable physical backends with common geometry and
 * capabilities. Owns the slot array, register/scratch buffers and capability
 * storage. Configuration is frozen before slots are allocated; reset clears
 * ownership and register contents without changing capabilities or geometry.
 */
typedef struct RISCVAMEBackendPoolCore {
    QemuMutex lock;
    bool lock_initialized;

    RISCVAMEBackendSlot *slots;
    uint8_t *mregs;
    uint8_t *accs;
    uint8_t *scratch;
    uint32_t num_slots;
    uint32_t num_attached_harts;
    uint64_t generation; /* Changes on release/reset so waiters can rescan. */

    bool enabled;
    bool configured;
    bool realized;
    RISCVAMEPoolConfigSource config_source;

    RISCVAMEWaiterList waiters; /* Harts with pending bounded acquires. */

    RISCVAMELayout layout;
    AmeCapabilities capabilities;
} RISCVAMEBackendPoolCore;

/*
 * Per-hart temporary memory-access bookkeeping.
 * chunks is an owned, reusable array of page-sized transfer descriptions.
 * fault_address is a guest address, valid only while fault_active is true.
 */
typedef struct RISCVAMEMemoryScratch {
    struct RISCVAMEMemoryChunk *chunks;
    size_t chunks_capacity; /* Number of allocated chunk entries. */
    uint64_t fault_address;
    bool fault_active;
} RISCVAMEMemoryScratch;

/* Result to retire when a pending acquire resumes; NONE keeps waiting. */
typedef enum RISCVAMEWaitTerminal {
    RISCV_AME_WAIT_NONE,
    RISCV_AME_WAIT_TIMEOUT,
    RISCV_AME_WAIT_INTERRUPTED,
} RISCVAMEWaitTerminal;

/*
 * Per-hart backend binding and wait bookkeeping. Tensor registers reside in
 * the acquired slot. The pool must outlive attached harts; only private_pool
 * is allocated and destroyed by this state.
 */
struct RISCVAMEState {
    /* VMState anchor and subsection presence marker. */
    uint8_t migration_anchor;
    bool migration_seen;

    RISCVAMEBackendPoolCore *pool; /* Attached pool; borrowed if shared. */
    RISCVAMEBackendPoolCore *private_pool; /* Owned one-slot pool, or NULL. */
    int32_t active_slot; /* Bound slot index; -1 after attach when unowned. */
    RISCVAMEMemoryScratch memory;

    /* Queue membership and timer exist only for host-side waiting. */
    QTAILQ_ENTRY(RISCVAMEState) waiter_link;
    QEMUTimer *wait_timer;
    void (*wait_kick)(Object *hart);
    Object *wait_hart; /* Referenced while queued; released on wait removal. */
    uint64_t wait_descriptor; /* Original ame.acquire descriptor for retry. */
    int64_t wait_deadline_ns; /* Absolute QEMU_CLOCK_VIRTUAL nanoseconds. */
    uint64_t wait_generation; /* Pool generation last checked by this waiter. */
    RISCVAMEWaitTerminal wait_terminal;
    bool wait_active; /* This state belongs to pool->waiters. */
    bool wait_wake;   /* Atomic hint to retry, not a promise of a free slot. */
};

/* Outcome of a backend acquisition attempt. */
typedef enum RISCVAMEAcquireResult {
    RISCV_AME_ACQUIRE_GRANTED,
    RISCV_AME_ACQUIRE_BUSY,
    RISCV_AME_ACQUIRE_UNSUPPORTED,
    RISCV_AME_ACQUIRE_DISABLED,
    RISCV_AME_ACQUIRE_WAIT, /* Suspend, then retry the same instruction. */
    RISCV_AME_ACQUIRE_TIMEOUT,
    RISCV_AME_ACQUIRE_INTERRUPTED,
} RISCVAMEAcquireResult;

/**
 * riscv_ame_pool_init: Initialize a backend pool's lock and slot-count policy
 * @pool: Zero-initialized pool
 * @num_slots: Number of interchangeable physical backends, in [1, INT32_MAX]
 * @enabled: Whether acquisitions are permitted after realization
 * @errp: Error destination
 *
 * Does not configure geometry or allocate registers. Configure, then realize
 * the pool before acquiring a slot.
 *
 * Returns: true on success; false on invalid input, setting @errp.
 */
bool riscv_ame_pool_init(RISCVAMEBackendPoolCore *pool, uint32_t num_slots,
                         bool enabled, Error **errp);

/**
 * riscv_ame_pool_configure: Set explicit geometry and capabilities for a pool
 * @pool: Initialized, not yet configured or realized pool
 * @n: N, the square side in elements
 * @u: U, the M cell width in bits
 * @num_mregs: NM, the physical M register count
 * @num_accregs: NA, the Acc register count
 * @capabilities: Constructed capability set to deep-copy
 * @errp: Error destination
 *
 * Validate geometry and freeze the configuration without allocating registers.
 * The caller retains ownership of @capabilities.
 *
 * Returns: true on success; false on invalid input or a frozen configuration,
 * setting @errp and leaving the pool's configuration unchanged.
 */
bool riscv_ame_pool_configure(RISCVAMEBackendPoolCore *pool, uint64_t n,
                              uint64_t u, uint32_t num_mregs,
                              uint32_t num_accregs,
                              const AmeCapabilities *capabilities,
                              Error **errp);

/**
 * riscv_ame_pool_configure_cpu: Apply or validate CPU configuration input
 * @pool: Initialized pool
 * @n: CPU's requested N, in elements per square side
 * @u: CPU's requested U, in bits per M cell
 * @num_mregs: CPU's requested NM
 * @num_accregs: CPU's requested NA
 * @policy: CPU capability policy
 * @requirements: Explicit CPU allowlist, or NULL for FOLLOW_MAX
 * @errp: Error destination
 *
 * An unconfigured pool copies the selected CPU capabilities and geometry.
 * For a configured pool, require matching geometry; EXPLICIT also requires
 * equal capabilities. FOLLOW_MAX accepts the existing pool's capabilities
 * and does not expand them to QEMU's maximum.
 *
 * Returns: true on success; false on invalid input or incompatibility,
 * setting @errp without changing an existing configuration.
 */
bool riscv_ame_pool_configure_cpu(RISCVAMEBackendPoolCore *pool, uint64_t n,
                                  uint64_t u, uint32_t num_mregs,
                                  uint32_t num_accregs,
                                  AmeCapabilityPolicy policy,
                                  const AmeCapabilities *requirements,
                                  Error **errp);

/**
 * riscv_ame_pool_realize: Allocate and initialize a pool's physical backends
 * @pool: Initialized pool that has not been realized
 * @errp: Error destination
 *
 * Configured pools allocate slot, register and scratch storage. An
 * unconfigured pool realizes without slots and cannot satisfy acquisitions.
 *
 * Returns: true on success; false on invalid state, size overflow or allocation
 * failure, setting @errp.
 */
bool riscv_ame_pool_realize(RISCVAMEBackendPoolCore *pool, Error **errp);

/**
 * riscv_ame_pool_reset: Clear all ownership, waits and backend register state
 * @pool: Pool to reset; may be NULL or unrealized
 *
 * Call with attached harts quiesced. Keep geometry and capabilities unchanged.
 */
void riscv_ame_pool_reset(RISCVAMEBackendPoolCore *pool);

/**
 * riscv_ame_pool_destroy: Free a pool's storage and lock, then zero its fields
 * @pool: Pool to destroy; may be NULL, zero-initialized or partly initialized
 *
 * All harts must be detached and no waiter may remain. Safe to call again
 * on a destroyed pool.
 */
void riscv_ame_pool_destroy(RISCVAMEBackendPoolCore *pool);

/**
 * riscv_ame_state_attach: Attach hart bookkeeping to a configured pool
 * @state: Zero-initialized, unattached hart state
 * @pool: Configured pool, kept alive by the caller until state destruction
 * @errp: Error destination
 *
 * Set the binding to unowned and create a virtual-clock wait timer in system
 * mode. No backend is acquired. Linux-user permits only a private one-slot
 * pool with no other attached hart.
 *
 * Returns: true on success; false on invalid state, setting @errp.
 */
bool riscv_ame_state_attach(RISCVAMEState *state,
                            RISCVAMEBackendPoolCore *pool, Error **errp);

/**
 * riscv_ame_state_init: Create and attach an owned, private one-slot pool
 * @state: Zero-initialized, unattached hart state
 * @n: N, the square side in elements
 * @u: U, the M cell width in bits
 * @num_mregs: NM, the M register count
 * @num_accregs: NA, the Acc register count
 * @policy: CPU capability policy
 * @requirements: Explicit CPU allowlist, or NULL for FOLLOW_MAX
 * @errp: Error destination
 *
 * Configure and realize the private pool, but do not acquire its slot. Release
 * all owned resources with riscv_ame_state_destroy().
 *
 * Returns: true on success; false on invalid input or allocation failure,
 * setting @errp and freeing resources created by this call.
 */
bool riscv_ame_state_init(RISCVAMEState *state, uint64_t n, uint64_t u,
                          uint32_t num_mregs, uint32_t num_accregs,
                          AmeCapabilityPolicy policy,
                          const AmeCapabilities *requirements, Error **errp);

/**
 * riscv_ame_state_layout: Get the attached pool's register geometry
 * @state: Hart state; may be NULL or unattached
 *
 * Returns: a borrowed layout pointer; NULL if no pool is attached.
 */
const RISCVAMELayout *riscv_ame_state_layout(const RISCVAMEState *state);

/**
 * riscv_ame_state_set_wait_kick: Set the callback that wakes a waiting hart
 * @state: Hart state with no active wait
 * @kick: Wake callback outside the pool lock; NULL disables callbacks
 */
void riscv_ame_state_set_wait_kick(RISCVAMEState *state,
                                   void (*kick)(Object *hart));

/**
 * riscv_ame_acquire_descriptor_decode: Decode WAIT_MODE and BOUNDED_WAIT
 * @descriptor: ame.acquire descriptor, with reserved upper bits zero
 * @bounded_wait: Optional output; true for bounded wait, false for no-wait
 * @timeout_ns: Optional output; timeout in nanoseconds, zero for no-wait
 *
 * Returns: true for an accepted descriptor; false for a reserved mode or
 * bounded timeout class, leaving outputs unchanged.
 */
bool riscv_ame_acquire_descriptor_decode(uint64_t descriptor,
                                         bool *bounded_wait,
                                         int64_t *timeout_ns);

/**
 * riscv_ame_acquire: Acquire any free backend or resume a bounded acquire
 * @state: Attached hart state; calls for this state must be serialized
 * @hart: Hart identity used for ownership
 * @bounded_wait: Wait mode already decoded from @descriptor
 * @descriptor: Original descriptor, unchanged across retries
 * @timeout_ns: Decoded virtual-clock timeout, unchanged across retries
 * @new_grant: Required output; true only when a new, reset slot is granted
 *
 * Reacquiring an already owned slot succeeds without resetting it. WAIT
 * keeps the instruction pending: the caller suspends the hart and retries
 * after a wake. BUSY, TIMEOUT and INTERRUPTED are completed failures.
 * The caller must validate the descriptor before this call. Linux-user
 * private pools cannot contend and do not use the waiting path.
 *
 * Returns: the acquisition outcome.
 */
RISCVAMEAcquireResult riscv_ame_acquire(RISCVAMEState *state, Object *hart,
                                        bool bounded_wait,
                                        uint64_t descriptor,
                                        int64_t timeout_ns,
                                        bool *new_grant);

/**
 * riscv_ame_release: Release a hart's slot and wake pending acquirers
 * @state: Attached hart state; calls for this state must be serialized
 * @hart: Expected owner
 *
 * Register contents are reset on the next new grant, not on release.
 *
 * Returns: true if the owned slot was released; false if no matching owner.
 */
bool riscv_ame_release(RISCVAMEState *state, Object *hart);

/**
 * riscv_ame_wait_has_wake: Check whether a pending acquire should retry
 * @state: Hart state; may be NULL
 *
 * Returns: the atomic wake hint; true does not guarantee a free backend.
 */
bool riscv_ame_wait_has_wake(const RISCVAMEState *state);

/**
 * riscv_ame_wait_interrupt: End a pending acquire before taking an interrupt
 * @state: Hart state
 * @hart: Hart whose interrupt is being checked
 *
 * Mark a nonterminal wait INTERRUPTED, without replacing an existing timeout.
 * The caller retries ame.acquire to retire its result before taking the trap.
 *
 * Returns: true if this hart has an active wait and interrupt delivery must
 * be deferred; otherwise false.
 */
bool riscv_ame_wait_interrupt(RISCVAMEState *state, Object *hart);

/**
 * riscv_ame_wait_cancel: Remove a pending acquire without retiring a result
 * @state: Hart state; may be NULL or unattached
 *
 * Remove queue membership, cancel the timer and release the waiting hart's
 * reference. Keep the reusable timer allocated for subsequent waits.
 */
void riscv_ame_wait_cancel(RISCVAMEState *state);

/**
 * riscv_ame_wait_restore: Rebuild a saved bounded acquire in system mode
 * @state: Attached, unowned hart state with no active wait
 * @hart: Waiting hart; a reference is taken on success
 * @descriptor: Original bounded-wait descriptor
 * @terminal: Saved terminal result, or NONE for a still-pending wait
 * @remaining_ns: Remaining virtual-clock timeout; zero for a terminal result
 * @errp: Error destination
 *
 * Validate the saved timeout, insert the waiter and arm its timer if needed.
 * Set the wake hint so the instruction rescans the pool after restoration.
 *
 * Returns: true on success; false on inconsistent saved state, setting @errp.
 */
bool riscv_ame_wait_restore(RISCVAMEState *state, Object *hart,
                            uint64_t descriptor,
                            RISCVAMEWaitTerminal terminal,
                            int64_t remaining_ns, Error **errp);

/**
 * riscv_ame_state_owned: Check the hart's atomically published slot ownership
 * @state: Hart state; may be NULL or unattached
 * @hart: Expected owner; may be NULL
 *
 * Returns: true if the bound slot is owned by @hart; otherwise false.
 */
bool riscv_ame_state_owned(const RISCVAMEState *state, const Object *hart);

/**
 * riscv_ame_state_backend: Get the slot currently owned by a hart
 * @state: Hart state
 * @hart: Expected owner
 *
 * Caller must prevent concurrent release/reset while using the returned slot.
 *
 * Returns: a borrowed slot pointer; NULL if ownership does not match.
 */
RISCVAMEBackendSlot *riscv_ame_state_backend(RISCVAMEState *state,
                                              const Object *hart);

/**
 * riscv_ame_state_backend_const: Get read-only access to a hart's owned slot
 * @state: Hart state
 * @hart: Expected owner
 *
 * Caller must prevent concurrent release/reset while using the returned slot.
 *
 * Returns: a borrowed slot pointer; NULL if ownership does not match.
 */
const RISCVAMEBackendSlot *riscv_ame_state_backend_const(
    const RISCVAMEState *state, const Object *hart);

/**
 * riscv_ame_state_reset: Cancel a hart's wait and release its backend binding
 * @state: Hart state to reset; may be NULL
 *
 * Call with this hart quiesced. Reset a private pool's register state; do not
 * reset other harts' shared slots. Retain allocated scratch and timer.
 */
void riscv_ame_state_reset(RISCVAMEState *state);

/**
 * riscv_ame_state_destroy: Detach a hart and release its owned host resources
 * @state: Quiesced state; may be NULL, zeroed or partly initialized
 *
 * Cancel its wait, release its binding, free memory-access scratch and timer,
 * and destroy a private pool if present. A shared pool is not destroyed.
 * Zero all fields; safe to call again on a destroyed state.
 */
void riscv_ame_state_destroy(RISCVAMEState *state);

#endif /* TARGET_RISCV_AME_BACKEND_H */
