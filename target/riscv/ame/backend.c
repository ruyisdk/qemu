/*
 * RISC-V Attached Matrix Extension (AME) backend state
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/atomic.h"
#include "qemu/host-utils.h"
#include "qemu/units.h"
#include "qom/object.h"
#include "target/riscv/ame/backend.h"

/* CPU-free unit tests exercise the complete system-mode waiting path. */
#ifdef COMPILING_PER_TARGET
#ifdef CONFIG_USER_ONLY
#define AME_SYSTEM_MODE 0
#else
#define AME_SYSTEM_MODE 1
#endif
#else
#define AME_SYSTEM_MODE 1
#endif

typedef struct RISCVAMEWaitHandle {
    QEMUTimer *timer;
    Object *hart;
} RISCVAMEWaitHandle;

typedef struct RISCVAMEWakeTarget {
    void (*kick)(Object *hart);
    Object *hart;
} RISCVAMEWakeTarget;

static void riscv_ame_slot_reset(const RISCVAMELayout *layout,
                                 RISCVAMEBackendSlot *slot)
{
    size_t i;

    memset(slot->mregs, 0, layout->mregs_bytes);
    memset(slot->accs, 0, layout->accs_bytes);
    memset(slot->scratch, 0, slot->scratch_bytes);
    memset(slot->md, 0, sizeof(slot->md));
    memset(slot->ad, 0, sizeof(slot->ad));
    for (i = 0; i < ARRAY_SIZE(slot->md_dtype_id); i++) {
        slot->md_dtype_id[i] = AME_DTYPE_ID_INVALID;
    }
    for (i = 0; i < ARRAY_SIZE(slot->ad_dtype_id); i++) {
        slot->ad_dtype_id[i] = AME_DTYPE_ID_INVALID;
    }
    slot->scalar_dtype = 0;
    slot->scalar_dtype_id = AME_DTYPE_ID_INVALID;
    slot->status = 0;
}

static RISCVAMEWaitHandle riscv_ame_waiter_remove_locked(
    RISCVAMEState *state)
{
    RISCVAMEWaitHandle handle = {
        .timer = state->wait_timer,
        .hart = state->wait_hart,
    };

    g_assert(state->wait_active);
    QTAILQ_REMOVE(&state->pool->waiters, state, waiter_link);
    state->wait_hart = NULL;
    state->wait_descriptor = 0;
    state->wait_deadline_ns = 0;
    state->wait_generation = 0;
    state->wait_terminal = RISCV_AME_WAIT_NONE;
    state->wait_active = false;
    qatomic_set(&state->wait_wake, false);
    return handle;
}

static void riscv_ame_waiter_finish(RISCVAMEWaitHandle *handle)
{
#if AME_SYSTEM_MODE
    if (handle->timer != NULL) {
        timer_del(handle->timer);
    }
#endif
    if (handle->hart != NULL) {
        object_unref(handle->hart);
        handle->hart = NULL;
    }
}

static void riscv_ame_kick_waiter(void (*kick)(Object *hart), Object *hart)
{
    if (kick != NULL) {
        kick(hart);
    }
}

#if AME_SYSTEM_MODE
static void riscv_ame_wait_timeout(void *opaque)
{
    RISCVAMEState *state = opaque;
    RISCVAMEBackendPoolCore *pool = state->pool;
    void (*kick)(Object *hart) = NULL;
    Object *hart = NULL;

    if (pool == NULL || !pool->lock_initialized) {
        return;
    }
    qemu_mutex_lock(&pool->lock);
    if (state->wait_active &&
        state->wait_terminal == RISCV_AME_WAIT_NONE) {
        if (state->wait_generation == pool->generation) {
            state->wait_terminal = RISCV_AME_WAIT_TIMEOUT;
        }
        qatomic_set(&state->wait_wake, true);
        hart = state->wait_hart;
        kick = state->wait_kick;
        g_assert(hart != NULL);
        object_ref(hart);
    }
    qemu_mutex_unlock(&pool->lock);

    if (hart != NULL) {
        riscv_ame_kick_waiter(kick, hart);
        object_unref(hart);
    }
}
#endif

bool riscv_ame_pool_init(RISCVAMEBackendPoolCore *pool, uint32_t num_slots,
                         bool enabled, Error **errp)
{
    if (pool == NULL) {
        error_setg(errp, "Ztt backend pool must be specified");
        return false;
    }
    if (pool->lock_initialized || pool->configured || pool->realized ||
        pool->slots != NULL) {
        error_setg(errp, "Ztt backend pool is already initialized");
        return false;
    }
    if (num_slots == 0 || num_slots > INT32_MAX) {
        error_setg(errp, "Ztt backend pool slot count must be in [1, %d]",
                   INT32_MAX);
        return false;
    }

    qemu_mutex_init(&pool->lock);
    QTAILQ_INIT(&pool->waiters);
    pool->lock_initialized = true;
    pool->num_slots = num_slots;
    pool->enabled = enabled;
    return true;
}

static bool pool_configure(RISCVAMEBackendPoolCore *pool, uint64_t n,
                             uint64_t u, uint32_t num_mregs,
                             uint32_t num_accregs,
                             const AmeCapabilities *capabilities,
                             RISCVAMEPoolConfigSource source, Error **errp)
{
    RISCVAMELayout layout;

    if (pool == NULL || !pool->lock_initialized) {
        error_setg(errp, "Ztt backend pool is not initialized");
        return false;
    }
    if (pool->configured || pool->realized) {
        error_setg(errp, "Ztt backend pool configuration is already frozen");
        return false;
    }
    if (!riscv_ame_compute_layout(n, u, num_mregs, num_accregs,
                                  capabilities, &layout, errp)) {
        return false;
    }

    riscv_ame_capabilities_copy(&pool->capabilities, capabilities);
    pool->layout = layout;
    pool->config_source = source;
    pool->configured = true;
    return true;
}

bool riscv_ame_pool_configure(RISCVAMEBackendPoolCore *pool, uint64_t n,
                              uint64_t u, uint32_t num_mregs,
                              uint32_t num_accregs,
                              const AmeCapabilities *capabilities,
                              Error **errp)
{
    return pool_configure(pool, n, u, num_mregs, num_accregs, capabilities,
                          RISCV_AME_POOL_EXPLICIT, errp);
}

bool riscv_ame_pool_configure_cpu(RISCVAMEBackendPoolCore *pool, uint64_t n,
                                  uint64_t u, uint32_t num_mregs,
                                  uint32_t num_accregs,
                                  AmeCapabilityPolicy policy,
                                  const AmeCapabilities *requirements,
                                  Error **errp)
{
    const AmeCapabilities *initial;
    RISCVAMELayout layout;

    if (pool == NULL || !pool->lock_initialized) {
        error_setg(errp, "Ztt backend pool is not initialized");
        return false;
    }
    initial = riscv_ame_capabilities_for_cpu(policy, requirements, errp);
    if (!initial) {
        return false;
    }
    if (!pool->configured) {
        return pool_configure(pool, n, u, num_mregs, num_accregs, initial,
                              RISCV_AME_POOL_AUTO, errp);
    }
    if (policy == AME_CAP_POLICY_EXPLICIT &&
        !riscv_ame_capabilities_equal(requirements, &pool->capabilities)) {
        error_setg(errp, "Ztt fixed CPU capabilities do not match pool");
        return false;
    }
    if (!riscv_ame_compute_layout(n, u, num_mregs, num_accregs,
                                  &pool->capabilities, &layout, errp)) {
        return false;
    }
    if (!riscv_ame_layout_equal(&layout, &pool->layout)) {
        error_setg(errp, "Ztt CPU geometry does not match backend pool");
        return false;
    }
    return true;
}

bool riscv_ame_pool_realize(RISCVAMEBackendPoolCore *pool, Error **errp)
{
    RISCVAMEBackendSlot *slots;
    uint8_t *mregs;
    uint8_t *accs;
    uint8_t *scratch;
    size_t mregs_bytes;
    size_t accs_bytes;
    size_t scratch_bytes, slot_scratch_bytes;
    uint32_t i;

    if (pool == NULL || !pool->lock_initialized) {
        error_setg(errp, "Ztt backend pool is not initialized");
        return false;
    }
    if (pool->realized) {
        error_setg(errp, "Ztt backend pool is already realized");
        return false;
    }
    if (!pool->configured) {
        pool->realized = true;
        return true;
    }
    slot_scratch_bytes = MAX(pool->layout.mregs_bytes, pool->layout.accs_bytes);
    if (__builtin_mul_overflow(pool->layout.mregs_bytes, pool->num_slots,
                               &mregs_bytes) ||
        __builtin_mul_overflow(pool->layout.accs_bytes, pool->num_slots,
                               &accs_bytes) ||
        __builtin_mul_overflow(slot_scratch_bytes, pool->num_slots,
                               &scratch_bytes)) {
        error_setg(errp, "Ztt total backend size calculation overflow");
        return false;
    }

    slots = g_try_new0(RISCVAMEBackendSlot, pool->num_slots);
    mregs = g_try_malloc0(mregs_bytes);
    accs = g_try_malloc0(accs_bytes);
    scratch = g_try_malloc0(scratch_bytes);
    if (slots == NULL || mregs == NULL || accs == NULL || scratch == NULL) {
        g_free(slots);
        g_free(mregs);
        g_free(accs);
        g_free(scratch);
        error_setg(errp, "Unable to allocate Ztt backend state");
        return false;
    }

    for (i = 0; i < pool->num_slots; i++) {
        slots[i].mregs = mregs + i * pool->layout.mregs_bytes;
        slots[i].accs = accs + i * pool->layout.accs_bytes;
        slots[i].scratch = scratch + i * slot_scratch_bytes;
        slots[i].scratch_bytes = slot_scratch_bytes;
        riscv_ame_slot_reset(&pool->layout, &slots[i]);
        slots[i].state = RISCV_AME_BACKEND_FREE;
    }
    pool->slots = slots;
    pool->mregs = mregs;
    pool->accs = accs;
    pool->scratch = scratch;
    pool->realized = true;
    return true;
}

void riscv_ame_pool_reset(RISCVAMEBackendPoolCore *pool)
{
    g_autoptr(GArray) cancelled = NULL;
    RISCVAMEState *waiter;
    uint32_t i;

    if (pool == NULL || !pool->lock_initialized || !pool->realized ||
        pool->slots == NULL) {
        return;
    }

    qemu_mutex_lock(&pool->lock);
    while ((waiter = QTAILQ_FIRST(&pool->waiters)) != NULL) {
        RISCVAMEWaitHandle handle;

        if (cancelled == NULL) {
            cancelled = g_array_new(false, false,
                                    sizeof(RISCVAMEWaitHandle));
        }
        handle = riscv_ame_waiter_remove_locked(waiter);
        g_array_append_val(cancelled, handle);
    }
    for (i = 0; i < pool->num_slots; i++) {
        RISCVAMEBackendSlot *slot = &pool->slots[i];

        if (slot->bound_state != NULL &&
            slot->bound_state->active_slot >= 0 &&
            (uint32_t)slot->bound_state->active_slot == i) {
            slot->bound_state->active_slot = -1;
        }
        qatomic_store_release(&slot->owner, NULL);
        slot->bound_state = NULL;
        slot->initializer = NULL;
        slot->state = RISCV_AME_BACKEND_INITIALIZING;
    }
    qemu_mutex_unlock(&pool->lock);

    if (cancelled != NULL) {
        for (i = 0; i < cancelled->len; i++) {
            riscv_ame_waiter_finish(
                &g_array_index(cancelled, RISCVAMEWaitHandle, i));
        }
    }
    for (i = 0; i < pool->num_slots; i++) {
        riscv_ame_slot_reset(&pool->layout, &pool->slots[i]);
    }

    qemu_mutex_lock(&pool->lock);
    for (i = 0; i < pool->num_slots; i++) {
        pool->slots[i].state = RISCV_AME_BACKEND_FREE;
    }
    pool->generation++;
    qemu_mutex_unlock(&pool->lock);
}

void riscv_ame_pool_destroy(RISCVAMEBackendPoolCore *pool)
{
    if (pool == NULL) {
        return;
    }

    g_assert(pool->num_attached_harts == 0);
    g_assert(!pool->lock_initialized || QTAILQ_EMPTY(&pool->waiters));
    g_free(pool->slots);
    g_free(pool->mregs);
    g_free(pool->accs);
    g_free(pool->scratch);
    riscv_ame_capabilities_clear(&pool->capabilities);
    if (pool->lock_initialized) {
        qemu_mutex_destroy(&pool->lock);
    }
    *pool = (RISCVAMEBackendPoolCore) { 0 };
}

bool riscv_ame_state_attach(RISCVAMEState *state,
                            RISCVAMEBackendPoolCore *pool, Error **errp)
{
    QEMUTimer *wait_timer = NULL;

    if (state == NULL) {
        error_setg(errp, "Ztt hart state must be specified");
        return false;
    }
    if (state->pool != NULL) {
        error_setg(errp, "Ztt hart state is already initialized");
        return false;
    }
    if (pool == NULL || !pool->lock_initialized || !pool->configured) {
        error_setg(errp, "Ztt hart requires a configured backend pool");
        return false;
    }

#if AME_SYSTEM_MODE
    wait_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL,
                              riscv_ame_wait_timeout, state);
#else
    /* A linux-user hart has its own slot and can never need to wait. */
    g_assert(state->private_pool == pool && pool->num_slots == 1);
    g_assert(pool->num_attached_harts == 0);
#endif

    qemu_mutex_lock(&pool->lock);
    if (pool->num_attached_harts == UINT32_MAX) {
        qemu_mutex_unlock(&pool->lock);
#if AME_SYSTEM_MODE
        timer_free(wait_timer);
#endif
        error_setg(errp, "Too many harts attached to Ztt backend pool");
        return false;
    }
    pool->num_attached_harts++;
    qemu_mutex_unlock(&pool->lock);
    state->pool = pool;
    state->active_slot = -1;
    state->wait_timer = wait_timer;
    return true;
}

bool riscv_ame_state_init(RISCVAMEState *state, uint64_t n, uint64_t u,
                          uint32_t num_mregs, uint32_t num_accregs,
                          AmeCapabilityPolicy policy,
                          const AmeCapabilities *requirements, Error **errp)
{
    RISCVAMEBackendPoolCore *pool;

    if (state == NULL) {
        error_setg(errp, "Ztt hart state must be specified");
        return false;
    }
    if (state->pool != NULL || state->private_pool != NULL) {
        error_setg(errp, "Ztt hart state is already initialized");
        return false;
    }

    pool = g_try_new0(RISCVAMEBackendPoolCore, 1);
    if (pool == NULL) {
        error_setg(errp, "Unable to allocate private Ztt backend pool");
        return false;
    }
    state->private_pool = pool;
    if (!riscv_ame_pool_init(pool, 1, true, errp) ||
        !riscv_ame_pool_configure_cpu(pool, n, u, num_mregs, num_accregs,
                                      policy, requirements, errp) ||
        !riscv_ame_pool_realize(pool, errp) ||
        !riscv_ame_state_attach(state, pool, errp)) {
        riscv_ame_state_destroy(state);
        return false;
    }
    return true;
}

const RISCVAMELayout *riscv_ame_state_layout(const RISCVAMEState *state)
{
    return state != NULL && state->pool != NULL ? &state->pool->layout : NULL;
}

void riscv_ame_state_set_wait_kick(RISCVAMEState *state,
                                   void (*kick)(Object *hart))
{
    g_assert(state != NULL && !state->wait_active);
    state->wait_kick = kick;
}

bool riscv_ame_acquire_descriptor_decode(uint64_t descriptor,
                                         bool *bounded_wait,
                                         int64_t *timeout_ns)
{
    static const uint16_t timeout_ms[] = { 1, 10, 100, 1000 };
    unsigned wait_mode = descriptor & 3;
    unsigned timeout_class = (descriptor >> 2) & 0xf;
    bool bounded = wait_mode == 1;

    if (descriptor >> 6 != 0 || wait_mode > 1 ||
        (bounded && timeout_class > 3 && timeout_class != 0xf)) {
        return false;
    }
    if (bounded_wait != NULL) {
        *bounded_wait = bounded;
    }
    if (timeout_ns != NULL) {
        *timeout_ns = bounded ?
            (timeout_class == 0xf ? 1000 : timeout_ms[timeout_class]) *
            SCALE_MS : 0;
    }
    return true;
}

bool riscv_ame_state_owned(const RISCVAMEState *state, const Object *hart)
{
    const RISCVAMEBackendSlot *slot;

    if (state == NULL || state->pool == NULL || hart == NULL ||
        state->active_slot < 0 ||
        (uint32_t)state->active_slot >= state->pool->num_slots ||
        state->pool->slots == NULL) {
        return false;
    }
    slot = &state->pool->slots[state->active_slot];
    return qatomic_load_acquire(&slot->owner) == hart;
}

RISCVAMEBackendSlot *riscv_ame_state_backend(RISCVAMEState *state,
                                              const Object *hart)
{
    return riscv_ame_state_owned(state, hart) ?
           &state->pool->slots[state->active_slot] : NULL;
}

const RISCVAMEBackendSlot *riscv_ame_state_backend_const(
    const RISCVAMEState *state, const Object *hart)
{
    return riscv_ame_state_owned(state, hart) ?
           &state->pool->slots[state->active_slot] : NULL;
}

bool riscv_ame_wait_has_wake(const RISCVAMEState *state)
{
    return state != NULL && qatomic_read(&state->wait_wake);
}

bool riscv_ame_wait_interrupt(RISCVAMEState *state, Object *hart)
{
    RISCVAMEBackendPoolCore *pool;
    bool defer_interrupt = false;

    if (state == NULL || hart == NULL || state->pool == NULL) {
        return false;
    }
    pool = state->pool;
    if (!pool->lock_initialized) {
        return false;
    }

    qemu_mutex_lock(&pool->lock);
    if (state->wait_active && state->wait_hart == hart) {
        /*
         * Retire the terminal acquire result before delivering the interrupt.
         */
        defer_interrupt = true;
        if (state->wait_terminal == RISCV_AME_WAIT_NONE) {
            state->wait_terminal = RISCV_AME_WAIT_INTERRUPTED;
            qatomic_set(&state->wait_wake, true);
        }
    }
    qemu_mutex_unlock(&pool->lock);
    return defer_interrupt;
}

void riscv_ame_wait_cancel(RISCVAMEState *state)
{
    RISCVAMEBackendPoolCore *pool;
    RISCVAMEWaitHandle handle = { 0 };

    if (state == NULL || state->pool == NULL) {
        return;
    }
    pool = state->pool;
    if (!pool->lock_initialized) {
        return;
    }

    qemu_mutex_lock(&pool->lock);
    if (state->wait_active) {
        handle = riscv_ame_waiter_remove_locked(state);
    }
    state->wait_terminal = RISCV_AME_WAIT_NONE;
    qemu_mutex_unlock(&pool->lock);
    riscv_ame_waiter_finish(&handle);
}

RISCVAMEAcquireResult riscv_ame_acquire(RISCVAMEState *state, Object *hart,
                                        bool bounded_wait,
                                        uint64_t descriptor,
                                        int64_t timeout_ns,
                                        bool *new_grant)
{
    RISCVAMEBackendPoolCore *pool;
    RISCVAMEBackendSlot *slot = NULL;
    RISCVAMEWaitHandle wait_handle = { 0 };
    uint32_t slot_index = 0;
    uint32_t i;
#if AME_SYSTEM_MODE
    RISCVAMEAcquireResult terminal_result = RISCV_AME_ACQUIRE_BUSY;
    bool arm_timer = false;
    int64_t now;
#endif

    if (new_grant == NULL) {
        return RISCV_AME_ACQUIRE_UNSUPPORTED;
    }
    *new_grant = false;
    if (state == NULL || hart == NULL || state->pool == NULL ||
        (bounded_wait && timeout_ns < 0)) {
        return RISCV_AME_ACQUIRE_UNSUPPORTED;
    }
    pool = state->pool;
    if (!pool->configured || !pool->realized || pool->slots == NULL) {
        return RISCV_AME_ACQUIRE_UNSUPPORTED;
    }
    if (!pool->enabled) {
        return RISCV_AME_ACQUIRE_DISABLED;
    }
    if (riscv_ame_state_owned(state, hart)) {
        return RISCV_AME_ACQUIRE_GRANTED;
    }
    if (state->active_slot >= 0) {
        return RISCV_AME_ACQUIRE_UNSUPPORTED;
    }

    qemu_mutex_lock(&pool->lock);
#if AME_SYSTEM_MODE
    if (state->wait_active) {
        g_assert(bounded_wait);
        g_assert(state->wait_hart == hart);
        g_assert(state->wait_descriptor == descriptor);
        now = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
        if (state->wait_terminal == RISCV_AME_WAIT_NONE &&
            state->wait_generation == pool->generation &&
            now >= state->wait_deadline_ns) {
            state->wait_terminal = RISCV_AME_WAIT_TIMEOUT;
            qatomic_set(&state->wait_wake, true);
        }
        if (state->wait_terminal != RISCV_AME_WAIT_NONE) {
            terminal_result =
                state->wait_terminal == RISCV_AME_WAIT_TIMEOUT ?
                RISCV_AME_ACQUIRE_TIMEOUT : RISCV_AME_ACQUIRE_INTERRUPTED;
            wait_handle = riscv_ame_waiter_remove_locked(state);
            qemu_mutex_unlock(&pool->lock);
            riscv_ame_waiter_finish(&wait_handle);
            return terminal_result;
        }
    }
#else
    g_assert(!state->wait_active);
#endif

    for (i = 0; i < pool->num_slots; i++) {
        if (pool->slots[i].state == RISCV_AME_BACKEND_FREE) {
            slot = &pool->slots[i];
            slot_index = i;
            slot->state = RISCV_AME_BACKEND_INITIALIZING;
            slot->initializer = hart;
            break;
        }
    }
#if AME_SYSTEM_MODE
    if (slot != NULL && state->wait_active) {
        wait_handle = riscv_ame_waiter_remove_locked(state);
    } else if (slot == NULL && bounded_wait) {
        now = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
        if (!state->wait_active) {
            state->wait_active = true;
            state->wait_hart = hart;
            state->wait_descriptor = descriptor;
            state->wait_deadline_ns =
                timeout_ns > INT64_MAX - now ? INT64_MAX : now + timeout_ns;
            state->wait_generation = pool->generation;
            state->wait_terminal = RISCV_AME_WAIT_NONE;
            object_ref(hart);
            QTAILQ_INSERT_TAIL(&pool->waiters, state, waiter_link);
            arm_timer = true;
        } else if (now >= state->wait_deadline_ns) {
            terminal_result = RISCV_AME_ACQUIRE_TIMEOUT;
            wait_handle = riscv_ame_waiter_remove_locked(state);
            qemu_mutex_unlock(&pool->lock);
            riscv_ame_waiter_finish(&wait_handle);
            return terminal_result;
        } else {
            state->wait_generation = pool->generation;
        }
        qatomic_set(&state->wait_wake, false);
    }
#else
    g_assert(slot != NULL);
#endif
    qemu_mutex_unlock(&pool->lock);

#if AME_SYSTEM_MODE
    if (slot == NULL) {
        if (!bounded_wait) {
            return RISCV_AME_ACQUIRE_BUSY;
        }
        if (arm_timer) {
            timer_mod_ns(state->wait_timer, state->wait_deadline_ns);
        }
        return RISCV_AME_ACQUIRE_WAIT;
    }
#endif
    riscv_ame_waiter_finish(&wait_handle);

    /* Backing initialization is deliberately outside the pool lock. */
    riscv_ame_slot_reset(&pool->layout, slot);

    qemu_mutex_lock(&pool->lock);
    g_assert(slot->state == RISCV_AME_BACKEND_INITIALIZING);
    g_assert(slot->initializer == hart);
    state->active_slot = slot_index;
    slot->bound_state = state;
    slot->state = RISCV_AME_BACKEND_OWNED;
    slot->initializer = NULL;
    qatomic_store_release(&slot->owner, hart);
    qemu_mutex_unlock(&pool->lock);
    *new_grant = true;
    return RISCV_AME_ACQUIRE_GRANTED;
}

bool riscv_ame_release(RISCVAMEState *state, Object *hart)
{
    RISCVAMEBackendPoolCore *pool;
    RISCVAMEBackendSlot *slot;
    RISCVAMEState *waiter;
    g_autoptr(GArray) wake_targets = NULL;

    if (state == NULL || hart == NULL || state->pool == NULL ||
        state->active_slot < 0) {
        return false;
    }
    pool = state->pool;
    if (pool->slots == NULL ||
        (uint32_t)state->active_slot >= pool->num_slots) {
        return false;
    }

    qemu_mutex_lock(&pool->lock);
    slot = &pool->slots[state->active_slot];
    if (qatomic_load_acquire(&slot->owner) != hart ||
        slot->bound_state != state) {
        qemu_mutex_unlock(&pool->lock);
        return false;
    }
    qatomic_store_release(&slot->owner, NULL);
    slot->bound_state = NULL;
    slot->state = RISCV_AME_BACKEND_FREE;
    state->active_slot = -1;
    pool->generation++;

    QTAILQ_FOREACH(waiter, &pool->waiters, waiter_link) {
        RISCVAMEWakeTarget target;

        if (waiter->wait_terminal != RISCV_AME_WAIT_NONE) {
            continue;
        }
        qatomic_set(&waiter->wait_wake, true);
        if (wake_targets == NULL) {
            wake_targets = g_array_new(false, false,
                                       sizeof(RISCVAMEWakeTarget));
        }
        target = (RISCVAMEWakeTarget) {
            .kick = waiter->wait_kick,
            .hart = waiter->wait_hart,
        };
        object_ref(target.hart);
        g_array_append_val(wake_targets, target);
    }
    qemu_mutex_unlock(&pool->lock);

    if (wake_targets != NULL) {
        unsigned i;

        for (i = 0; i < wake_targets->len; i++) {
            RISCVAMEWakeTarget *target = &g_array_index(
                wake_targets, RISCVAMEWakeTarget, i);

            riscv_ame_kick_waiter(target->kick, target->hart);
            object_unref(target->hart);
        }
    }
    return true;
}

void riscv_ame_state_reset(RISCVAMEState *state)
{
    RISCVAMEBackendPoolCore *pool;

    if (state == NULL) {
        return;
    }

    pool = state->pool;
    riscv_ame_wait_cancel(state);
    if (pool != NULL && state->active_slot >= 0 && pool->slots != NULL &&
        (uint32_t)state->active_slot < pool->num_slots) {
        Object *owner = qatomic_load_acquire(
            &pool->slots[state->active_slot].owner);

        if (owner != NULL) {
            riscv_ame_release(state, owner);
        }
    }
    state->active_slot = -1;
    state->memory.fault_address = 0;
    state->memory.fault_active = false;
    if (state->pool != NULL && state->pool == state->private_pool) {
        riscv_ame_pool_reset(state->pool);
    }
}

void riscv_ame_state_destroy(RISCVAMEState *state)
{
    RISCVAMEBackendPoolCore *pool;
    RISCVAMEBackendPoolCore *private_pool;

    if (state == NULL) {
        return;
    }

    pool = state->pool;
    private_pool = state->private_pool;
    riscv_ame_state_reset(state);
    g_free(state->memory.chunks);
#if AME_SYSTEM_MODE
    if (state->wait_timer != NULL) {
        timer_free(state->wait_timer);
    }
#endif
    if (pool != NULL) {
        qemu_mutex_lock(&pool->lock);
        g_assert(pool->num_attached_harts != 0);
        pool->num_attached_harts--;
        qemu_mutex_unlock(&pool->lock);
    }
    *state = (RISCVAMEState) { 0 };

    if (private_pool != NULL) {
        riscv_ame_pool_destroy(private_pool);
        g_free(private_pool);
    }
}
