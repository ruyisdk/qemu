/*
 * RISC-V Attached Matrix Extension migration state
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "cpu.h"
#include "qemu/atomic.h"
#include "qemu/bswap.h"
#include "qemu/error-report.h"
#include "migration/qemu-file-types.h"
#include "target/riscv/ame/cap-wire.h"
#include "target/riscv/ame/pool.h"
#include "target/riscv/ame/migration.h"

static int ame_migration_put_manifest(QEMUFile *f,
                                      const AmeCapabilities *capabilities)
{
    Error *err = NULL;
    g_autoptr(GByteArray) manifest =
        riscv_ame_capabilities_manifest(capabilities, &err);

    if (manifest == NULL) {
        error_report_err(err);
        return -EINVAL;
    }
    qemu_put_buffer(f, manifest->data, manifest->len);
    return 0;
}

static int ame_migration_get_manifest(QEMUFile *f,
                                      const AmeCapabilities *capabilities)
{
    Error *err = NULL;
    g_autoptr(GByteArray) manifest =
        riscv_ame_capabilities_manifest(capabilities, &err);
    uint8_t buffer[4096];
    size_t offset = 0;

    if (manifest == NULL) {
        error_report_err(err);
        return -EINVAL;
    }
    /*
     * Both ends use the canonical encoding. All sizes come from our frozen
     * configuration, never from incoming counts. This also rejects reordered,
     * duplicate and reserved fields without constructing an incoming set.
     */
    while (offset < manifest->len) {
        size_t count = MIN(sizeof(buffer), manifest->len - offset);

        if (qemu_get_buffer(f, buffer, count) != count ||
            memcmp(buffer, manifest->data + offset, count) != 0) {
            error_report("RISC-V Ztt migration capability manifest mismatch");
            return -EINVAL;
        }
        offset += count;
    }
    return 0;
}

static bool ame_migration_dtype_valid(const AmeCapabilities *capabilities,
                                      uint32_t encoding, uint16_t *id)
{
    const AmeDTypeDesc *dtype;

    if (encoding == 0) {
        *id = AME_DTYPE_ID_INVALID;
        return true;
    }
    dtype = riscv_ame_dtype_by_encoding(encoding);
    if (dtype == NULL ||
        !riscv_ame_dtype_set_has(&capabilities->dtypes, dtype->id)) {
        return false;
    }
    *id = dtype->id;
    return true;
}

static uint16_t ame_migration_scalar_dtype_id(
    const AmeCapabilities *capabilities,
    uint32_t encoding)
{
    const AmeDTypeDesc *dtype = riscv_ame_dtype_by_encoding(encoding);

    if (dtype == NULL ||
        !riscv_ame_dtype_set_has(&capabilities->dtypes, dtype->id)) {
        return AME_DTYPE_ID_INVALID;
    }
    return dtype->id;
}

static bool ame_migration_m_dtype_valid(
    const AmeCapabilities *capabilities,
    const RISCVAMELayout *layout,
    uint32_t index, uint32_t encoding, uint16_t *id)
{
    const AmeDTypeDesc *dtype;
    uint32_t group_size;

    if (!ame_migration_dtype_valid(capabilities, encoding, id) ||
        encoding == 0) {
        return encoding == 0;
    }
    dtype = riscv_ame_dtype_by_id(*id);
    g_assert(dtype != NULL);
    group_size = DIV_ROUND_UP(dtype->bits, layout->u);
    return index % group_size == 0 &&
           group_size <= layout->num_mregs - index;
}

typedef struct AmeMigrationSlot {
    RISCVAMEBackendSlot backend;
    uint64_t owner_hartid;
    RISCVCPU *owner;
    bool occupied;
} AmeMigrationSlot;

static void ame_migration_slots_free(AmeMigrationSlot *slots,
                                     uint32_t num_slots)
{
    uint32_t i;

    if (slots == NULL) {
        return;
    }
    for (i = 0; i < num_slots; i++) {
        g_free(slots[i].backend.mregs);
        g_free(slots[i].backend.accs);
    }
    g_free(slots);
}

static bool ame_migration_slot_dtypes_valid(
    const AmeCapabilities *capabilities,
    const RISCVAMELayout *layout,
    AmeMigrationSlot *slot)
{
    size_t i;

    memset(slot->backend.md_dtype_id, 0xff,
           sizeof(slot->backend.md_dtype_id));
    memset(slot->backend.ad_dtype_id, 0xff,
           sizeof(slot->backend.ad_dtype_id));
    for (i = 0; i < layout->num_mregs; i++) {
        if (!ame_migration_m_dtype_valid(capabilities, layout, i,
                                         slot->backend.md[i],
                                         &slot->backend.md_dtype_id[i])) {
            return false;
        }
    }
    for (i = 0; i < layout->num_accregs; i++) {
        if (!ame_migration_dtype_valid(capabilities, slot->backend.ad[i],
                                       &slot->backend.ad_dtype_id[i])) {
            return false;
        }
    }
    /*
     * amestype is an unconstrained read/write CSR.  Unsupported raw
     * encodings are architectural state: instructions consume the derived
     * invalid ID and report UN, but migration must preserve the CSR value.
     */
    slot->backend.scalar_dtype_id = ame_migration_scalar_dtype_id(
        capabilities, slot->backend.scalar_dtype);
    return true;
}

static int ame_pool_migration_put(QEMUFile *f, void *opaque, size_t size,
                                  const VMStateField *field,
                                  JSONWriter *vmdesc)
{
    RISCVAMEBackendPool *pool = container_of(
        opaque, RISCVAMEBackendPool, migration_anchor);
    RISCVAMEBackendPoolCore *core = &pool->core;
    const RISCVAMELayout *layout = &core->layout;
    uint32_t i;

    qemu_put_byte(f, core->enabled);
    qemu_put_byte(f, core->configured);
    qemu_put_be32(f, core->num_slots);
    if (!core->configured) {
        return 0;
    }

    qemu_put_be64(f, layout->n);
    qemu_put_be64(f, layout->u);
    qemu_put_byte(f, layout->num_mregs);
    qemu_put_byte(f, layout->num_accregs);
    if (ame_migration_put_manifest(f, &core->capabilities) < 0) {
        return -EINVAL;
    }

    for (i = 0; i < core->num_slots; i++) {
        RISCVAMEBackendSlot *slot = &core->slots[i];
        Object *owner_obj = qatomic_load_acquire(&slot->owner);
        RISCVCPU *owner = owner_obj != NULL &&
                          object_dynamic_cast(owner_obj, TYPE_RISCV_CPU) ?
                          RISCV_CPU(owner_obj) : NULL;
        bool occupied = slot->state == RISCV_AME_BACKEND_OWNED;
        size_t j;

        if (slot->state == RISCV_AME_BACKEND_INITIALIZING ||
            (!core->enabled && occupied) ||
            occupied != (owner_obj != NULL) ||
            (occupied && owner == NULL) ||
            (occupied &&
             (slot->bound_state != &owner->ame ||
              !owner->cfg.ext_ztt || owner->ame.pool != core ||
              (uint32_t)owner->ame.active_slot != i ||
              owner->ame.wait_active ||
              !riscv_cpu_ame_enabled(&owner->env))) ||
            (!occupied && slot->bound_state != NULL)) {
            error_report("Invalid live RISC-V Ztt backend state");
            return -EINVAL;
        }
        qemu_put_byte(f, occupied);
        if (!occupied) {
            continue;
        }
        qemu_put_be64(f, owner->env.mhartid);
        for (j = 0; j < layout->num_mregs; j++) {
            qemu_put_be32(f, slot->md[j]);
        }
        for (j = 0; j < layout->num_accregs; j++) {
            qemu_put_be32(f, slot->ad[j]);
        }
        qemu_put_be32(f, slot->scalar_dtype);
        qemu_put_be32(f, slot->status);
        qemu_put_buffer(f, slot->mregs, layout->mregs_bytes);
        qemu_put_buffer(f, slot->accs, layout->accs_bytes);
    }
    return 0;
}

static int ame_pool_migration_get(QEMUFile *f, void *opaque, size_t size,
                                  const VMStateField *field)
{
    RISCVAMEBackendPool *pool = container_of(
        opaque, RISCVAMEBackendPool, migration_anchor);
    RISCVAMEBackendPoolCore *core = &pool->core;
    const RISCVAMELayout *layout = &core->layout;
    g_autoptr(GHashTable) owners = g_hash_table_new(g_direct_hash,
                                                    g_direct_equal);
    AmeMigrationSlot *slots = NULL;
    uint8_t enabled, configured;
    uint32_t num_slots, i;
    int ret = -EINVAL;

    enabled = qemu_get_byte(f);
    configured = qemu_get_byte(f);
    num_slots = qemu_get_be32(f);
    if (enabled > 1 || configured > 1 || enabled != core->enabled ||
        configured != core->configured || num_slots != core->num_slots) {
        error_report("RISC-V Ztt backend pool configuration mismatch");
        return -EINVAL;
    }
    if (!configured) {
        return qemu_file_get_error(f) < 0 ? -EINVAL : 0;
    }
    if (qemu_get_be64(f) != layout->n ||
        qemu_get_be64(f) != layout->u ||
        qemu_get_byte(f) != layout->num_mregs ||
        qemu_get_byte(f) != layout->num_accregs) {
        error_report("RISC-V Ztt backend layout mismatch");
        return -EINVAL;
    }
    if (ame_migration_get_manifest(f, &core->capabilities) < 0) {
        return -EINVAL;
    }

    slots = g_try_new0(AmeMigrationSlot, num_slots);
    if (slots == NULL) {
        error_report("Unable to allocate RISC-V Ztt migration staging");
        return -ENOMEM;
    }
    for (i = 0; i < num_slots; i++) {
        AmeMigrationSlot *incoming = &slots[i];
        CPUState *cs;
        uint8_t occupied;
        size_t j;

        incoming->backend.mregs = g_try_malloc0(layout->mregs_bytes);
        incoming->backend.accs = g_try_malloc0(layout->accs_bytes);
        if (incoming->backend.mregs == NULL ||
            incoming->backend.accs == NULL) {
            error_report("Unable to allocate RISC-V Ztt migration backing");
            ret = -ENOMEM;
            goto out;
        }
        memset(incoming->backend.md_dtype_id, 0xff,
               sizeof(incoming->backend.md_dtype_id));
        memset(incoming->backend.ad_dtype_id, 0xff,
               sizeof(incoming->backend.ad_dtype_id));
        incoming->backend.scalar_dtype_id = AME_DTYPE_ID_INVALID;

        occupied = qemu_get_byte(f);
        if (occupied > 1) {
            goto invalid_state;
        }
        incoming->occupied = occupied;
        if (!incoming->occupied) {
            continue;
        }
        if (!enabled) {
            goto invalid_state;
        }
        incoming->owner_hartid = qemu_get_be64(f);
        for (j = 0; j < layout->num_mregs; j++) {
            incoming->backend.md[j] = qemu_get_be32(f);
        }
        for (j = 0; j < layout->num_accregs; j++) {
            incoming->backend.ad[j] = qemu_get_be32(f);
        }
        incoming->backend.scalar_dtype = qemu_get_be32(f);
        incoming->backend.status = qemu_get_be32(f);
        if (incoming->backend.status & ~AME_STATUS_UN ||
            !ame_migration_slot_dtypes_valid(&core->capabilities, layout,
                                             incoming)) {
            goto invalid_state;
        }

        if (qemu_get_buffer(f, incoming->backend.mregs,
                            layout->mregs_bytes) != layout->mregs_bytes ||
            qemu_get_buffer(f, incoming->backend.accs,
                            layout->accs_bytes) != layout->accs_bytes) {
            goto invalid_state;
        }

        cs = cpu_by_arch_id(incoming->owner_hartid);
        if (cs == NULL || !object_dynamic_cast(OBJECT(cs), TYPE_RISCV_CPU)) {
            goto invalid_state;
        }
        incoming->owner = RISCV_CPU(cs);
        if (!incoming->owner->cfg.ext_ztt ||
            incoming->owner->ame.pool != core ||
            incoming->owner->ame.wait_active ||
            incoming->owner->ame.active_slot >= 0 ||
            g_hash_table_contains(owners, incoming->owner)) {
            goto invalid_state;
        }
        /*
         * A private pool is a CPU child and its VMState is loaded before the
         * parent CPU restores mstatus.  The shared machine pool is loaded
         * after CPU state, so its owner MS value is already authoritative.
         * riscv_cpu_post_load() performs the deferred private-pool check.
         */
        if (incoming->owner->private_ame_pool != pool &&
            !riscv_cpu_ame_enabled(&incoming->owner->env)) {
            goto invalid_state;
        }
        g_hash_table_add(owners, incoming->owner);
    }

    if (qemu_file_get_error(f) < 0) {
        goto invalid_state;
    }

    qemu_mutex_lock(&core->lock);
    for (i = 0; i < num_slots; i++) {
        AmeMigrationSlot *incoming = &slots[i];
        RISCVAMEBackendSlot *slot = &core->slots[i];
        RISCVAMEState *old_state = slot->bound_state;

        if (old_state != NULL && old_state->active_slot >= 0 &&
            (uint32_t)old_state->active_slot == i) {
            old_state->active_slot = -1;
        }
        memcpy(slot->mregs, incoming->backend.mregs,
               layout->mregs_bytes);
        memcpy(slot->accs, incoming->backend.accs,
               layout->accs_bytes);
        memcpy(slot->md, incoming->backend.md, sizeof(slot->md));
        memcpy(slot->ad, incoming->backend.ad, sizeof(slot->ad));
        memcpy(slot->md_dtype_id, incoming->backend.md_dtype_id,
               sizeof(slot->md_dtype_id));
        memcpy(slot->ad_dtype_id, incoming->backend.ad_dtype_id,
               sizeof(slot->ad_dtype_id));
        slot->scalar_dtype = incoming->backend.scalar_dtype;
        slot->scalar_dtype_id = incoming->backend.scalar_dtype_id;
        slot->status = incoming->backend.status;
        /* Staging is not part of the stream and never survives a load. */
        memset(slot->scratch, 0, slot->scratch_bytes);
        slot->initializer = NULL;
        if (incoming->occupied) {
            slot->state = RISCV_AME_BACKEND_OWNED;
            incoming->owner->ame.active_slot = i;
            slot->bound_state = &incoming->owner->ame;
            qatomic_store_release(&slot->owner, OBJECT(incoming->owner));
        } else {
            slot->state = RISCV_AME_BACKEND_FREE;
            slot->bound_state = NULL;
            qatomic_store_release(&slot->owner, NULL);
        }
    }
    core->generation++;
    qemu_mutex_unlock(&core->lock);
    ret = 0;
    goto out;

invalid_state:
    error_report("Invalid RISC-V Ztt backend pool migration state");
out:
    ame_migration_slots_free(slots, num_slots);
    return ret;
}

static const VMStateInfo vmstate_info_ame_pool = {
    .name = "riscv-ame-backend-pool",
    .get = ame_pool_migration_get,
    .put = ame_pool_migration_put,
};

static bool ame_pool_needed(void *opaque)
{
    RISCVAMEBackendPool *pool = opaque;

    return pool->core.configured;
}

static const VMStateDescription vmstate_ame_pool = {
    .name = "riscv-ame-backend-pool",
    .version_id = 1,
    .minimum_version_id = 1,
    .needed = ame_pool_needed,
    .fields = (const VMStateField[]) {
        VMSTATE_SINGLE(migration_anchor, RISCVAMEBackendPool, 1,
                       vmstate_info_ame_pool, uint8_t),
        VMSTATE_END_OF_LIST()
    }
};

const VMStateDescription *riscv_ame_backend_pool_vmstate(void)
{
    return &vmstate_ame_pool;
}

static int ame_hart_migration_put(QEMUFile *f, void *opaque, size_t size,
                                  const VMStateField *field,
                                  JSONWriter *vmdesc)
{
    RISCVAMEState *ame = container_of(opaque, RISCVAMEState,
                                      migration_anchor);
    RISCVCPU *cpu = container_of(ame, RISCVCPU, ame);
    bool bounded_wait;
    int64_t timeout_ns;
    int64_t remaining_ns = 0;

    if (ame->wait_active && ame->active_slot >= 0) {
        error_report("RISC-V Ztt owner has a pending acquire");
        return -EINVAL;
    }
    if (ame->wait_active &&
        (ame->pool == NULL || !ame->pool->enabled ||
         ame->wait_hart != OBJECT(cpu) ||
         !riscv_ame_acquire_descriptor_decode(ame->wait_descriptor,
                                               &bounded_wait,
                                               &timeout_ns) ||
         !bounded_wait ||
         ame->wait_terminal > RISCV_AME_WAIT_INTERRUPTED)) {
        error_report("Invalid live RISC-V Ztt pending acquire state");
        return -EINVAL;
    }
    if (ame->wait_active &&
        ame->wait_terminal == RISCV_AME_WAIT_NONE) {
        remaining_ns = MAX(ame->wait_deadline_ns -
                           qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL), 0);
        if (remaining_ns > timeout_ns) {
            error_report("Invalid live RISC-V Ztt pending acquire timeout");
            return -EINVAL;
        }
    }
    qemu_put_byte(f, cpu->env.amefflags);
    qemu_put_byte(f, cpu->env.amexsat);
    qemu_put_byte(f, ame->wait_active);
    if (!ame->wait_active) {
        return 0;
    }
    qemu_put_be64(f, ame->wait_descriptor);
    qemu_put_byte(f, ame->wait_terminal);
    qemu_put_be64(f, remaining_ns);
    return 0;
}

static int ame_hart_migration_get(QEMUFile *f, void *opaque, size_t size,
                                  const VMStateField *field)
{
    RISCVAMEState *ame = container_of(opaque, RISCVAMEState,
                                      migration_anchor);
    RISCVCPU *cpu = container_of(ame, RISCVCPU, ame);
    uint8_t fflags, xsat, pending;
    uint64_t descriptor, remaining_ns;
    RISCVAMEWaitTerminal terminal;
    Error *local_err = NULL;

    ame->migration_seen = true;
    if (!cpu->cfg.ext_ztt || ame->pool == NULL) {
        error_report("RISC-V Ztt hart migration state received by "
                     "disabled CPU");
        return -EINVAL;
    }
    fflags = qemu_get_byte(f);
    xsat = qemu_get_byte(f);
    pending = qemu_get_byte(f);
    if (fflags & ~AME_FFLAGS_MASK || xsat > 1 || pending > 1) {
        goto invalid_state;
    }
    if (!pending) {
        if (qemu_file_get_error(f) < 0) {
            goto invalid_state;
        }
        cpu->env.amefflags = fflags;
        cpu->env.amexsat = xsat;
        return 0;
    }

    descriptor = qemu_get_be64(f);
    terminal = qemu_get_byte(f);
    remaining_ns = qemu_get_be64(f);
    if (terminal > RISCV_AME_WAIT_INTERRUPTED ||
        remaining_ns > INT64_MAX ||
        (terminal != RISCV_AME_WAIT_NONE && remaining_ns != 0) ||
        qemu_file_get_error(f) < 0 ||
        !riscv_ame_wait_restore(ame, OBJECT(cpu), descriptor, terminal,
                                remaining_ns, &local_err)) {
        if (local_err != NULL) {
            error_report_err(local_err);
        }
        goto invalid_state;
    }
    cpu->env.amefflags = fflags;
    cpu->env.amexsat = xsat;
    return 0;

invalid_state:
    error_report("Invalid RISC-V Ztt hart migration state");
    return -EINVAL;
}

static const VMStateInfo vmstate_info_ame_hart = {
    .name = "riscv-ame-hart",
    .get = ame_hart_migration_get,
    .put = ame_hart_migration_put,
};

static bool ame_needed(void *opaque)
{
    RISCVCPU *cpu = opaque;

    return cpu->cfg.ext_ztt;
}

const VMStateDescription vmstate_riscv_ame_hart = {
    .name = "cpu/ame-hart",
    .version_id = 1,
    .minimum_version_id = 1,
    .needed = ame_needed,
    .fields = (const VMStateField[]) {
        VMSTATE_SINGLE(ame.migration_anchor, RISCVCPU, 1,
                       vmstate_info_ame_hart, uint8_t),
        VMSTATE_END_OF_LIST()
    }
};

void riscv_ame_migration_pre_load(RISCVCPU *cpu)
{
    cpu->ame.migration_seen = false;
}

int riscv_ame_migration_post_load(RISCVCPU *cpu)
{
    CPURISCVState *env = &cpu->env;

    if (!cpu->cfg.ext_ztt) {
        return 0;
    }
    if (!cpu->ame.migration_seen) {
        error_report("RISC-V Ztt migration state missing from source CPU");
        return -EINVAL;
    }
    if (cpu->ame.active_slot >= 0 &&
        (!riscv_ame_state_owned(&cpu->ame, OBJECT(cpu)) ||
         cpu->ame.wait_active)) {
        error_report("Invalid RISC-V Ztt owner migration state");
        return -EINVAL;
    }
    if ((cpu->ame.active_slot >= 0 || cpu->ame.wait_active) &&
        (cpu->ame.pool == NULL || !cpu->ame.pool->enabled ||
         !riscv_cpu_ame_enabled(env))) {
        error_report("Disabled RISC-V Ztt migration state is active");
        return -EINVAL;
    }
    return 0;
}
