/*
 * RISC-V Attached Matrix Extension backend pool device
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef TARGET_RISCV_AME_POOL_H
#define TARGET_RISCV_AME_POOL_H

#include "hw/core/qdev.h"
#include "target/riscv/ame/backend.h"

#define TYPE_RISCV_AME_BACKEND_POOL "riscv-ame-backend-pool"
OBJECT_DECLARE_SIMPLE_TYPE(RISCVAMEBackendPool, RISCV_AME_BACKEND_POOL)

/*
 * Machine-created QOM device wrapping the shared backend pool core.
 * num_slots and enabled are device properties copied into core at preparation.
 * Harts attach to core; the device must outlive all attached harts.
 */
struct RISCVAMEBackendPool {
    DeviceState parent_obj;

    /* VMState anchor for the explicit variable-length wire format. */
    uint8_t migration_anchor;
    uint32_t num_slots;
    bool enabled;
    RISCVAMEBackendPoolCore core;
};

/**
 * riscv_ame_backend_pool_configure: Set explicit pool device capabilities
 * @pool: Machine-created pool device, not yet configured or realized
 * @n: N, the square side in elements
 * @u: U, the M cell width in bits
 * @num_mregs: NM, the M register count
 * @num_accregs: NA, the Acc register count
 * @capabilities: Constructed capability set to deep-copy
 * @errp: Error destination
 *
 * Prepare the core from device properties if necessary, then configure it.
 * Register storage is allocated by device realization, not this function.
 *
 * Returns: true on success; false on invalid configuration, setting @errp.
 */
bool riscv_ame_backend_pool_configure(
    RISCVAMEBackendPool *pool, uint64_t n, uint64_t u,
    uint32_t num_mregs, uint32_t num_accregs,
    const AmeCapabilities *capabilities, Error **errp);
/**
 * riscv_ame_backend_pool_configure_cpu: Configure or check a pool for a CPU
 * @pool: Machine-created pool device
 * @n: CPU's requested N, in elements per square side
 * @u: CPU's requested U, in bits per M cell
 * @num_mregs: CPU's requested NM
 * @num_accregs: CPU's requested NA
 * @policy: CPU capability policy
 * @requirements: Explicit CPU allowlist, or NULL for FOLLOW_MAX
 * @errp: Error destination
 *
 * Prepare the core if necessary. An unconfigured core copies the CPU input;
 * later CPUs validate against the frozen pool without changing it.
 *
 * Returns: true on success; false on invalid input or incompatibility,
 * setting @errp.
 */
bool riscv_ame_backend_pool_configure_cpu(
    RISCVAMEBackendPool *pool, uint64_t n, uint64_t u,
    uint32_t num_mregs, uint32_t num_accregs, AmeCapabilityPolicy policy,
    const AmeCapabilities *requirements, Error **errp);
/**
 * riscv_ame_backend_pool_core: Get the backend core embedded in a pool device
 * @pool: Pool device
 *
 * Does not prepare or realize the core, and does not acquire a backend.
 *
 * Returns: a borrowed pointer with the device's lifetime.
 */
RISCVAMEBackendPoolCore *riscv_ame_backend_pool_core(
    RISCVAMEBackendPool *pool);

#endif /* TARGET_RISCV_AME_POOL_H */
