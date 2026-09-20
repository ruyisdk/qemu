/*
 * RISC-V Attached Matrix Extension migration state
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef TARGET_RISCV_AME_MIGRATION_H
#define TARGET_RISCV_AME_MIGRATION_H

#include "migration/vmstate.h"

typedef struct ArchCPU RISCVCPU;

/* CPU subsection for per-hart flags and a pending bounded acquire. */
extern const VMStateDescription vmstate_riscv_ame_hart;

/**
 * riscv_ame_backend_pool_vmstate: Get the shared pool's migration schema
 *
 * The pool saves geometry, capabilities, slot ownership and matrix state once,
 * separately from the per-hart CPU subsection.
 *
 * Returns: a borrowed static VMStateDescription for DeviceClass.vmsd.
 */
const VMStateDescription *riscv_ame_backend_pool_vmstate(void);

/**
 * riscv_ame_migration_pre_load: Clear the AME CPU subsection's presence marker
 * @cpu: Destination CPU before loading its migration state
 *
 * Lets post-load distinguish a received subsection from missing source state.
 */
void riscv_ame_migration_pre_load(RISCVCPU *cpu);

/**
 * riscv_ame_migration_post_load: Validate restored hart state and binding
 * @cpu: Destination CPU after its migration state has been loaded
 *
 * Check subsection presence, ownership/wait consistency and enable state.
 *
 * Returns: zero on success or when Ztt is disabled; -EINVAL on invalid state,
 * reporting the migration error.
 */
int riscv_ame_migration_post_load(RISCVCPU *cpu);

#endif /* TARGET_RISCV_AME_MIGRATION_H */
