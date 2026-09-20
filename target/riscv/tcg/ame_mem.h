/*
 * RISC-V Attached Matrix Extension (AME) memory transactions
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef TARGET_RISCV_TCG_AME_MEM_H
#define TARGET_RISCV_TCG_AME_MEM_H

#include "cpu.h"

/**
 * riscv_ame_memory_read: Load planned guest RAM segments into a host buffer
 * @env: Executing hart, supplying its MMU context and memory-access scratch
 * @base: Guest address of the first segment, already alignment-checked
 * @stride: Byte displacement between segment starts, with XLEN wrapping
 * @plan: Validated transfer sizes and segment count
 * @buffer: Host output buffer of at least plan->total_bytes
 * @ra: TCG helper return address used to restore faulting instruction state
 *
 * Probe every segment before copying. Pack segments adjacently in @buffer;
 * register layout conversion is separate. Translation and non-RAM failures
 * raise a guest exception rather than returning an error code.
 */
void riscv_ame_memory_read(CPURISCVState *env, target_ulong base,
                           target_ulong stride, const AmeMemPlan *plan,
                           uint8_t *buffer, uintptr_t ra);
/**
 * riscv_ame_memory_write: Store a host buffer into planned guest RAM segments
 * @env: Executing hart, supplying its MMU context and memory-access scratch
 * @base: Guest address of the first segment, already alignment-checked
 * @stride: Byte displacement between segment starts, with XLEN wrapping
 * @plan: Validated transfer sizes and segment count
 * @buffer: Host input buffer of plan->total_bytes, with adjacent segments
 * @ra: TCG helper return address used to restore faulting instruction state
 *
 * Probe every segment before copying. Translation and non-RAM failures raise
 * a guest exception rather than returning an error code. Register layout
 * conversion is the caller's responsibility.
 */
void riscv_ame_memory_write(CPURISCVState *env, target_ulong base,
                            target_ulong stride, const AmeMemPlan *plan,
                            const uint8_t *buffer, uintptr_t ra);

#endif /* TARGET_RISCV_TCG_AME_MEM_H */
