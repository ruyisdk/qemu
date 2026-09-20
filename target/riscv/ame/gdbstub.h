/*
 * RISC-V Attached Matrix Extension GDB register access
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef TARGET_RISCV_AME_GDBSTUB_H
#define TARGET_RISCV_AME_GDBSTUB_H

#include <stdbool.h>

#include "qemu/typedefs.h"

/**
 * riscv_ame_gdb_is_csr: Identify CSRs exposed by the dedicated AME GDB feature
 * @csrno: Architectural CSR number
 *
 * Returns: true for AME status, scalar datatype, flags, geometry or ownership
 * CSRs; otherwise false.
 */
bool riscv_ame_gdb_is_csr(int csrno);

/**
 * riscv_ame_register_gdb_regs: Register the CPU's dynamic AME GDB feature
 * @cs: Ztt-enabled CPU with initialized backend geometry
 *
 * Expose M/Acc data, datatype registers and AME state through a feature sized
 * for this backend. The caller checks extension enablement before registering.
 */
void riscv_ame_register_gdb_regs(CPUState *cs);

/**
 * riscv_ame_gdb_finalize: Free the CPU's dynamic AME GDB description
 * @cs: CPU being finalized, whether or not its AME feature was registered
 *
 * Release the generated XML, register names and register-name array.
 */
void riscv_ame_gdb_finalize(CPUState *cs);

#endif /* TARGET_RISCV_AME_GDBSTUB_H */
