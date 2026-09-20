/*
 * RISC-V Ztt packed move test for a single-Acc implementation
 *
 * Run with N=4, U=8, NM=16, and NA=1.
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

    .option norvc

    .text
    .global _start
_start:
    li      t0, MSTATUS_AME_MS_INITIAL
    csrs    mstatus, t0
    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail

    li      t0, 0x40000004
    .word   ZTT_MSETTYP(0, ZTT_R_T0)
    .word   ZTT_ASETTYP(0, ZTT_R_T0)

    /* Int4 needs two Acc squares; NA=1 reports UN before any transfer. */
    li      t1, 0
    li      a2, 16
.Lfill:
    li      t0, 0xa5
    .word   ZTT_MMOVE8_M_X(0, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    bne     t1, a2, .Lfill

    csrwi   CSR_AMESTATUS, 0
    .word   ZTT_MMOV_M_A(0, 0)
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    beqz    t0, fail

    li      t1, 0
.Lcheck:
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 0, ZTT_R_T1)
    li      t2, 0xa5
    bne     t0, t2, fail
    addi    t1, t1, 1
    bne     t1, a2, .Lcheck

    csrwi   CSR_AMESTATUS, 0
    .word   ZTT_MMOV_A_M(0, 0)
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    beqz    t0, fail

    .word   ZTT_AME_RELEASE
    li      a0, 0
    j       semihost_exit

fail:
    li      a0, 1

semihost_exit:
    lla     a1, semiargs
    li      t0, 0x20026
#if __riscv_xlen == 64
    sd      t0, 0(a1)
    sd      a0, 8(a1)
#else
    sw      t0, 0(a1)
    sw      a0, 4(a1)
#endif
    li      a0, 0x20
    .balign 16
    slli    zero, zero, 0x1f
    ebreak
    srai    zero, zero, 0x7
    j       .

    .bss
    .balign 16
semiargs:
#if __riscv_xlen == 64
    .space  16
#else
    .space  8
#endif
