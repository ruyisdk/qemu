/*
 * RISC-V Ztt maximum accumulator-register index test
 *
 * Run with N=4, U=8, NM=16, and NA=16.
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

    .option norvc

    .macro EXPECT_ILLEGAL insn
    li      s0, 1
    lla     s1, .Lresume\@
    lla     s2, .Lfault\@
.Lfault\@:
    .word   \insn
    j       fail
.Lresume\@:
    li      t0, 2
    bne     s0, t0, fail
    .endm

    .text
    .global _start
_start:
    lla     t0, trap_handler
    csrw    mtvec, t0
    li      t0, MSTATUS_AME_MS_INITIAL
    csrs    mstatus, t0

    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail

    li      t0, 0x40000008
    .word   ZTT_MSETTYP(0, ZTT_R_T0)
    .word   ZTT_MSETTYP(1, ZTT_R_T0)
    .word   ZTT_ASETTYP(15, ZTT_R_T0)

    li      t1, 0
.Lfill:
    addi    t0, t1, 1
    .word   ZTT_MMOVE8_M_X(0, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    li      t2, 16
    bne     t1, t2, .Lfill

    .word   ZTT_MMOV_A_M(15, 0)
    .word   ZTT_MMOV_M_A(1, 15)
    li      t1, 0
.Lcheck:
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 1, ZTT_R_T1)
    addi    t2, t1, 1
    bne     t0, t2, fail
    addi    t1, t1, 1
    li      t2, 16
    bne     t1, t2, .Lcheck

    /* Index 16 is encodable but outside NA=16 for either operand slot. */
    li      t0, 0x40000008
    EXPECT_ILLEGAL ZTT_ASETTYP(16, ZTT_R_T0)
    EXPECT_ILLEGAL ZTT_AGETTYP(ZTT_R_T0, 16)
    EXPECT_ILLEGAL ZTT_MZERO_2D_ACC(16)
    EXPECT_ILLEGAL ZTT_MMOV_A_M(16, 0)
    EXPECT_ILLEGAL ZTT_MMOV_M_A(0, 16)
    EXPECT_ILLEGAL ZTT_MMULACC_2D(16, 0, 0)

    .word   ZTT_AME_RELEASE
    li      a0, 0
    j       semihost_exit

trap_handler:
    csrr    t0, mcause
    li      t1, 2
    bne     t0, t1, fail
    csrr    t0, mepc
    bne     t0, s2, fail
    li      t0, 1
    bne     s0, t0, fail
    li      s0, 2
    csrw    mepc, s1
    mret

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
