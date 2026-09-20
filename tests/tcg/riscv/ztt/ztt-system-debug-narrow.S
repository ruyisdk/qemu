/*
 * RISC-V Ztt fixed-width debug move narrow-row test
 *
 * Run with N=1, U=8, and NM=16.
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

    /* One row contains only eight bits, so only the 8-bit form is legal. */
    li      t0, 0xa5
    li      t1, -1
    .word   ZTT_MMOVE8_M_X(0, ZTT_R_T0, ZTT_R_T1)
    .word   ZTT_MMOVE8_X_M(ZTT_R_T2, 0, ZTT_R_T1)
    bne     t2, t0, fail

    li      t0, 0x1234
    EXPECT_ILLEGAL ZTT_MMOVE16_M_X(0, ZTT_R_T0, ZTT_R_T1)
    EXPECT_ILLEGAL ZTT_MMOVE16_X_M(ZTT_R_T2, 0, ZTT_R_T1)
    EXPECT_ILLEGAL ZTT_MMOVE32_M_X(0, ZTT_R_T0, ZTT_R_T1)
    EXPECT_ILLEGAL ZTT_MMOVE32_X_M(ZTT_R_T2, 0, ZTT_R_T1)
    EXPECT_ILLEGAL ZTT_MMOVE64_M_X(0, ZTT_R_T0, ZTT_R_T1)
    EXPECT_ILLEGAL ZTT_MMOVE64_X_M(ZTT_R_T2, 0, ZTT_R_T1)

    /* Failed wider forms leave the byte and status unchanged. */
    .word   ZTT_MMOVE8_X_M(ZTT_R_T2, 0, ZTT_R_T1)
    li      t0, 0xa5
    bne     t2, t0, fail
    csrr    t0, CSR_AMESTATUS
    bnez    t0, fail

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
