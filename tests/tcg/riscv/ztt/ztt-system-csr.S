/*
 * RISC-V Ztt MS and provisional CSR system-mode test
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

    .option norvc

    .macro EXPECT_ILLEGAL insn:vararg
    li      s0, 1
    lla     s1, .Lresume\@
    \insn
    j       fail
.Lresume\@:
    li      t0, 2
    bne     s0, t0, fail
    .endm

    .macro CHECK_GEOMETRY
    csrr    t0, CSR_AMENLEN
    li      t1, 4
    bne     t0, t1, fail
    csrr    t0, CSR_AMEUDSZ
    li      t1, 8
    bne     t0, t1, fail
    .endm

    .text
    .global _start
_start:
    lla     t0, trap_handler
    csrw    mtvec, t0

    /* Reset leaves MS Off, so every AME CSR is inaccessible. */
    EXPECT_ILLEGAL csrr t0, CSR_AMENLEN
    EXPECT_ILLEGAL csrr t0, CSR_AMEUDSZ
    EXPECT_ILLEGAL csrr t0, CSR_AMEFFLAGS
    EXPECT_ILLEGAL csrw CSR_AMEFFLAGS, zero
    EXPECT_ILLEGAL csrr t0, CSR_AMEXSAT
    EXPECT_ILLEGAL csrw CSR_AMEXSAT, zero

    li      t0, MSTATUS_AME_MS_INITIAL
    csrs    mstatus, t0

    CHECK_GEOMETRY
    csrr    t0, CSR_AMEOWN
    bnez    t0, fail

    /* Context and hart-local flag CSRs all require ownership. */
    EXPECT_ILLEGAL csrr t0, CSR_AMESTATUS
    EXPECT_ILLEGAL csrr t0, CSR_AMESTYPE
    li      s3, 0x55
    EXPECT_ILLEGAL csrr s3, CSR_AMEFFLAGS
    EXPECT_ILLEGAL csrr s3, CSR_AMEXSAT
    EXPECT_ILLEGAL csrw CSR_AMEFFLAGS, s3
    EXPECT_ILLEGAL csrw CSR_AMEXSAT, s3
    li      t0, 0x55
    bne     s3, t0, fail
    csrr    t0, mstatus
    li      t1, MSTATUS_AME_MS_MASK
    and     t0, t0, t1
    li      t1, MSTATUS_AME_MS_INITIAL
    bne     t0, t1, fail

    /* sstatus shadows MS and may change it while no backend is owned. */
    csrr    t0, sstatus
    li      t1, MSTATUS_AME_MS_MASK
    not     t1, t1
    and     t0, t0, t1
    li      t1, MSTATUS_AME_MS_CLEAN
    or      t0, t0, t1
    csrw    sstatus, t0
    EXPECT_ILLEGAL csrr s3, CSR_AMEFFLAGS
    EXPECT_ILLEGAL csrw CSR_AMEXSAT, s3
    csrr    t0, mstatus
    li      t1, MSTATUS_AME_MS_MASK
    and     t0, t0, t1
    li      t1, MSTATUS_AME_MS_CLEAN
    bne     t0, t1, fail

    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail

    /* An owner can restore flags; writes dirty MS and set SD. */
    li      t0, -1
    csrw    CSR_AMEFFLAGS, t0
    csrr    t0, CSR_AMEFFLAGS
    li      t1, AME_FFLAGS_MASK
    bne     t0, t1, fail
    csrr    t0, mstatus
    li      t1, MSTATUS_AME_MS_MASK
    and     t0, t0, t1
    li      t1, MSTATUS_AME_MS_DIRTY
    bne     t0, t1, fail
    csrr    t0, mstatus
    li      t1, 1
#if __riscv_xlen == 64
    slli    t1, t1, 63
#else
    slli    t1, t1, 31
#endif
    and     t0, t0, t1
    beqz    t0, fail

    /* Changing MS to Clean preserves flags and ownership. */
    csrr    t0, sstatus
    li      t1, MSTATUS_AME_MS_MASK
    not     t1, t1
    and     t0, t0, t1
    li      t1, MSTATUS_AME_MS_CLEAN
    or      t0, t0, t1
    csrw    sstatus, t0
    csrr    t0, mstatus
    li      t1, MSTATUS_AME_MS_MASK
    and     t0, t0, t1
    li      t1, MSTATUS_AME_MS_CLEAN
    bne     t0, t1, fail

    li      t0, AME_XSAT_SAT
    csrw    CSR_AMEXSAT, t0
    csrr    t0, CSR_AMEXSAT
    li      t1, AME_XSAT_SAT
    bne     t0, t1, fail

    /* The three capability/ownership CSRs use read-only CSR encodings. */
    EXPECT_ILLEGAL csrw CSR_AMENLEN, zero
    EXPECT_ILLEGAL csrw CSR_AMEUDSZ, zero
    EXPECT_ILLEGAL csrw CSR_AMEOWN, zero

    .word   ZTT_AME_RELEASE
    CHECK_GEOMETRY
    EXPECT_ILLEGAL csrr s3, CSR_AMEFFLAGS
    EXPECT_ILLEGAL csrw CSR_AMEFFLAGS, s3
    EXPECT_ILLEGAL csrr s3, CSR_AMEXSAT
    EXPECT_ILLEGAL csrw CSR_AMEXSAT, s3

    /* MS Off still gates geometry and flags after release. */
    li      t0, MSTATUS_AME_MS_MASK
    csrc    mstatus, t0
    EXPECT_ILLEGAL csrr t0, CSR_AMEOWN
    EXPECT_ILLEGAL csrr t0, CSR_AMEXSAT
    li      t0, MSTATUS_AME_MS_INITIAL
    csrs    mstatus, t0

    /* A new grant clears hart-local flags as well as backend context. */
    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail
    csrr    t0, CSR_AMEFFLAGS
    bnez    t0, fail
    csrr    t0, CSR_AMEXSAT
    bnez    t0, fail
    CHECK_GEOMETRY
    .word   ZTT_AME_RELEASE
    CHECK_GEOMETRY

    li      a0, 0
    j       semihost_exit

trap_handler:
    csrr    t0, mcause
    li      t1, 2
    bne     t0, t1, fail
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
