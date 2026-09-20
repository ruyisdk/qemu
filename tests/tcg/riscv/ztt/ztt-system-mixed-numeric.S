/*
 * Ztt mixed numeric instruction test. Scalar FS remains Off.
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"
#include "include/ztt-mixed-numeric.S"

    .option norvc
    .text
    .global _start
_start:
    li      s0, 1
    lla     t0, fail
    csrw    mtvec, t0
    li      t0, MSTATUS_AME_MS_INITIAL
    csrs    mstatus, t0
    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail
    MIX_TESTS
    .word   ZTT_AME_RELEASE
    li      a0, 0
    j       semihost_exit
fail:
    mv      a0, s0
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
    .space 16
