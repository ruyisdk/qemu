/*
 * RISC-V Ztt linux-user context and control instruction test
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

#define NR_EXIT 93

    .option norvc

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
    CHECK_GEOMETRY
    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail
    csrr    t0, CSR_AMEOWN
    li      t1, 1
    bne     t0, t1, fail
    CHECK_GEOMETRY

    .word   ZTT_MGETTYP(ZTT_R_T0, 0)
    bnez    t0, fail
    .word   ZTT_AGETTYP(ZTT_R_T0, 0)
    bnez    t0, fail

    /* RV64 also proves that settyp ignores upper X-register bits. */
    li      t0, 0x40000008
#if __riscv_xlen == 64
    li      t1, 1
    slli    t1, t1, 40
    or      t0, t0, t1
#endif
    .word   ZTT_MSETTYP(0, ZTT_R_T0)
    .word   ZTT_MGETTYP(ZTT_R_T0, 0)
    li      t1, 0x40000008
    bne     t0, t1, fail

    li      t0, 0x20300120      /* FP32. */
    .word   ZTT_ASETTYP(0, ZTT_R_T0)
    .word   ZTT_AGETTYP(ZTT_R_T1, 0)
    bne     t0, t1, fail

    csrwi   CSR_AMESTATUS, 0
    li      t0, 0x11500108      /* OCP E4M3 with unsupported RNO. */
    .word   ZTT_MSETTYP(0, ZTT_R_T0)
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    beqz    t0, fail
    .word   ZTT_MGETTYP(ZTT_R_T0, 0)
    li      t1, 0x40000008
    bne     t0, t1, fail

    csrwi   CSR_AMESTATUS, 0
    .word   ZTT_MZERO_M(0)
    csrr    t0, CSR_AMESTATUS
    bnez    t0, fail

    li      t0, AME_FFLAG_NV
    csrw    CSR_AMEFFLAGS, t0
    li      t0, AME_XSAT_SAT
    csrw    CSR_AMEXSAT, t0

    /* Bounded-wait acquire immediately preserves the private binding. */
    li      a1, 1
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail
    csrr    t0, CSR_AMEFFLAGS
    li      t1, AME_FFLAG_NV
    bne     t0, t1, fail
    .word   ZTT_MGETTYP(ZTT_R_T0, 0)
    li      t1, 0x40000008
    bne     t0, t1, fail

    .word   ZTT_AME_RELEASE
    csrr    t0, CSR_AMEOWN
    bnez    t0, fail
    CHECK_GEOMETRY

    /* A new bounded-wait grant succeeds without a timer and clears state. */
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail
    .word   ZTT_MGETTYP(ZTT_R_T0, 0)
    bnez    t0, fail
    csrr    t0, CSR_AMEFFLAGS
    bnez    t0, fail
    csrr    t0, CSR_AMEXSAT
    bnez    t0, fail
    .word   ZTT_AME_RELEASE

    li      a0, 0
    j       exit

fail:
    li      a0, 1
exit:
    li      a7, NR_EXIT
    ecall
