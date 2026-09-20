/*
 * RISC-V Ztt linux-user reduction and scatter test
 *
 * Run with N=4, U=8, and NM=16.
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

#define NR_EXIT          93
#define DTYPE_INT8       0x40000008

    .option norvc

    .macro SET_M_DTYPE mreg
    li      t0, DTYPE_INT8
    .word   ZTT_MSETTYP(\mreg, ZTT_R_T0)
    .endm

    .text
    .global _start
_start:
    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail

    SET_M_DTYPE 0
    SET_M_DTYPE 1
    SET_M_DTYPE 2
    li      t1, 0
    li      a2, 16
.Lfill_source:
    mv      t0, t1
    .word   ZTT_MMOVE8_M_X(0, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    bne     t1, a2, .Lfill_source

    .word   ZTT_MPREFIXADD_ROW(1, 0)
    li      t1, 3
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 1, ZTT_R_T1)
    li      t2, 6
    bne     t0, t2, fail

    li      t1, 0
    li      a2, 16
.Lfill_indices:
    li      t0, 3
    .word   ZTT_MMOVE8_M_X(2, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    bne     t1, a2, .Lfill_indices
    .word   ZTT_MZERO_M(1)
    .word   ZTT_MCOLSCATADD_EW(1, 0, 2)
    li      t1, 3
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 1, ZTT_R_T1)
    li      t2, 6
    bne     t0, t2, fail

    .word   ZTT_MREDUCEMIN_COL(1, 0)
    li      t1, 12
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 1, ZTT_R_T1)
    bnez    t0, fail

    .word   ZTT_AME_RELEASE
    li      a0, 0
    j       exit

fail:
    li      a0, 1
exit:
    li      a7, NR_EXIT
    ecall
