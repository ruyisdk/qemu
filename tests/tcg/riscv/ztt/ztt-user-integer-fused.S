/*
 * RISC-V Ztt linux-user integer fused and scaling test
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

    .macro SET_M_DTYPE mreg, encoding
    li      t0, \encoding
    .word   ZTT_MSETTYP(\mreg, ZTT_R_T0)
    .endm

    .macro FILL_RAW_CONST mreg, value
    li      t1, 0
    li      a2, 16
.Lfill_raw\@:
    li      t0, \value
    .word   ZTT_MMOVE8_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    bne     t1, a2, .Lfill_raw\@
    .endm

    .macro CHECK_RAW_CONST mreg, value
    li      t1, 0
    li      a2, 16
.Lcheck_raw\@:
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, \mreg, ZTT_R_T1)
    li      t2, \value
    bne     t0, t2, fail
    addi    t1, t1, 1
    bne     t1, a2, .Lcheck_raw\@
    .endm

    .text
    .global _start
_start:
    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail

    SET_M_DTYPE 0, DTYPE_INT8
    SET_M_DTYPE 1, DTYPE_INT8
    SET_M_DTYPE 2, DTYPE_INT8
    FILL_RAW_CONST 0, 10
    FILL_RAW_CONST 1, -3
    FILL_RAW_CONST 2, 4
    .word   ZTT_MMULACC_EW(0, 1, 2)
    CHECK_RAW_CONST 0, 0xfe

    FILL_RAW_CONST 0, 10
    .word   ZTT_MCMOVLT_EW(0, 1, 2)
    CHECK_RAW_CONST 0, 4

    FILL_RAW_CONST 1, 3
    FILL_RAW_CONST 2, 1
    .word   ZTT_MRDEXP_EW(0, 1, 2)
    CHECK_RAW_CONST 0, 2

    csrwi   CSR_AMESTYPE, 0
    FILL_RAW_CONST 0, 10
    li      t0, 1
    .word   ZTT_MLDEXPACC_EW_X(0, ZTT_R_T0, 1)
    CHECK_RAW_CONST 0, 16

    .word   ZTT_AME_RELEASE
    li      a0, 0
    j       exit

fail:
    li      a0, 1
exit:
    li      a7, NR_EXIT
    ecall
