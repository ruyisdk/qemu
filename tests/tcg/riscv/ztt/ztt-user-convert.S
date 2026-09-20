/*
 * RISC-V Ztt linux-user integer conversion test
 *
 * Run with N=4, U=8, and NM=16.
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

#define NR_EXIT          93
#define DTYPE_INT4       0x40000004
#define DTYPE_INT4_SAT   0x60000004
#define DTYPE_INT8       0x40000008
#define DTYPE_FP32       0x20300120

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

    SET_M_DTYPE 0, DTYPE_INT4_SAT
    SET_M_DTYPE 2, DTYPE_INT8
    FILL_RAW_CONST 2, 0xf7
    FILL_RAW_CONST 3, 0x08
    csrwi   CSR_AMEXSAT, 0
    .word   ZTT_MCONV_EW(0, 2)
    csrr    t0, CSR_AMEXSAT
    andi    t0, t0, AME_XSAT_SAT
    beqz    t0, fail
    li      t1, 0
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 0, ZTT_R_T1)
    li      t2, 0x88
    bne     t0, t2, fail
    li      t1, 8
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 0, ZTT_R_T1)
    li      t2, 0x77
    bne     t0, t2, fail

    /* Floating sqrt requires a floating destination; UN preserves it. */
    SET_M_DTYPE 4, DTYPE_INT8
    SET_M_DTYPE 8, DTYPE_FP32
    FILL_RAW_CONST 4, 0x5a
    csrwi   CSR_AMESTATUS, 0
    .word   ZTT_MSQRT_EW(4, 8)
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    beqz    t0, fail
    CHECK_RAW_CONST 4, 0x5a

    .word   ZTT_AME_RELEASE
    li      a0, 0
    j       exit

fail:
    li      a0, 1
exit:
    li      a7, NR_EXIT
    ecall
