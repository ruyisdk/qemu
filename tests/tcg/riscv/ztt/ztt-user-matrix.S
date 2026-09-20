/*
 * RISC-V Ztt linux-user integer matrix operation test
 *
 * Run with N=4, U=8, NM=16, and NA=1.
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

#define NR_EXIT          93
#define DTYPE_INT8       0x40000008
#define DTYPE_INT16      0x40000010

    .option norvc

    .macro SET_M_DTYPE mreg, encoding
    li      t0, \encoding
    .word   ZTT_MSETTYP(\mreg, ZTT_R_T0)
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
    SET_M_DTYPE 2, DTYPE_INT16
    li      t0, DTYPE_INT16
    .word   ZTT_ASETTYP(0, ZTT_R_T0)
    lla     t0, matrix_a
    .word   ZTT_MLS_RM(0, ZTT_R_T0)
    lla     t0, identity
    .word   ZTT_MLS_RM(1, ZTT_R_T0)

    .word   ZTT_MMULACC_2D(0, 0, 1)
    .word   ZTT_MMOV_M_A(2, 0)
    li      t1, 0
.Lcheck:
    .word   ZTT_MMOVE16_X_M(ZTT_R_T0, 2, ZTT_R_T1)
    addi    t2, t1, 1
    bne     t0, t2, fail
    addi    t1, t1, 1
    li      t2, 8
    bne     t1, t2, .Lcheck
    li      t1, 0
.Lcheck_high:
    .word   ZTT_MMOVE16_X_M(ZTT_R_T0, 3, ZTT_R_T1)
    addi    t2, t1, 9
    bne     t0, t2, fail
    addi    t1, t1, 1
    li      t2, 8
    bne     t1, t2, .Lcheck_high

    .word   ZTT_AME_RELEASE
    li      a0, 0
    j       exit

fail:
    li      a0, 1
exit:
    li      a7, NR_EXIT
    ecall

    .section .rodata
    .balign 256
matrix_a:
    .byte   1, 2, 3, 4, 5, 6, 7, 8
    .byte   9, 10, 11, 12, 13, 14, 15, 16
identity:
    .byte   1, 0, 0, 0, 0, 1, 0, 0
    .byte   0, 0, 1, 0, 0, 0, 0, 1
