/*
 * RISC-V Ztt integer matrix operation test
 *
 * Run with N=4, U=8, NM=32, and NA=1.
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

#define DTYPE_INT4       0x40000004
#define DTYPE_INT8       0x40000008
#define DTYPE_INT16      0x40000010
#define DTYPE_UINT8      0x00000008

    .option norvc

    .macro SET_M_DTYPE mreg, encoding
    li      t0, \encoding
    .word   ZTT_MSETTYP(\mreg, ZTT_R_T0)
    .endm

    .macro SET_A_DTYPE areg, encoding
    li      t0, \encoding
    .word   ZTT_ASETTYP(\areg, ZTT_R_T0)
    .endm

    .macro LOAD_M mreg, address
    lla     t0, \address
    .word   ZTT_MLS_RM(\mreg, ZTT_R_T0)
    .endm

    .macro CHECK_ACC expected
    .word   ZTT_MMOV_M_A(2, 0)
    lla     t0, output
    .word   ZTT_MSS_RM(2, ZTT_R_T0)
    lla     t3, output
    lla     t4, \expected
    li      t5, 16
.Lcheck_acc\@:
    lhu     t0, 0(t3)
    lhu     t1, 0(t4)
    bne     t0, t1, fail
    addi    t3, t3, 2
    addi    t4, t4, 2
    addi    t5, t5, -1
    bnez    t5, .Lcheck_acc\@
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

    SET_M_DTYPE 0, DTYPE_INT8
    SET_M_DTYPE 1, DTYPE_INT8
    SET_M_DTYPE 2, DTYPE_INT16
    SET_A_DTYPE 0, DTYPE_INT16
    LOAD_M  0, matrix_a
    LOAD_M  1, matrix_b

    /* Every v0.6 matrix opcode accumulates into Acc0. */
    .word   ZTT_MZERO_ACC(0)
    .word   ZTT_MMULACC_2D(0, 0, 1)
    CHECK_ACC expected_ab
    .word   ZTT_MMULACC_2D(0, 0, 1)
    CHECK_ACC expected_2ab
    .word   ZTT_MMULACCNEG_2D(0, 0, 1)
    CHECK_ACC expected_ab
    .word   ZTT_MZERO_ACC(0)
    .word   ZTT_MMULACCNEG_2D(0, 0, 1)
    CHECK_ACC expected_neg_ab
    .word   ZTT_MZERO_ACC(0)
    .word   ZTT_MMULATACC_2D(0, 0, 1)
    CHECK_ACC expected_at_b
    .word   ZTT_MMULATACC_2D(0, 0, 1)
    CHECK_ACC expected_2at_b
    .word   ZTT_MMULATACCNEG_2D(0, 0, 1)
    CHECK_ACC expected_at_b
    .word   ZTT_MZERO_ACC(0)
    .word   ZTT_MMULATACCNEG_2D(0, 0, 1)
    CHECK_ACC expected_neg_at_b
    .word   ZTT_MZERO_ACC(0)
    .word   ZTT_MMULBTACC_2D(0, 0, 1)
    CHECK_ACC expected_a_bt
    .word   ZTT_MMULBTACC_2D(0, 0, 1)
    CHECK_ACC expected_2a_bt
    .word   ZTT_MMULBTACCNEG_2D(0, 0, 1)
    CHECK_ACC expected_a_bt
    .word   ZTT_MZERO_ACC(0)
    .word   ZTT_MMULBTACCNEG_2D(0, 0, 1)
    CHECK_ACC expected_neg_a_bt

    /* Signed -1 and unsigned 255 multiply as values, not raw-bit Int8. */
    SET_M_DTYPE 1, DTYPE_UINT8
    LOAD_M  0, all_ones
    LOAD_M  1, all_ones
    .word   ZTT_MZERO_ACC(0)
    .word   ZTT_MMULACC_2D(0, 0, 1)
    CHECK_ACC expected_mixed

    /*
     * Int4 contributes two packed squares from M4.  Int8 contributes
     * the matching two squares from consecutive physical registers M5/M6.
     */
    SET_M_DTYPE 4, DTYPE_INT4
    SET_M_DTYPE 5, DTYPE_INT8
    SET_M_DTYPE 6, DTYPE_INT8
    LOAD_M  4, packed_ones
    LOAD_M  5, identity
    LOAD_M  6, identity
    .word   ZTT_MZERO_ACC(0)
    .word   ZTT_MMULACC_2D(0, 4, 5)
    CHECK_ACC expected_twos

    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    bnez    t0, fail
    .word   ZTT_AME_RELEASE
    li      a0, 0
    j       semihost_exit

trap_handler:
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

    .section .rodata
    .balign 256
matrix_a:
    .byte   1, 2, 3, 4, 5, 6, 7, 8
    .byte   9, 10, 11, 12, 13, 14, 15, 16
matrix_b:
    .byte   0, 1, 0, 0, 0, 0, 1, 0
    .byte   0, 0, 0, 1, 1, 0, 0, 0
all_ones:
    .rept   16
    .byte   255
    .endr
expected_mixed:
    .rept   16
    .half   -1020
    .endr
packed_ones:
    .rept   16
    .byte   0x11
    .endr
identity:
    .byte   1, 0, 0, 0, 0, 1, 0, 0
    .byte   0, 0, 1, 0, 0, 0, 0, 1
expected_ab:
    .2byte  4, 1, 2, 3, 8, 5, 6, 7
    .2byte  12, 9, 10, 11, 16, 13, 14, 15
expected_2ab:
    .2byte  8, 2, 4, 6, 16, 10, 12, 14
    .2byte  24, 18, 20, 22, 32, 26, 28, 30
expected_neg_ab:
    .2byte  -4, -1, -2, -3, -8, -5, -6, -7
    .2byte  -12, -9, -10, -11, -16, -13, -14, -15
expected_at_b:
    .2byte  13, 1, 5, 9, 14, 2, 6, 10
    .2byte  15, 3, 7, 11, 16, 4, 8, 12
expected_2at_b:
    .2byte  26, 2, 10, 18, 28, 4, 12, 20
    .2byte  30, 6, 14, 22, 32, 8, 16, 24
expected_neg_at_b:
    .2byte  -13, -1, -5, -9, -14, -2, -6, -10
    .2byte  -15, -3, -7, -11, -16, -4, -8, -12
expected_a_bt:
    .2byte  2, 3, 4, 1, 6, 7, 8, 5
    .2byte  10, 11, 12, 9, 14, 15, 16, 13
expected_2a_bt:
    .2byte  4, 6, 8, 2, 12, 14, 16, 10
    .2byte  20, 22, 24, 18, 28, 30, 32, 26
expected_neg_a_bt:
    .2byte  -2, -3, -4, -1, -6, -7, -8, -5
    .2byte  -10, -11, -12, -9, -14, -15, -16, -13
expected_twos:
    .rept   16
    .2byte  2
    .endr

    .bss
    .balign 256
output:
    .space  32
    .balign 16
semiargs:
#if __riscv_xlen == 64
    .space  16
#else
    .space  8
#endif
