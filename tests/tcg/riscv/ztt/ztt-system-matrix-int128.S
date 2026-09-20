/*
 * RISC-V Ztt Int128 matrix accumulation test
 *
 * Run with N=2, U=64, NM=32, and NA=1.
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

#define DTYPE_INT128       0x40000080
#define DTYPE_INT128_SAT   0x60000080

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
    .word   ZTT_MMOV_M_A(4, 0)
    lla     t0, output
    .word   ZTT_MSS_RM(4, ZTT_R_T0)
    lla     t3, output
    lla     t4, \expected
    li      t5, 16
.Lcheck_acc\@:
    lw      t0, 0(t3)
    lw      t1, 0(t4)
    bne     t0, t1, fail
    addi    t3, t3, 4
    addi    t4, t4, 4
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

    SET_M_DTYPE 0, DTYPE_INT128
    SET_M_DTYPE 2, DTYPE_INT128
    SET_M_DTYPE 4, DTYPE_INT128
    SET_A_DTYPE 0, DTYPE_INT128
    LOAD_M  0, matrix_a
    LOAD_M  2, matrix_b

    /* MIN*MIN + MIN*MAX is exactly +2^127 before final conversion. */
    csrwi   CSR_AMEXSAT, 0
    .word   ZTT_MZERO_ACC(0)
    .word   ZTT_MMULACC_2D(0, 0, 2)
    CHECK_ACC expected_wrapped
    csrr    t0, CSR_AMEXSAT
    andi    t0, t0, AME_XSAT_SAT
    bnez    t0, fail

    SET_M_DTYPE 4, DTYPE_INT128_SAT
    SET_A_DTYPE 0, DTYPE_INT128_SAT
    csrwi   CSR_AMEXSAT, 0
    .word   ZTT_MZERO_ACC(0)
    .word   ZTT_MMULACC_2D(0, 0, 2)
    CHECK_ACC expected_saturated
    csrr    t0, CSR_AMEXSAT
    andi    t0, t0, AME_XSAT_SAT
    beqz    t0, fail

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
    .8byte  0, 0x8000000000000000
    .8byte  0, 0x8000000000000000
    .8byte  0, 0
    .8byte  0, 0
matrix_b:
    .8byte  0, 0x8000000000000000
    .8byte  0, 0
    .8byte  -1, 0x7fffffffffffffff
    .8byte  0, 0
expected_wrapped:
    .8byte  0, 0x8000000000000000
    .rept   3
    .8byte  0, 0
    .endr
expected_saturated:
    .8byte  -1, 0x7fffffffffffffff
    .rept   3
    .8byte  0, 0
    .endr

    .bss
    .balign 256
output:
    .space  64
    .balign 16
semiargs:
#if __riscv_xlen == 64
    .space  16
#else
    .space  8
#endif
