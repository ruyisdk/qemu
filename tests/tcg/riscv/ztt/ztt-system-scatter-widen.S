/*
 * RISC-V Ztt widening scatter-max test
 *
 * Run with N=4, U=8, NM=32, and NA=1.
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

#define DTYPE_INT8       0x40000008
#define DTYPE_INT16      0x40000010
#define DTYPE_UINT64     0x00000040

    .option norvc

    .macro SET_M_DTYPE mreg, encoding
    li      t0, \encoding
    .word   ZTT_MSETTYP(\mreg, ZTT_R_T0)
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

    SET_M_DTYPE 0, DTYPE_INT16
    SET_M_DTYPE 2, DTYPE_INT8
    SET_M_DTYPE 8, DTYPE_UINT64

    li      t0, 50
    li      t1, DTYPE_INT8
    .word   ZTT_MBCAST_M_X(0, ZTT_R_T0, ZTT_R_T1)
    li      t1, 0
    li      a2, 16
.Lfill_source:
    srli    t2, t1, 2
    li      t0, 10
    beqz    t2, .Lstore_source
    li      t0, 100
    li      a3, 1
    beq     t2, a3, .Lstore_source
    li      t0, -5
    li      a3, 2
    beq     t2, a3, .Lstore_source
    li      t0, 90
.Lstore_source:
    .word   ZTT_MMOVE8_M_X(2, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    bne     t1, a2, .Lfill_source

    lla     t0, indices
    .word   ZTT_MLS_RM(8, ZTT_R_T0)
    .word   ZTT_MROWSCATMAX_EW(0, 2, 8)
    lla     t0, output
    .word   ZTT_MSS_RM(0, ZTT_R_T0)

    lla     t3, output
    li      t1, 0
    li      a2, 16
.Lcheck_output:
    srli    t2, t1, 2
    li      t4, 50
    li      a3, 2
    bne     t2, a3, .Lcompare_output
    li      t4, 100
.Lcompare_output:
    lhu     t0, 0(t3)
    bne     t0, t4, fail
    addi    t3, t3, 2
    addi    t1, t1, 1
    bne     t1, a2, .Lcheck_output

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
    .balign 128
indices:
    .rept   16
    .8byte  2
    .endr

    .bss
    .balign 128
output:
    .space  32
    .balign 16
semiargs:
#if __riscv_xlen == 64
    .space  16
#else
    .space  8
#endif
