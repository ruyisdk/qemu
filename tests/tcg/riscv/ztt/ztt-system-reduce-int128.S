/*
 * RISC-V Ztt Int128 reduction test
 *
 * Run with N=4, U=8, NM=32, and NA=1.
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

#define DTYPE_UINT128    0x00000080

    .option norvc

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

    li      t0, DTYPE_UINT128
    .word   ZTT_MSETTYP(0, ZTT_R_T0)
    .word   ZTT_MSETTYP(16, ZTT_R_T0)
    lla     t0, input
    .word   ZTT_MLS_RM(0, ZTT_R_T0)
    .word   ZTT_MREDUCEADD_ROW(16, 0)
    lla     t0, output
    .word   ZTT_MSS_RM(16, ZTT_R_T0)

    /* UINT64_MAX + 1 carries into bit 64, then repeats across each row. */
    lla     t3, output
    li      t4, 16
.Lcheck_output:
    lw      t0, 0(t3)
    bnez    t0, fail
    lw      t0, 4(t3)
    bnez    t0, fail
    lw      t0, 8(t3)
    li      t1, 1
    bne     t0, t1, fail
    lw      t0, 12(t3)
    bnez    t0, fail
    addi    t3, t3, 16
    addi    t4, t4, -1
    bnez    t4, .Lcheck_output

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
input:
    .rept   4
    .8byte  0xffffffffffffffff, 0
    .8byte  1, 0
    .8byte  0, 0
    .8byte  0, 0
    .endr

    .bss
    .balign 256
output:
    .space  256
    .balign 16
semiargs:
#if __riscv_xlen == 64
    .space  16
#else
    .space  8
#endif
