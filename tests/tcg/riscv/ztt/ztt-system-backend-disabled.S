/*
 * RISC-V Ztt disabled backend result test
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

    .option norvc

    .text
    .global _start
_start:
    li      t0, MSTATUS_AME_MS_INITIAL
    csrs    mstatus, t0
    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_DISABLED
    bne     a0, t0, fail
    csrr    t0, CSR_AMEOWN
    bnez    t0, fail
    /* Disabling acquisition does not hide the configured capabilities. */
    csrr    t0, CSR_AMENLEN
    li      t1, 4
    bne     t0, t1, fail
    csrr    t0, CSR_AMEUDSZ
    li      t1, 8
    bne     t0, t1, fail
    li      a0, 0
    j       semihost_exit

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

    .bss
    .balign 16
semiargs:
#if __riscv_xlen == 64
    .space  16
#else
    .space  8
#endif
