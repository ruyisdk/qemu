/*
 * RISC-V Ztt shared backend MTTCG stress test
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

    .option norvc

#ifndef ZTT_STRESS_BACKENDS
#define ZTT_STRESS_BACKENDS 2
#endif

    .equ ITERATIONS, 2000

    .text
    .global _start
_start:
    csrr    s0, mhartid
    li      t0, 3
    bgeu    s0, t0, park
    li      t0, MSTATUS_AME_MS_INITIAL
    csrs    mstatus, t0
    lla     s1, active_owners
    lla     s2, failures
    lla     s3, completed
    li      s4, ITERATIONS

iteration:
    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_BUSY
    beq     a0, t0, iteration
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail_owned

    li      t0, 1
    amoadd.w.aq t1, t0, (s1)
    li      t2, ZTT_STRESS_BACKENDS
    bgeu    t1, t2, fail_counted

    csrr    t0, CSR_AMEOWN
    li      t1, 1
    bne     t0, t1, fail_counted

    .word   ZTT_MGETTYP(ZTT_R_T0, 0)
    bnez    t0, fail_counted
    li      s5, 0x40000008
    slli    t0, s0, 27
    or      s5, s5, t0
    .word   ZTT_MSETTYP(0, ZTT_R_S5)
    andi    t0, s0, 1
    csrw    CSR_AMEXSAT, t0

    /* Keep the grant long enough for other MTTCG threads to overlap. */
    li      t2, 64
1:
    addi    t2, t2, -1
    bnez    t2, 1b

    .word   ZTT_MGETTYP(ZTT_R_T0, 0)
    bne     t0, s5, fail_counted
    csrr    t0, CSR_AMEXSAT
    andi    t1, s0, 1
    bne     t0, t1, fail_counted

    li      t0, -1
    amoadd.w.rl zero, t0, (s1)
    .word   ZTT_AME_RELEASE
    addi    s4, s4, -1
    bnez    s4, iteration

complete:
    li      t0, 1
    amoadd.w.rl zero, t0, (s3)
    bnez    s0, park
2:
    lw      t0, 0(s3)
    li      t1, 3
    bne     t0, t1, 2b
    lw      a0, 0(s2)
    lw      t0, 0(s1)
    or      a0, a0, t0
    snez    a0, a0
    j       semihost_exit

fail_counted:
    li      t0, -1
    amoadd.w.rl zero, t0, (s1)
fail_owned:
    li      t0, 1
    amoswap.w.rl zero, t0, (s2)
    csrr    t0, CSR_AMEOWN
    beqz    t0, complete
    .word   ZTT_AME_RELEASE
    j       complete

park:
    wfi
    j       park

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
active_owners:
    .word   0
failures:
    .word   0
completed:
    .word   0
semiargs:
#if __riscv_xlen == 64
    .space  16
#else
    .space  8
#endif
