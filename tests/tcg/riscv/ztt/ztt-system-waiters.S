/*
 * RISC-V Ztt multiple bounded waiters test
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
    csrr    s0, mhartid
    beqz    s0, owner
    li      t0, 3
    bgeu    s0, t0, park
    j       waiter

owner:
    li      t0, MSTATUS_AME_MS_INITIAL
    csrs    mstatus, t0
    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail_owner
    lla     s1, phase
    li      t0, 1
    amoswap.w.rl zero, t0, (s1)

    lla     s2, ready
1:
    lw      t0, 0(s2)
    li      t1, 2
    bne     t0, t1, 1b
    /* Let both harts retire TRY and enter their bounded acquire helper. */
    li      t0, 4096
2:
    addi    t0, t0, -1
    bnez    t0, 2b
    .word   ZTT_AME_RELEASE

    lla     s3, winners
    lla     s4, failures
3:
    lw      t0, 0(s4)
    bnez    t0, fail_owner
    lw      t0, 0(s3)
    beqz    t0, 3b
    li      t1, 1
    bne     t0, t1, fail_owner

    /* The second waiter must remain pending while the winner holds the slot. */
    li      t0, 16384
4:
    addi    t0, t0, -1
    bnez    t0, 4b
    lw      t0, 0(s3)
    li      t1, 1
    bne     t0, t1, fail_owner
    lw      t0, 0(s4)
    bnez    t0, fail_owner
    li      a0, 0
    j       semihost_exit

waiter:
    li      t0, MSTATUS_AME_MS_INITIAL
    csrs    mstatus, t0
    lla     s1, phase
5:
    lw      t0, 0(s1)
    li      t1, 1
    bne     t0, t1, 5b

    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_BUSY
    bne     a0, t0, fail_waiter
    lla     s2, ready
    li      t0, 1
    amoadd.w.rl zero, t0, (s2)

    li      a1, 13              /* BOUNDED_WAIT, VERY_LONG */
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail_waiter
    lla     s3, winners
    li      t0, 1
    amoadd.w.aq t1, t0, (s3)
    bnez    t1, fail_waiter
6:
    wfi
    j       6b

fail_waiter:
    lla     t0, failures
    li      t1, 1
    amoswap.w.rl zero, t1, (t0)
    j       park

fail_owner:
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

park:
    wfi
    j       park

    .bss
    .balign 16
phase:
    .word   0
ready:
    .word   0
winners:
    .word   0
failures:
    .word   0
semiargs:
#if __riscv_xlen == 64
    .space  16
#else
    .space  8
#endif
