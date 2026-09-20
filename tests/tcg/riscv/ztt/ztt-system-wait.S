/*
 * RISC-V Ztt bounded acquire wait test
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

    .option norvc

    .equ CLINT_MSIP_HART1, 0x02000004

    .text
    .global _start
_start:
    csrr    s0, mhartid
    beqz    s0, hart0
    li      t0, 1
    beq     s0, t0, hart1
park:
    wfi
    j       park

hart0:
    lla     sp, stack0_end
    li      t0, MSTATUS_AME_MS_INITIAL
    csrs    mstatus, t0
    lla     s1, phase

    /* Hold the only slot while hart1 enters BOUNDED_WAIT. */
    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail0
    li      t0, 1
    amoswap.w.rl zero, t0, (s1)
1:
    lw      t0, 0(s1)
    bltz    t0, fail0
    li      t1, 2
    bne     t0, t1, 1b
    .word   ZTT_AME_RELEASE

    /* Hart1 must wake, win the released slot, and release it. */
2:
    lw      t0, 0(s1)
    bltz    t0, fail0
    li      t1, 3
    bne     t0, t1, 2b

    /* Hold the slot beyond the SHORT deadline. */
    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail0
    li      t0, 4
    amoswap.w.rl zero, t0, (s1)
3:
    lw      t0, 0(s1)
    bltz    t0, fail0
    li      t1, 5
    bne     t0, t1, 3b

    /* A deliverable interrupt terminates the wait before slot release. */
    li      t0, 6
    amoswap.w.rl zero, t0, (s1)
4:
    lw      t0, 0(s1)
    bltz    t0, fail0
    li      t1, 7
    bne     t0, t1, 4b
    li      t0, CLINT_MSIP_HART1
    li      t1, 1
    sw      t1, 0(t0)
5:
    lw      t0, 0(s1)
    bltz    t0, fail0
    li      t1, 8
    bne     t0, t1, 5b

    .word   ZTT_AME_RELEASE
    li      a0, 0
    j       semihost_exit

hart1:
    lla     sp, stack1_end
    li      t0, MSTATUS_AME_MS_INITIAL
    csrs    mstatus, t0
    lla     s1, phase
6:
    lw      t0, 0(s1)
    li      t1, 1
    bne     t0, t1, 6b

    li      t0, 2
    amoswap.w.rl zero, t0, (s1)
    li      a1, 13              /* BOUNDED_WAIT, VERY_LONG */
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail1

    /* All standard bounded timeout classes are valid for an owner. */
    li      a1, 1
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    bne     a0, t0, fail1
    li      a1, 5
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    bne     a0, t0, fail1
    li      a1, 9
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    bne     a0, t0, fail1
    li      a1, 13
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    bne     a0, t0, fail1
    li      a1, 61              /* BOUNDED_WAIT, implementation-defined */
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    bne     a0, t0, fail1
    .word   ZTT_AME_RELEASE
    li      t0, 3
    amoswap.w.rl zero, t0, (s1)

7:
    lw      t0, 0(s1)
    li      t1, 4
    bne     t0, t1, 7b
    li      a1, 1               /* BOUNDED_WAIT, SHORT */
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_TIMEOUT
    bne     a0, t0, fail1
    li      t0, 5
    amoswap.w.rl zero, t0, (s1)

8:
    lw      t0, 0(s1)
    li      t1, 6
    bne     t0, t1, 8b
    lla     t0, irq_handler
    csrw    mtvec, t0
    li      t0, 1 << 3          /* mie.MSIE */
    csrs    mie, t0
    li      t0, 1 << 3          /* mstatus.MIE */
    csrs    mstatus, t0
    li      t0, 7
    amoswap.w.rl zero, t0, (s1)
    li      a1, 13              /* BOUNDED_WAIT, VERY_LONG */
    li      a0, 0x55            /* Handler must observe the final writeback. */
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
interrupted_acquire_retired:
    li      t0, AME_ACQUIRE_INTERRUPTED
    bne     a0, t0, fail1
    lla     t0, irq_seen
    lw      t1, 0(t0)
    li      t2, 1
    bne     t1, t2, fail1
    li      t0, 8
    amoswap.w.rl zero, t0, (s1)
9:
    wfi
    j       9b

    .balign 4
irq_handler:
    li      t0, AME_ACQUIRE_INTERRUPTED
    bne     a0, t0, fail1
    csrr    t0, mepc
    lla     t1, interrupted_acquire_retired
    bne     t0, t1, fail1
    lla     t0, irq_seen
    li      t1, 1
    sw      t1, 0(t0)
    li      t0, CLINT_MSIP_HART1
    sw      zero, 0(t0)
    mret

fail1:
    li      t0, -1
    amoswap.w.rl zero, t0, (s1)
10:
    wfi
    j       10b

fail0:
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
phase:
    .word   0
irq_seen:
    .word   0
    .balign 16
stack0:
    .space  4096
stack0_end:
stack1:
    .space  4096
stack1_end:
semiargs:
#if __riscv_xlen == 64
    .space  16
#else
    .space  8
#endif
