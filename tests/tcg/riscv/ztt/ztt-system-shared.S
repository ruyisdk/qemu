/*
 * RISC-V Ztt shared backend ownership test
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

    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail0

    /* Dirty the slot so hart1 can verify that a new grant resets it. */
    li      t0, 0x40000008
    csrw    CSR_AMESTYPE, t0
    .word   ZTT_MSETTYP(0, ZTT_R_T0)
    li      t0, 3
    csrw    CSR_AMEFFLAGS, t0
    li      t0, 1
    csrw    CSR_AMEXSAT, t0

    /* Repeated acquire keeps the existing binding and all state. */
    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail0
    csrr    t0, CSR_AMESTYPE
    li      t1, 0x40000008
    bne     t0, t1, fail0
    .word   ZTT_MGETTYP(ZTT_R_T0, 0)
    li      t1, 0x40000008
    bne     t0, t1, fail0
    csrr    t0, CSR_AMEFFLAGS
    li      t1, 3
    bne     t0, t1, fail0
    csrr    t0, CSR_AMEXSAT
    li      t1, 1
    bne     t0, t1, fail0

    lla     t0, phase
    li      t1, 1
    amoswap.w.rl zero, t1, (t0)

1:
    lw      t1, 0(t0)
    bltz    t1, fail0
    li      t2, 2
    bne     t1, t2, 1b

    .word   ZTT_AME_RELEASE
    li      t1, 3
    amoswap.w.rl zero, t1, (t0)

2:
    lw      t1, 0(t0)
    bltz    t1, fail0
    li      t2, 4
    bne     t1, t2, 2b

    li      a0, 0
    j       semihost_exit

hart1:
    lla     sp, stack1_end
    li      t0, MSTATUS_AME_MS_INITIAL
    csrs    mstatus, t0
    lla     s1, phase
    lla     t0, hart1_trap
    csrw    mtvec, t0
    li      s2, 0

3:
    lw      t0, 0(s1)
    li      t1, 1
    bne     t0, t1, 3b

    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_BUSY
    bne     a0, t0, fail1
    csrr    t0, CSR_AMEOWN
    bnez    t0, fail1

    /* The pool geometry is readable even while another hart owns it. */
    csrr    t0, CSR_AMENLEN
    li      t1, 4
    bne     t0, t1, fail1
    csrr    t0, CSR_AMEUDSZ
    li      t1, 8
    bne     t0, t1, fail1

    /* Backend state and ordinary instructions require ownership. */
    csrr    t0, CSR_AMESTYPE
    .word   ZTT_MZERO_M(0)
    li      t0, 2
    bne     s2, t0, fail1

    /* Unowned flag CSR reads and writes are also Illegal and preserve X. */
    li      s3, 0x55
    csrr    s3, CSR_AMEFFLAGS
    csrw    CSR_AMEFFLAGS, s3
    csrr    s3, CSR_AMEXSAT
    csrw    CSR_AMEXSAT, s3
    li      t0, 6
    bne     s2, t0, fail1
    li      t0, 0x55
    bne     s3, t0, fail1
    csrr    t0, mstatus
    li      t1, MSTATUS_AME_MS_MASK
    and     t0, t0, t1
    li      t1, MSTATUS_AME_MS_INITIAL
    bne     t0, t1, fail1

    li      t0, 2
    amoswap.w.rl zero, t0, (s1)

4:
    lw      t0, 0(s1)
    li      t1, 3
    bne     t0, t1, 4b

    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail1

    /* A slot handed to a new owner is always in architectural initial state. */
    csrr    t0, CSR_AMESTYPE
    bnez    t0, fail1
    csrr    t0, CSR_AMEFFLAGS
    bnez    t0, fail1
    csrr    t0, CSR_AMEXSAT
    bnez    t0, fail1
    .word   ZTT_MGETTYP(ZTT_R_T0, 0)
    bnez    t0, fail1
    .word   ZTT_AGETTYP(ZTT_R_T0, 0)
    bnez    t0, fail1
    csrr    t0, CSR_AMESTATUS
    bnez    t0, fail1

    .word   ZTT_AME_RELEASE
    li      t0, 4
    amoswap.w.rl zero, t0, (s1)
5:
    wfi
    j       5b

    .balign 4
hart1_trap:
    csrr    t0, mcause
    li      t1, 2
    bne     t0, t1, fail1
    addi    s2, s2, 1
    csrr    t0, mepc
    addi    t0, t0, 4
    csrw    mepc, t0
    mret

fail1:
    li      t0, -1
    amoswap.w.rl zero, t0, (s1)
6:
    wfi
    j       6b

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
