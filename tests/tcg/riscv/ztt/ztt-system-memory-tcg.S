/*
 * Ztt memory integration with TCG watchpoints, plugins and translated code.
 *
 * Copyright (c) 2026 Alibaba Group
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

    .option norvc
    .text
    .global _start
_start:
    li      s0, 1
    lla     t0, fail
    csrw    mtvec, t0
    li      t0, MSTATUS_AME_MS_INITIAL
    csrs    mstatus, t0
    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail

    /* N=4, U=8: a raw M register contains sixteen bytes. */
    lla     t0, memory_input
    .global memory_load
memory_load:
    .word   ZTT_MLS_1R(0, ZTT_R_T0)
    .global memory_after_load
memory_after_load:
    lla     t0, memory_output
    .global memory_store
memory_store:
    .word   ZTT_MSS_1R(0, ZTT_R_T0)
    .global memory_after_store
memory_after_store:
    fence   rw, rw
    li      s0, 2
    lla     t0, memory_output
    li      t1, 0
1:
    lbu     t2, 0(t0)
    bne     t2, t1, fail
    addi    t0, t0, 1
    addi    t1, t1, 1
    li      t2, 16
    bne     t1, t2, 1b

    /* Populate a TB on a different page, then replace it with an AME store. */
    li      s0, 3
    lla     s1, memory_code
    jalr    s1
    li      t0, 1
    bne     a0, t0, fail
    lla     t0, replacement_code
    .word   ZTT_MLS_1R(0, ZTT_R_T0)
    mv      t0, s1
    .word   ZTT_MSS_1R(0, ZTT_R_T0)
    fence   rw, rw
    /*
     * Deliberately omit fence.i to test TCG's write-to-code invalidation,
     * not the ISA's weaker guarantees for self-modifying guest programs.
     */
    jalr    s1
    li      t0, 42
    bne     a0, t0, fail

    /*
     * For instructions later in this TB, use the architectural fence.i.
     * RISC-V does not request TCG's precise_smc re-execution behavior.
     */
    li      s0, 4
    lla     t0, replacement_inline
    .word   ZTT_MLS_1R(0, ZTT_R_T0)
    lla     t0, inline_code
    .word   ZTT_MSS_1R(0, ZTT_R_T0)
    .word   0x0000100f             /* fence.i */
inline_code:
    li      a0, 1
    nop
    nop
    nop
    li      t0, 42
    bne     a0, t0, fail
    csrr    t0, CSR_AMESTATUS
    bnez    t0, fail
    .word   ZTT_AME_RELEASE
    li      a0, 0
    j       semihost_exit

fail:
    mv      a0, s0
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

    .data
    .balign 4096
memory_code:
    li      a0, 1
    ret
    nop
    nop
replacement_code:
    li      a0, 42
    ret
    nop
    nop
replacement_inline:
    li      a0, 42
    nop
    nop
    nop
    .balign 4096
    .space  4096 - 8
    .global memory_input
memory_input:
    .byte   0, 1, 2, 3, 4, 5, 6, 7
    .byte   8, 9, 10, 11, 12, 13, 14, 15
    .balign 4096
    .space  4096 - 8
    .global memory_output
memory_output:
    .fill   16, 1, 0xa5

    .bss
    .balign 16
semiargs:
    .space  16
