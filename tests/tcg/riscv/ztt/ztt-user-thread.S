/*
 * RISC-V Ztt linux-user thread lifecycle test
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

#define NR_EXIT                 93
#define NR_CLONE                220

#define CLONE_VM                0x00000100
#define CLONE_FS                0x00000200
#define CLONE_FILES             0x00000400
#define CLONE_SIGHAND           0x00000800
#define CLONE_THREAD            0x00010000
#define CLONE_SYSVSEM           0x00040000
#define CLONE_CHILD_CLEARTID    0x00200000
#define CLONE_CHILD_SETTID      0x01000000
#define THREAD_FLAGS            (CLONE_VM | CLONE_FS | CLONE_FILES | \
                                 CLONE_SIGHAND | CLONE_THREAD | \
                                 CLONE_SYSVSEM | CLONE_CHILD_CLEARTID | \
                                 CLONE_CHILD_SETTID)

#define THREAD_ITERATIONS       8
#define MREG_BYTES              16 /* N=4, U=8. */

    .option norvc
    .option norelax

    .macro CHECK_BYTES actual, expected, failure
    lla     t0, \actual
    lla     t1, \expected
    li      t2, MREG_BYTES
1:
    lbu     t3, 0(t0)
    lbu     t4, 0(t1)
    bne     t3, t4, \failure
    addi    t0, t0, 1
    addi    t1, t1, 1
    addi    t2, t2, -1
    bnez    t2, 1b
    .endm

    .text
    .global _start
_start:
    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail

    /* Allocate the parent's memory staging before copying its CPU state. */
    li      t0, 0x40000008
    .word   ZTT_MSETTYP(0, ZTT_R_T0)
    lla     t0, parent_input
    .word   ZTT_MLS_1R(0, ZTT_R_T0)
    li      t0, AME_FFLAG_NV
    csrw    CSR_AMEFFLAGS, t0
    li      t0, AME_XSAT_SAT
    csrw    CSR_AMEXSAT, t0

    li      s0, THREAD_ITERATIONS
thread_loop:
    lla     t0, child_result
    sw      zero, 0(t0)
    lla     t0, child_tid
    li      t1, 1
    sw      t1, 0(t0)
    fence   rw, rw

    li      a0, THREAD_FLAGS
    lla     a1, child_stack_end
    li      a2, 0
    li      a3, 0
    lla     a4, child_tid
    li      a7, NR_CLONE
    ecall
    beqz    a0, child
    bltz    a0, fail

wait_child:
    lla     t0, child_tid
    lw      t1, 0(t0)
    bnez    t1, wait_child
    fence   rw, rw
    lla     t0, child_result
    lw      t1, 0(t0)
    bnez    t1, fail

    csrr    t0, CSR_AMEOWN
    li      t1, 1
    bne     t0, t1, fail
    csrr    t0, CSR_AMEFFLAGS
    li      t1, AME_FFLAG_NV
    bne     t0, t1, fail
    csrr    t0, CSR_AMEXSAT
    li      t1, AME_XSAT_SAT
    bne     t0, t1, fail

    /* Child writes and destruction must leave the parent's storage intact. */
    .word   ZTT_MGETTYP(ZTT_R_T0, 0)
    li      t1, 0x40000008
    bne     t0, t1, fail
    lla     t0, parent_output
    .word   ZTT_MSS_1R(0, ZTT_R_T0)
    CHECK_BYTES parent_output, parent_input, fail

    addi    s0, s0, -1
    bnez    s0, thread_loop

    .word   ZTT_AME_RELEASE
    csrr    t0, CSR_AMEOWN
    bnez    t0, fail
    li      a0, 0
    j       exit

child:
    csrr    t0, CSR_AMEOWN
    bnez    t0, child_fail

    /* Flag CSRs are inaccessible until the child's first, clearing grant. */
    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, child_fail
    csrr    t0, CSR_AMEOWN
    li      t1, 1
    bne     t0, t1, child_fail
    csrr    t0, CSR_AMEFFLAGS
    bnez    t0, child_fail
    csrr    t0, CSR_AMEXSAT
    bnez    t0, child_fail

    .word   ZTT_MGETTYP(ZTT_R_T0, 0)
    bnez    t0, child_fail
    lla     t0, child_output
    .word   ZTT_MSS_1R(0, ZTT_R_T0)
    CHECK_BYTES child_output, zero_input, child_fail

    /* The child has its own descriptors, register data and memory staging. */
    li      t0, 0x40000010
    .word   ZTT_MSETTYP(0, ZTT_R_T0)
    lla     t0, child_input
    .word   ZTT_MLS_1R(0, ZTT_R_T0)
    lla     t0, child_output
    .word   ZTT_MSS_1R(0, ZTT_R_T0)
    CHECK_BYTES child_output, child_input, child_fail
    li      t0, AME_FFLAG_DZ
    csrw    CSR_AMEFFLAGS, t0

    /* Alternate explicit release and cleanup while still owning a backend. */
    andi    t0, s0, 1
    beqz    t0, child_success
    .word   ZTT_AME_RELEASE
    csrr    t0, CSR_AMEOWN
    bnez    t0, child_fail

child_success:
    li      t1, 0
    j       child_exit

child_fail:
    li      t1, 1

child_exit:
    lla     t0, child_result
    sw      t1, 0(t0)
    fence   rw, rw
    li      a0, 0
    li      a7, NR_EXIT
    ecall

fail:
    li      a0, 1
exit:
    li      a7, NR_EXIT
    ecall

    .data
    .balign 16
parent_input:
    .byte   0, 1, 2, 3, 4, 5, 6, 7
    .byte   8, 9, 10, 11, 12, 13, 14, 15
child_input:
    .fill   MREG_BYTES, 1, 0x5a
zero_input:
    .zero   MREG_BYTES

    .bss
    .balign 16
parent_output:
    .space  MREG_BYTES
child_output:
    .space  MREG_BYTES

    .balign 4
child_tid:
    .word   0
child_result:
    .word   0

    .balign 16
child_stack:
    .space  4096
child_stack_end:
