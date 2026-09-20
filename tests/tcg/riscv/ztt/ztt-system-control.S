/*
 * RISC-V Ztt context and control instruction test
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

    .option norvc

    .macro EXPECT_ILLEGAL insn:vararg
    li      s0, 1
    lla     s1, .Lresume\@
    lla     s2, .Lfault\@
.Lfault\@:
    \insn
    j       fail
.Lresume\@:
    li      t0, 2
    bne     s0, t0, fail
    .endm

    .macro EXPECT_ILLEGAL_WORD insn
    EXPECT_ILLEGAL .word \insn
    .endm

    .text
    .global _start
_start:
    lla     t0, trap_handler
    csrw    mtvec, t0

    /* MS Off rejects even backend-management instructions without writeback. */
    li      a0, 0x55
    li      a1, 0
    EXPECT_ILLEGAL_WORD ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, 0x55
    bne     a0, t0, fail

    li      t0, MSTATUS_AME_MS_INITIAL
    csrs    mstatus, t0

    /* Failed acquire requests preserve ownership, MS, and writable state. */
    li      a1, 64              /* Reserved descriptor bit. */
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_BAD_DESC
    bne     a0, t0, fail
    li      a1, 2               /* Reserved WAIT_MODE. */
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    bne     a0, t0, fail
    li      a1, 17              /* Reserved BOUNDED_WAIT class. */
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    bne     a0, t0, fail
    csrr    t0, CSR_AMEOWN
    bnez    t0, fail
    csrr    t0, mstatus
    li      t1, MSTATUS_AME_MS_MASK
    and     t0, t0, t1
    li      t1, MSTATUS_AME_MS_INITIAL
    bne     t0, t1, fail

    /* TRY ignores TIMEOUT_CLASS. A new grant establishes initial state. */
    li      a1, 60
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail
    csrr    t0, CSR_AMEOWN
    li      t1, 1
    bne     t0, t1, fail
    csrr    t0, CSR_AMEFFLAGS
    bnez    t0, fail
    csrr    t0, CSR_AMEXSAT
    bnez    t0, fail
    csrr    t0, CSR_AMESTATUS
    bnez    t0, fail
    csrr    t0, CSR_AMESTYPE
    bnez    t0, fail
    .word   ZTT_MGETTYP(ZTT_R_T0, 0)
    bnez    t0, fail
    .word   ZTT_AGETTYP(ZTT_R_T0, 0)
    bnez    t0, fail

    /* Set/get uses the low 32 descriptor bits and dirties the context. */
    li      t0, 0x40000008      /* Int8. */
    .word   ZTT_MSETTYP(0, ZTT_R_T0)
    .word   ZTT_MGETTYP(ZTT_R_T1, 0)
    bne     t0, t1, fail
    li      t0, 0x40000020      /* Int32. */
    .word   ZTT_ASETTYP(0, ZTT_R_T0)
    .word   ZTT_AGETTYP(ZTT_R_T1, 0)
    bne     t0, t1, fail
    li      t0, 0x40000010      /* Int16, valid group at m2. */
    .word   ZTT_MSETTYP(2, ZTT_R_T0)
    .word   ZTT_MGETTYP(ZTT_R_T1, 2)
    bne     t0, t1, fail
    .word   ZTT_MZERO_M(2)
    .word   ZTT_MZERO_ACC(0)

    li      t0, 0x40000008
    csrw    CSR_AMESTYPE, t0
    li      t0, 7
    csrw    CSR_AMEFFLAGS, t0
    li      t0, AME_XSAT_SAT
    csrw    CSR_AMEXSAT, t0

    /* A bad descriptor also preserves an existing binding and its flags. */
    li      a1, 64
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_BAD_DESC
    bne     a0, t0, fail

    /* Reacquiring the current binding preserves all state and MS. */
    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail
    .word   ZTT_MGETTYP(ZTT_R_T0, 0)
    li      t1, 0x40000008
    bne     t0, t1, fail
    csrr    t0, CSR_AMESTYPE
    bne     t0, t1, fail
    csrr    t0, CSR_AMEFFLAGS
    li      t1, 7
    bne     t0, t1, fail
    csrr    t0, CSR_AMEXSAT
    li      t1, AME_XSAT_SAT
    bne     t0, t1, fail
    csrr    t0, mstatus
    li      t1, MSTATUS_AME_MS_MASK
    and     t0, t0, t1
    li      t1, MSTATUS_AME_MS_DIRTY
    bne     t0, t1, fail

    /* Unsupported datatypes set UN and preserve descriptors. */
    csrwi   CSR_AMESTATUS, 0
    li      t0, 0x11500108      /* OCP E4M3 with unsupported RNO. */
    .word   ZTT_MSETTYP(0, ZTT_R_T0)
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    beqz    t0, fail
    .word   ZTT_MGETTYP(ZTT_R_T0, 0)
    li      t1, 0x40000008
    bne     t0, t1, fail

    csrwi   CSR_AMESTATUS, 0
    li      t0, 0
    .word   ZTT_ASETTYP(0, ZTT_R_T0)
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    beqz    t0, fail
    .word   ZTT_AGETTYP(ZTT_R_T0, 0)
    li      t1, 0x40000020
    bne     t0, t1, fail

    /* A supported wide datatype with an invalid group is Illegal and no-op. */
    csrwi   CSR_AMESTATUS, 0
    li      t0, 0x40000010
    EXPECT_ILLEGAL_WORD ZTT_MSETTYP(1, ZTT_R_T0)
    .word   ZTT_MGETTYP(ZTT_R_T0, 1)
    bnez    t0, fail
    csrr    t0, CSR_AMESTATUS
    bnez    t0, fail
    li      t0, 0x40000010
    EXPECT_ILLEGAL_WORD ZTT_MSETTYP(15, ZTT_R_T0)
    .word   ZTT_MGETTYP(ZTT_R_T0, 15)
    bnez    t0, fail

    /* An uninitialized zero tuple is unsupported and leaves Md unchanged. */
    csrwi   CSR_AMESTATUS, 0
    .word   ZTT_MZERO_M(4)
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    beqz    t0, fail
    .word   ZTT_MGETTYP(ZTT_R_T0, 4)
    bnez    t0, fail

    /* Encoded physical-register bounds are checked before execution. */
    li      s3, 0x123
    EXPECT_ILLEGAL_WORD ZTT_MGETTYP(ZTT_R_S3, 31)
    li      t0, 0x123
    bne     s3, t0, fail
    EXPECT_ILLEGAL_WORD ZTT_AGETTYP(ZTT_R_S3, 1)
    li      t0, 0x123
    bne     s3, t0, fail

    /* Release preserves hart flags but resets ownership and MS to Initial. */
    .word   ZTT_AME_RELEASE
    csrr    t0, CSR_AMEOWN
    bnez    t0, fail
    EXPECT_ILLEGAL csrr t0, CSR_AMEFFLAGS
    EXPECT_ILLEGAL csrr t0, CSR_AMEXSAT
    csrr    t0, mstatus
    li      t1, MSTATUS_AME_MS_MASK
    and     t0, t0, t1
    li      t1, MSTATUS_AME_MS_INITIAL
    bne     t0, t1, fail

    /* A no-owner context instruction is Illegal with no X writeback. */
    li      s3, 0x456
    EXPECT_ILLEGAL_WORD ZTT_MGETTYP(ZTT_R_S3, 0)
    li      t0, 0x456
    bne     s3, t0, fail

    /* Release by a non-owner is a no-op, including the current MS value. */
    li      t0, MSTATUS_AME_MS_MASK
    csrc    mstatus, t0
    li      t0, MSTATUS_AME_MS_CLEAN
    csrs    mstatus, t0
    .word   ZTT_AME_RELEASE
    csrr    t0, mstatus
    li      t1, MSTATUS_AME_MS_MASK
    and     t0, t0, t1
    li      t1, MSTATUS_AME_MS_CLEAN
    bne     t0, t1, fail

    li      a0, 0
    j       semihost_exit

trap_handler:
    csrr    t0, mcause
    li      t1, 2
    bne     t0, t1, fail
    csrr    t0, mepc
    bne     t0, s2, fail
    li      t0, 1
    bne     s0, t0, fail
    li      s0, 2
    csrw    mepc, s1
    mret

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
