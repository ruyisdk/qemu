/*
 * RISC-V Ztt fixed-width debug move and fence test
 *
 * Run with N=4, U=16, and NM=16.
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

    .option norvc

    .macro EXPECT_ILLEGAL insn
    li      s0, 1
    li      s3, 2
    lla     s1, .Lresume\@
    lla     s2, .Lfault\@
.Lfault\@:
    .word   \insn
    j       fail
.Lresume\@:
    li      t0, 2
    bne     s0, t0, fail
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

    /* Position uses only the low log2(elements-per-M) bits. */
    li      t0, 0x1ab
    li      t1, 33              /* 8-bit position 1 modulo 32. */
    .word   ZTT_MMOVE8_M_X(0, ZTT_R_T0, ZTT_R_T1)
    .word   ZTT_MMOVE8_X_M(ZTT_R_T2, 0, ZTT_R_T1)
    li      t0, 0xab
    bne     t2, t0, fail

    li      t0, 0x1234cdef
    li      t1, 19              /* 16-bit position 3 modulo 16. */
    .word   ZTT_MMOVE16_M_X(0, ZTT_R_T0, ZTT_R_T1)
    .word   ZTT_MMOVE16_X_M(ZTT_R_T2, 0, ZTT_R_T1)
    li      t0, 0xcdef
    bne     t2, t0, fail

    li      t0, 0x89abcdef
    li      t1, 13              /* 32-bit position 5 modulo 8. */
    .word   ZTT_MMOVE32_M_X(0, ZTT_R_T0, ZTT_R_T1)
    .word   ZTT_MMOVE32_X_M(ZTT_R_T2, 0, ZTT_R_T1)
    li      t0, 0x89abcdef
    bne     t2, t0, fail

#if __riscv_xlen == 64
    li      t0, 0x1122334455667788
    li      t1, 7               /* 64-bit position 3 modulo 4. */
    .word   ZTT_MMOVE64_M_X(0, ZTT_R_T0, ZTT_R_T1)
    .word   ZTT_MMOVE64_X_M(ZTT_R_T2, 0, ZTT_R_T1)
    bne     t2, t0, fail
#else
    li      t0, 0x55667788
    li      t1, 7
    EXPECT_ILLEGAL ZTT_MMOVE64_M_X(0, ZTT_R_T0, ZTT_R_T1)
    EXPECT_ILLEGAL ZTT_MMOVE64_X_M(ZTT_R_T2, 0, ZTT_R_T1)
#endif

    /* The raw image proves little-endian slicing and Md independence. */
    lla     t0, raw_output
    .word   ZTT_MSS_1R(0, ZTT_R_T0)
    fence   rw, rw
    lla     a0, raw_output
    lla     a1, raw_expected
    li      a2, 32
    jal     ra, compare_bytes
    csrr    t0, CSR_AMESTATUS
    bnez    t0, fail
    .word   ZTT_MGETTYP(ZTT_R_T0, 0)
    bnez    t0, fail

    /* At U=16, .1r uses a 16-bit row-major image and two-byte alignment. */
    lla     t0, raw_input
    .word   ZTT_MLS_1R(0, ZTT_R_T0)
    lla     t0, raw_output
    .word   ZTT_MSS_1R(0, ZTT_R_T0)
    fence   rw, rw
    lla     a0, raw_output
    lla     a1, raw_input
    li      a2, 32
    jal     ra, compare_bytes
    .word   ZTT_MGETTYP(ZTT_R_T0, 0)
    bnez    t0, fail

    li      s0, 1
    li      s3, 4
    lla     s4, raw_input
    addi    s4, s4, 1
    mv      t0, s4
    lla     s1, .Lafter_misaligned_load
    lla     s2, .Lmisaligned_load
.Lmisaligned_load:
    .word   ZTT_MLS_1R(0, ZTT_R_T0)
    j       fail
.Lafter_misaligned_load:
    li      t0, 2
    bne     s0, t0, fail
    lla     t0, raw_output
    .word   ZTT_MSS_1R(0, ZTT_R_T0)
    fence   rw, rw
    lla     a0, raw_output
    lla     a1, raw_input
    li      a2, 32
    jal     ra, compare_bytes

    li      s0, 1
    li      s3, 6
    lla     s4, misaligned_output
    addi    s4, s4, 1
    mv      t0, s4
    lla     s1, .Lafter_misaligned_store
    lla     s2, .Lmisaligned_store
.Lmisaligned_store:
    .word   ZTT_MSS_1R(0, ZTT_R_T0)
    j       fail
.Lafter_misaligned_store:
    li      t0, 2
    bne     s0, t0, fail
    lla     a0, misaligned_output
    lla     a1, misaligned_expected
    li      a2, 34
    jal     ra, compare_bytes

    /* AME memory accesses use the standard RISC-V fence model. */
    fence   w, r
    fence   r, w
    fence   rw, rw
    csrr    t0, CSR_AMESTATUS
    bnez    t0, fail

    .word   ZTT_AME_RELEASE
    /* The AME fence encoding is reserved. */
    EXPECT_ILLEGAL 0xa620002b
    li      a0, 0
    j       semihost_exit

trap_handler:
    csrr    t0, mcause
    bne     t0, s3, fail
    csrr    t0, mepc
    bne     t0, s2, fail
    li      t0, 2
    beq     s3, t0, .Ltrap_checked
    csrr    t0, mtval
    bne     t0, s4, fail
.Ltrap_checked:
    li      t0, 1
    bne     s0, t0, fail
    li      s0, 2
    csrw    mepc, s1
    mret

compare_bytes:
    beqz    a2, .Lcompare_done
.Lcompare_loop:
    lbu     t0, 0(a0)
    lbu     t1, 0(a1)
    bne     t0, t1, fail
    addi    a0, a0, 1
    addi    a1, a1, 1
    addi    a2, a2, -1
    bnez    a2, .Lcompare_loop
.Lcompare_done:
    ret

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

    .data
    .balign 32
raw_expected:
    .byte   0x00, 0xab, 0x00, 0x00, 0x00, 0x00, 0xef, 0xcd
    .byte   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    .byte   0x00, 0x00, 0x00, 0x00, 0xef, 0xcd, 0xab, 0x89
#if __riscv_xlen == 64
    .byte   0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11
#else
    .fill   8, 1, 0
#endif
raw_output:
    .space  32
raw_input:
    .byte   0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07
    .byte   0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f
    .byte   0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17
    .byte   0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f
misaligned_output:
    .fill   34, 1, 0xa5
misaligned_expected:
    .fill   34, 1, 0xa5

    .bss
    .balign 16
semiargs:
#if __riscv_xlen == 64
    .space  16
#else
    .space  8
#endif
