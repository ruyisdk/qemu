/*
 * RISC-V Ztt linux-user memory instruction test
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

#define NR_EXIT 93

    .option norvc
    .option norelax

    .macro SET_M_DTYPE mreg, encoding
    li      t0, \encoding
    .word   ZTT_MSETTYP(\mreg, ZTT_R_T0)
    .endm

    .text
    .global _start
_start:
    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail

    /* Contiguous row-major and column-major transfers. */
    SET_M_DTYPE 0, 0x40000008
    lla     t0, input_rm
    .word   ZTT_MLS_RM(0, ZTT_R_T0)
    lla     t0, output_rm
    .word   ZTT_MSS_RM(0, ZTT_R_T0)
    lla     t0, output_cm
    .word   ZTT_MSS_CM(0, ZTT_R_T0)
    fence   rw, rw
    lla     a0, output_rm
    lla     a1, input_rm
    li      a2, 16
    jal     ra, compare_bytes
    lla     a0, output_cm
    lla     a1, input_cm
    li      a2, 16
    jal     ra, compare_bytes

    SET_M_DTYPE 1, 0x40000008
    lla     t0, input_cm
    .word   ZTT_MLS_CM(1, ZTT_R_T0)
    lla     t0, output_general
    .word   ZTT_MSS_RM(1, ZTT_R_T0)
    fence   rw, rw
    lla     a0, output_general
    lla     a1, input_rm
    li      a2, 16
    jal     ra, compare_bytes

    /* Strided rows and columns ignore the gaps between segments. */
    SET_M_DTYPE 2, 0x40000008
    lla     t0, input_st
    li      t1, 8
    .word   ZTT_MLS_ST(2, ZTT_R_T0, ZTT_R_T1)
    lla     t0, output_general
    .word   ZTT_MSS_RM(2, ZTT_R_T0)
    fence   rw, rw
    lla     a0, output_general
    lla     a1, input_rm
    li      a2, 16
    jal     ra, compare_bytes

    SET_M_DTYPE 3, 0x40000008
    lla     t0, input_tst
    li      t1, 8
    .word   ZTT_MLS_TST(3, ZTT_R_T0, ZTT_R_T1)
    lla     t0, output_general
    .word   ZTT_MSS_RM(3, ZTT_R_T0)
    fence   rw, rw
    lla     a0, output_general
    lla     a1, input_rm
    li      a2, 16
    jal     ra, compare_bytes

    /* .1r uses a fixed unit-datatype row-major image without consulting Md. */
    lla     t0, input_raw
    .word   ZTT_MLS_1R(4, ZTT_R_T0)
    lla     t0, output_raw
    .word   ZTT_MSS_1R(4, ZTT_R_T0)
    fence   rw, rw
    lla     a0, output_raw
    lla     a1, input_raw
    li      a2, 16
    jal     ra, compare_bytes

    .word   ZTT_AME_RELEASE
    li      a0, 0
    j       exit

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
exit:
    li      a7, NR_EXIT
    ecall

    .data
    .balign 32
input_rm:
    .byte   0, 1, 2, 3, 4, 5, 6, 7
    .byte   8, 9, 10, 11, 12, 13, 14, 15
input_cm:
    .byte   0, 4, 8, 12, 1, 5, 9, 13
    .byte   2, 6, 10, 14, 3, 7, 11, 15
input_raw:
    .byte   0x81, 0x72, 0x63, 0x54, 0x45, 0x36, 0x27, 0x18
    .byte   0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87

    .balign 16
input_st:
    .byte   0, 1, 2, 3, 0xee, 0xee, 0xee, 0xee
    .byte   4, 5, 6, 7, 0xee, 0xee, 0xee, 0xee
    .byte   8, 9, 10, 11, 0xee, 0xee, 0xee, 0xee
    .byte   12, 13, 14, 15, 0xee, 0xee, 0xee, 0xee
input_tst:
    .byte   0, 4, 8, 12, 0xee, 0xee, 0xee, 0xee
    .byte   1, 5, 9, 13, 0xee, 0xee, 0xee, 0xee
    .byte   2, 6, 10, 14, 0xee, 0xee, 0xee, 0xee
    .byte   3, 7, 11, 15, 0xee, 0xee, 0xee, 0xee

    .balign 32
output_rm:
    .space  16
output_cm:
    .space  16
output_general:
    .space  16
output_raw:
    .space  16
