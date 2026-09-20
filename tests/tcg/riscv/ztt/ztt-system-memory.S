/*
 * RISC-V Ztt memory instruction test
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

    .option norvc

    .macro SET_M_DTYPE mreg, encoding
    li      t0, \encoding
    .word   ZTT_MSETTYP(\mreg, ZTT_R_T0)
    .endm

    .macro CHECK_NO_UN
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    bnez    t0, fail
    .endm

    .macro CHECK_UN
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    beqz    t0, fail
    .endm

    .text
    .global _start
_start:
    li      s0, 1
    li      t0, MSTATUS_AME_MS_INITIAL
    csrs    mstatus, t0
    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail

    /* Contiguous row-major load/store and column-major export. */
    li      s0, 2
    SET_M_DTYPE 0, 0x40000008
    lla     t0, input_rm
    .word   ZTT_MLS_RM(0, ZTT_R_T0)
    li      t1, 0
    .word   ZTT_MMOVE8_X_M(ZTT_R_T2, 0, ZTT_R_T1)
    bnez    t2, fail
    li      t1, 15
    .word   ZTT_MMOVE8_X_M(ZTT_R_T2, 0, ZTT_R_T1)
    li      t0, 15
    bne     t2, t0, fail
    lla     t0, output_rm
    .word   ZTT_MSS_RM(0, ZTT_R_T0)
    /* Order the AME store before the following scalar loads. */
    fence   rw, rw
    lla     a0, output_rm
    lla     a1, input_rm
    li      a2, 16
    jal     ra, compare_bytes

    lla     t0, output_cm
    .word   ZTT_MSS_CM(0, ZTT_R_T0)
    fence   rw, rw
    lla     a0, output_cm
    lla     a1, input_cm
    li      a2, 16
    jal     ra, compare_bytes
    CHECK_NO_UN

    /* Column-major load reconstructs the same logical square. */
    li      s0, 3
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
    CHECK_NO_UN

    /* Strided rows preserve gaps and transposed-strided uses columns. */
    li      s0, 4
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

    lla     t0, output_st
    li      t1, 8
    .word   ZTT_MSS_ST(2, ZTT_R_T0, ZTT_R_T1)
    fence   rw, rw
    lla     a0, output_st
    lla     a1, expected_st
    li      a2, 32
    jal     ra, compare_bytes

    li      s0, 5
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

    lla     t0, output_tst
    li      t1, 8
    .word   ZTT_MSS_TST(3, ZTT_R_T0, ZTT_R_T1)
    fence   rw, rw
    lla     a0, output_tst
    lla     a1, expected_tst
    li      a2, 32
    jal     ra, compare_bytes
    CHECK_NO_UN

    /* With overlapping segments, the later segment wins byte by byte. */
    li      s0, 6
    lla     t0, output_overlap_st
    li      t1, 0
    .word   ZTT_MSS_ST(2, ZTT_R_T0, ZTT_R_T1)
    fence   rw, rw
    lla     a0, output_overlap_st
    lla     a1, expected_overlap_st
    li      a2, 4
    jal     ra, compare_bytes

    lla     t0, output_overlap_tst
    li      t1, 0
    .word   ZTT_MSS_TST(2, ZTT_R_T0, ZTT_R_T1)
    fence   rw, rw
    lla     a0, output_overlap_tst
    lla     a1, expected_overlap_tst
    li      a2, 4
    jal     ra, compare_bytes
    CHECK_NO_UN

    /* Address generation accepts a negative XLEN stride. */
    li      s0, 7
    SET_M_DTYPE 4, 0x40000008
    lla     t0, input_st
    addi    t0, t0, 24
    li      t1, -8
    .word   ZTT_MLS_ST(4, ZTT_R_T0, ZTT_R_T1)
    lla     t0, output_general
    .word   ZTT_MSS_RM(4, ZTT_R_T0)
    fence   rw, rw
    lla     a0, output_general
    lla     a1, reverse_rm
    li      a2, 16
    jal     ra, compare_bytes
    CHECK_NO_UN

    /* .1r uses a fixed unit-datatype row-major image and ignores Md. */
    li      s0, 8
    lla     t0, input_raw
    .word   ZTT_MLS_1R(5, ZTT_R_T0)
    lla     t0, output_raw
    .word   ZTT_MSS_1R(5, ZTT_R_T0)
    fence   rw, rw
    lla     a0, output_raw
    lla     a1, input_raw
    li      a2, 16
    jal     ra, compare_bytes
    CHECK_NO_UN

    /* Packed Int4 transfers two squares and pads each TST segment. */
    li      s0, 9
    SET_M_DTYPE 6, 0x40000004
    lla     t0, input_i4
    .word   ZTT_MLS_RM(6, ZTT_R_T0)
    lla     t0, output_i4
    .word   ZTT_MSS_RM(6, ZTT_R_T0)
    fence   rw, rw
    lla     a0, output_i4
    lla     a1, input_i4
    li      a2, 16
    jal     ra, compare_bytes

    lla     t0, output_i4_tst
    li      t1, 4
    .word   ZTT_MSS_TST(6, ZTT_R_T0, ZTT_R_T1)
    fence   rw, rw
    lla     a0, output_i4_tst
    lla     a1, expected_i4_tst
    li      a2, 32
    jal     ra, compare_bytes

    SET_M_DTYPE 7, 0x40000004
    lla     t0, output_i4_tst
    li      t1, 4
    .word   ZTT_MLS_TST(7, ZTT_R_T0, ZTT_R_T1)
    lla     t0, output_i4
    .word   ZTT_MSS_RM(7, ZTT_R_T0)
    fence   rw, rw
    lla     a0, output_i4
    lla     a1, input_i4
    li      a2, 16
    jal     ra, compare_bytes
    CHECK_NO_UN

    /* IEEE data movement is bit-exact even though FP computation is UN. */
    li      s0, 10
    SET_M_DTYPE 8, 0x14300110
    lla     t0, input_fp16
    .word   ZTT_MLS_RM(8, ZTT_R_T0)
    lla     t0, output_fp16
    .word   ZTT_MSS_RM(8, ZTT_R_T0)
    fence   rw, rw
    lla     a0, output_fp16
    lla     a1, input_fp16
    li      a2, 32
    jal     ra, compare_bytes
    CHECK_NO_UN

    /* An unprogrammed datatype reports UN without touching M or memory. */
    li      s0, 11
    lla     t0, input_raw
    .word   ZTT_MLS_1R(10, ZTT_R_T0)
    csrwi   CSR_AMESTATUS, 0
    lla     t0, input_rm
    .word   ZTT_MLS_RM(10, ZTT_R_T0)
    CHECK_UN
    lla     t0, output_unsupported_load
    .word   ZTT_MSS_1R(10, ZTT_R_T0)
    fence   rw, rw
    lla     a0, output_unsupported_load
    lla     a1, input_raw
    li      a2, 16
    jal     ra, compare_bytes

    csrwi   CSR_AMESTATUS, 0
    lla     t0, output_unsupported_store
    .word   ZTT_MSS_RM(10, ZTT_R_T0)
    CHECK_UN
    fence   rw, rw
    lla     a0, output_unsupported_store
    lla     a1, sentinel16
    li      a2, 16
    jal     ra, compare_bytes

    .word   ZTT_AME_RELEASE
    li      a0, 0
    j       semihost_exit

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
input_i4:
    .byte   0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe
    .byte   0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01

    .balign 32
input_fp16:
    .2byte  0x0000, 0x8000, 0x3c00, 0xbc00
    .2byte  0x7c00, 0xfc00, 0x7e01, 0x0001
    .2byte  0x3555, 0xb555, 0x0400, 0x8400
    .2byte  0x7bff, 0xfbff, 0x1234, 0xabcd

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
reverse_rm:
    .byte   12, 13, 14, 15, 8, 9, 10, 11
    .byte   4, 5, 6, 7, 0, 1, 2, 3
expected_st:
    .byte   0, 1, 2, 3, 0xa5, 0xa5, 0xa5, 0xa5
    .byte   4, 5, 6, 7, 0xa5, 0xa5, 0xa5, 0xa5
    .byte   8, 9, 10, 11, 0xa5, 0xa5, 0xa5, 0xa5
    .byte   12, 13, 14, 15, 0xa5, 0xa5, 0xa5, 0xa5
expected_tst:
    .byte   0, 4, 8, 12, 0xa5, 0xa5, 0xa5, 0xa5
    .byte   1, 5, 9, 13, 0xa5, 0xa5, 0xa5, 0xa5
    .byte   2, 6, 10, 14, 0xa5, 0xa5, 0xa5, 0xa5
    .byte   3, 7, 11, 15, 0xa5, 0xa5, 0xa5, 0xa5
expected_i4_tst:
    .byte   0x40, 0xc8, 0xa5, 0xa5, 0x51, 0xd9, 0xa5, 0xa5
    .byte   0x62, 0xea, 0xa5, 0xa5, 0x73, 0xfb, 0xa5, 0xa5
    .byte   0xbf, 0x37, 0xa5, 0xa5, 0xae, 0x26, 0xa5, 0xa5
    .byte   0x9d, 0x15, 0xa5, 0xa5, 0x8c, 0x04, 0xa5, 0xa5
expected_overlap_st:
    .byte   12, 13, 14, 15
expected_overlap_tst:
    .byte   3, 7, 11, 15
sentinel16:
    .fill   16, 1, 0xa5

    .balign 32
output_rm:
    .space  16
output_cm:
    .space  16
output_general:
    .space  32
output_raw:
    .space  16
output_i4:
    .space  16
output_fp16:
    .space  32
output_unsupported_load:
    .space  16
output_unsupported_store:
    .fill   16, 1, 0xa5
output_st:
    .fill   32, 1, 0xa5
output_tst:
    .fill   32, 1, 0xa5
output_i4_tst:
    .fill   32, 1, 0xa5
output_overlap_st:
    .fill   4, 1, 0xa5
output_overlap_tst:
    .fill   4, 1, 0xa5

    .bss
    .balign 16
semiargs:
#if __riscv_xlen == 64
    .space  16
#else
    .space  8
#endif
