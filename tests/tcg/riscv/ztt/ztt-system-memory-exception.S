/*
 * RISC-V Ztt precise memory exception test
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

#define MSTATUS_MPP_MASK       (3 << 11)
#define MSTATUS_MPP_S          (1 << 11)
#define MSTATUS_MPRV           (1 << 17)

#define EXCP_LOAD_MISALIGNED   4
#define EXCP_LOAD_ACCESS       5
#define EXCP_STORE_MISALIGNED  6
#define EXCP_STORE_ACCESS      7
#define EXCP_LOAD_PAGE         13
#define EXCP_STORE_PAGE        15

    .option norvc

    .macro SET_M_DTYPE mreg, encoding
    li      t0, \encoding
    .word   ZTT_MSETTYP(\mreg, ZTT_R_T0)
    .endm

    .macro EXPECT_TRAP cause, base, resume
    li      s1, \cause
    mv      s2, \base
    lla     s3, \resume
    li      s4, 0
    lla     s6, .Lfault\@
.Lfault\@:
    .endm

    .macro CHECK_TRAP
    li      t0, 1
    bne     s4, t0, fail
    .endm

    .macro SET_TRANSACTION_MARKERS
    li      t0, AME_FFLAGS_MASK
    csrw    CSR_AMEFFLAGS, t0
    li      t0, AME_XSAT_SAT
    csrw    CSR_AMEXSAT, t0
    .endm

    .macro CHECK_TRANSACTION_MARKERS
    csrr    t0, CSR_AMEFFLAGS
    li      t1, AME_FFLAGS_MASK
    bne     t0, t1, fail
    csrr    t0, CSR_AMEXSAT
    li      t1, AME_XSAT_SAT
    bne     t0, t1, fail
    .endm

    .macro ENABLE_MPRV
    li      t0, MSTATUS_MPP_MASK
    csrc    mstatus, t0
    li      t0, MSTATUS_MPP_S | MSTATUS_MPRV
    csrs    mstatus, t0
    .endm

    .macro DISABLE_MPRV
    li      t0, MSTATUS_MPRV
    csrc    mstatus, t0
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
    SET_M_DTYPE 0, 0x40000010

    /* Int16 needs two-byte, not transfer-size, base alignment. */
    lla     t0, element_aligned_input
    addi    t0, t0, 2
    .word   ZTT_MLS_RM(0, ZTT_R_T0)
    lla     t0, element_aligned_output
    addi    t0, t0, 2
    .word   ZTT_MSS_RM(0, ZTT_R_T0)
    fence   rw, rw
    lla     a0, element_aligned_output
    addi    a0, a0, 2
    lla     a1, element_aligned_input
    addi    a1, a1, 2
    li      a2, 32
    jal     ra, compare_bytes

    /* Alignment faults precede memory access and preserve both sides. */
    li      s0, 1
    lla     t0, input_rm
    addi    t0, t0, 1
    EXPECT_TRAP EXCP_LOAD_MISALIGNED, t0, .Lafter_misaligned_load
    .word   ZTT_MLS_RM(0, ZTT_R_T0)
    j       fail
.Lafter_misaligned_load:
    CHECK_TRAP
    lla     t0, output_group
    .word   ZTT_MSS_RM(0, ZTT_R_T0)
    fence   rw, rw
    lla     a0, output_group
    lla     a1, element_aligned_input
    addi    a1, a1, 2
    li      a2, 32
    jal     ra, compare_bytes

    li      s0, 2
    lla     t0, misaligned_output
    addi    t0, t0, 1
    EXPECT_TRAP EXCP_STORE_MISALIGNED, t0, .Lafter_misaligned_store
    .word   ZTT_MSS_RM(0, ZTT_R_T0)
    j       fail
.Lafter_misaligned_store:
    CHECK_TRAP
    lla     a0, misaligned_output
    lla     a1, sentinel32
    li      a2, 32
    jal     ra, compare_bytes

    SET_M_DTYPE 0, 0x40000008
    lla     t0, input_raw
    .word   ZTT_MLS_1R(0, ZTT_R_T0)

    /* A later MMIO segment is a PMA failure; neither side is modified. */
    li      s0, 3
    lla     t0, input_raw
    li      t1, 0x10000000
    sub     t1, t1, t0
    li      t2, 0x10000000
    EXPECT_TRAP EXCP_LOAD_ACCESS, t2, .Lafter_pma_load
    .word   ZTT_MLS_ST(0, ZTT_R_T0, ZTT_R_T1)
    j       fail
.Lafter_pma_load:
    CHECK_TRAP
    lla     t0, output_raw
    .word   ZTT_MSS_1R(0, ZTT_R_T0)
    fence   rw, rw
    lla     a0, output_raw
    lla     a1, input_raw
    li      a2, 16
    jal     ra, compare_bytes

    li      s0, 4
    lla     t0, pma_output
    li      t1, 0x10000000
    sub     t1, t1, t0
    li      t2, 0x10000000
    EXPECT_TRAP EXCP_STORE_ACCESS, t2, .Lafter_pma_store
    .word   ZTT_MSS_ST(0, ZTT_R_T0, ZTT_R_T1)
    j       fail
.Lafter_pma_store:
    CHECK_TRAP
    lla     a0, pma_output
    lla     a1, sentinel16
    li      a2, 16
    jal     ra, compare_bytes

    /* Permit S-mode data and page-table walks before testing MPRV. */
    csrw    pmpcfg0, zero
    li      t0, -1
    csrw    pmpaddr0, t0
    li      t0, 0x1f
    csrw    pmpcfg0, t0
    jal     ra, build_page_tables
    mv      s5, a0

    /* The third strided segment is unmapped: load has no partial write. */
    li      s0, 5
    csrw    satp, s5
    sfence.vma zero, zero
    SET_TRANSACTION_MARKERS
    ENABLE_MPRV
    li      t0, 0x40001000
    addi    t0, t0, -16
    li      t1, 8
    li      t2, 0x40001000
    EXPECT_TRAP EXCP_LOAD_PAGE, t2, .Lafter_page_load
    .word   ZTT_MLS_ST(0, ZTT_R_T0, ZTT_R_T1)
    j       fail
.Lafter_page_load:
    CHECK_TRAP
    CHECK_TRANSACTION_MARKERS
    csrw    satp, zero
    sfence.vma zero, zero
    lla     t0, output_raw
    .word   ZTT_MSS_1R(0, ZTT_R_T0)
    fence   rw, rw
    lla     a0, output_raw
    lla     a1, input_raw
    li      a2, 16
    jal     ra, compare_bytes

    /* The same faulting store leaves the two mapped segments untouched. */
    li      s0, 6
    csrw    satp, s5
    sfence.vma zero, zero
    SET_TRANSACTION_MARKERS
    ENABLE_MPRV
    li      t0, 0x40001000
    addi    t0, t0, -16
    li      t1, 8
    li      t2, 0x40001000
    EXPECT_TRAP EXCP_STORE_PAGE, t2, .Lafter_page_store
    .word   ZTT_MSS_ST(0, ZTT_R_T0, ZTT_R_T1)
    j       fail
.Lafter_page_store:
    CHECK_TRAP
    CHECK_TRANSACTION_MARKERS
    csrw    satp, zero
    sfence.vma zero, zero
    lla     t0, fault_page
    li      t1, 4096 - 16
    add     a0, t0, t1
    lla     a1, sentinel16
    li      a2, 16
    jal     ra, compare_bytes

    /* Segment address addition wraps naturally at the active XLEN. */
    li      s0, 7
    csrw    satp, s5
    sfence.vma zero, zero
    ENABLE_MPRV
    li      t0, -16
    li      t1, 8
    .word   ZTT_MLS_ST(0, ZTT_R_T0, ZTT_R_T1)
    DISABLE_MPRV
    csrw    satp, zero
    sfence.vma zero, zero
    lla     t0, output_wrap
    .word   ZTT_MSS_RM(0, ZTT_R_T0)
    fence   rw, rw
    lla     a0, output_wrap
    lla     a1, input_rm
    li      a2, 16
    jal     ra, compare_bytes

    /* A denied PMP segment reports its failing address and stores nothing. */
    li      s0, 8
    csrw    pmpcfg0, zero
    lla     t0, pmp_deny
    srli    t0, t0, 2
    ori     t0, t0, 1
    csrw    pmpaddr0, t0
    li      t0, -1
    csrw    pmpaddr1, t0
    li      t0, 0x1f18
    csrw    pmpcfg0, t0

    ENABLE_MPRV
    lla     t0, pmp_output
    lla     t1, pmp_deny
    sub     t1, t1, t0
    add     t2, t0, t1
    EXPECT_TRAP EXCP_STORE_ACCESS, t2, .Lafter_pmp_store
    .word   ZTT_MSS_ST(0, ZTT_R_T0, ZTT_R_T1)
    j       fail
.Lafter_pmp_store:
    CHECK_TRAP
    lla     a0, pmp_output
    lla     a1, sentinel16
    li      a2, 16
    jal     ra, compare_bytes

    /* A denied PMP load segment preserves the complete M destination. */
    li      s0, 9
    SET_TRANSACTION_MARKERS
    ENABLE_MPRV
    lla     t0, pmp_output
    lla     t1, pmp_deny
    sub     t1, t1, t0
    add     t2, t0, t1
    EXPECT_TRAP EXCP_LOAD_ACCESS, t2, .Lafter_pmp_load
    .word   ZTT_MLS_ST(0, ZTT_R_T0, ZTT_R_T1)
    j       fail
.Lafter_pmp_load:
    CHECK_TRAP
    CHECK_TRANSACTION_MARKERS
    lla     t0, output_wrap
    .word   ZTT_MSS_RM(0, ZTT_R_T0)
    fence   rw, rw
    lla     a0, output_wrap
    lla     a1, input_rm
    li      a2, 16
    jal     ra, compare_bytes

    csrw    pmpcfg0, zero
    li      t0, -1
    csrw    pmpaddr0, t0
    li      t0, 0x1f
    csrw    pmpcfg0, t0
    .word   ZTT_AME_RELEASE
    li      a0, 0
    j       semihost_exit

trap_handler:
    csrr    t0, mcause
    bne     t0, s1, fail
    csrr    t0, mepc
    bne     t0, s6, fail
    csrr    t0, mtval
    bne     t0, s2, fail
    li      s4, 1
    li      t0, MSTATUS_MPRV
    csrc    mstatus, t0
    csrw    mepc, s3
    mret

/* Return the XLEN-specific SATP value for three mapped virtual regions. */
build_page_tables:
    lla     t0, fault_page
    li      t1, 0xa5a5a5a5
    li      t2, 4096 - 16
    add     t0, t0, t2
    sw      t1, 0(t0)
    sw      t1, 4(t0)
    sw      t1, 8(t0)
    sw      t1, 12(t0)

    lla     t0, wrap_hi_page
    li      t1, 4096 - 16
    add     t0, t0, t1
    li      t1, 0x03020100
    sw      t1, 0(t0)
    li      t1, 0x07060504
    sw      t1, 8(t0)
    lla     t0, wrap_lo_page
    li      t1, 0x0b0a0908
    sw      t1, 0(t0)
    li      t1, 0x0f0e0d0c
    sw      t1, 8(t0)

#if __riscv_xlen == 64
    lla     t0, pt_pf_l1
    srli    t1, t0, 2
    ori     t1, t1, 1
    lla     t2, pt_root
    sd      t1, 8(t2)             /* VA 0x40000000: VPN[2] = 1 */
    lla     t0, pt_pf_l0
    srli    t1, t0, 2
    ori     t1, t1, 1
    lla     t2, pt_pf_l1
    sd      t1, 0(t2)
    lla     t0, fault_page
    srli    t1, t0, 2
    ori     t1, t1, 0xc7
    lla     t2, pt_pf_l0
    sd      t1, 0(t2)

    lla     t0, pt_hi_l1
    srli    t1, t0, 2
    ori     t1, t1, 1
    lla     t2, pt_root
    li      t3, 4088
    add     t2, t2, t3
    sd      t1, 0(t2)
    lla     t0, pt_hi_l0
    srli    t1, t0, 2
    ori     t1, t1, 1
    lla     t2, pt_hi_l1
    li      t3, 4088
    add     t2, t2, t3
    sd      t1, 0(t2)
    lla     t0, wrap_hi_page
    srli    t1, t0, 2
    ori     t1, t1, 0xc7
    lla     t2, pt_hi_l0
    li      t3, 4088
    add     t2, t2, t3
    sd      t1, 0(t2)

    lla     t0, pt_lo_l1
    srli    t1, t0, 2
    ori     t1, t1, 1
    lla     t2, pt_root
    sd      t1, 0(t2)
    lla     t0, pt_lo_l0
    srli    t1, t0, 2
    ori     t1, t1, 1
    lla     t2, pt_lo_l1
    sd      t1, 0(t2)
    lla     t0, wrap_lo_page
    srli    t1, t0, 2
    ori     t1, t1, 0xc7
    lla     t2, pt_lo_l0
    sd      t1, 0(t2)

    lla     a0, pt_root
    srli    a0, a0, 12
    li      t0, 8
    slli    t0, t0, 60
    or      a0, a0, t0
#else
    lla     t0, pt_pf_l0
    srli    t1, t0, 2
    ori     t1, t1, 1
    lla     t2, pt_root
    sw      t1, 1024(t2)          /* VA 0x40000000: VPN[1] = 256 */
    lla     t0, fault_page
    srli    t1, t0, 2
    ori     t1, t1, 0xc7
    lla     t2, pt_pf_l0
    sw      t1, 0(t2)

    lla     t0, pt_hi_l0
    srli    t1, t0, 2
    ori     t1, t1, 1
    lla     t2, pt_root
    li      t3, 4092
    add     t2, t2, t3
    sw      t1, 0(t2)
    lla     t0, wrap_hi_page
    srli    t1, t0, 2
    ori     t1, t1, 0xc7
    lla     t2, pt_hi_l0
    li      t3, 4092
    add     t2, t2, t3
    sw      t1, 0(t2)

    lla     t0, pt_lo_l0
    srli    t1, t0, 2
    ori     t1, t1, 1
    lla     t2, pt_root
    sw      t1, 0(t2)
    lla     t0, wrap_lo_page
    srli    t1, t0, 2
    ori     t1, t1, 0xc7
    lla     t2, pt_lo_l0
    sw      t1, 0(t2)

    lla     a0, pt_root
    srli    a0, a0, 12
    li      t0, 1
    slli    t0, t0, 31
    or      a0, a0, t0
#endif
    ret

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
element_aligned_input:
    .2byte  0xa5a5
    .byte   0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07
    .byte   0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f
    .byte   0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17
    .byte   0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f
    .2byte  0xa5a5
element_aligned_output:
    .fill   36, 1, 0xa5
    .balign 32
output_group:
    .space  32
input_rm:
    .byte   0, 1, 2, 3, 4, 5, 6, 7
    .byte   8, 9, 10, 11, 12, 13, 14, 15
input_raw:
    .byte   0x81, 0x72, 0x63, 0x54, 0x45, 0x36, 0x27, 0x18
    .byte   0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87
sentinel16:
    .fill   16, 1, 0xa5
sentinel32:
    .fill   32, 1, 0xa5

    .balign 32
output_raw:
    .space  16
output_wrap:
    .space  16
misaligned_output:
    .fill   32, 1, 0xa5
pma_output:
    .fill   16, 1, 0xa5
pmp_output:
    .fill   16, 1, 0xa5
    .balign 16
pmp_deny:
    .fill   16, 1, 0xa5

    .bss
    .balign 4096
pt_root:
    .space  4096
pt_pf_l1:
    .space  4096
pt_pf_l0:
    .space  4096
pt_hi_l1:
    .space  4096
pt_hi_l0:
    .space  4096
pt_lo_l1:
    .space  4096
pt_lo_l0:
    .space  4096
fault_page:
    .space  4096
wrap_hi_page:
    .space  4096
wrap_lo_page:
    .space  4096
semiargs:
#if __riscv_xlen == 64
    .space  16
#else
    .space  8
#endif
