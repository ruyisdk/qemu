/*
 * RISC-V Ztt S-mode and VS-mode state-enablement test
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

#define CSR_VSSTATUS            0x200
#define CSR_HSTATUS             0x600
#define CSR_SEPC                0x141
#define CSR_MSTATUSH            0x310
#define SSTATUS_SPP             (1 << 8)
#define HSTATUS_SPV             (1 << 7)
#define MSTATUS_MPP_MASK        (3 << 11)
#define MSTATUS_MPP_S           (1 << 11)
#define MSTATUS_MPV_RV64        (1ULL << 39)
#define MSTATUSH_MPV_RV32       (1 << 7)

    .option norvc

    .macro ENTER_LOWER_MODE target
    csrr    t0, mstatus
    li      t1, MSTATUS_MPP_MASK
    not     t1, t1
    and     t0, t0, t1
    li      t1, MSTATUS_MPP_S
    or      t0, t0, t1
    csrw    mstatus, t0
#if __riscv_xlen == 64
    li      t0, MSTATUS_MPV_RV64
    csrc    mstatus, t0
#else
    li      t0, MSTATUSH_MPV_RV32
    csrc    CSR_MSTATUSH, t0
#endif
    lla     t0, \target
    csrw    mepc, t0
    mret
    .endm

    .text
    .global _start
_start:
    lla     t0, machine_trap
    csrw    mtvec, t0
    sw      zero, phase, t0

    /* Allow S/VS mode to execute and access the test image. */
    li      t0, -1
    csrw    pmpaddr0, t0
    li      t0, 0x0f
    csrw    pmpcfg0, t0

    li      t0, MSTATUS_AME_MS_INITIAL
    csrs    mstatus, t0
    ENTER_LOWER_MODE s_entry

s_entry:
    csrr    t0, sstatus
    li      t1, MSTATUS_AME_MS_MASK
    and     t0, t0, t1
    li      t1, MSTATUS_AME_MS_INITIAL
    bne     t0, t1, fail
    csrr    t0, CSR_AMENLEN
    li      t1, 4
    bne     t0, t1, fail

    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail
    li      t0, 1
    csrw    CSR_AMEFFLAGS, t0
    csrr    t0, sstatus
    li      t1, MSTATUS_AME_MS_MASK
    and     t0, t0, t1
    li      t1, MSTATUS_AME_MS_DIRTY
    bne     t0, t1, fail
    csrr    t0, sstatus
#if __riscv_xlen == 64
    srli    t0, t0, 63
#else
    srli    t0, t0, 31
#endif
    beqz    t0, fail

    /* Without ownership, S may restore MS to Initial. */
    .word   ZTT_AME_RELEASE
    li      t0, MSTATUS_AME_MS_MASK
    csrc    sstatus, t0
    li      t0, MSTATUS_AME_MS_INITIAL
    csrs    sstatus, t0
    ecall
    j       fail

hs_entry:
    /* Enter VS through HS.  Guest MS is Initial; HS MS is Off. */
    li      t0, HSTATUS_SPV
    csrs    CSR_HSTATUS, t0
    li      t0, SSTATUS_SPP
    csrs    sstatus, t0
    lla     t0, vs_entry
    csrw    CSR_SEPC, t0
    sret

vs_entry:
    li      s0, 1
    lla     s1, vs_enabled
    csrr    t0, CSR_AMENLEN
    j       fail

vs_enabled:
    li      t0, 2
    bne     s0, t0, fail
    csrr    t0, CSR_AMENLEN
    li      t1, 4
    bne     t0, t1, fail

    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail
    li      t0, 3
    csrw    CSR_AMEFFLAGS, t0
    csrr    t0, sstatus
    li      t1, MSTATUS_AME_MS_MASK
    and     t0, t0, t1
    li      t1, MSTATUS_AME_MS_DIRTY
    bne     t0, t1, fail
    csrr    t0, sstatus
#if __riscv_xlen == 64
    srli    t0, t0, 63
#else
    srli    t0, t0, 31
#endif
    beqz    t0, fail
    ecall
    j       fail

machine_trap:
    csrr    t0, mcause
    li      t1, 2
    beq     t0, t1, handle_illegal
    li      t1, 9
    beq     t0, t1, handle_s_ecall
    li      t1, 10
    beq     t0, t1, handle_vs_ecall
    j       fail

handle_illegal:
    li      t0, 1
    bne     s0, t0, fail
    lla     t0, phase
    lw      t1, 0(t0)
    li      t2, 1
    bne     t1, t2, fail

    /* Enable the HS field; the saved guest field remains Initial. */
    li      t1, MSTATUS_AME_MS_INITIAL
    csrs    mstatus, t1
    li      t1, 2
    sw      t1, 0(t0)
    li      s0, 2
    csrw    mepc, s1
    mret

handle_s_ecall:
    lla     t0, phase
    lw      t1, 0(t0)
    bnez    t1, fail
    li      t1, 1
    sw      t1, 0(t0)

    /* Prepare a guest Initial field and an HS Off field. */
    li      t1, MSTATUS_AME_MS_MASK
    csrc    mstatus, t1
    li      t1, MSTATUS_AME_MS_INITIAL
    csrw    CSR_VSSTATUS, t1
    ENTER_LOWER_MODE hs_entry

handle_vs_ecall:
    lla     t0, phase
    lw      t1, 0(t0)
    li      t2, 2
    bne     t1, t2, fail

    /* A VS AME state write dirties both the guest and HS fields. */
    csrr    t0, mstatus
    li      t1, MSTATUS_AME_MS_MASK
    and     t0, t0, t1
    li      t1, MSTATUS_AME_MS_DIRTY
    bne     t0, t1, fail
    csrr    t0, CSR_VSSTATUS
    li      t1, MSTATUS_AME_MS_MASK
    and     t0, t0, t1
    li      t1, MSTATUS_AME_MS_DIRTY
    bne     t0, t1, fail
    csrr    t0, CSR_VSSTATUS
#if __riscv_xlen == 64
    srli    t0, t0, 63
#else
    srli    t0, t0, 31
#endif
    beqz    t0, fail
    csrr    t0, CSR_AMEFFLAGS
    li      t1, 3
    bne     t0, t1, fail

    li      a0, 0
    j       semihost_exit

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
    .balign 4
phase:
    .word   0

    .bss
    .balign 16
semiargs:
#if __riscv_xlen == 64
    .space  16
#else
    .space  8
#endif
