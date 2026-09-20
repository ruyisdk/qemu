/*
 * RISC-V Ztt GDB execution fixture
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

    .option norvc
    .option norelax
    .text
    .global _start
_start:
    j       .

    .global gdb_ame_acquire
gdb_ame_acquire:
    li      t0, MSTATUS_AME_MS_INITIAL
    csrs    mstatus, t0
    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    .global gdb_ame_after_acquire
gdb_ame_after_acquire:
    j       .

    .global gdb_ame_execute
gdb_ame_execute:
    .word   ZTT_MZERO_M(0)
    .global gdb_ame_after_execute
gdb_ame_after_execute:
    j       .

    /*
     * Hart 0 owns the only backend before these entry points are selected by
     * the debugger.  Hart 1 enters BOUNDED_WAIT and hart 0 stops at a
     * breakpoint only after the waiter has reached the acquire instruction.
     */
    .global gdb_ame_owner_hold
gdb_ame_owner_hold:
    lla     t0, gdb_ame_wait_ready
1:
    lw      t1, 0(t0)
    beqz    t1, 1b
    li      t1, 65536
2:
    addi    t1, t1, -1
    bnez    t1, 2b
    .global gdb_ame_owner_wait_stopped
gdb_ame_owner_wait_stopped:
    j       .

    .global gdb_ame_owner_release
gdb_ame_owner_release:
    .word   ZTT_AME_RELEASE
    lla     t0, gdb_ame_wait_done
3:
    lw      t1, 0(t0)
    beqz    t1, 3b
    .global gdb_ame_after_wait
gdb_ame_after_wait:
    j       .

    .global gdb_ame_waiter
gdb_ame_waiter:
    li      t0, MSTATUS_AME_MS_INITIAL
    csrs    mstatus, t0
    lla     t0, gdb_ame_wait_ready
    li      t1, 1
    sw      t1, 0(t0)
    li      a1, 13
    li      a0, 0x55
    .global gdb_ame_wait_acquire
gdb_ame_wait_acquire:
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    lla     t0, gdb_ame_wait_result
    sw      a0, 0(t0)
    lla     t0, gdb_ame_wait_done
    li      t1, 1
    sw      t1, 0(t0)
    .global gdb_ame_waiter_done
gdb_ame_waiter_done:
    j       .

    /*
     * A debugger that redirects a pending waiter here must cancel the old
     * acquire rather than leave a hidden waiter attached to the hart.
     */
    .global gdb_ame_wait_cancelled
gdb_ame_wait_cancelled:
    li      a0, 0x77
    lla     t0, gdb_ame_wait_result
    sw      a0, 0(t0)
    lla     t0, gdb_ame_wait_done
    li      t1, 1
    sw      t1, 0(t0)
    j       gdb_ame_waiter_done

    .global gdb_ame_waiter_release
gdb_ame_waiter_release:
    .word   ZTT_AME_RELEASE
    .global gdb_ame_after_waiter_release
gdb_ame_after_waiter_release:
    j       .

    .bss
    .balign 4
gdb_ame_wait_ready:
    .word   0
gdb_ame_wait_done:
    .word   0
gdb_ame_wait_result:
    .word   0
