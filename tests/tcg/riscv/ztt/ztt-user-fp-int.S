/*
 * Ztt linux-user floating-to-integer conversion test.
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"
#include "include/ztt-fp-int.S"

    .option norvc
    .text
    .global _start
_start:
    li      s0, 1
    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail
    F2I_TESTS
    .word   ZTT_AME_RELEASE
    li      a0, 0
    j       exit
fail:
    mv      a0, s0
exit:
    li      a7, 93
    ecall
