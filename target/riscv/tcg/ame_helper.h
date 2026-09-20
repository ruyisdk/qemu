/*
 * RISC-V Attached Matrix Extension (Ztt) helper parameters
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef TARGET_RISCV_TCG_AME_HELPER_H
#define TARGET_RISCV_TCG_AME_HELPER_H

/*
 * How an elementwise helper interprets the integer-register operand.
 * Data scalars use amestype, then convert to the tensor source datatype.
 * Control scalars supply shift counts; exponents use AME_MAX_INT_DTYPE.
 * FIRST/SECOND place a data scalar in the numerical operation's source order.
 */
typedef enum AmeEWScalarKind {
    AME_EW_DATA_SCALAR_FIRST,
    AME_EW_DATA_SCALAR_SECOND,
    AME_EW_CONTROL_SCALAR,
    AME_EW_EXPONENT_SCALAR,
} AmeEWScalarKind;

#endif
