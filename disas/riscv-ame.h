/*
 * QEMU RISC-V Attached Matrix Extension disassembler
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef DISAS_RISCV_AME_H
#define DISAS_RISCV_AME_H

#include "disas/riscv.h"

/* Mnemonics and operand formats indexed by the op value from decode_ame(). */
extern const rv_opcode_data ame_opcode_data[];

/**
 * decode_ame: Select the AME opcode for a raw instruction
 * @dec: Read inst and replace op; operand decoding is separate
 * @isa: ISA width; AME opcode selection is width-independent
 *
 * Set op to an index in ame_opcode_data[], or rv_op_illegal for an unrecognized
 * encoding. Extension enablement is checked by the disassembler's caller.
 */
void decode_ame(rv_decode *dec, rv_isa isa);

#endif /* DISAS_RISCV_AME_H */
