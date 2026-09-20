/*
 * RISC-V Attached Matrix Extension capability wire representation
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef TARGET_RISCV_AME_CAP_WIRE_H
#define TARGET_RISCV_AME_CAP_WIRE_H

#include "target/riscv/ame/cap-set.h"

/**
 * riscv_ame_capabilities_manifest: Encode capabilities for migration comparison
 * @set: Constructed, normalized capability set
 * @errp: Error destination
 *
 * Serialize datatype encodings and explicit operation/form/category wire IDs in
 * deterministic order.
 *
 * Returns: a newly allocated byte array, released with g_byte_array_unref();
 * NULL for invalid entries or an excessive wire size, setting @errp.
 */
GByteArray *riscv_ame_capabilities_manifest(const AmeCapabilities *set,
                                          Error **errp);

#endif /* TARGET_RISCV_AME_CAP_WIRE_H */
