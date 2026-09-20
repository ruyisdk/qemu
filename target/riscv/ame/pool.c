/*
 * RISC-V Attached Matrix Extension backend pool device
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "hw/core/qdev-properties.h"
#include "hw/core/resettable.h"
#include "qemu/module.h"
#include "target/riscv/ame/pool.h"

static bool riscv_ame_backend_pool_prepare(RISCVAMEBackendPool *pool,
                                           Error **errp)
{
    if (pool->core.lock_initialized) {
        return true;
    }
    return riscv_ame_pool_init(&pool->core, pool->num_slots, pool->enabled,
                               errp);
}

bool riscv_ame_backend_pool_configure(
    RISCVAMEBackendPool *pool, uint64_t n, uint64_t u,
    uint32_t num_mregs, uint32_t num_accregs,
    const AmeCapabilities *capabilities, Error **errp)
{
    return riscv_ame_backend_pool_prepare(pool, errp) &&
           riscv_ame_pool_configure(&pool->core, n, u, num_mregs,
                                    num_accregs, capabilities, errp);
}

bool riscv_ame_backend_pool_configure_cpu(
    RISCVAMEBackendPool *pool, uint64_t n, uint64_t u,
    uint32_t num_mregs, uint32_t num_accregs, AmeCapabilityPolicy policy,
    const AmeCapabilities *requirements, Error **errp)
{
    return riscv_ame_backend_pool_prepare(pool, errp) &&
           riscv_ame_pool_configure_cpu(&pool->core, n, u, num_mregs,
                                        num_accregs, policy, requirements,
                                        errp);
}

RISCVAMEBackendPoolCore *riscv_ame_backend_pool_core(
    RISCVAMEBackendPool *pool)
{
    return &pool->core;
}

static void riscv_ame_backend_pool_realize(DeviceState *dev, Error **errp)
{
    RISCVAMEBackendPool *pool = RISCV_AME_BACKEND_POOL(dev);

    if (!riscv_ame_backend_pool_prepare(pool, errp)) {
        return;
    }
    riscv_ame_pool_realize(&pool->core, errp);
}

static void riscv_ame_backend_pool_unrealize(DeviceState *dev)
{
    RISCVAMEBackendPool *pool = RISCV_AME_BACKEND_POOL(dev);

    riscv_ame_pool_destroy(&pool->core);
}

static void riscv_ame_backend_pool_finalize(Object *obj)
{
    RISCVAMEBackendPool *pool = RISCV_AME_BACKEND_POOL(obj);

    /* Cover configure failures that never reach device unrealize. */
    riscv_ame_pool_destroy(&pool->core);
}

static void riscv_ame_backend_pool_reset_hold(Object *obj, ResetType type)
{
    RISCVAMEBackendPool *pool = RISCV_AME_BACKEND_POOL(obj);

    riscv_ame_pool_reset(&pool->core);
}

static const Property riscv_ame_backend_pool_properties[] = {
    DEFINE_PROP_UINT32("num-slots", RISCVAMEBackendPool, num_slots, 1),
    DEFINE_PROP_BOOL("enabled", RISCVAMEBackendPool, enabled, true),
};

static void riscv_ame_backend_pool_class_init(ObjectClass *oc,
                                               const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(oc);
    ResettableClass *rc = RESETTABLE_CLASS(oc);

    dc->realize = riscv_ame_backend_pool_realize;
    dc->unrealize = riscv_ame_backend_pool_unrealize;
    dc->user_creatable = false;
    device_class_set_props(dc, riscv_ame_backend_pool_properties);
    rc->phases.hold = riscv_ame_backend_pool_reset_hold;
}

static const TypeInfo riscv_ame_backend_pool_type_info = {
    .name = TYPE_RISCV_AME_BACKEND_POOL,
    .parent = TYPE_DEVICE,
    .instance_size = sizeof(RISCVAMEBackendPool),
    .instance_finalize = riscv_ame_backend_pool_finalize,
    .class_init = riscv_ame_backend_pool_class_init,
};

static void riscv_ame_backend_pool_register_types(void)
{
    type_register_static(&riscv_ame_backend_pool_type_info);
}

type_init(riscv_ame_backend_pool_register_types)
