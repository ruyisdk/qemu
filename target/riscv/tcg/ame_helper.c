/*
 * RISC-V Attached Matrix Extension (Ztt) emulation helpers
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/bitops.h"
#include "qemu/bswap.h"
#include "cpu.h"
#include "internals.h"
#include "ame_mem.h"
#include "ame_helper.h"
#include "accel/tcg/cpu-loop.h"
#include "exec/helper-proto.h"
#include "target/riscv/ame/exec.h"
#include "target/riscv/ame/fp.h"

#define AME_ACQUIRE_GRANTED       1
#define AME_ACQUIRE_BUSY          (1 << 1)
#define AME_ACQUIRE_UNSUPPORTED   (2 << 1)
#define AME_ACQUIRE_DISABLED      (3 << 1)
#define AME_ACQUIRE_BAD_DESC      (4 << 1)
#define AME_ACQUIRE_TIMEOUT       (5 << 1)
#define AME_ACQUIRE_INTERRUPTED   (6 << 1)

static G_NORETURN void ame_illegal(CPURISCVState *env, uintptr_t ra)
{
    riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, ra);
}

static void ame_require_enabled(CPURISCVState *env, uintptr_t ra)
{
    if (!riscv_cpu_ame_enabled(env)) {
        ame_illegal(env, ra);
    }
}

static RISCVAMEBackendSlot *ame_require_context(CPURISCVState *env,
                                                uintptr_t ra)
{
    RISCVCPU *cpu = env_archcpu(env);
    RISCVAMEBackendSlot *slot;

    ame_require_enabled(env, ra);
    slot = riscv_ame_state_backend(&cpu->ame, OBJECT(cpu));
    if (slot == NULL) {
        ame_illegal(env, ra);
    }
    return slot;
}

static void ame_set_un(CPURISCVState *env, RISCVAMEBackendSlot *slot)
{
    slot->status |= AME_STATUS_UN;
    riscv_ame_mark_dirty(env);
}

static G_NORETURN void ame_memory_misaligned(CPURISCVState *env,
                                             target_ulong address,
                                             bool store, uintptr_t ra)
{
#ifndef CONFIG_USER_ONLY
    int mmu_idx = riscv_env_mmu_index(env, false);
#endif

    env->badaddr = address;
#ifndef CONFIG_USER_ONLY
    env->two_stage_lookup = mmuidx_2stage(mmu_idx);
    env->two_stage_indirect_lookup = false;
#endif
    riscv_raise_exception(env,
                          store ? RISCV_EXCP_STORE_AMO_ADDR_MIS :
                          RISCV_EXCP_LOAD_ADDR_MIS,
                          ra);
}

static const AmeDTypeDesc *ame_supported_dtype(CPURISCVState *env,
                                               uint32_t encoding)
{
    RISCVAMEBackendPoolCore *pool = env_archcpu(env)->ame.pool;
    const AmeDTypeDesc *dtype;

    g_assert(pool != NULL);
    dtype = riscv_ame_dtype_by_encoding(encoding);
    if (dtype == NULL ||
        !riscv_ame_dtype_set_has(&pool->capabilities.dtypes, dtype->id)) {
        return NULL;
    }
    return dtype;
}

/* The tuple has already been authorized; select only its numeric path. */
static AmeImplClass ame_select_impl(AmeOperation op, uint16_t dest,
                                    uint16_t source1, uint16_t source2)
{
    return riscv_ame_exec_impl(op, riscv_ame_dtype_by_id(dest),
                               riscv_ame_dtype_by_id(source1),
                               riscv_ame_dtype_by_id(source2));
}

static AmeImplClass ame_base_impl(const AmeCapabilities *capabilities,
                                  AmeOperation op, uint16_t dest,
                                  uint16_t source1, uint16_t source2)
{
    if (!riscv_ame_capabilities_has(capabilities, op, AME_FORM_BASE,
                                   dest, source1, source2)) {
        return AME_IMPL_UNSUPPORTED;
    }
    return ame_select_impl(op, dest, source1, source2);
}

static float_status *ame_local_fp_status(AmeImplClass impl,
                                         const AmeDTypeDesc *round_dtype,
                                         float_status *storage)
{
    if (!riscv_ame_impl_is_fp(impl)) {
        return NULL;
    }
    g_assert(riscv_ame_fp_status_prepare(storage, round_dtype));
    return storage;
}

static void ame_m_group_view(CPURISCVState *env, AmeMView *view,
                             uint32_t base, const AmeDTypeDesc *dtype,
                             uintptr_t ra)
{
    const RISCVAMELayout *layout =
        riscv_ame_state_layout(&env_archcpu(env)->ame);
    AmeDTypeShape shape;

    if (!riscv_ame_dtype_shape(layout, dtype, &shape) ||
        !riscv_ame_m_view_init(layout, view, base, dtype,
                               shape.pack_factor)) {
        ame_illegal(env, ra);
    }
}

static target_ulong ame_acquire_insn(CPURISCVState *env,
                                     target_ulong descriptor, uintptr_t ra)
{
    RISCVCPU *cpu = env_archcpu(env);
    RISCVAMEAcquireResult result;
    bool new_grant;
    bool bounded_wait;
    int64_t timeout_ns;

    ame_require_enabled(env, ra);
    if (cpu->ame.wait_active && cpu->ame.wait_descriptor != descriptor) {
        riscv_ame_wait_cancel(&cpu->ame);
    }
    if (!riscv_ame_acquire_descriptor_decode(descriptor, &bounded_wait,
                                              &timeout_ns)) {
        return AME_ACQUIRE_BAD_DESC;
    }

    result = riscv_ame_acquire(&cpu->ame, OBJECT(cpu),
                               bounded_wait, descriptor, timeout_ns,
                               &new_grant);
    if (new_grant) {
        env->amefflags = 0;
        env->amexsat = false;
        riscv_ame_set_ms(env, EXT_STATUS_INITIAL);
    }

    switch (result) {
    case RISCV_AME_ACQUIRE_GRANTED:
        return AME_ACQUIRE_GRANTED;
    case RISCV_AME_ACQUIRE_BUSY:
        return AME_ACQUIRE_BUSY;
    case RISCV_AME_ACQUIRE_UNSUPPORTED:
        return AME_ACQUIRE_UNSUPPORTED;
    case RISCV_AME_ACQUIRE_DISABLED:
        return AME_ACQUIRE_DISABLED;
    case RISCV_AME_ACQUIRE_WAIT:
#ifndef CONFIG_USER_ONLY
        env_cpu(env)->halted = 1;
        env_cpu(env)->exception_index = EXCP_HLT;
        cpu_loop_exit_restore(env_cpu(env), ra);
#else
        g_assert_not_reached();
#endif
    case RISCV_AME_ACQUIRE_TIMEOUT:
        return AME_ACQUIRE_TIMEOUT;
    case RISCV_AME_ACQUIRE_INTERRUPTED:
        return AME_ACQUIRE_INTERRUPTED;
    default:
        g_assert_not_reached();
    }
}

static void ame_release_insn(CPURISCVState *env, uintptr_t ra)
{
    RISCVCPU *cpu = env_archcpu(env);

    ame_require_enabled(env, ra);
    if (riscv_ame_release(&cpu->ame, OBJECT(cpu))) {
        riscv_ame_set_ms(env, EXT_STATUS_INITIAL);
    }
}

static void ame_settyp_m(CPURISCVState *env, RISCVAMEBackendSlot *slot,
                         uint32_t index, target_ulong raw_dtype,
                         uintptr_t ra)
{
    const RISCVAMELayout *layout =
        riscv_ame_state_layout(&env_archcpu(env)->ame);
    uint32_t encoding = (uint32_t)raw_dtype;
    const AmeDTypeDesc *dtype = ame_supported_dtype(env, encoding);
    AmeMView view;

    if (dtype == NULL) {
        ame_set_un(env, slot);
        return;
    }
    ame_m_group_view(env, &view, index, dtype, ra);

    memset(slot->mregs + index * layout->mreg_bytes, 0,
           riscv_ame_m_view_bytes(layout, &view));
    slot->md[index] = encoding;
    slot->md_dtype_id[index] = dtype->id;
    riscv_ame_mark_dirty(env);
}

static void ame_settyp_acc(CPURISCVState *env, RISCVAMEBackendSlot *slot,
                           uint32_t index, target_ulong raw_dtype)
{
    const RISCVAMELayout *layout =
        riscv_ame_state_layout(&env_archcpu(env)->ame);
    uint32_t encoding = (uint32_t)raw_dtype;
    const AmeDTypeDesc *dtype = ame_supported_dtype(env, encoding);

    if (dtype == NULL) {
        ame_set_un(env, slot);
        return;
    }

    memset(slot->accs + index * layout->acc_bytes, 0, layout->acc_bytes);
    slot->ad[index] = encoding;
    slot->ad_dtype_id[index] = dtype->id;
    riscv_ame_mark_dirty(env);
}

static void ame_zero_m(CPURISCVState *env, RISCVAMEBackendSlot *slot,
                       uint32_t index, uintptr_t ra)
{
    RISCVAMEBackendPoolCore *pool = env_archcpu(env)->ame.pool;
    const RISCVAMELayout *layout = &pool->layout;
    uint16_t dtype_id = slot->md_dtype_id[index];
    const AmeDTypeDesc *dtype;
    AmeMView view;

    if (!riscv_ame_capabilities_has(&pool->capabilities, AME_OP_ZERO,
                                    AME_FORM_BASE, dtype_id,
                                    AME_DTYPE_ID_INVALID,
                                    AME_DTYPE_ID_INVALID)) {
        ame_set_un(env, slot);
        return;
    }

    dtype = riscv_ame_dtype_by_id(dtype_id);
    g_assert(dtype != NULL);
    ame_m_group_view(env, &view, index, dtype, ra);
    memset(slot->mregs + index * layout->mreg_bytes, 0,
           riscv_ame_m_view_bytes(layout, &view));
    riscv_ame_mark_dirty(env);
}

static void ame_zero_acc(CPURISCVState *env, RISCVAMEBackendSlot *slot,
                         uint32_t index)
{
    RISCVAMEBackendPoolCore *pool = env_archcpu(env)->ame.pool;
    uint16_t dtype_id = slot->ad_dtype_id[index];

    if (!riscv_ame_capabilities_has(&pool->capabilities, AME_OP_ZERO,
                                    AME_FORM_BASE, dtype_id,
                                    AME_DTYPE_ID_INVALID,
                                    AME_DTYPE_ID_INVALID)) {
        ame_set_un(env, slot);
        return;
    }

    memset(slot->accs + index * pool->layout.acc_bytes, 0,
           pool->layout.acc_bytes);
    riscv_ame_mark_dirty(env);
}

static const AmeDTypeDesc *ame_move_dtype(CPURISCVState *env,
                                          RISCVAMEBackendSlot *slot,
                                          uint16_t destination,
                                          uint16_t source)
{
    RISCVAMEBackendPoolCore *pool = env_archcpu(env)->ame.pool;
    const AmeDTypeDesc *dtype;

    if (destination != source ||
        !riscv_ame_dtype_set_has(&pool->capabilities.dtypes, destination)) {
        ame_set_un(env, slot);
        return NULL;
    }
    dtype = riscv_ame_dtype_by_id(destination);
    g_assert(dtype != NULL);
    return dtype;
}

static void ame_mmov_m_m(CPURISCVState *env, RISCVAMEBackendSlot *slot,
                         uint32_t md, uint32_t ms, uintptr_t ra)
{
    const RISCVAMELayout *layout =
        riscv_ame_state_layout(&env_archcpu(env)->ame);
    const AmeDTypeDesc *dtype = ame_move_dtype(
        env, slot, slot->md_dtype_id[md], slot->md_dtype_id[ms]);
    AmeMView destination, source;

    if (dtype == NULL) {
        return;
    }
    ame_m_group_view(env, &destination, md, dtype, ra);
    ame_m_group_view(env, &source, ms, dtype, ra);
    riscv_ame_move_m_to_m(layout, slot->mregs, &destination, &source);
    riscv_ame_mark_dirty(env);
}

static bool ame_move_acc_span(CPURISCVState *env,
                              RISCVAMEBackendSlot *slot,
                              const AmeDTypeDesc *dtype, uint32_t acc,
                              uint32_t *count, uintptr_t ra)
{
    const RISCVAMELayout *layout =
        riscv_ame_state_layout(&env_archcpu(env)->ame);
    AmeDTypeShape shape;

    g_assert(riscv_ame_dtype_shape(layout, dtype, &shape));
    if (shape.pack_factor > layout->num_accregs) {
        ame_set_un(env, slot);
        return false;
    }
    if (shape.pack_factor > layout->num_accregs - acc) {
        ame_illegal(env, ra);
    }
    *count = shape.pack_factor;
    return true;
}

static void ame_mmov_m_a(CPURISCVState *env, RISCVAMEBackendSlot *slot,
                         uint32_t md, uint32_t acc, uintptr_t ra)
{
    const RISCVAMELayout *layout =
        riscv_ame_state_layout(&env_archcpu(env)->ame);
    const AmeDTypeDesc *dtype = ame_move_dtype(
        env, slot, slot->md_dtype_id[md], slot->ad_dtype_id[acc]);
    AmeMView destination;
    uint32_t count;

    if (dtype == NULL ||
        !ame_move_acc_span(env, slot, dtype, acc, &count, ra)) {
        return;
    }
    if (!riscv_ame_m_view_init(layout, &destination, md, dtype, count)) {
        ame_illegal(env, ra);
    }
    riscv_ame_move_acc_to_m(layout, slot->mregs, &destination, slot->accs,
                            acc, count, slot->scratch,
                            slot->scratch_bytes);
    riscv_ame_mark_dirty(env);
}

static void ame_mmov_a_m(CPURISCVState *env, RISCVAMEBackendSlot *slot,
                         uint32_t acc, uint32_t ms, uintptr_t ra)
{
    const RISCVAMELayout *layout =
        riscv_ame_state_layout(&env_archcpu(env)->ame);
    const AmeDTypeDesc *dtype = ame_move_dtype(
        env, slot, slot->ad_dtype_id[acc], slot->md_dtype_id[ms]);
    AmeMView source;
    uint32_t count;

    if (dtype == NULL ||
        !ame_move_acc_span(env, slot, dtype, acc, &count, ra)) {
        return;
    }
    if (!riscv_ame_m_view_init(layout, &source, ms, dtype, count)) {
        ame_illegal(env, ra);
    }
    riscv_ame_move_m_to_acc(layout, slot->accs, acc, count, slot->mregs,
                            &source, slot->scratch,
                            slot->scratch_bytes);
    riscv_ame_mark_dirty(env);
}

static void ame_m_result_commit(CPURISCVState *env,
                                RISCVAMEBackendSlot *slot,
                                const AmeMView *destination,
                                AmeExecFlags flags)
{
    const RISCVAMELayout *layout =
        riscv_ame_state_layout(&env_archcpu(env)->ame);
    size_t bytes = riscv_ame_m_view_bytes(layout, destination);

    memcpy(slot->mregs + destination->base * layout->mreg_bytes,
           slot->scratch, bytes);
    env->amefflags |= flags.fflags;
    env->amexsat |= flags.saturated;
    riscv_ame_mark_dirty(env);
}

static void ame_acc_result_commit(CPURISCVState *env,
                                  RISCVAMEBackendSlot *slot,
                                  const AmeAccView *destination,
                                  AmeExecFlags flags)
{
    const RISCVAMELayout *layout =
        riscv_ame_state_layout(&env_archcpu(env)->ame);

    memcpy(slot->accs + destination->index * layout->acc_bytes,
           slot->scratch, layout->acc_bytes);
    env->amefflags |= flags.fflags;
    env->amexsat |= flags.saturated;
    riscv_ame_mark_dirty(env);
}

static void ame_m_pair_result_commit(CPURISCVState *env,
                                     RISCVAMEBackendSlot *slot,
                                     const AmeMView *first,
                                     const AmeMView *second)
{
    const RISCVAMELayout *layout =
        riscv_ame_state_layout(&env_archcpu(env)->ame);
    size_t first_bytes = riscv_ame_m_view_bytes(layout, first);
    size_t second_bytes = riscv_ame_m_view_bytes(layout, second);

    memcpy(slot->mregs + first->base * layout->mreg_bytes,
           slot->scratch, first_bytes);
    memcpy(slot->mregs + second->base * layout->mreg_bytes,
           slot->scratch + first_bytes, second_bytes);
    riscv_ame_mark_dirty(env);
}

static void ame_mconv_ew(CPURISCVState *env, RISCVAMEBackendSlot *slot,
                         uint32_t md, uint32_t ms1, uintptr_t ra)
{
    RISCVAMEBackendPoolCore *pool = env_archcpu(env)->ame.pool;
    const RISCVAMELayout *layout = &pool->layout;
    const AmeDTypeDesc *dtypes[2];
    AmeMView destination, source;
    AmeImplClass impl;
    uint32_t n_sq;
    AmeExecFlags flags;
    bool valid;

    impl = ame_base_impl(&pool->capabilities,
                         AME_OP_CONV, slot->md_dtype_id[md],
                         slot->md_dtype_id[ms1],
                         AME_DTYPE_ID_INVALID);
    if (impl == AME_IMPL_UNSUPPORTED) {
        ame_set_un(env, slot);
        return;
    }

    dtypes[0] = riscv_ame_dtype_by_id(slot->md_dtype_id[md]);
    dtypes[1] = riscv_ame_dtype_by_id(slot->md_dtype_id[ms1]);
    g_assert(dtypes[0] != NULL && dtypes[1] != NULL);
    valid = riscv_ame_n_sq(layout, dtypes, ARRAY_SIZE(dtypes), &n_sq);
    g_assert(valid);
    if (!riscv_ame_m_view_init(layout, &destination, md, dtypes[0], n_sq) ||
        !riscv_ame_m_view_init(layout, &source, ms1, dtypes[1], n_sq)) {
        ame_illegal(env, ra);
    }

    valid = riscv_ame_exec_m_convert(
        impl, layout, slot->mregs, &destination, &source,
        slot->scratch, slot->scratch_bytes,
        &flags);
    g_assert(valid);
    ame_m_result_commit(env, slot, &destination, flags);
}

static void ame_ew_unary(CPURISCVState *env, RISCVAMEBackendSlot *slot,
                         AmeOperation op, uint32_t md, uint32_t ms1,
                         uintptr_t ra)
{
    RISCVAMEBackendPoolCore *pool = env_archcpu(env)->ame.pool;
    const RISCVAMELayout *layout = &pool->layout;
    const AmeDTypeDesc *dtypes[2];
    AmeMView destination, source;
    AmeImplClass impl;
    uint32_t n_sq;
    AmeExecFlags flags;
    bool valid;

    impl = ame_base_impl(&pool->capabilities, op,
                         slot->md_dtype_id[md],
                         slot->md_dtype_id[ms1],
                         AME_DTYPE_ID_INVALID);
    if (impl == AME_IMPL_UNSUPPORTED) {
        ame_set_un(env, slot);
        return;
    }
    dtypes[0] = riscv_ame_dtype_by_id(slot->md_dtype_id[md]);
    dtypes[1] = riscv_ame_dtype_by_id(slot->md_dtype_id[ms1]);
    g_assert(dtypes[0] != NULL && dtypes[1] != NULL);
    valid = riscv_ame_n_sq(layout, dtypes, ARRAY_SIZE(dtypes), &n_sq);
    g_assert(valid);
    if (!riscv_ame_m_view_init(layout, &destination, md, dtypes[0], n_sq) ||
        !riscv_ame_m_view_init(layout, &source, ms1, dtypes[1], n_sq)) {
        ame_illegal(env, ra);
    }

    valid = riscv_ame_exec_m_unary(
        impl, op, layout, slot->mregs, &destination, &source,
        slot->scratch, slot->scratch_bytes,
        &flags);
    g_assert(valid);
    ame_m_result_commit(env, slot, &destination, flags);
}

static void ame_ew_binary(CPURISCVState *env, RISCVAMEBackendSlot *slot,
                          AmeOperation op, uint32_t md, uint32_t ms1,
                          uint32_t ms2, uintptr_t ra)
{
    RISCVAMEBackendPoolCore *pool = env_archcpu(env)->ame.pool;
    const RISCVAMELayout *layout = &pool->layout;
    const AmeDTypeDesc *dtypes[3];
    AmeMView destination, source1, source2;
    AmeImplClass impl;
    uint32_t n_sq;
    AmeExecFlags flags;
    bool valid;

    impl = ame_base_impl(&pool->capabilities, op,
                         slot->md_dtype_id[md],
                         slot->md_dtype_id[ms1],
                         slot->md_dtype_id[ms2]);
    if (impl == AME_IMPL_UNSUPPORTED) {
        ame_set_un(env, slot);
        return;
    }
    dtypes[0] = riscv_ame_dtype_by_id(slot->md_dtype_id[md]);
    dtypes[1] = riscv_ame_dtype_by_id(slot->md_dtype_id[ms1]);
    dtypes[2] = riscv_ame_dtype_by_id(slot->md_dtype_id[ms2]);
    g_assert(dtypes[0] != NULL && dtypes[1] != NULL && dtypes[2] != NULL);
    valid = riscv_ame_n_sq(layout, dtypes, ARRAY_SIZE(dtypes), &n_sq);
    g_assert(valid);
    if (!riscv_ame_m_view_init(layout, &destination, md, dtypes[0], n_sq) ||
        !riscv_ame_m_view_init(layout, &source1, ms1, dtypes[1], n_sq) ||
        !riscv_ame_m_view_init(layout, &source2, ms2, dtypes[2], n_sq)) {
        ame_illegal(env, ra);
    }

    valid = riscv_ame_exec_m_binary(
        impl, op, layout, slot->mregs, &destination, &source1, &source2,
        slot->scratch, slot->scratch_bytes,
        &flags);
    g_assert(valid);
    ame_m_result_commit(env, slot, &destination, flags);
}

static void ame_ew_scalar(CPURISCVState *env, RISCVAMEBackendSlot *slot,
                          AmeOperation op, AmeEWScalarKind kind,
                          uint32_t md, target_ulong scalar, uint32_t ms2,
                          uintptr_t ra)
{
    RISCVAMEBackendPoolCore *pool = env_archcpu(env)->ame.pool;
    const RISCVAMELayout *layout = &pool->layout;
    const AmeDTypeDesc *dtypes[2];
    const AmeDTypeDesc *scalar_dtype;
    AmeMView destination, source;
    AmeValue scalar_value = { .lo = scalar };
    AmeExecResult converted;
    float_status fp_status;
    AmeImplClass impl;
    uint16_t dest_id = slot->md_dtype_id[md];
    uint16_t source_id = slot->md_dtype_id[ms2];
    bool data = kind == AME_EW_DATA_SCALAR_FIRST ||
                kind == AME_EW_DATA_SCALAR_SECOND;
    AmeCapForm form = data ? AME_FORM_DATA_SCALAR : AME_FORM_CONTROL_SCALAR;
    uint32_t n_sq;
    AmeExecFlags flags;
    bool valid;

    if (!riscv_ame_capabilities_has(&pool->capabilities, op, form,
                                    dest_id, source_id,
                                    data ? slot->scalar_dtype_id :
                                    AME_DTYPE_ID_INVALID)) {
        ame_set_un(env, slot);
        return;
    }

    dtypes[0] = riscv_ame_dtype_by_id(dest_id);
    dtypes[1] = riscv_ame_dtype_by_id(source_id);
    switch (kind) {
    case AME_EW_DATA_SCALAR_FIRST:
    case AME_EW_DATA_SCALAR_SECOND:
        scalar_dtype = riscv_ame_dtype_by_id(slot->scalar_dtype_id);
        break;
    case AME_EW_CONTROL_SCALAR:
        scalar_dtype = dtypes[1];
        break;
    case AME_EW_EXPONENT_SCALAR:
        scalar_dtype = ame_supported_dtype(
            env, pool->capabilities.max_int_dtype);
        break;
    default:
        g_assert_not_reached();
    }
    g_assert(dtypes[0] != NULL && dtypes[1] != NULL && scalar_dtype != NULL);
    /* Data scalars execute as (D,T,T) after the separate S-to-T conversion. */
    impl = ame_select_impl(op, dest_id, source_id,
                           data ? source_id : scalar_dtype->id);
    if (impl == AME_IMPL_UNSUPPORTED) {
        ame_set_un(env, slot);
        return;
    }
    valid = riscv_ame_n_sq(layout, dtypes, ARRAY_SIZE(dtypes), &n_sq);
    g_assert(valid);
    if (!riscv_ame_m_view_init(layout, &destination, md, dtypes[0], n_sq) ||
        !riscv_ame_m_view_init(layout, &source, ms2, dtypes[1], n_sq)) {
        ame_illegal(env, ra);
    }

    if (data) {
        AmeImplClass convert_impl = ame_select_impl(
            AME_OP_CONV, source_id, slot->scalar_dtype_id,
            AME_DTYPE_ID_INVALID);
        float_status *status;

        if (convert_impl == AME_IMPL_UNSUPPORTED) {
            ame_set_un(env, slot);
            return;
        }
        status = ame_local_fp_status(convert_impl, dtypes[1], &fp_status);

        /* These are primitives of the authorized scalar instruction. */
        valid = riscv_ame_exec_convert(convert_impl, dtypes[1],
                                       scalar_dtype, &scalar_value,
                                       &converted, status);
        g_assert(valid);
        scalar_value = converted.value;
        scalar_dtype = dtypes[1];
    }
    valid = riscv_ame_exec_m_binary_scalar(
        impl, op, layout, slot->mregs, &destination, &source,
        scalar_dtype, &scalar_value,
        kind == AME_EW_DATA_SCALAR_FIRST,
        slot->scratch, slot->scratch_bytes,
        &flags);
    g_assert(valid);
    if (data) {
        flags.fflags |= converted.fflags;
        flags.saturated |= converted.saturated;
    }
    ame_m_result_commit(env, slot, &destination, flags);
}

static void ame_ew_ternary(CPURISCVState *env,
                           RISCVAMEBackendSlot *slot, AmeOperation op,
                           uint32_t md, uint32_t ms1, uint32_t ms2,
                           uintptr_t ra)
{
    RISCVAMEBackendPoolCore *pool = env_archcpu(env)->ame.pool;
    const RISCVAMELayout *layout = &pool->layout;
    const AmeDTypeDesc *dtypes[3];
    AmeMView destination, source1, source2;
    AmeImplClass impl;
    uint32_t n_sq;
    AmeExecFlags flags;
    bool valid;

    impl = ame_base_impl(&pool->capabilities, op,
                         slot->md_dtype_id[md],
                         slot->md_dtype_id[ms1],
                         slot->md_dtype_id[ms2]);
    if (impl == AME_IMPL_UNSUPPORTED) {
        ame_set_un(env, slot);
        return;
    }
    dtypes[0] = riscv_ame_dtype_by_id(slot->md_dtype_id[md]);
    dtypes[1] = riscv_ame_dtype_by_id(slot->md_dtype_id[ms1]);
    dtypes[2] = riscv_ame_dtype_by_id(slot->md_dtype_id[ms2]);
    g_assert(dtypes[0] != NULL && dtypes[1] != NULL && dtypes[2] != NULL);
    valid = riscv_ame_n_sq(layout, dtypes, ARRAY_SIZE(dtypes), &n_sq);
    g_assert(valid);
    if (!riscv_ame_m_view_init(layout, &destination, md, dtypes[0], n_sq) ||
        !riscv_ame_m_view_init(layout, &source1, ms1, dtypes[1], n_sq) ||
        !riscv_ame_m_view_init(layout, &source2, ms2, dtypes[2], n_sq)) {
        ame_illegal(env, ra);
    }

    valid = riscv_ame_exec_m_ternary(
        impl, op, layout, slot->mregs, &destination, &source1, &source2,
        slot->scratch, slot->scratch_bytes, &flags);
    g_assert(valid);
    ame_m_result_commit(env, slot, &destination, flags);
}

static void ame_ew_ternary_scalar(CPURISCVState *env,
                                  RISCVAMEBackendSlot *slot,
                                  AmeOperation op, AmeEWScalarKind kind,
                                  uint32_t md, target_ulong scalar,
                                  uint32_t ms2, uintptr_t ra)
{
    RISCVAMEBackendPoolCore *pool = env_archcpu(env)->ame.pool;
    const RISCVAMELayout *layout = &pool->layout;
    const AmeDTypeDesc *dtypes[2];
    const AmeDTypeDesc *scalar_dtype;
    AmeMView destination, source;
    AmeValue scalar_value = { .lo = scalar };
    AmeExecResult converted;
    float_status fp_status;
    AmeImplClass impl;
    uint16_t dest_id = slot->md_dtype_id[md];
    uint16_t source_id = slot->md_dtype_id[ms2];
    bool data = kind == AME_EW_DATA_SCALAR_FIRST;
    AmeCapForm form = data ? AME_FORM_DATA_SCALAR : AME_FORM_CONTROL_SCALAR;
    uint32_t n_sq;
    AmeExecFlags flags;
    bool valid;

    g_assert(data || kind == AME_EW_EXPONENT_SCALAR);
    if (!riscv_ame_capabilities_has(&pool->capabilities, op, form,
                                    dest_id, source_id,
                                    data ? slot->scalar_dtype_id :
                                    AME_DTYPE_ID_INVALID)) {
        ame_set_un(env, slot);
        return;
    }

    dtypes[0] = riscv_ame_dtype_by_id(dest_id);
    dtypes[1] = riscv_ame_dtype_by_id(source_id);
    scalar_dtype = data ?
        riscv_ame_dtype_by_id(slot->scalar_dtype_id) :
        ame_supported_dtype(env, pool->capabilities.max_int_dtype);
    g_assert(dtypes[0] != NULL && dtypes[1] != NULL && scalar_dtype != NULL);
    impl = ame_select_impl(op, dest_id, source_id,
                           data ? source_id : scalar_dtype->id);
    if (impl == AME_IMPL_UNSUPPORTED) {
        ame_set_un(env, slot);
        return;
    }
    valid = riscv_ame_n_sq(layout, dtypes, ARRAY_SIZE(dtypes), &n_sq);
    g_assert(valid);
    if (!riscv_ame_m_view_init(layout, &destination, md, dtypes[0], n_sq) ||
        !riscv_ame_m_view_init(layout, &source, ms2, dtypes[1], n_sq)) {
        ame_illegal(env, ra);
    }

    if (data) {
        AmeImplClass convert_impl = ame_select_impl(
            AME_OP_CONV, source_id, slot->scalar_dtype_id,
            AME_DTYPE_ID_INVALID);
        float_status *status;

        if (convert_impl == AME_IMPL_UNSUPPORTED) {
            ame_set_un(env, slot);
            return;
        }
        status = ame_local_fp_status(convert_impl, dtypes[1], &fp_status);

        valid = riscv_ame_exec_convert(convert_impl, dtypes[1],
                                       scalar_dtype, &scalar_value,
                                       &converted, status);
        g_assert(valid);
        scalar_value = converted.value;
        scalar_dtype = dtypes[1];
    }
    valid = riscv_ame_exec_m_ternary_scalar(
        impl, op, layout, slot->mregs, &destination, &source,
        scalar_dtype, &scalar_value, data,
        slot->scratch, slot->scratch_bytes, &flags);
    g_assert(valid);
    if (data) {
        flags.fflags |= converted.fflags;
        flags.saturated |= converted.saturated;
    }
    ame_m_result_commit(env, slot, &destination, flags);
}

static bool ame_structural_dtypes_unpacked(const RISCVAMELayout *layout,
                                           const AmeDTypeDesc *const *dtypes,
                                           size_t count)
{
    size_t i;

    for (i = 0; i < count; i++) {
        AmeDTypeShape shape;

        if (!riscv_ame_dtype_shape(layout, dtypes[i], &shape) ||
            shape.pack_factor != 1) {
            return false;
        }
    }
    return true;
}

static void ame_permute(CPURISCVState *env, RISCVAMEBackendSlot *slot,
                        AmeOperation op, uint32_t md, uint32_t ms1,
                        target_ulong raw_control, uintptr_t ra)
{
    RISCVAMEBackendPoolCore *pool = env_archcpu(env)->ame.pool;
    const RISCVAMELayout *layout = &pool->layout;
    const AmeDTypeDesc *dtypes[2];
    AmeMView destination, source;
    AmeImplClass impl;
    bool valid;

    impl = ame_base_impl(&pool->capabilities, op,
                         slot->md_dtype_id[md],
                         slot->md_dtype_id[ms1],
                         AME_DTYPE_ID_INVALID);
    if (impl == AME_IMPL_UNSUPPORTED) {
        ame_set_un(env, slot);
        return;
    }

    dtypes[0] = riscv_ame_dtype_by_id(slot->md_dtype_id[md]);
    dtypes[1] = riscv_ame_dtype_by_id(slot->md_dtype_id[ms1]);
    g_assert(dtypes[0] != NULL && dtypes[1] != NULL);
    if (!ame_structural_dtypes_unpacked(layout, dtypes,
                                        ARRAY_SIZE(dtypes))) {
        ame_set_un(env, slot);
        return;
    }
    if ((op == AME_OP_COLBCAST || op == AME_OP_ROWBCAST) &&
        (uint32_t)raw_control >= layout->n) {
        ame_illegal(env, ra);
    }
    if (!riscv_ame_m_view_init(layout, &destination, md, dtypes[0], 1) ||
        !riscv_ame_m_view_init(layout, &source, ms1, dtypes[1], 1)) {
        ame_illegal(env, ra);
    }

    valid = riscv_ame_exec_m_permute(
        impl, op, layout, slot->mregs, &destination, &source,
        (int32_t)(uint32_t)raw_control, slot->scratch,
        slot->scratch_bytes);
    g_assert(valid);
    ame_m_result_commit(env, slot, &destination, (AmeExecFlags) { 0 });
}

static void ame_gather(CPURISCVState *env, RISCVAMEBackendSlot *slot,
                       AmeOperation op, uint32_t md, uint32_t ms1,
                       uint32_t ms2, uintptr_t ra)
{
    RISCVAMEBackendPoolCore *pool = env_archcpu(env)->ame.pool;
    const RISCVAMELayout *layout = &pool->layout;
    const AmeDTypeDesc *dtypes[3];
    AmeMView destination, source, indices;
    AmeImplClass impl;
    bool valid;

    impl = ame_base_impl(&pool->capabilities, op,
                         slot->md_dtype_id[md],
                         slot->md_dtype_id[ms1],
                         slot->md_dtype_id[ms2]);
    if (impl == AME_IMPL_UNSUPPORTED) {
        ame_set_un(env, slot);
        return;
    }

    dtypes[0] = riscv_ame_dtype_by_id(slot->md_dtype_id[md]);
    dtypes[1] = riscv_ame_dtype_by_id(slot->md_dtype_id[ms1]);
    dtypes[2] = riscv_ame_dtype_by_id(slot->md_dtype_id[ms2]);
    g_assert(dtypes[0] != NULL && dtypes[1] != NULL && dtypes[2] != NULL);
    if (!ame_structural_dtypes_unpacked(layout, dtypes,
                                        ARRAY_SIZE(dtypes))) {
        ame_set_un(env, slot);
        return;
    }
    if (!riscv_ame_m_view_init(layout, &destination, md, dtypes[0], 1) ||
        !riscv_ame_m_view_init(layout, &source, ms1, dtypes[1], 1) ||
        !riscv_ame_m_view_init(layout, &indices, ms2, dtypes[2], 1)) {
        ame_illegal(env, ra);
    }

    valid = riscv_ame_exec_m_gather(
        impl, op, layout, slot->mregs, &destination, &source, &indices,
        slot->scratch, slot->scratch_bytes);
    g_assert(valid);
    ame_m_result_commit(env, slot, &destination, (AmeExecFlags) { 0 });
}

static void ame_index(CPURISCVState *env, RISCVAMEBackendSlot *slot,
                      AmeOperation op, uint32_t md, uintptr_t ra)
{
    RISCVAMEBackendPoolCore *pool = env_archcpu(env)->ame.pool;
    const RISCVAMELayout *layout = &pool->layout;
    const AmeDTypeDesc *dtype;
    AmeDTypeShape shape;
    AmeMView destination;
    AmeImplClass impl;
    uint64_t maximum;
    unsigned int magnitude_bits;
    bool valid;

    impl = ame_base_impl(&pool->capabilities, op,
                         slot->md_dtype_id[md],
                         AME_DTYPE_ID_INVALID,
                         AME_DTYPE_ID_INVALID);
    if (impl == AME_IMPL_UNSUPPORTED) {
        ame_set_un(env, slot);
        return;
    }

    dtype = riscv_ame_dtype_by_id(slot->md_dtype_id[md]);
    g_assert(dtype != NULL && dtype->class == AME_DTYPE_CLASS_INTEGER);
    magnitude_bits = dtype->bits - dtype->integer.is_signed;
    maximum = magnitude_bits >= 64 ? UINT64_MAX :
              MAKE_64BIT_MASK(0, magnitude_bits);
    if (layout->n - 1 > maximum) {
        ame_set_un(env, slot);
        return;
    }

    g_assert(riscv_ame_dtype_shape(layout, dtype, &shape));
    if (!riscv_ame_m_view_init(layout, &destination, md, dtype,
                               shape.pack_factor)) {
        ame_illegal(env, ra);
    }
    valid = riscv_ame_exec_m_index(
        impl, op, layout, &destination, slot->scratch,
        slot->scratch_bytes);
    g_assert(valid);
    ame_m_result_commit(env, slot, &destination, (AmeExecFlags) { 0 });
}

static void ame_zip(CPURISCVState *env, RISCVAMEBackendSlot *slot,
                    AmeOperation op, uint32_t ms1, uint32_t ms2,
                    uintptr_t ra)
{
    RISCVAMEBackendPoolCore *pool = env_archcpu(env)->ame.pool;
    const RISCVAMELayout *layout = &pool->layout;
    const AmeDTypeDesc *dtypes[2];
    AmeMView first, second;
    AmeImplClass impl;
    bool valid;

    if (ms1 == ms2) {
        ame_illegal(env, ra);
    }
    impl = ame_base_impl(&pool->capabilities, op,
                         slot->md_dtype_id[ms1],
                         slot->md_dtype_id[ms2],
                         AME_DTYPE_ID_INVALID);
    if (impl == AME_IMPL_UNSUPPORTED) {
        ame_set_un(env, slot);
        return;
    }

    dtypes[0] = riscv_ame_dtype_by_id(slot->md_dtype_id[ms1]);
    dtypes[1] = riscv_ame_dtype_by_id(slot->md_dtype_id[ms2]);
    g_assert(dtypes[0] != NULL && dtypes[1] != NULL);
    if (!ame_structural_dtypes_unpacked(layout, dtypes,
                                        ARRAY_SIZE(dtypes))) {
        ame_set_un(env, slot);
        return;
    }
    if (!riscv_ame_m_view_init(layout, &first, ms1, dtypes[0], 1) ||
        !riscv_ame_m_view_init(layout, &second, ms2, dtypes[1], 1)) {
        ame_illegal(env, ra);
    }

    valid = riscv_ame_exec_m_zip(
        impl, op, layout, slot->mregs, &first, &second,
        slot->scratch, slot->scratch_bytes);
    g_assert(valid);
    ame_m_pair_result_commit(env, slot, &first, &second);
}

static void ame_bcast(CPURISCVState *env, RISCVAMEBackendSlot *slot,
                      uint32_t md, target_ulong scalar,
                      target_ulong raw_dtype, uintptr_t ra)
{
    RISCVAMEBackendPoolCore *pool = env_archcpu(env)->ame.pool;
    const RISCVAMELayout *layout = &pool->layout;
    const AmeDTypeDesc *destination_dtype;
    const AmeDTypeDesc *source_dtype = ame_supported_dtype(
        env, (uint32_t)raw_dtype);
    AmeMView destination;
    AmeValue raw = { .lo = scalar };
    AmeExecResult converted;
    AmeExecFlags flags;
    float_status fp_status;
    float_status *status;
    AmeImplClass impl;
    bool valid;

    if (source_dtype == NULL) {
        ame_set_un(env, slot);
        return;
    }
    impl = ame_base_impl(&pool->capabilities,
                         AME_OP_CONV, slot->md_dtype_id[md],
                         source_dtype->id, AME_DTYPE_ID_INVALID);
    if (impl == AME_IMPL_UNSUPPORTED) {
        ame_set_un(env, slot);
        return;
    }

    destination_dtype = riscv_ame_dtype_by_id(slot->md_dtype_id[md]);
    g_assert(destination_dtype != NULL);
    ame_m_group_view(env, &destination, md, destination_dtype, ra);
    status = ame_local_fp_status(impl, destination_dtype, &fp_status);
    valid = riscv_ame_exec_convert(impl, destination_dtype, source_dtype,
                                   &raw, &converted, status);
    g_assert(valid);
    valid = riscv_ame_exec_m_fill(
        layout, &destination, &converted.value, slot->scratch,
        slot->scratch_bytes);
    g_assert(valid);
    flags = (AmeExecFlags) {
        .fflags = converted.fflags,
        .saturated = converted.saturated,
    };
    ame_m_result_commit(env, slot, &destination, flags);
}

static void ame_pack(CPURISCVState *env, RISCVAMEBackendSlot *slot,
                     uint32_t md, uint32_t ms1, target_ulong raw_index,
                     bool unpack, uintptr_t ra)
{
    RISCVAMEBackendPoolCore *pool = env_archcpu(env)->ame.pool;
    const RISCVAMELayout *layout = &pool->layout;
    const AmeDTypeDesc *destination_dtype;
    const AmeDTypeDesc *source_dtype;
    AmeDTypeShape destination_shape, source_shape;
    AmeMView destination, source;
    AmeExecFlags flags;
    AmeImplClass impl;
    uint32_t pack_factor, index;
    bool valid;

    impl = ame_base_impl(&pool->capabilities,
                         AME_OP_CONV, slot->md_dtype_id[md],
                         slot->md_dtype_id[ms1],
                         AME_DTYPE_ID_INVALID);
    if (impl == AME_IMPL_UNSUPPORTED) {
        ame_set_un(env, slot);
        return;
    }

    destination_dtype = riscv_ame_dtype_by_id(slot->md_dtype_id[md]);
    source_dtype = riscv_ame_dtype_by_id(slot->md_dtype_id[ms1]);
    g_assert(destination_dtype != NULL && source_dtype != NULL);
    g_assert(riscv_ame_dtype_shape(layout, destination_dtype,
                                   &destination_shape));
    g_assert(riscv_ame_dtype_shape(layout, source_dtype, &source_shape));
    if ((!unpack && (destination_shape.pack_factor <= 1 ||
                     source_shape.pack_factor != 1)) ||
        (unpack && (destination_shape.pack_factor != 1 ||
                    source_shape.pack_factor <= 1))) {
        ame_set_un(env, slot);
        return;
    }

    pack_factor = unpack ? source_shape.pack_factor :
                           destination_shape.pack_factor;
    index = (uint32_t)raw_index & (pack_factor - 1);
    if (!riscv_ame_m_view_init(
            layout, &destination, md, destination_dtype,
            unpack ? 1 : destination_shape.pack_factor) ||
        !riscv_ame_m_view_init(
            layout, &source, ms1, source_dtype,
            unpack ? source_shape.pack_factor : 1)) {
        ame_illegal(env, ra);
    }

    valid = unpack ?
        riscv_ame_exec_m_unpack(
            impl, layout, slot->mregs, &destination, &source, index,
            slot->scratch, slot->scratch_bytes, &flags) :
        riscv_ame_exec_m_pack(
            impl, layout, slot->mregs, &destination, &source, index,
            slot->scratch, slot->scratch_bytes, &flags);
    g_assert(valid);
    ame_m_result_commit(env, slot, &destination, flags);
}

static void ame_scan(CPURISCVState *env, RISCVAMEBackendSlot *slot,
                     AmeOperation op, uint32_t md, uint32_t ms1,
                     uintptr_t ra)
{
    RISCVAMEBackendPoolCore *pool = env_archcpu(env)->ame.pool;
    const RISCVAMELayout *layout = &pool->layout;
    const AmeDTypeDesc *dtypes[2];
    AmeMView destination, source;
    AmeExecFlags flags;
    AmeImplClass impl;
    uint32_t n_sq;
    bool valid;

    impl = ame_base_impl(&pool->capabilities, op,
                         slot->md_dtype_id[md],
                         slot->md_dtype_id[ms1],
                         AME_DTYPE_ID_INVALID);
    if (impl != AME_IMPL_INT_REDUCE && impl != AME_IMPL_FP_REDUCE) {
        ame_set_un(env, slot);
        return;
    }

    dtypes[0] = riscv_ame_dtype_by_id(slot->md_dtype_id[md]);
    dtypes[1] = riscv_ame_dtype_by_id(slot->md_dtype_id[ms1]);
    g_assert(dtypes[0] != NULL && dtypes[1] != NULL);
    valid = riscv_ame_n_sq(layout, dtypes, ARRAY_SIZE(dtypes), &n_sq);
    g_assert(valid);
    if (!riscv_ame_m_view_init(layout, &destination, md, dtypes[0], n_sq) ||
        !riscv_ame_m_view_init(layout, &source, ms1, dtypes[1], n_sq)) {
        ame_illegal(env, ra);
    }

    valid = riscv_ame_exec_m_scan(
        impl, op, layout, slot->mregs, &destination, &source,
        slot->scratch, slot->scratch_bytes, &flags);
    g_assert(valid);
    ame_m_result_commit(env, slot, &destination, flags);
}

static void ame_scatter(CPURISCVState *env, RISCVAMEBackendSlot *slot,
                        AmeOperation op, uint32_t md, uint32_t ms1,
                        uint32_t ms2, uintptr_t ra)
{
    RISCVAMEBackendPoolCore *pool = env_archcpu(env)->ame.pool;
    const RISCVAMELayout *layout = &pool->layout;
    const AmeDTypeDesc *dtypes[3];
    AmeMView destination, source, indices;
    AmeExecFlags flags;
    AmeImplClass impl;
    bool valid;

    impl = ame_base_impl(&pool->capabilities, op,
                         slot->md_dtype_id[md],
                         slot->md_dtype_id[ms1],
                         slot->md_dtype_id[ms2]);
    if (impl != AME_IMPL_INT_SCATTER && impl != AME_IMPL_FP_SCATTER) {
        ame_set_un(env, slot);
        return;
    }

    dtypes[0] = riscv_ame_dtype_by_id(slot->md_dtype_id[md]);
    dtypes[1] = riscv_ame_dtype_by_id(slot->md_dtype_id[ms1]);
    dtypes[2] = riscv_ame_dtype_by_id(slot->md_dtype_id[ms2]);
    g_assert(dtypes[0] != NULL && dtypes[1] != NULL && dtypes[2] != NULL);
    if (!ame_structural_dtypes_unpacked(layout, dtypes,
                                        ARRAY_SIZE(dtypes))) {
        ame_set_un(env, slot);
        return;
    }
    if (!riscv_ame_m_view_init(layout, &destination, md, dtypes[0], 1) ||
        !riscv_ame_m_view_init(layout, &source, ms1, dtypes[1], 1) ||
        !riscv_ame_m_view_init(layout, &indices, ms2, dtypes[2], 1)) {
        ame_illegal(env, ra);
    }

    valid = riscv_ame_exec_m_scatter(
        impl, op, layout, slot->mregs, &destination, &source, &indices,
        slot->scratch, slot->scratch_bytes, &flags);
    g_assert(valid);
    ame_m_result_commit(env, slot, &destination, flags);
}

static void ame_matmul(CPURISCVState *env, RISCVAMEBackendSlot *slot,
                       AmeOperation op, uint32_t acc, uint32_t ms1,
                       uint32_t ms2, uintptr_t ra)
{
    RISCVAMEBackendPoolCore *pool = env_archcpu(env)->ame.pool;
    const RISCVAMELayout *layout = &pool->layout;
    const AmeDTypeDesc *dtypes[3];
    const AmeDTypeDesc *source_dtypes[2];
    AmeAccView destination;
    AmeMView source1, source2;
    AmeExecFlags flags;
    AmeImplClass impl;
    uint32_t n_sq;
    bool valid;

    impl = ame_base_impl(&pool->capabilities, op,
                         slot->ad_dtype_id[acc],
                         slot->md_dtype_id[ms1],
                         slot->md_dtype_id[ms2]);
    if (impl != AME_IMPL_INT_MATMUL && impl != AME_IMPL_FP_MATMUL) {
        ame_set_un(env, slot);
        return;
    }

    dtypes[0] = riscv_ame_dtype_by_id(slot->ad_dtype_id[acc]);
    dtypes[1] = riscv_ame_dtype_by_id(slot->md_dtype_id[ms1]);
    dtypes[2] = riscv_ame_dtype_by_id(slot->md_dtype_id[ms2]);
    g_assert(dtypes[0] != NULL && dtypes[1] != NULL && dtypes[2] != NULL);
    source_dtypes[0] = dtypes[1];
    source_dtypes[1] = dtypes[2];
    valid = riscv_ame_n_sq(layout, source_dtypes,
                           ARRAY_SIZE(source_dtypes), &n_sq);
    g_assert(valid);
    valid = riscv_ame_acc_view_init(layout, &destination, acc, dtypes[0]);
    g_assert(valid);
    if (!riscv_ame_m_view_init(layout, &source1, ms1, dtypes[1], n_sq) ||
        !riscv_ame_m_view_init(layout, &source2, ms2, dtypes[2], n_sq)) {
        ame_illegal(env, ra);
    }

    valid = riscv_ame_exec_acc_matmul(
        impl, op, layout, slot->mregs, slot->accs, &destination,
        &source1, &source2, slot->scratch,
        slot->scratch_bytes, &flags);
    g_assert(valid);
    ame_acc_result_commit(env, slot, &destination, flags);
}

static bool ame_prepare_memory(CPURISCVState *env,
                               RISCVAMEBackendSlot *slot, uint32_t mreg,
                               AmeMemFormat format, AmeMView *view,
                               AmeMemPlan *plan, uintptr_t ra)
{
    RISCVAMEBackendPoolCore *pool = env_archcpu(env)->ame.pool;
    const RISCVAMELayout *layout = &pool->layout;
    const AmeDTypeDesc *dtype;
    AmeDTypeShape shape;
    uint16_t dtype_id;

    if (format == AME_MEM_WHOLE_REGISTER) {
        g_assert(riscv_ame_mem_plan_init(layout, NULL, format, plan));
        return true;
    }

    dtype_id = slot->md_dtype_id[mreg];
    if (!riscv_ame_dtype_set_has(&pool->capabilities.dtypes, dtype_id)) {
        ame_set_un(env, slot);
        return false;
    }
    dtype = riscv_ame_dtype_by_id(dtype_id);
    g_assert(dtype != NULL);
    g_assert(riscv_ame_dtype_shape(layout, dtype, &shape));
    if (!riscv_ame_m_view_init(layout, view, mreg, dtype,
                               shape.pack_factor)) {
        ame_illegal(env, ra);
    }
    g_assert(riscv_ame_mem_plan_init(layout, dtype, format, plan));
    return true;
}

static void ame_memory_insn(CPURISCVState *env, RISCVAMEBackendSlot *slot,
                            uint32_t mreg, target_ulong address,
                            target_ulong stride, AmeMemFormat format,
                            bool store, uintptr_t ra)
{
    const RISCVAMELayout *layout =
        riscv_ame_state_layout(&env_archcpu(env)->ame);
    uint8_t *scratch = slot->scratch;
    AmeMView view;
    AmeMemPlan plan;

    if (!ame_prepare_memory(env, slot, mreg, format, &view, &plan, ra)) {
        return;
    }
    g_assert(plan.total_bytes <= slot->scratch_bytes);
    if (address % plan.alignment != 0) {
        ame_memory_misaligned(env, address, store, ra);
    }

    if (store) {
        if (format == AME_MEM_WHOLE_REGISTER) {
            memcpy(scratch, slot->mregs + mreg * layout->mreg_bytes,
                   plan.total_bytes);
        } else {
            g_assert(riscv_ame_m_export(layout, &view, slot->mregs,
                                        &plan, scratch));
        }
        riscv_ame_memory_write(env, address, stride, &plan, scratch, ra);
        return;
    }

    riscv_ame_memory_read(env, address, stride, &plan, scratch, ra);
    if (format == AME_MEM_WHOLE_REGISTER) {
        memcpy(slot->mregs + mreg * layout->mreg_bytes, scratch,
               plan.total_bytes);
    } else {
        g_assert(riscv_ame_m_import(layout, &view, slot->mregs,
                                    &plan, scratch));
    }
    riscv_ame_mark_dirty(env);
}

static void ame_debug_check(CPURISCVState *env, uint32_t element_bits,
                            uintptr_t ra)
{
    const RISCVAMELayout *layout =
        riscv_ame_state_layout(&env_archcpu(env)->ame);

    g_assert(element_bits == 8 || element_bits == 16 ||
             element_bits == 32 || element_bits == 64);
    if (element_bits > riscv_cpu_xlen(env) ||
        layout->n * layout->u < element_bits) {
        ame_illegal(env, ra);
    }
}

static size_t ame_debug_offset(const RISCVAMELayout *layout,
                               uint32_t element_bits,
                               target_ulong position)
{
    size_t total_elements = layout->mreg_bytes * CHAR_BIT / element_bits;

    g_assert(total_elements != 0 &&
             (total_elements & (total_elements - 1)) == 0);
    return ((uint64_t)position & (total_elements - 1)) *
           (element_bits / CHAR_BIT);
}

static uint64_t ame_debug_load(const uint8_t *ptr, uint32_t element_bits)
{
    switch (element_bits) {
    case 8:
        return *ptr;
    case 16:
        return lduw_le_p(ptr);
    case 32:
        return (uint32_t)ldl_le_p(ptr);
    case 64:
        return ldq_le_p(ptr);
    default:
        g_assert_not_reached();
    }
}

static void ame_debug_store(uint8_t *ptr, uint32_t element_bits,
                            uint64_t value)
{
    switch (element_bits) {
    case 8:
        *ptr = value;
        break;
    case 16:
        stw_le_p(ptr, value);
        break;
    case 32:
        stl_le_p(ptr, value);
        break;
    case 64:
        stq_le_p(ptr, value);
        break;
    default:
        g_assert_not_reached();
    }
}

static void ame_debug_write(CPURISCVState *env, RISCVAMEBackendSlot *slot,
                            uint32_t mreg, target_ulong value,
                            target_ulong position, uint32_t element_bits,
                            uintptr_t ra)
{
    const RISCVAMELayout *layout =
        riscv_ame_state_layout(&env_archcpu(env)->ame);
    size_t offset;

    ame_debug_check(env, element_bits, ra);
    offset = ame_debug_offset(layout, element_bits, position);
    ame_debug_store(slot->mregs + mreg * layout->mreg_bytes + offset,
                    element_bits, value);
    riscv_ame_mark_dirty(env);
}

static target_ulong ame_debug_read(CPURISCVState *env,
                                    RISCVAMEBackendSlot *slot,
                                    uint32_t mreg, target_ulong position,
                                    uint32_t element_bits, uintptr_t ra)
{
    const RISCVAMELayout *layout =
        riscv_ame_state_layout(&env_archcpu(env)->ame);
    size_t offset;

    ame_debug_check(env, element_bits, ra);
    offset = ame_debug_offset(layout, element_bits, position);
    return ame_debug_load(slot->mregs + mreg * layout->mreg_bytes + offset,
                          element_bits);
}

/* Encoded indices are checked after ownership, before any dtype lookup. */
static void ame_check_mreg(CPURISCVState *env, uint32_t reg, uintptr_t ra)
{
    if (reg >= riscv_ame_state_layout(&env_archcpu(env)->ame)->num_mregs) {
        ame_illegal(env, ra);
    }
}

static void ame_check_accreg(CPURISCVState *env, uint32_t reg, uintptr_t ra)
{
    if (reg >= riscv_ame_state_layout(&env_archcpu(env)->ame)->num_accregs) {
        ame_illegal(env, ra);
    }
}

target_ulong helper_ame_acquire(CPURISCVState *env, target_ulong descriptor)
{
    return ame_acquire_insn(env, descriptor, GETPC());
}

void helper_ame_release(CPURISCVState *env)
{
    ame_release_insn(env, GETPC());
}

target_ulong helper_ame_mgettyp(CPURISCVState *env, uint32_t ms)
{
    uintptr_t ra = GETPC();
    RISCVAMEBackendSlot *slot = ame_require_context(env, ra);

    ame_check_mreg(env, ms, ra);
    return slot->md[ms];
}

target_ulong helper_ame_agettyp(CPURISCVState *env, uint32_t as)
{
    uintptr_t ra = GETPC();
    RISCVAMEBackendSlot *slot = ame_require_context(env, ra);

    ame_check_accreg(env, as, ra);
    return slot->ad[as];
}

void helper_ame_msettyp(CPURISCVState *env, uint32_t md, target_ulong dtype)
{
    uintptr_t ra = GETPC();
    RISCVAMEBackendSlot *slot = ame_require_context(env, ra);

    ame_check_mreg(env, md, ra);
    ame_settyp_m(env, slot, md, dtype, ra);
}

void helper_ame_asettyp(CPURISCVState *env, uint32_t ad, target_ulong dtype)
{
    uintptr_t ra = GETPC();
    RISCVAMEBackendSlot *slot = ame_require_context(env, ra);

    ame_check_accreg(env, ad, ra);
    ame_settyp_acc(env, slot, ad, dtype);
}

void helper_ame_zero_m(CPURISCVState *env, uint32_t md)
{
    uintptr_t ra = GETPC();
    RISCVAMEBackendSlot *slot = ame_require_context(env, ra);

    ame_check_mreg(env, md, ra);
    ame_zero_m(env, slot, md, ra);
}

void helper_ame_zero_acc(CPURISCVState *env, uint32_t ad)
{
    uintptr_t ra = GETPC();
    RISCVAMEBackendSlot *slot = ame_require_context(env, ra);

    ame_check_accreg(env, ad, ra);
    ame_zero_acc(env, slot, ad);
}

#define AME_HELPER_MM(name, exec) \
void helper_ame_##name(CPURISCVState *env, uint32_t md, uint32_t ms) \
{ \
    uintptr_t ra = GETPC(); \
    RISCVAMEBackendSlot *slot = ame_require_context(env, ra); \
    ame_check_mreg(env, md, ra); \
    ame_check_mreg(env, ms, ra); \
    exec(env, slot, md, ms, ra); \
}
AME_HELPER_MM(mov_m_m, ame_mmov_m_m)
AME_HELPER_MM(convert, ame_mconv_ew)
#undef AME_HELPER_MM

void helper_ame_mov_m_a(CPURISCVState *env, uint32_t md, uint32_t as)
{
    uintptr_t ra = GETPC();
    RISCVAMEBackendSlot *slot = ame_require_context(env, ra);

    ame_check_mreg(env, md, ra);
    ame_check_accreg(env, as, ra);
    ame_mmov_m_a(env, slot, md, as, ra);
}

void helper_ame_mov_a_m(CPURISCVState *env, uint32_t ad, uint32_t ms)
{
    uintptr_t ra = GETPC();
    RISCVAMEBackendSlot *slot = ame_require_context(env, ra);

    ame_check_accreg(env, ad, ra);
    ame_check_mreg(env, ms, ra);
    ame_mmov_a_m(env, slot, ad, ms, ra);
}

#define AME_HELPER_OP_MM(name, exec) \
void helper_ame_##name(CPURISCVState *env, uint32_t op, \
                      uint32_t m0, uint32_t m1) \
{ \
    uintptr_t ra = GETPC(); \
    RISCVAMEBackendSlot *slot = ame_require_context(env, ra); \
    ame_check_mreg(env, m0, ra); \
    ame_check_mreg(env, m1, ra); \
    exec(env, slot, op, m0, m1, ra); \
}
AME_HELPER_OP_MM(unary, ame_ew_unary)
AME_HELPER_OP_MM(scan, ame_scan)
AME_HELPER_OP_MM(zip, ame_zip)
#undef AME_HELPER_OP_MM

#define AME_HELPER_OP_MMM(name, exec) \
void helper_ame_##name(CPURISCVState *env, uint32_t op, uint32_t md, \
                      uint32_t ms1, uint32_t ms2) \
{ \
    uintptr_t ra = GETPC(); \
    RISCVAMEBackendSlot *slot = ame_require_context(env, ra); \
    ame_check_mreg(env, md, ra); \
    ame_check_mreg(env, ms1, ra); \
    ame_check_mreg(env, ms2, ra); \
    exec(env, slot, op, md, ms1, ms2, ra); \
}
AME_HELPER_OP_MMM(binary, ame_ew_binary)
AME_HELPER_OP_MMM(ternary, ame_ew_ternary)
AME_HELPER_OP_MMM(gather, ame_gather)
AME_HELPER_OP_MMM(scatter, ame_scatter)
#undef AME_HELPER_OP_MMM

#define AME_HELPER_SCALAR(name, exec) \
void helper_ame_##name(CPURISCVState *env, uint32_t op, uint32_t kind, \
                      uint32_t md, target_ulong scalar, uint32_t ms2) \
{ \
    uintptr_t ra = GETPC(); \
    RISCVAMEBackendSlot *slot = ame_require_context(env, ra); \
    ame_check_mreg(env, md, ra); \
    ame_check_mreg(env, ms2, ra); \
    exec(env, slot, op, kind, md, scalar, ms2, ra); \
}
AME_HELPER_SCALAR(scalar, ame_ew_scalar)
AME_HELPER_SCALAR(ternary_scalar, ame_ew_ternary_scalar)
#undef AME_HELPER_SCALAR

void helper_ame_matmul(CPURISCVState *env, uint32_t op, uint32_t ad,
                      uint32_t ms1, uint32_t ms2)
{
    uintptr_t ra = GETPC();
    RISCVAMEBackendSlot *slot = ame_require_context(env, ra);

    ame_check_accreg(env, ad, ra);
    ame_check_mreg(env, ms1, ra);
    ame_check_mreg(env, ms2, ra);
    ame_matmul(env, slot, op, ad, ms1, ms2, ra);
}

void helper_ame_permute(CPURISCVState *env, uint32_t op, uint32_t md,
                       target_ulong control, uint32_t ms2)
{
    uintptr_t ra = GETPC();
    RISCVAMEBackendSlot *slot = ame_require_context(env, ra);

    ame_check_mreg(env, md, ra);
    ame_check_mreg(env, ms2, ra);
    ame_permute(env, slot, op, md, ms2, control, ra);
}

void helper_ame_index(CPURISCVState *env, uint32_t op, uint32_t md)
{
    uintptr_t ra = GETPC();
    RISCVAMEBackendSlot *slot = ame_require_context(env, ra);

    ame_check_mreg(env, md, ra);
    ame_index(env, slot, op, md, ra);
}

void helper_ame_bcast(CPURISCVState *env, uint32_t md,
                     target_ulong scalar, target_ulong dtype)
{
    uintptr_t ra = GETPC();
    RISCVAMEBackendSlot *slot = ame_require_context(env, ra);

    ame_check_mreg(env, md, ra);
    ame_bcast(env, slot, md, scalar, dtype, ra);
}

void helper_ame_pack(CPURISCVState *env, uint32_t unpack, uint32_t md,
                    target_ulong index, uint32_t ms2)
{
    uintptr_t ra = GETPC();
    RISCVAMEBackendSlot *slot = ame_require_context(env, ra);

    ame_check_mreg(env, md, ra);
    ame_check_mreg(env, ms2, ra);
    ame_pack(env, slot, md, ms2, index, unpack, ra);
}

void helper_ame_load(CPURISCVState *env, uint32_t md, target_ulong address,
                    target_ulong stride, uint32_t format)
{
    uintptr_t ra = GETPC();
    RISCVAMEBackendSlot *slot = ame_require_context(env, ra);

    ame_check_mreg(env, md, ra);
    ame_memory_insn(env, slot, md, address, stride, format, false, ra);
}

void helper_ame_store(CPURISCVState *env, uint32_t ms, target_ulong address,
                     target_ulong stride, uint32_t format)
{
    uintptr_t ra = GETPC();
    RISCVAMEBackendSlot *slot = ame_require_context(env, ra);

    ame_check_mreg(env, ms, ra);
    ame_memory_insn(env, slot, ms, address, stride, format, true, ra);
}

void helper_ame_debug_write(CPURISCVState *env, uint32_t md,
                            target_ulong value, target_ulong position,
                            uint32_t element_bits)
{
    uintptr_t ra = GETPC();
    RISCVAMEBackendSlot *slot = ame_require_context(env, ra);

    ame_check_mreg(env, md, ra);
    ame_debug_write(env, slot, md, value, position, element_bits, ra);
}

target_ulong helper_ame_debug_read(CPURISCVState *env, uint32_t ms,
                                    target_ulong position,
                                    uint32_t element_bits)
{
    uintptr_t ra = GETPC();
    RISCVAMEBackendSlot *slot = ame_require_context(env, ra);

    ame_check_mreg(env, ms, ra);
    return ame_debug_read(env, slot, ms, position, element_bits, ra);
}
