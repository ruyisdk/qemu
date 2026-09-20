/*
 * RISC-V Attached Matrix Extension (AME) memory transactions
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "cpu.h"
#include "internals.h"
#include "ame_mem.h"
#include "accel/tcg/cpu-ldst.h"
#include "accel/tcg/cpu-ops.h"
#include "accel/tcg/probe.h"
#include "exec/target_page.h"
#include "exec/tlb-flags.h"

typedef struct RISCVAMEMemoryChunk {
    uint8_t *host;
    target_ulong guest;
    size_t buffer_offset;
    size_t length;
    bool slow;
} RISCVAMEMemoryChunk;

static G_NORETURN void ame_memory_access_fault(CPURISCVState *env,
                                                target_ulong address,
                                                MMUAccessType access_type,
                                                int mmu_idx, uintptr_t ra)
{
    env->badaddr = address;
#ifndef CONFIG_USER_ONLY
    env->two_stage_lookup = mmuidx_2stage(mmu_idx);
    env->two_stage_indirect_lookup = false;
#else
    (void)mmu_idx;
#endif
    riscv_raise_exception(env,
                          access_type == MMU_DATA_LOAD ?
                          RISCV_EXCP_LOAD_ACCESS_FAULT :
                          RISCV_EXCP_STORE_AMO_ACCESS_FAULT,
                          ra);
}

static RISCVAMEMemoryChunk *ame_reserve_chunk(CPURISCVState *env,
                                               size_t index)
{
    RISCVAMEMemoryScratch *scratch = &env_archcpu(env)->ame.memory;
    RISCVAMEMemoryChunk *chunks = scratch->chunks;

    if (index >= scratch->chunks_capacity) {
        size_t doubled, capacity;

        g_assert(index < SIZE_MAX);
        if (__builtin_mul_overflow(scratch->chunks_capacity,
                                   (size_t)2, &doubled)) {
            doubled = SIZE_MAX;
        }
        capacity = MAX(index + 1, doubled);
        capacity = MAX(capacity, (size_t)64);
        g_assert(capacity <= SIZE_MAX / sizeof(*chunks));
        chunks = g_renew(RISCVAMEMemoryChunk, chunks, capacity);
        scratch->chunks = chunks;
        scratch->chunks_capacity = capacity;
    }
    return &chunks[index];
}

static void ame_probe_ram(CPURISCVState *env, RISCVAMEMemoryChunk *chunk,
                          target_ulong address, size_t length,
                          MMUAccessType access_type,
                          int mmu_idx, uintptr_t ra)
{
    RISCVAMEMemoryScratch *scratch = &env_archcpu(env)->ame.memory;
    void *host = NULL;
    int flags;

    g_assert(length > 0 && length <= TARGET_PAGE_SIZE && length <= INT_MAX);

retry:
    flags = probe_access_flags(env, address, length, access_type, mmu_idx,
                               true, &host, ra);
    if (flags & TLB_INVALID_MASK) {
        scratch->fault_address = address;
        scratch->fault_active = true;
        probe_access(env, address, length, access_type, mmu_idx, ra);
        scratch->fault_active = false;
        goto retry;
    }

    chunk->slow = false;
#ifdef CONFIG_USER_ONLY
    chunk->slow = flags & TLB_FORCE_SLOW;
#else
    if (flags & TLB_MMIO) {
        void *raw_host = NULL;
        int raw_flags = probe_access_full_mmu(env, address, length,
                                              access_type, mmu_idx,
                                              &raw_host, NULL);

        if (raw_flags & TLB_INVALID_MASK) {
            goto retry;
        }
        if (raw_flags & TLB_MMIO || raw_host == NULL) {
            ame_memory_access_fault(env, address, access_type, mmu_idx, ra);
        }
        if (raw_flags & TLB_WATCHPOINT) {
            int wp_access = access_type == MMU_DATA_STORE ?
                            BP_MEM_WRITE : BP_MEM_READ;

            cpu_check_watchpoint(env_cpu(env), address, length,
                                 MEMTXATTRS_UNSPECIFIED, wp_access, ra);
        }
        host = raw_host;
        chunk->slow = true;
    } else {
        host = probe_access(env, address, length, access_type, mmu_idx, ra);
    }
#endif
    scratch->fault_active = false;
    if (host == NULL) {
        ame_memory_access_fault(env, address, access_type, mmu_idx, ra);
    }
    chunk->host = host;
    chunk->guest = address;
}

static size_t ame_probe_segments(CPURISCVState *env, target_ulong base,
                                 target_ulong stride,
                                 const AmeMemPlan *plan,
                                 MMUAccessType access_type, uintptr_t ra)
{
    int mmu_idx = riscv_env_mmu_index(env, false);
    unsigned int xlen = riscv_cpu_xlen(env);
    size_t num_chunks = 0;
    uint32_t segment;

    for (segment = 0; segment < plan->num_segments; segment++) {
        target_ulong address = riscv_ame_xlen_madd(
            base, segment, stride, xlen);
        size_t done = 0;

        while (done < plan->segment_bytes) {
            size_t page_bytes = TARGET_PAGE_SIZE -
                                (address & ~TARGET_PAGE_MASK);
            size_t length = MIN(page_bytes, plan->segment_bytes - done);
            size_t segment_offset, buffer_offset;
            RISCVAMEMemoryChunk *chunk;

            g_assert(!__builtin_mul_overflow((size_t)segment,
                                             plan->segment_bytes,
                                             &segment_offset));
            g_assert(!__builtin_add_overflow(segment_offset, done,
                                             &buffer_offset));
            g_assert(buffer_offset <= plan->total_bytes);
            g_assert(length <= plan->total_bytes - buffer_offset);
            g_assert(num_chunks < SIZE_MAX);
            chunk = ame_reserve_chunk(env, num_chunks++);

            ame_probe_ram(env, chunk, address, length,
                          access_type, mmu_idx, ra);
            chunk->buffer_offset = buffer_offset;
            chunk->length = length;
            address = riscv_ame_xlen_madd(address, length, 1, xlen);
            done += length;
        }
    }
    return num_chunks;
}

static void ame_memory_transfer(CPURISCVState *env, target_ulong base,
                                target_ulong stride, const AmeMemPlan *plan,
                                uint8_t *buffer, bool store, uintptr_t ra)
{
    MMUAccessType access_type = store ? MMU_DATA_STORE : MMU_DATA_LOAD;
    unsigned int xlen = riscv_cpu_xlen(env);
    RISCVAMEMemoryChunk *chunks;
    size_t num_chunks, index;

    g_assert(plan != NULL && buffer != NULL && plan->total_bytes > 0);
    num_chunks = ame_probe_segments(env, base, stride, plan,
                                    access_type, ra);
    chunks = env_archcpu(env)->ame.memory.chunks;

    for (index = 0; index < num_chunks; index++) {
        uint8_t *data = buffer + chunks[index].buffer_offset;
        size_t byte;

        if (!chunks[index].slow) {
            if (store) {
                memcpy(chunks[index].host, data, chunks[index].length);
            } else {
                memcpy(data, chunks[index].host, chunks[index].length);
            }
            continue;
        }
        for (byte = 0; byte < chunks[index].length; byte++) {
            target_ulong address = riscv_ame_xlen_madd(
                chunks[index].guest, byte, 1, xlen);

            if (store) {
                cpu_stb_data_ra(env, address, data[byte], ra);
            } else {
                data[byte] = cpu_ldub_data_ra(env, address, ra);
            }
        }
    }
}

void riscv_ame_memory_read(CPURISCVState *env, target_ulong base,
                           target_ulong stride, const AmeMemPlan *plan,
                           uint8_t *buffer, uintptr_t ra)
{
    ame_memory_transfer(env, base, stride, plan, buffer, false, ra);
}

void riscv_ame_memory_write(CPURISCVState *env, target_ulong base,
                            target_ulong stride, const AmeMemPlan *plan,
                            const uint8_t *buffer, uintptr_t ra)
{
    ame_memory_transfer(env, base, stride, plan, (uint8_t *)buffer, true, ra);
}
