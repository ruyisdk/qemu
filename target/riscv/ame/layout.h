/*
 * RISC-V Attached Matrix Extension (AME) tensor layout model
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef TARGET_RISCV_AME_LAYOUT_H
#define TARGET_RISCV_AME_LAYOUT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "qapi/error.h"
#include "target/riscv/ame/ame.h"
#include "target/riscv/ame/cap-set.h"

/*
 * Register geometry and host storage sizes for one backend.
 * N is the side of each N x N square; U is the physical M cell width in bits.
 * Each M register holds N*N*U bits. Each Acc register reserves enough bytes
 * for one square of the widest datatype accepted by the backend.
 */
typedef struct RISCVAMELayout {
    uint32_t n;
    uint32_t u;
    uint8_t num_mregs;
    uint8_t num_accregs;
    size_t mreg_bytes;  /* Bytes in one physical M register. */
    size_t mregs_bytes; /* Bytes in the complete M register file. */
    size_t acc_bytes;   /* Reserved bytes per Acc. */
    size_t accs_bytes;  /* Bytes in the complete Acc register file. */
} RISCVAMELayout;

/* Packing/grouping needed to store a datatype in physical M registers. */
typedef struct AmeDTypeShape {
    uint32_t pack_factor; /* Squares per register when datatype width < U. */
    uint32_t group_size;  /* Registers per square when datatype width > U. */
} AmeDTypeShape;

/*
 * Typed view of n_sq consecutive N x N squares in the M register file.
 * Owns no data. Elements are square-major, then row-major within each square.
 * Narrow elements are packed; wide elements span aligned register groups.
 */
typedef struct AmeMView {
    const AmeDTypeDesc *dtype;
    uint32_t base; /* First physical M register number. */
    uint32_t pack_factor;
    uint32_t group_size;
    uint32_t n_sq; /* Number of logical squares used by this operand. */
    uint32_t span; /* Number of consecutive physical M registers occupied. */
    uint32_t elements_per_square;
    uint32_t total_elements;
} AmeMView;

/* Typed view of one Acc square; Acc operands do not expand with n_sq. */
typedef struct AmeAccView {
    const AmeDTypeDesc *dtype;
    uint32_t index; /* Physical Acc register number. */
    uint32_t elements_per_square;
} AmeAccView;

/* AME load/store layouts; strided forms transfer one segment per row/column. */
typedef enum AmeMemFormat {
    AME_MEM_WHOLE_REGISTER,
    AME_MEM_ROW_MAJOR,
    AME_MEM_COLUMN_MAJOR,
    AME_MEM_STRIDED,
    AME_MEM_TRANSPOSED_STRIDED,
    AME_MEM_FORMAT_MAX,
} AmeMemFormat;

/*
 * Size and segmentation of a load/store, without guest addresses or storage.
 * A transfer buffer holds num_segments adjacent segments of segment_bytes
 * bytes. Any partial last byte is padding; segment_bits counts only data.
 * Guest segment i starts at base + i*stride, with XLEN address wrapping.
 */
typedef struct AmeMemPlan {
    AmeMemFormat format;
    uint32_t num_segments;
    uint32_t pack_factor;
    uint16_t element_bits;
    size_t segment_bits;
    size_t segment_bytes;
    size_t total_bytes;
    size_t alignment; /* Required guest base-address alignment, in bytes. */
} AmeMemPlan;

/**
 * riscv_ame_compute_layout: Validate geometry and calculate storage sizes
 * @n: N, the square side in elements
 * @u: U, the physical M cell width in bits
 * @num_mregs: NM, the physical M register count
 * @num_accregs: NA, the Acc register count
 * @capabilities: Constructed, nonempty backend capability set
 * @layout: Calculated geometry and byte sizes; unchanged on failure
 * @errp: Error destination
 *
 * Check parameter ranges, datatype compatibility with U, byte alignment and
 * size arithmetic.
 *
 * Returns: true on success; false on invalid geometry, setting @errp.
 */
bool riscv_ame_compute_layout(uint64_t n, uint64_t u, uint32_t num_mregs,
                              uint32_t num_accregs,
                              const AmeCapabilities *capabilities,
                              RISCVAMELayout *layout, Error **errp);
/**
 * riscv_ame_layout_equal: Compare register geometry and storage sizes
 * @left: First layout; may be NULL
 * @right: Second layout; may be NULL
 *
 * Returns: true for identical geometry and sizes, including two NULL pointers.
 */
bool riscv_ame_layout_equal(const RISCVAMELayout *left,
                            const RISCVAMELayout *right);

/**
 * riscv_ame_dtype_shape: Calculate M register packing for a datatype
 * @layout: Geometry; only its U field is used
 * @dtype: Element datatype
 * @shape: Packing and grouping factors; unchanged on failure
 *
 * Both factors are at least one; at most one exceeds one.
 *
 * Returns: true if the element width is supported by the storage model and
 * compatible with U; otherwise false.
 */
bool riscv_ame_dtype_shape(const RISCVAMELayout *layout,
                           const AmeDTypeDesc *dtype,
                           AmeDTypeShape *shape);
/**
 * riscv_ame_n_sq: Calculate the common square count for M operands
 * @layout: Backend geometry
 * @dtypes: Datatypes of the participating M operands; exclude Acc operands
 * @num_dtypes: Nonzero number of entries in @dtypes
 * @n_sq: Greatest operand pack_factor; unchanged on failure
 *
 * Returns: true if every datatype has a valid shape; otherwise false.
 */
bool riscv_ame_n_sq(const RISCVAMELayout *layout,
                    const AmeDTypeDesc *const *dtypes, size_t num_dtypes,
                    uint32_t *n_sq);

/**
 * riscv_ame_m_view_init: Describe a typed span of physical M registers
 * @layout: Backend geometry
 * @view: Result; unchanged on failure
 * @base: First physical M register number
 * @dtype: Element datatype; the view borrows this descriptor
 * @n_sq: Logical square count, a power of two divisible by pack_factor
 *
 * Check register-group alignment and that the entire span fits in the file.
 *
 * Returns: true for a valid view; otherwise false.
 */
bool riscv_ame_m_view_init(const RISCVAMELayout *layout, AmeMView *view,
                           uint32_t base, const AmeDTypeDesc *dtype,
                           uint32_t n_sq);
/**
 * riscv_ame_m_view_bytes: Get the physical storage size occupied by an M view
 * @layout: Backend geometry
 * @view: Validated M view
 *
 * Returns: span multiplied by bytes per physical M register.
 */
size_t riscv_ame_m_view_bytes(const RISCVAMELayout *layout,
                              const AmeMView *view);

/**
 * riscv_ame_m_read: Read one element's raw bits from the M register file
 * @layout: Backend geometry
 * @mregs: Start of the complete physical M register file
 * @view: Validated typed view
 * @element: Square-major, row-major index below view->total_elements
 *
 * Returns: element bits in AmeValue, zero-extended above the datatype width.
 */
AmeValue riscv_ame_m_read(const RISCVAMELayout *layout,
                          const uint8_t *mregs, const AmeMView *view,
                          size_t element);
/**
 * riscv_ame_m_write: Write one element's raw bits to the M register file
 * @layout: Backend geometry
 * @mregs: Start of the complete physical M register file
 * @view: Validated typed view
 * @element: Square-major, row-major index below view->total_elements
 * @value: Raw bits; bits above the datatype width are ignored
 *
 * Preserve all other elements, including the adjacent nibble of an Int4 byte.
 */
void riscv_ame_m_write(const RISCVAMELayout *layout, uint8_t *mregs,
                       const AmeMView *view, size_t element,
                       const AmeValue *value);

/**
 * riscv_ame_acc_view_init: Describe one typed Acc square
 * @layout: Backend geometry
 * @view: Result; unchanged on failure
 * @index: Physical Acc register number
 * @dtype: Element datatype; the view borrows this descriptor
 *
 * Returns: true if the register exists and one typed square fits; otherwise
 * false.
 */
bool riscv_ame_acc_view_init(const RISCVAMELayout *layout, AmeAccView *view,
                             uint32_t index, const AmeDTypeDesc *dtype);
/**
 * riscv_ame_acc_read: Read one element's raw bits from an Acc square
 * @layout: Backend geometry
 * @accs: Start of the complete physical Acc register file
 * @view: Validated typed Acc view
 * @element: Row-major index below view->elements_per_square
 *
 * Returns: element bits in AmeValue, zero-extended above the datatype width.
 */
AmeValue riscv_ame_acc_read(const RISCVAMELayout *layout,
                            const uint8_t *accs, const AmeAccView *view,
                            size_t element);
/**
 * riscv_ame_acc_write: Write one element's raw bits to an Acc square
 * @layout: Backend geometry
 * @accs: Start of the complete physical Acc register file
 * @view: Validated typed Acc view
 * @element: Row-major index below view->elements_per_square
 * @value: Raw bits; bits above the datatype width are ignored
 *
 * Preserve all other elements and unused bytes in the Acc register.
 */
void riscv_ame_acc_write(const RISCVAMELayout *layout, uint8_t *accs,
                         const AmeAccView *view, size_t element,
                         const AmeValue *value);

/**
 * riscv_ame_move_m_to_m: Copy an M span, allowing source/destination overlap
 * @layout: Backend geometry
 * @mregs: Complete M register file to update
 * @destination: Validated destination view
 * @source: Validated source with the same datatype and square count
 *
 * Source and destination may overlap. The caller validates all views and
 * their matching sizes before calling; failed preconditions assert.
 */
void riscv_ame_move_m_to_m(const RISCVAMELayout *layout, uint8_t *mregs,
                           const AmeMView *destination,
                           const AmeMView *source);
/**
 * riscv_ame_move_acc_to_m: Copy consecutive Acc squares into an M span
 * @layout: Backend geometry
 * @mregs: Complete M register file to update
 * @destination: Validated M view; its datatype interprets every source Acc
 * @accs: Complete Acc register file to read
 * @acc_base: First source Acc register
 * @count: Acc register count, equal to destination->n_sq
 * @scratch: Temporary buffer that must not overlap either register file
 * @scratch_bytes: Buffer capacity, at least the destination span's byte size
 *
 * The caller validates the register range, datatype and buffer size. Copy
 * squares in increasing register order; no datatype conversion is performed.
 */
void riscv_ame_move_acc_to_m(const RISCVAMELayout *layout, uint8_t *mregs,
                             const AmeMView *destination,
                             const uint8_t *accs, uint32_t acc_base,
                             uint32_t count, uint8_t *scratch,
                             size_t scratch_bytes);
/**
 * riscv_ame_move_m_to_acc: Copy an M span into consecutive Acc squares
 * @layout: Backend geometry
 * @accs: Complete Acc register file to update
 * @acc_base: First destination Acc register
 * @count: Acc register count, equal to source->n_sq
 * @mregs: Complete M register file to read
 * @source: Validated M view; its datatype interprets every destination Acc
 * @scratch: Temporary buffer that must not overlap either register file
 * @scratch_bytes: Buffer capacity, at least count * layout->acc_bytes
 *
 * The caller validates the register range, datatype and buffer size. Preserve
 * unused Acc bytes. No datatype conversion is performed.
 */
void riscv_ame_move_m_to_acc(const RISCVAMELayout *layout, uint8_t *accs,
                             uint32_t acc_base, uint32_t count,
                             const uint8_t *mregs, const AmeMView *source,
                             uint8_t *scratch, size_t scratch_bytes);

/**
 * riscv_ame_mem_plan_init: Calculate load/store segment and buffer sizes
 * @layout: Backend geometry
 * @dtype: Element datatype; ignored and may be NULL for WHOLE_REGISTER
 * @format: Load/store layout
 * @plan: Result; unchanged on failure
 *
 * WHOLE_REGISTER transfers one physical M register without interpreting its
 * datatype. Other formats transfer pack_factor logical squares. No guest
 * memory is accessed and no buffer is allocated.
 *
 * Returns: true for a valid plan; false for invalid input or size overflow.
 */
bool riscv_ame_mem_plan_init(const RISCVAMELayout *layout,
                             const AmeDTypeDesc *dtype, AmeMemFormat format,
                             AmeMemPlan *plan);
/**
 * riscv_ame_xlen_madd: Calculate a guest address with XLEN wrapping
 * @base: Base guest address
 * @index: Segment or byte index
 * @stride: Address increment in bytes
 * @xlen: Guest address width, 32 or 64
 *
 * Returns: (base + index * stride) modulo 2^XLEN, zero-extended to uint64_t.
 */
uint64_t riscv_ame_xlen_madd(uint64_t base, uint64_t index, uint64_t stride,
                             unsigned int xlen);
/**
 * riscv_ame_m_import: Arrange a load buffer's elements in the M register file
 * @layout: Backend geometry
 * @view: Validated destination view with n_sq equal to pack_factor
 * @mregs: Complete M register file to update
 * @plan: Matching plan for a typed format, not WHOLE_REGISTER
 * @memory: Host buffer of plan->total_bytes; must not overlap @mregs
 *
 * This only rearranges raw element bits; it does not access guest addresses.
 *
 * Returns: true on success; false on a plan/view mismatch, without writing.
 */
bool riscv_ame_m_import(const RISCVAMELayout *layout, const AmeMView *view,
                        uint8_t *mregs, const AmeMemPlan *plan,
                        const uint8_t *memory);
/**
 * riscv_ame_m_export: Arrange M elements in a store buffer
 * @layout: Backend geometry
 * @view: Validated source view with n_sq equal to pack_factor
 * @mregs: Complete M register file to read
 * @plan: Matching plan for a typed format, not WHOLE_REGISTER
 * @memory: Host output buffer of plan->total_bytes; must not overlap @mregs
 *
 * This only rearranges raw element bits; it does not access guest addresses.
 * Clear padding bits in each segment's last byte.
 *
 * Returns: true on success; false on a plan/view mismatch, without writing.
 */
bool riscv_ame_m_export(const RISCVAMELayout *layout, const AmeMView *view,
                        const uint8_t *mregs, const AmeMemPlan *plan,
                        uint8_t *memory);

#endif /* TARGET_RISCV_AME_LAYOUT_H */
