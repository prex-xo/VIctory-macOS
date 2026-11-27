#define RS "RootConstants(num32BitConstants=5, b0), SRV(t0), UAV(u1), UAV(u0), SRV(t1)"
// `widest`: the most groups in x a dispatch runs (`indirect::widest`).
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint flags; uint widest; };
// The title's records, read only; the clamped ones go to the list's own.
ByteAddressBuffer args : register(t0);
RWByteAddressBuffer dest : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
// The title's record count: read only if `flags & 1`.
ByteAddressBuffer countWord : register(t1);
static const uint BUDGET = 1u << 27;
// Draws of MOST_DRAW vertices/indices per instance or more are unfilled slots.
static const uint MOST_DRAW = 1u << 23;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
// Clamps the record at `at` into `dest`; its cost.
uint clamped(uint at) {
    uint cost;
    uint base_bytes;
    if (kind <= 1u) {
        // Draws: vertex or index count, then instances.
        uint raw_count = args.Load(at);
        uint raw_instances = args.Load(at + 4u);
        uint count = raw_count;
        uint instances = 0u;
        if (raw_count >= MOST_DRAW) {
            count = 0u;
        } else {
            instances = min(raw_instances, (1u << 26) / max(count, 1u));
        }
        cost = count * instances;
        dest.Store2(at, uint2(count, instances));
        // The rest of the record as it is: first vertex and instance (4 words
        // in all), or first index, base vertex and first instance (5).
        if (kind == 1u) {
            dest.Store3(at + 8u, args.Load3(at + 8u));
            base_bytes = 20u;
        } else {
            dest.Store2(at + 8u, args.Load2(at + 8u));
            base_bytes = 16u;
        }
    } else if (kind == 2u) {
        uint3 g = args.Load3(at);
        // Up to `widest` groups in x run as asked; a wider record is cut to
        // D3D12's 65,535 as before.
        g = uint3(g.x <= widest ? g.x : 65535u, min(g.y, 65535u), min(g.z, 65535u));
        // x * y fits 32 bits where it is looked at; the whole count may not.
        if ((g.y != 0u && g.x > GROUPS / g.y) || (g.z != 0u && g.x * g.y > GROUPS / g.z)) {
            g = uint3(0u, 0u, 0u);
        }
        cost = g.x * g.y * g.z * 64u;
        dest.Store3(at, g);
        base_bytes = 12u;
    } else {
        // Mesh dispatches: the console's are one-dimensional; unfilled slots
        // (`> 65535`, e.g. `0xffffffff`) run empty instead of clamping to 65535.
        uint3 raw_g = args.Load3(at);
        uint3 g = uint3(raw_g.x, min(raw_g.y, 1u), min(raw_g.z, 1u));
        if (raw_g.x > 65535u) {
            g = uint3(0u, 0u, 0u);
        }
        cost = g.x * g.y * g.z * 128u;
        dest.Store3(at, g);
        base_bytes = 12u;
    }
    for (uint extra = base_bytes; extra < stride; extra += 4u) {
        dest.Store(at + extra, args.Load(at + extra));
    }
    return cost;
}
// A record past the budget (no cost reaches it).
static const uint PAST = 0xffffffffu;
groupshared uint costs[64];
// One group walks the records in tiles of 64: every lane clamps its own
// record, then the first lane charges the tile's records in order. No
// other group shares the budget: a shared add wraps once enough records
// are rejected, and a compare-exchange retries more the more records fit.
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    uint left = 0u;
    if (lane == 0u) {
        left = BUDGET - min(budget.Load(0u), BUDGET);
    }
    // ExecuteIndirect runs min(max, count) records: no others are clamped or
    // charged. The loop still counts tiles up to `records`, a constant, as
    // the compiler cannot prove that a value read from memory is the same
    // in every lane, and barriers need that; tiles past `walked` only wait
    // at them.
    uint walked = records;
    if ((flags & 1u) != 0u) {
        walked = min(records, countWord.Load(0u));
    }
    uint take_rest = flags & 2u;
    for (uint tile = 0u; tile < (records + 63u) / 64u; tile++) {
        uint record = tile * 64u + lane;
        uint at = record * stride;
        uint cost = 0u;
        if (record < walked) {
            cost = clamped(at);
        }
        costs[lane] = cost;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0u && tile * 64u < walked) {
            // The tile's walked records only: the slots past them cost nothing.
            uint own = min(64u, walked - tile * 64u);
            for (uint i = 0u; i < own; i++) {
                if (costs[i] > left) {
                    costs[i] = PAST;
                    if (take_rest != 0u) {
                        left = 0u;
                    }
                } else {
                    left -= costs[i];
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (record < walked && costs[lane] == PAST) {
            if (kind <= 1u) {
                dest.Store2(at, uint2(0u, 0u));
            } else {
                dest.Store3(at, uint3(0u, 0u, 0u));
            }
        }
    }
    if (lane == 0u) {
        budget.Store(0u, BUDGET - left);
    }
}
