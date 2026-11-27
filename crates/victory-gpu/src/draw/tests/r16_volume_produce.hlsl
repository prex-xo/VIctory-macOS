RWTexture3D<float> volume : register(u0);
groupshared uint cell[64];

[numthreads(2, 2, 16)]
void main(uint3 pt : SV_DispatchThreadID, uint lane : SV_GroupIndex) {
    if ((lane & 1) == 0) cell[lane] = lane;
    GroupMemoryBarrierWithGroupSync();
    uint distance_seed = cell[(lane & 62) ^ 32];
    volume[pt] = 1 + (3 * pt.x + 5 * pt.y + 7 * pt.z + distance_seed) % 1536;
}
