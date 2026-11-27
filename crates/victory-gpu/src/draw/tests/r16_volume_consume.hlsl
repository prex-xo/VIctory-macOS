Texture3D<float> volume : register(t0);
RWByteAddressBuffer output : register(u1);
static const uint slices[8] = {0, 7, 8, 15, 16, 40, 255, 511};

[numthreads(8, 8, 1)]
void main(uint3 pt : SV_DispatchThreadID) {
    if (pt.x >= 60 || pt.y >= 34) return;
    for (uint i = 0; i < 8; i++) {
        float value = volume.Load(int4(pt.x, pt.y, slices[i], 0));
        output.Store((i * 60 * 34 + pt.y * 60 + pt.x) * 4, asuint(value));
    }
}
