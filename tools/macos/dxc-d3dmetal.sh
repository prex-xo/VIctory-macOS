#!/bin/sh
# Builds dxc-d3dmetal, a minimal DXC front end over the libdxcompiler.dylib
# that D3DMetal ships, for VICTORY_HELPER_DXC (see docs/macos.md):
#   tools/macos/dxc-d3dmetal.sh OUTPUT
# The source is kept here because the repository's asset policy allows shell
# scripts but not C++ files. The library is x86_64 only, so this builds for
# x86_64 and the helper runs under Rosetta.
set -e
out="$1"
[ -n "$out" ] || { echo "usage: $0 OUTPUT" >&2; exit 2; }
/usr/bin/clang++ -arch x86_64 -std=c++17 -O2 -x c++ -o "$out" - <<'CPP'
// A minimal DXC front end over the libdxcompiler.dylib that D3DMetal ships,
// for `VICTORY_HELPER_DXC`: `dxc-d3dmetal -T PROFILE -E ENTRY -Fo OUTPUT
// INPUT`, the form crates/victory-gpu/build.rs uses. The library is x86_64
// only, so this builds for x86_64 and runs under Rosetta. DXC_LIB names the
// library; the build script sets it.
//
// The declarations below follow DXC's public dxcapi.h. Wide strings are
// wchar_t. Unlike DXC's own WinAdapter.h, D3DMetal's build has no virtual
// destructor in IUnknown (with those two slots Compile crashed).
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <string>
#include <vector>

typedef int32_t HRESULT;
typedef uint32_t ULONG;
struct GUID {
    uint32_t a;
    uint16_t b, c;
    uint8_t d[8];
};

struct IUnknown {
    virtual HRESULT QueryInterface(const GUID &iid, void **out) = 0;
    virtual ULONG AddRef() = 0;
    virtual ULONG Release() = 0;
};
struct IDxcBlob : IUnknown {
    virtual void *GetBufferPointer() = 0;
    virtual size_t GetBufferSize() = 0;
};
struct IDxcIncludeHandler;
struct DxcBuffer {
    const void *Ptr;
    size_t Size;
    uint32_t Encoding;
};
struct IDxcOperationResult : IUnknown {
    virtual HRESULT GetStatus(HRESULT *status) = 0;
    virtual HRESULT GetResult(IDxcBlob **result) = 0;
    virtual HRESULT GetErrorBuffer(IDxcBlob **errors) = 0;
};
struct IDxcCompiler3 : IUnknown {
    virtual HRESULT Compile(const DxcBuffer *source, const wchar_t **args, uint32_t count,
                            IDxcIncludeHandler *include, const GUID &iid, void **result) = 0;
};

// {73e22d93-e6ce-47f3-b5bf-f0664f39c1b0}
static const GUID CLSID_DxcCompiler = {
    0x73e22d93, 0xe6ce, 0x47f3, {0xb5, 0xbf, 0xf0, 0x66, 0x4f, 0x39, 0xc1, 0xb0}};
// {228B4687-5A6A-4730-900C-9702B2203F54}
static const GUID IID_IDxcCompiler3 = {
    0x228b4687, 0x5a6a, 0x4730, {0x90, 0x0c, 0x97, 0x02, 0xb2, 0x20, 0x3f, 0x54}};
// {CEDB484A-D4E9-445A-B991-CA21CA157DC2}
static const GUID IID_IDxcOperationResult = {
    0xcedb484a, 0xd4e9, 0x445a, {0xb9, 0x91, 0xca, 0x21, 0xca, 0x15, 0x7d, 0xc2}};

typedef HRESULT (*CreateInstance)(const GUID &clsid, const GUID &iid, void **out);

static std::wstring wide(const char *s) {
    std::wstring w;
    while (*s) w.push_back(static_cast<wchar_t>(static_cast<unsigned char>(*s++)));
    return w;
}

int main(int argc, char **argv) {
    if (argc != 8 || strcmp(argv[1], "-T") || strcmp(argv[3], "-E") || strcmp(argv[5], "-Fo")) {
        fprintf(stderr, "Expected -T PROFILE -E ENTRY -Fo OUTPUT INPUT\n");
        return 2;
    }
    const char *lib = getenv("DXC_LIB");
    if (!lib) {
        fprintf(stderr, "Set DXC_LIB to D3DMetal's libdxcompiler.dylib\n");
        return 2;
    }
    void *dll = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!dll) {
        fprintf(stderr, "cannot load %s: %s\n", lib, dlerror());
        return 1;
    }
    auto create = reinterpret_cast<CreateInstance>(dlsym(dll, "DxcCreateInstance"));
    if (!create) {
        fprintf(stderr, "%s has no DxcCreateInstance\n", lib);
        return 1;
    }

    FILE *in = fopen(argv[7], "rb");
    if (!in) {
        perror(argv[7]);
        return 1;
    }
    std::vector<char> text;
    char chunk[4096];
    for (size_t n; (n = fread(chunk, 1, sizeof chunk, in)) > 0;) text.insert(text.end(), chunk, chunk + n);
    fclose(in);

    IDxcCompiler3 *compiler = nullptr;
    HRESULT hr = create(CLSID_DxcCompiler, IID_IDxcCompiler3, reinterpret_cast<void **>(&compiler));
    if (hr < 0 || !compiler) {
        fprintf(stderr, "DxcCreateInstance failed: 0x%08x\n", static_cast<unsigned>(hr));
        return 1;
    }
    std::wstring profile = wide(argv[2]), entry = wide(argv[4]), name = wide(argv[7]);
    const wchar_t *args[] = {name.c_str(), L"-T", profile.c_str(), L"-E", entry.c_str(), L"-O3"};
    DxcBuffer source = {text.data(), text.size(), 65001 /* UTF-8 */};
    IDxcOperationResult *result = nullptr;
    hr = compiler->Compile(&source, args, sizeof args / sizeof *args, nullptr, IID_IDxcOperationResult,
                           reinterpret_cast<void **>(&result));
    if (hr < 0 || !result) {
        fprintf(stderr, "Compile failed: 0x%08x\n", static_cast<unsigned>(hr));
        return 1;
    }
    HRESULT status = 0;
    result->GetStatus(&status);
    IDxcBlob *errors = nullptr;
    if (result->GetErrorBuffer(&errors) >= 0 && errors && errors->GetBufferSize() > 0) {
        fwrite(errors->GetBufferPointer(), 1, errors->GetBufferSize(), stderr);
    }
    if (status < 0) {
        fprintf(stderr, "%s: compilation failed: 0x%08x\n", argv[7], static_cast<unsigned>(status));
        return 1;
    }
    IDxcBlob *object = nullptr;
    if (result->GetResult(&object) < 0 || !object) {
        fprintf(stderr, "no compiled object\n");
        return 1;
    }
    FILE *out = fopen(argv[6], "wb");
    if (!out || fwrite(object->GetBufferPointer(), 1, object->GetBufferSize(), out) != object->GetBufferSize()) {
        perror(argv[6]);
        return 1;
    }
    fclose(out);
    return 0;
}
CPP
