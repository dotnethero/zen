#pragma once

#include "helpers.h"

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal.h>

using Element = float;
using Layout  = cutlass::layout::RowMajor;
using Gemm    = cutlass::gemm::device::Gemm<
    Element, Layout,
    Element, Layout,
    Element, Layout>;

using GemmHandle = zen::Handle<Gemm>;

extern "C" {
    
    DLL_EXPORT void zenCreateGemm(
        GemmHandle** handle,
        int m,
        int n,
        int k,
        const Element* a, int lda,
        const Element* b, int ldb,
              Element* c, int ldc) {
        
        Gemm* gemm = new Gemm();
        Gemm::Arguments args {
            cutlass::gemm::GemmCoord(m, n, k),
            {a, lda},
            {b, ldb},
            {c, ldc},
            {c, ldc},
            {1.0f, 0.0f}
        };

        auto status = gemm->initialize(args);
        
        if (status != cutlass::Status::kSuccess) {
            delete gemm;
            return;
        }
        
        *handle = new GemmHandle(gemm);
    }

    DLL_EXPORT void zenExecuteGemm(GemmHandle* handle, cudaStream_t stream) {
        handle->run(stream);
    }

    DLL_EXPORT void zenDestroyGemm(GemmHandle* handle) {
        delete handle->gemm;
        delete handle;
    }
}