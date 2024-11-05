#pragma once

#define DLL_EXPORT __declspec(dllexport)

namespace zen
{
    template<typename DeviceFunction>
    struct Handle {
        DeviceFunction* gemm;
        explicit Handle(DeviceFunction* gemm)
        {
            this->gemm = gemm;
        }
        void run(cudaStream_t stream) const
        {
            this->gemm->run(stream);
        }
    };
}
