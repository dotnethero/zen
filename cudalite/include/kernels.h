#pragma once

#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>

namespace zen
{
    namespace device
    {
        using IdentitySwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

        template<
            typename ElementA, typename LayoutA,
            typename ElementB, typename LayoutB,
            typename ElementC, typename LayoutC,
            typename ElementAccumulator = ElementC,
            typename OpClass = cutlass::arch::OpClassSimt,
            typename ArchTag = cutlass::arch::Sm70,
            typename MathTag = cutlass::arch::OpMultiplyAdd,
            typename Config = cutlass::gemm::device::DefaultGemmConfiguration<
                OpClass,
                ArchTag,
                ElementA,
                ElementB,
                ElementC,
                ElementAccumulator>>
        using Conv2dFprop = typename cutlass::conv::kernel::DefaultConv2dFprop<
            ElementA, LayoutA,
            ElementB, LayoutB,
            ElementC, LayoutC,
            ElementAccumulator,
            OpClass,
            ArchTag,
            typename Config::ThreadblockShape,
            typename Config::WarpShape,
            typename Config::InstructionShape,
            typename Config::EpilogueOutputOp,
            IdentitySwizzle,
            3,
            MathTag>::Kernel;

        template<
            typename ElementA, typename LayoutA,
            typename ElementB, typename LayoutB,
            typename ElementC, typename LayoutC,
            typename ElementAccumulator = ElementC,
            typename OpClass = cutlass::arch::OpClassSimt,
            typename ArchTag = cutlass::arch::Sm70,
            typename MathTag = cutlass::arch::OpMultiplyAdd>
        using Conv2d = cutlass::conv::device::ImplicitGemmConvolution<
            Conv2dFprop<
                ElementA, LayoutA,
                ElementB, LayoutB,
                ElementC, LayoutC,
                ElementAccumulator,
                OpClass,
                ArchTag,
                MathTag>>;
    } // device
} // zen
