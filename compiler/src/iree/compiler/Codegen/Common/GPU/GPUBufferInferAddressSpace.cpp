// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-buffer-infer-addressspace"

namespace mlir::iree_compiler {

static constexpr int64_t kCudaWarpSize = 32;

namespace {

void modifyAllocOp(memref::AllocOp allocOp) {
  auto alloced = allocOp.getResult();
  auto memRefType = alloced.getType();
  if (memRefType.getMemorySpace()) {
    // llvm::outs() << "Pass " << memRefType << "\n";
    return;
  }
  // llvm::outs() << "Before " << memRefType << "\n";
  bool threadFind = false, blockFind = false;
  auto visitDef = [&](auto self, Value val) -> void {
    if (!llvm::isa_and_nonnull<IndexType>(val.getType())) {
      return;
    }
    Operation *defOp = val.getDefiningOp();
    if (auto arg = val.dyn_cast<BlockArgument>()) {
      auto bOp = arg.getOwner()->getParentOp();
      int argNum = arg.getArgNumber();
      if (auto forallOp = llvm::dyn_cast_or_null<scf::ForallOp>(bOp)) {
        auto map = forallOp.getMapping();
        if (!map || argNum >= map->size())
          return;
        auto mapped = map->getValue()[argNum];
        if (llvm::isa_and_nonnull<gpu::GPUThreadMappingAttr>(mapped)) {
          threadFind = true;
        }
      }
    } else if (!defOp || defOp == allocOp.getOperation()) {
      return;
    } else if (llvm::dyn_cast<gpu::ThreadIdOp>(defOp)) {
      threadFind = true;
    } else if (llvm::dyn_cast<iree_compiler::IREE::HAL::InterfaceWorkgroupIDOp>(
                   defOp)) {
      blockFind = true;
    } else {
      for (auto operand : defOp->getOperands()) {
        self(self, operand);
      }
    }
  };

  auto visitUse = [&](auto self, Value val) -> void {
    if (!llvm::isa_and_nonnull<MemRefType>(val.getType()))
      return;
    for (Operation *use : val.getUsers()) {
      if (auto subviewOp = llvm::dyn_cast<memref::SubViewOp>(use)) {
        self(self, subviewOp.getResult());
        for (auto operand : subviewOp.getOffsets()) {
          visitDef(visitDef, operand);
        }
      } else if (auto viewOp = llvm::dyn_cast<memref::ViewOp>(use)) {
        self(self, viewOp.getResult());
        visitDef(visitDef, viewOp.getByteShift());
      } else if (auto copyOp = llvm::dyn_cast<memref::CopyOp>(use)) {
      } else {
        for (auto operand : use->getOperands()) {
          visitDef(visitDef, operand);
        }
      }
    }
  };

  visitUse(visitUse, alloced);
  gpu::AddressSpaceAttr memorySpace = gpu::AddressSpaceAttr::get(
      allocOp.getContext(), gpu::GPUDialect::getPrivateAddressSpace());
  blockFind = false;
  if (blockFind) {
    memorySpace = gpu::AddressSpaceAttr::get(allocOp.getContext(),
                                             gpu::AddressSpace::Global);
  } else if (threadFind) {
    memorySpace = gpu::AddressSpaceAttr::get(
        allocOp.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  }

  auto replaceSubview = [&](auto self, TypedValue<MemRefType> val) -> void {
    MemRefType originType = val.getType();
    MemRefType allocType =
        MemRefType::get(originType.getShape(), originType.getElementType(),
                        originType.getLayout(), memorySpace);
    val.setType(allocType);
    for (Operation *use : val.getUsers()) {
      if (auto subviewOp = llvm::dyn_cast<memref::SubViewOp>(use)) {
        self(self, subviewOp.getResult());
      } else if (auto viewOp = llvm::dyn_cast<memref::ViewOp>(use)) {
        self(self, viewOp.getResult());
      }
    }
  };
  replaceSubview(replaceSubview, alloced);

  // llvm::outs() << "After " << alloced.getType() << "\n";
  return;
}
struct GPUBufferInferAddressSpace
    : public GPUBufferInferAddressSpaceBase<GPUBufferInferAddressSpace> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, gpu::GPUDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();
    funcOp->walk([](memref::AllocOp op) { modifyAllocOp(op); });
  };
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createGPUBufferInferAddressSpace() {
  return std::make_unique<GPUBufferInferAddressSpace>();
}

} // namespace mlir::iree_compiler
