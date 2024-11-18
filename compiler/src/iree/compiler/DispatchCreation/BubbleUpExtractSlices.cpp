// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_BUBBLEUPEXTRACTSLICESPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

// Convert extract_slice(dequant) to dequant(extract_slice)
//
// Because `extract_slice` ops and dequantize-like ops get cloned into regions
// later, it's okay to bubble up through multi-use dequant ops.
struct BubbleUpExtract : OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const final {
    Value source = sliceOp.getSource();
    auto genericOp = source.getDefiningOp<linalg::GenericOp>();
    if (!genericOp || genericOp->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          sliceOp, "expected source to implement `linalg::LinalgOp` and have a "
                   "single result");
    }

    if (!IREE::LinalgExt::isBitExtendOp(genericOp) && !genericOp->hasOneUse()) {
      return rewriter.notifyMatchFailure(
          sliceOp,
          "expected source to be dequantize-like op or have a single use");
    }

    if (!sliceOp.hasUnitStride()) {
      return rewriter.notifyMatchFailure(sliceOp, "expected unit stride");
    }

    if (!llvm::all_of(genericOp.getIndexingMapsArray(), [](AffineMap map) {
          return map.isProjectedPermutation();
        })) {
      return rewriter.notifyMatchFailure(
          genericOp,
          "expected generic op to have all projected permutation maps");
    }

    if (genericOp.hasIndexSemantics()) {
      return rewriter.notifyMatchFailure(
          genericOp, "pattern doesn't support index semantics");
    }

    Value replacement;
    linalg::GenericOp swappedOp;
    {
      FailureOr<TilingResult> tilingResult =
          tensor::replaceExtractSliceWithTiledProducer(rewriter, sliceOp,
                                                       genericOp->getResult(0));
      assert(succeeded(tilingResult) && "failed to swap extract_slice with op");
      assert(tilingResult->tiledOps.size() == 1);
      replacement = tilingResult->tiledValues[0];
      swappedOp = cast<linalg::GenericOp>(tilingResult->tiledOps[0]);
    }

    // Check if this is a rank-reducing slice, if so we need to fold the unit
    // dimensions of the op.
    // This is necessary because `replaceExtractSliceWithTiledProducer` does not
    // take into account the `extract_slice`'s implicit rank reduction. The
    // operations generated by that function will have any unit dims that were
    // removed by the original `extract_slice`. Folding them away ensures that
    // the types match.
    if (sliceOp.getSourceType().getRank() !=
        sliceOp.getResultType().getRank()) {

      llvm::SmallBitVector droppedDims = sliceOp.getDroppedDims();
      // Get the indexing map for the result.
      AffineMap resultMap =
          swappedOp.getIndexingMapMatchingResult(swappedOp->getResult(0));
      linalg::ControlDropUnitDims options;
      options.rankReductionStrategy = linalg::ControlDropUnitDims::
          RankReductionStrategy::ExtractInsertSlice;
      options.controlFn = [&](Operation *op) -> SmallVector<unsigned> {
        SmallVector<unsigned> droppedDimsVec;
        for (auto [index, expr] : llvm::enumerate(resultMap.getResults())) {
          if (!droppedDims.test(index)) {
            continue;
          }
          auto dimExpr = cast<AffineDimExpr>(expr);
          droppedDimsVec.push_back(dimExpr.getPosition());
        }
        return droppedDimsVec;
      };
      FailureOr<linalg::DropUnitDimsResult> dropUnitDims =
          linalg::dropUnitDims(rewriter, swappedOp, options);
      assert(succeeded(dropUnitDims) &&
             "failed to drop unit dims of produced operation");
      swappedOp = dropUnitDims->resultOp;
      replacement = swappedOp->getResult(0);
    }
    rewriter.replaceOp(sliceOp, replacement);
    return success();
  }
};

/// Swaps tensor.extract_slice(linalg.fill(%cst, %init)) into linalg.fill(%cst,
/// tensor.extract_slice(%init)) even when the linalg.fill has multiple users.
/// Bubbles up tensor.extract_slice when encountered with linalg.fill and the
/// former can be folded away.
struct SwapExtractSliceOfFill final
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractOp,
                                PatternRewriter &rewriter) const override {
    auto fillOp = extractOp.getSource().getDefiningOp<linalg::FillOp>();
    if (!fillOp)
      return failure();

    auto newExtractOp = rewriter.create<tensor::ExtractSliceOp>(
        extractOp.getLoc(), extractOp.getType(), fillOp.getOutputs()[0],
        extractOp.getMixedOffsets(), extractOp.getMixedSizes(),
        extractOp.getMixedStrides());
    rewriter.replaceOpWithNewOp<linalg::FillOp>(
        extractOp, fillOp.getInputs(), ValueRange{newExtractOp.getResult()});
    return success();
  }
};

struct BubbleUpExtractSlicesPass
    : impl::BubbleUpExtractSlicesPassBase<BubbleUpExtractSlicesPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    {
      RewritePatternSet patterns(context);
      patterns.insert<BubbleUpExtract>(context);
      patterns.insert<SwapExtractSliceOfFill>(context);
      tensor::populateFoldTensorEmptyPatterns(patterns, false);
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
