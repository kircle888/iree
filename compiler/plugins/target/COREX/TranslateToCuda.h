#ifndef IREE_COMPILER_PLUGINS_TARGET_COREX_TRANSLATETOCUDA_H_
#define IREE_COMPILER_PLUGINS_TARGET_COREX_TRANSLATETOCUDA_H_

#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"
namespace mlir {
namespace emitcuda {
llvm::LogicalResult translateToCuda(mlir::Operation *op, llvm::raw_ostream &os);
}
} // namespace mlir
#endif // IREE_COMPILER_PLUGINS_TARGET_COREX_TRANSLATETOCUDA_H_