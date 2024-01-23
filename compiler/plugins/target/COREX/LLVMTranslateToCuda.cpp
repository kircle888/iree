#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/LLVMLinkerUtils.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/compiler/Utils/ModuleUtils.h"
#include "iree/compiler/Utils/StringUtils.h"
#include "iree/compiler/Utils/ToolUtils.h"
#include "iree/schemas/cuda_executable_def_builder.h"
#include "iree_cuda/libdevice_embedded.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
namespace mlir {
namespace emitcuda {
class CudaEmitter {

public:
  CudaEmitter(raw_ostream &os) : os(os) {
    valueInScopeCount.push(0);
    labelInScopeCount.push(0);
  }
  LogicalResult emitOperation(Operation &op, bool trailingSemicolon);
  LogicalResult emitAssignPrefix(Operation &op);
  LogicalResult emitVariableDeclaration(OpResult result,
                                        bool trailingSemicolon);

  /// Return the existing or a new name for a Value.
  StringRef getOrCreateName(Value val);

  /// Return the existing or a new label of a Block.
  StringRef getOrCreateName(Block &block);
  /// Whether to map an mlir integer to a unsigned integer in C++.
  bool shouldMapToUnsigned(IntegerType::SignednessSemantics val);
  LogicalResult emitType(Location loc, Type type);
  /// Emits array of types as a std::tuple of the emitted types independently of
  /// the array size.
  LogicalResult emitTupleType(Location loc, ArrayRef<Type> types);
  LogicalResult emitLabel(Block &block);

  /// Returns wether the Value is assigned to a C++ variable in the scope.
  bool hasValueInScope(Value val);

  // Returns whether a label is assigned to the block.
  bool hasBlockLabel(Block &block);
  /// Emits attribute or returns failure.
  LogicalResult emitAttribute(Location loc, Attribute attr);
  raw_ostream &ostream() { return os; }
  struct Scope {
    Scope(CudaEmitter &emitter)
        : valueMapperScope(emitter.valueMapper),
          blockMapperScope(emitter.blockMapper), emitter(emitter) {
      emitter.valueInScopeCount.push(emitter.valueInScopeCount.top());
      emitter.labelInScopeCount.push(emitter.labelInScopeCount.top());
    }
    ~Scope() {
      emitter.valueInScopeCount.pop();
      emitter.labelInScopeCount.pop();
    }

  private:
    llvm::ScopedHashTableScope<Value, std::string> valueMapperScope;
    llvm::ScopedHashTableScope<Block *, std::string> blockMapperScope;
    CudaEmitter &emitter;
  };

private:
  raw_ostream &os;
  using ValueMapper = llvm::ScopedHashTable<Value, std::string>;
  using BlockMapper = llvm::ScopedHashTable<Block *, std::string>;

  /// Map from value to name of C++ variable that contain the name.
  ValueMapper valueMapper;

  /// Map from block to name of C++ label.
  BlockMapper blockMapper;

  /// The number of values in the current scope. This is used to declare the
  /// names of values in a scope.
  std::stack<int64_t> valueInScopeCount;
  std::stack<int64_t> labelInScopeCount;
};
/// Convenience functions to produce interleaved output with functions returning
/// a LogicalResult. This is different than those in STLExtras as functions used
/// on each element doesn't return a string.
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor>
inline LogicalResult
interleaveWithError(ForwardIterator begin, ForwardIterator end,
                    UnaryFunctor eachFn, NullaryFunctor betweenFn) {
  if (begin == end)
    return success();
  if (failed(eachFn(*begin)))
    return failure();
  ++begin;
  for (; begin != end; ++begin) {
    betweenFn();
    if (failed(eachFn(*begin)))
      return failure();
  }
  return success();
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult interleaveWithError(const Container &c,
                                         UnaryFunctor eachFn,
                                         NullaryFunctor betweenFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, betweenFn);
}

template <typename Container, typename UnaryFunctor>
inline LogicalResult interleaveCommaWithError(const Container &c,
                                              raw_ostream &os,
                                              UnaryFunctor eachFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, [&]() { os << ", "; });
}
static LogicalResult printBinaryOperation(CudaEmitter &emitter,
                                          Operation *operation,
                                          StringRef binaryOperator) {
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << emitter.getOrCreateName(operation->getOperand(0));
  os << " " << binaryOperator;
  os << " " << emitter.getOrCreateName(operation->getOperand(1));

  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, ModuleOp moduleOp) {
  CudaEmitter::Scope scope(emitter);

  for (Operation &op : moduleOp) {
    if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/false)))
      return failure();
  }
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, LLVM::AShrOp op) {
  Operation *operation = op.getOperation();

  return printBinaryOperation(emitter, operation, "<<");
}
static LogicalResult printOperation(CudaEmitter &emitter, LLVM::AddOp op) {
  Operation *operation = op.getOperation();

  return printBinaryOperation(emitter, operation, "+");
}
static LogicalResult printOperation(CudaEmitter &emitter, LLVM::AndOp op) {
  Operation *operation = op.getOperation();

  return printBinaryOperation(emitter, operation, "&");
}
static LogicalResult printOperation(CudaEmitter &emitter, LLVM::MulOp op) {
  Operation *operation = op.getOperation();

  return printBinaryOperation(emitter, operation, "*");
}
static LogicalResult printOperation(CudaEmitter &emitter, LLVM::SubOp op) {
  Operation *operation = op.getOperation();

  return printBinaryOperation(emitter, operation, "-");
}
static LogicalResult printOperation(CudaEmitter &emitter, LLVM::GEPOp op) {
  Operation *operation = op.getOperation();
  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  auto &os = emitter.ostream();
  os << "(";
  if (failed(emitter.emitType(op->getLoc(), op.getElemType())))
    return failure();
  os << "*)";
  os << emitter.getOrCreateName(operation->getOperand(0));
  os << "+";
  os << operation->getNumOperands();
  return success();
  // return printBinaryOperation(emitter, operation, "+");
}
static LogicalResult printOperation(CudaEmitter &emitter, LLVM::StoreOp op) {
  Operation *operation = op.getOperation();
  auto &os = emitter.ostream();
  os << "*(";
  if (failed(emitter.emitType(op->getLoc(), op->getOperand(0).getType())))
    return failure();
  os << "*)";
  os << emitter.getOrCreateName(operation->getOperand(1));
  os << " = ";
  os << emitter.getOrCreateName(operation->getOperand(0));
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, LLVM::LoadOp op) {
  Operation *operation = op.getOperation();
  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  auto &os = emitter.ostream();
  os << "*(";
  if (failed(emitter.emitType(op->getLoc(), op->getResult(0).getType())))
    return failure();
  os << "*)";
  os << emitter.getOrCreateName(operation->getOperand(0));
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, LLVM::ReturnOp op) {
  auto &os = emitter.ostream();
  os << "return";
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, LLVM::BrOp branchOp) {
  raw_ostream &os = emitter.ostream();
  Block &successor = *branchOp.getSuccessor();

  for (auto pair :
       llvm::zip(branchOp.getOperands(), successor.getArguments())) {
    Value &operand = std::get<0>(pair);
    BlockArgument &argument = std::get<1>(pair);
    os << emitter.getOrCreateName(argument) << " = "
       << emitter.getOrCreateName(operand) << ";\n";
  }

  os << "goto ";
  if (!(emitter.hasBlockLabel(successor)))
    return branchOp.emitOpError("unable to find label for successor block");
  os << emitter.getOrCreateName(successor);
  return success();
}

static LogicalResult printOperation(CudaEmitter &emitter,
                                    LLVM::CondBrOp condBranchOp) {
  raw_ostream &os = emitter.ostream();
  Block &trueSuccessor = *condBranchOp.getTrueDest();
  Block &falseSuccessor = *condBranchOp.getFalseDest();

  os << "if (" << emitter.getOrCreateName(condBranchOp.getCondition())
     << ") {\n";

  // If condition is true.
  for (auto pair : llvm::zip(condBranchOp.getTrueDestOperands(),
                             trueSuccessor.getArguments())) {
    Value &operand = std::get<0>(pair);
    BlockArgument &argument = std::get<1>(pair);
    os << emitter.getOrCreateName(argument) << " = "
       << emitter.getOrCreateName(operand) << ";\n";
  }

  os << "goto ";
  if (!(emitter.hasBlockLabel(trueSuccessor))) {
    return condBranchOp.emitOpError("unable to find label for successor block");
  }
  os << emitter.getOrCreateName(trueSuccessor) << ";\n";
  os << "} else {\n";
  // If condition is false.
  for (auto pair : llvm::zip(condBranchOp.getFalseDestOperands(),
                             falseSuccessor.getArguments())) {
    Value &operand = std::get<0>(pair);
    BlockArgument &argument = std::get<1>(pair);
    os << emitter.getOrCreateName(argument) << " = "
       << emitter.getOrCreateName(operand) << ";\n";
  }

  os << "goto ";
  if (!(emitter.hasBlockLabel(falseSuccessor))) {
    return condBranchOp.emitOpError()
           << "unable to find label for successor block";
  }
  os << emitter.getOrCreateName(falseSuccessor) << ";\n";
  os << "}";
  return success();
}

static LogicalResult printConstantOp(CudaEmitter &emitter, Operation *operation,
                                     Attribute value) {
  // Emit a variable declaration.
  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  return emitter.emitAttribute(operation->getLoc(), value);
}
static LogicalResult printOperation(CudaEmitter &emitter,
                                    LLVM::ConstantOp constantOp) {
  Operation *operation = constantOp.getOperation();
  Attribute value = constantOp.getValue();

  return printConstantOp(emitter, operation, value);
}
static LogicalResult printOperation(CudaEmitter &emitter,
                                    LLVM::ExtractElementOp exeOp) {
  Operation *operation = exeOp.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << emitter.getOrCreateName(operation->getOperand(0));
  os << "[";
  os << emitter.getOrCreateName(operation->getOperand(1));
  os << "]";
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, LLVM::SelectOp op) {
  Operation *operation = op.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << emitter.getOrCreateName(operation->getOperand(0));
  os << "?";
  os << emitter.getOrCreateName(operation->getOperand(1));
  os << ":";
  os << emitter.getOrCreateName(operation->getOperand(2));
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter,
                                    LLVM::ExtractValueOp exeOp) {
  Operation *operation = exeOp.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << emitter.getOrCreateName(operation->getOperand(1));
  os << "[";
  if (failed(emitter.emitAttribute(operation->getLoc(),
                                   exeOp.getPositionAttr()))) {
    return failure();
  }
  os << "]";
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, LLVM::FAddOp op) {
  Operation *operation = op.getOperation();

  return printBinaryOperation(emitter, operation, "+");
}
static LogicalResult printOperation(CudaEmitter &emitter, LLVM::FCmpOp op) {
  Operation *operation = op.getOperation();
  auto predicate = op.getPredicate();
  std::string boperator = "";
  if (predicate == LLVM::FCmpPredicate::_false) {
    Operation *operation = op.getOperation();
    raw_ostream &os = emitter.ostream();

    if (failed(emitter.emitAssignPrefix(*operation)))
      return failure();
    os << "false";
    return success();
  } else if (predicate == LLVM::FCmpPredicate::oeq)
    return printBinaryOperation(emitter, operation, "==");
  else if (predicate == LLVM::FCmpPredicate::ogt)
    return printBinaryOperation(emitter, operation, ">");
  else if (predicate == LLVM::FCmpPredicate::oge)
    return printBinaryOperation(emitter, operation, ">=");
  else if (predicate == LLVM::FCmpPredicate::olt)
    return printBinaryOperation(emitter, operation, "<");
  else if (predicate == LLVM::FCmpPredicate::ole)
    return printBinaryOperation(emitter, operation, "<=");
  else if (predicate == LLVM::FCmpPredicate::one)
    return printBinaryOperation(emitter, operation, "!=");
  // else if (predicate == LLVM::FCmpPredicate::ord)
  //   return printBinaryOperation(emitter, operation, "<=");
  else if (predicate == LLVM::FCmpPredicate::ugt)
    return printBinaryOperation(emitter, operation, ">");
  else if (predicate == LLVM::FCmpPredicate::uge)
    return printBinaryOperation(emitter, operation, ">=");
  else if (predicate == LLVM::FCmpPredicate::ult)
    return printBinaryOperation(emitter, operation, "<");
  else if (predicate == LLVM::FCmpPredicate::ule)
    return printBinaryOperation(emitter, operation, "<=");
  else if (predicate == LLVM::FCmpPredicate::une)
    return printBinaryOperation(emitter, operation, "!=");
  // else if (predicate == LLVM::FCmpPredicate::uno)
  //   return printBinaryOperation(emitter, operation, ">=");
  else if (predicate == LLVM::FCmpPredicate::_true) {
    Operation *operation = op.getOperation();
    raw_ostream &os = emitter.ostream();

    if (failed(emitter.emitAssignPrefix(*operation)))
      return failure();
    os << "true";
    return success();
  } else
    return failure();
}
static LogicalResult printOperation(CudaEmitter &emitter, LLVM::FDivOp op) {
  Operation *operation = op.getOperation();

  return printBinaryOperation(emitter, operation, "/");
}
static LogicalResult printOperation(CudaEmitter &emitter, LLVM::FMulOp op) {
  Operation *operation = op.getOperation();

  return printBinaryOperation(emitter, operation, "*");
}
static LogicalResult printOperation(CudaEmitter &emitter, LLVM::FNegOp op) {
  Operation *operation = op.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << "-" << emitter.getOrCreateName(operation->getOperand(0));
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, LLVM::FPToSIOp op) {
  Operation *operation = op.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << emitter.getOrCreateName(operation->getOperand(0));
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, LLVM::FPToUIOp op) {
  Operation *operation = op.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << emitter.getOrCreateName(operation->getOperand(0));
  return success();
}

static LogicalResult printOperation(CudaEmitter &emitter, LLVM::FSubOp op) {
  Operation *operation = op.getOperation();

  return printBinaryOperation(emitter, operation, "-");
}
static LogicalResult printOperation(CudaEmitter &emitter, LLVM::ICmpOp op) {
  Operation *operation = op.getOperation();
  auto predicate = op.getPredicate();
  std::string boperator = "";
  if (predicate == LLVM::ICmpPredicate::eq)
    return printBinaryOperation(emitter, operation, "==");
  else if (predicate == LLVM::ICmpPredicate::ne)
    return printBinaryOperation(emitter, operation, "!=");
  else if (predicate == LLVM::ICmpPredicate::slt)
    return printBinaryOperation(emitter, operation, "<");
  else if (predicate == LLVM::ICmpPredicate::sle)
    return printBinaryOperation(emitter, operation, "<=");
  else if (predicate == LLVM::ICmpPredicate::sgt)
    return printBinaryOperation(emitter, operation, ">");
  else if (predicate == LLVM::ICmpPredicate::sge)
    return printBinaryOperation(emitter, operation, ">=");
  else if (predicate == LLVM::ICmpPredicate::ult)
    return printBinaryOperation(emitter, operation, "<");
  else if (predicate == LLVM::ICmpPredicate::ule)
    return printBinaryOperation(emitter, operation, "<=");
  else if (predicate == LLVM::ICmpPredicate::ugt)
    return printBinaryOperation(emitter, operation, ">");
  else if (predicate == LLVM::ICmpPredicate::uge)
    return printBinaryOperation(emitter, operation, ">=");
  else
    return failure();
}
static LogicalResult printOperation(CudaEmitter &emitter, LLVM::PtrToIntOp op) {
  Operation *operation = op.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << emitter.getOrCreateName(operation->getOperand(0));
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, LLVM::SExtOp op) {
  Operation *operation = op.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << emitter.getOrCreateName(operation->getOperand(0));
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter,
                                    NVVM::ThreadIdXOp op) {
  Operation *operation = op.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << "threadIdx.x";
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter,
                                    NVVM::ThreadIdYOp op) {
  Operation *operation = op.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << "threadIdx.y";
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter,
                                    NVVM::ThreadIdZOp op) {
  Operation *operation = op.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << "threadIdx.z";
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, NVVM::BlockIdXOp op) {
  Operation *operation = op.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << "blockIdx.x";
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, NVVM::BlockIdYOp op) {
  Operation *operation = op.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << "blockIdx.y";
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, NVVM::BlockIdZOp op) {
  Operation *operation = op.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << "blockIdx.z";
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter,
                                    NVVM::BlockDimXOp op) {
  Operation *operation = op.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << "blockDim.x";
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter,
                                    NVVM::BlockDimYOp op) {
  Operation *operation = op.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << "blockDim.y";
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter,
                                    NVVM::BlockDimZOp op) {
  Operation *operation = op.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << "blockDim.z";
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, NVVM::GridDimXOp op) {
  Operation *operation = op.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << "gridDim.x";
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, NVVM::GridDimYOp op) {
  Operation *operation = op.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << "gridDim.y";
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, NVVM::GridDimZOp op) {
  Operation *operation = op.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << "gridDim.z";
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter,
                                    NVVM::BlockInClusterIdXOp op) {
  Operation *operation = op.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << "gridDim.x";
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter,
                                    NVVM::BlockInClusterIdYOp op) {
  Operation *operation = op.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << "gridDim.y";
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter,
                                    NVVM::BlockInClusterIdZOp op) {
  Operation *operation = op.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << "gridDim.z";
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter,
                                    LLVM::LLVMFuncOp functionOp) {
  CudaEmitter::Scope scope(emitter);
  auto &os = emitter.ostream();
  os << "__global__ ";
  if (failed(emitter.emitType(functionOp.getLoc(),
                              functionOp.getFunctionType().getReturnType())))
    return failure();
  os << " " << functionOp.getName();

  os << "(";
  if (failed(interleaveCommaWithError(
          functionOp.getArguments(), os,
          [&](BlockArgument arg) -> LogicalResult {
            if (failed(emitter.emitType(functionOp.getLoc(), arg.getType())))
              return failure();
            os << " " << emitter.getOrCreateName(arg);
            return success();
          })))
    return failure();
  os << ") {\n";

  Region::BlockListType &blocks = functionOp.getBlocks();
  // Create label names for basic blocks.
  for (Block &block : blocks) {
    emitter.getOrCreateName(block);
  }

  // Declare variables for basic block arguments.
  for (Block &block : llvm::drop_begin(blocks)) {
    for (BlockArgument &arg : block.getArguments()) {
      if (emitter.hasValueInScope(arg))
        return functionOp.emitOpError(" block argument #")
               << arg.getArgNumber() << " is out of scope";
      if (failed(
              emitter.emitType(block.getParentOp()->getLoc(), arg.getType()))) {
        return failure();
      }
      os << " " << emitter.getOrCreateName(arg) << ";\n";
    }
  }

  for (Block &block : blocks) {
    // Only print a label if the block has predecessors.
    if (!block.hasNoPredecessors()) {
      if (failed(emitter.emitLabel(block)))
        return failure();
    }
    for (Operation &op : block.getOperations()) {
      // When generating code for an emitc.if or cf.cond_br op no semicolon
      // needs to be printed after the closing brace.
      // When generating code for an emitc.for op, printing a trailing semicolon
      // is handled within the printOperation function.
      bool trailingSemicolon = true;

      if (failed(emitter.emitOperation(
              op, /*trailingSemicolon=*/trailingSemicolon)))
        return failure();
    }
  }
  os << "}\n";
  return success();
}
LogicalResult CudaEmitter::emitLabel(Block &block) {
  if (!hasBlockLabel(block))
    return block.getParentOp()->emitError("label for block not found");
  // FIXME: Add feature in `raw_indented_ostream` to ignore indent for block
  // label instead of using `getOStream`.
  os << getOrCreateName(block) << ":\n";
  return success();
}
bool CudaEmitter::hasValueInScope(Value val) { return valueMapper.count(val); }

bool CudaEmitter::hasBlockLabel(Block &block) {
  return blockMapper.count(&block);
}
bool CudaEmitter::shouldMapToUnsigned(IntegerType::SignednessSemantics val) {
  switch (val) {
  case IntegerType::Signless:
    return false;
  case IntegerType::Signed:
    return false;
  case IntegerType::Unsigned:
    return true;
  }
  llvm_unreachable("Unexpected IntegerType::SignednessSemantics");
}
/// Return the existing or a new name for a Value.
StringRef CudaEmitter::getOrCreateName(Value val) {
  if (!valueMapper.count(val)) {
    valueMapper.insert(val, llvm::formatv("v{0}", ++valueInScopeCount.top()));
  }
  return *valueMapper.begin(val);
}

/// Return the existing or a new label for a Block.
StringRef CudaEmitter::getOrCreateName(Block &block) {
  if (!blockMapper.count(&block))
    blockMapper.insert(&block,
                       llvm::formatv("label{0}", ++labelInScopeCount.top()));
  return *blockMapper.begin(&block);
}
LogicalResult CudaEmitter::emitAttribute(Location loc, Attribute attr) {
  auto printInt = [&](const APInt &val, bool isUnsigned) {
    if (val.getBitWidth() == 1) {
      if (val.getBoolValue())
        os << "true";
      else
        os << "false";
    } else {
      SmallString<128> strValue;
      val.toString(strValue, 10, !isUnsigned, false);
      os << strValue;
    }
  };

  auto printFloat = [&](const APFloat &val) {
    if (val.isFinite()) {
      SmallString<128> strValue;
      // Use default values of toString except don't truncate zeros.
      val.toString(strValue, 0, 0, false);
      switch (llvm::APFloatBase::SemanticsToEnum(val.getSemantics())) {
      case llvm::APFloatBase::S_IEEEsingle:
        os << "(float)";
        break;
      case llvm::APFloatBase::S_IEEEdouble:
        os << "(double)";
        break;
      default:
        break;
      };
      os << strValue;
    } else if (val.isNaN()) {
      os << "NAN";
    } else if (val.isInfinity()) {
      if (val.isNegative())
        os << "-";
      os << "INFINITY";
    }
  };

  // Print floating point attributes.
  if (auto fAttr = dyn_cast<FloatAttr>(attr)) {
    printFloat(fAttr.getValue());
    return success();
  }
  if (auto dense = dyn_cast<DenseFPElementsAttr>(attr)) {
    os << '{';
    interleaveComma(dense, os, [&](const APFloat &val) { printFloat(val); });
    os << '}';
    return success();
  }

  // Print integer attributes.
  if (auto iAttr = dyn_cast<IntegerAttr>(attr)) {
    if (auto iType = dyn_cast<IntegerType>(iAttr.getType())) {
      printInt(iAttr.getValue(), shouldMapToUnsigned(iType.getSignedness()));
      return success();
    }
    if (auto iType = dyn_cast<IndexType>(iAttr.getType())) {
      printInt(iAttr.getValue(), false);
      return success();
    }
  }
  if (auto dense = dyn_cast<DenseIntElementsAttr>(attr)) {
    if (auto iType = dyn_cast<IntegerType>(
            cast<TensorType>(dense.getType()).getElementType())) {
      os << '{';
      interleaveComma(dense, os, [&](const APInt &val) {
        printInt(val, shouldMapToUnsigned(iType.getSignedness()));
      });
      os << '}';
      return success();
    }
    if (auto iType = dyn_cast<IndexType>(
            cast<TensorType>(dense.getType()).getElementType())) {
      os << '{';
      interleaveComma(dense, os,
                      [&](const APInt &val) { printInt(val, false); });
      os << '}';
      return success();
    }
  }

  // Print symbolic reference attributes.
  if (auto sAttr = dyn_cast<SymbolRefAttr>(attr)) {
    if (sAttr.getNestedReferences().size() > 1)
      return emitError(loc, "attribute has more than 1 nested reference");
    os << sAttr.getRootReference().getValue();
    return success();
  }

  // Print type attributes.
  if (auto type = dyn_cast<TypeAttr>(attr))
    return emitType(loc, type.getValue());

  return emitError(loc, "cannot emit attribute: ") << attr;
}
LogicalResult CudaEmitter::emitAssignPrefix(Operation &op) {
  switch (op.getNumResults()) {
  case 0:
    break;
  case 1: {
    OpResult result = op.getResult(0);

    if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/false)))
      return failure();
    os << " = ";

    break;
  }
  default:
    for (OpResult result : op.getResults()) {
      if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/true)))
        return failure();
    }
    os << "std::tie(";
    interleaveComma(op.getResults(), os,
                    [&](Value result) { os << getOrCreateName(result); });
    os << ") = ";
  }
  return success();
}

LogicalResult CudaEmitter::emitTupleType(Location loc, ArrayRef<Type> types) {
  os << "std::tuple<";
  if (failed(interleaveCommaWithError(
          types, os, [&](Type type) { return emitType(loc, type); })))
    return failure();
  os << ">";
  return success();
}

LogicalResult CudaEmitter::emitType(Location loc, Type type) {
  if (auto vType = dyn_cast<LLVM::LLVMVoidType>(type)) {
    os << "void";
    return success();
  }
  if (auto iType = dyn_cast<IntegerType>(type)) {
    switch (iType.getWidth()) {
    case 1:
      return (os << "bool"), success();
    case 8:
    case 16:
    case 32:
    case 64:
      if (shouldMapToUnsigned(iType.getSignedness()))
        return (os << "uint" << iType.getWidth() << "_t"), success();
      else
        return (os << "int" << iType.getWidth() << "_t"), success();
    default:
      return emitError(loc, "cannot emit integer type ") << type;
    }
  }
  if (auto fType = dyn_cast<FloatType>(type)) {
    switch (fType.getWidth()) {
    case 32:
      return (os << "float"), success();
    case 64:
      return (os << "double"), success();
    default:
      return emitError(loc, "cannot emit float type ") << type;
    }
  }
  if (auto iType = dyn_cast<IndexType>(type))
    return (os << "size_t"), success();
  if (auto tType = dyn_cast<TensorType>(type)) {
    if (!tType.hasRank())
      return emitError(loc, "cannot emit unranked tensor type");
    if (!tType.hasStaticShape())
      return emitError(loc, "cannot emit tensor type with non static shape");
    os << "Tensor<";
    if (failed(emitType(loc, tType.getElementType())))
      return failure();
    auto shape = tType.getShape();
    for (auto dimSize : shape) {
      os << ", ";
      os << dimSize;
    }
    os << ">";
    return success();
  }
  if (auto tType = dyn_cast<TupleType>(type))
    return emitTupleType(loc, tType.getTypes());
  if (auto pType = dyn_cast<LLVM::LLVMPointerType>(type)) {
    os << "void *";
    return success();
  }
  return emitError(loc, "cannot emit type ") << type;
}
LogicalResult CudaEmitter::emitVariableDeclaration(OpResult result,
                                                   bool trailingSemicolon) {
  if (hasValueInScope(result)) {
    return result.getDefiningOp()->emitError(
        "result variable for the operation already declared");
  }
  if (failed(emitType(result.getOwner()->getLoc(), result.getType())))
    return failure();
  os << " " << getOrCreateName(result);
  if (trailingSemicolon)
    os << ";\n";
  return success();
}
LogicalResult CudaEmitter::emitOperation(Operation &op,
                                         bool trailingSemicolon) {
  llvm::errs() << op.getName() << "\n";
  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          // Builtin ops.
          .Case<ModuleOp>([&](auto op) { return printOperation(*this, op); })
          .Case<LLVM::GlobalOp>([&](auto op) {
            // os << "//";
            // op.print(os);
            return success();
          })
          .Case<LLVM::LLVMFuncOp>(
              [&](auto op) { return printOperation(*this, op); })
          // CF ops.
          .Case<LLVM::BrOp, LLVM::CondBrOp, LLVM::AShrOp, LLVM::AddOp,
                LLVM::AndOp, LLVM::FAddOp, LLVM::FCmpOp, LLVM::FDivOp,
                LLVM::FMulOp, LLVM::FNegOp, LLVM::FPToSIOp, LLVM::FPToUIOp,
                LLVM::GEPOp, LLVM::FSubOp, LLVM::ICmpOp, LLVM::ConstantOp,
                LLVM::PtrToIntOp, LLVM::SExtOp, LLVM::MulOp, LLVM::SubOp,
                LLVM::StoreOp, LLVM::ReturnOp, LLVM::SelectOp, LLVM::LoadOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<NVVM::ThreadIdXOp, NVVM::ThreadIdYOp, NVVM::ThreadIdZOp,
                NVVM::BlockIdXOp, NVVM::BlockIdYOp, NVVM::BlockIdZOp,
                NVVM::BlockDimXOp, NVVM::BlockDimYOp, NVVM::BlockDimZOp,
                NVVM::GridDimXOp, NVVM::GridDimYOp, NVVM::GridDimZOp,
                NVVM::BlockInClusterIdXOp, NVVM::BlockInClusterIdYOp,
                NVVM::BlockInClusterIdZOp>(
              [&](auto op) { return printOperation(*this, op); })
          // Func ops.
          // .Case<func::CallOp, func::ConstantOp, func::FuncOp,
          // func::ReturnOp>(
          //     [&](auto op) { return printOperation(*this, op); })
          .Default([&](Operation *) {
            os << "//";
            os << op.getName();
            return success();
            // return op.emitOpError("unable to find printer for op");
          });

  if (failed(status))
    return failure();

  os << (trailingSemicolon ? ";\n" : "\n");
  return success();
}
/// Translates the given module op to CUDA kernel code. The operation or
/// operations in the region of 'op' need all be in LLVM or NVVM dialect.
LogicalResult translateToCuda(Operation *op, raw_ostream &os) {
  CudaEmitter emmiter(os);
  return emmiter.emitOperation(*op, false);
}
} // namespace emitcuda
} // namespace mlir