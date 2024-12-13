#include <iree/compiler/Dialect/HAL/IR/HALOps.h>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <queue>
#include <vector>
#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/HAL/Target/LLVMLinkerUtils.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/compiler/Utils/ModuleUtils.h"
#include "iree/compiler/Utils/StringUtils.h"
#include "iree/compiler/Utils/ToolUtils.h"
#include "iree/schemas/cuda_executable_def_builder.h"
#include "iree_cuda/libdevice_embedded.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
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
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

namespace mlir {
namespace emitcuda {
using llvm::formatv;
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

namespace {
/// Emitter that uses dialect specific emitters to emit C++ code.
struct CudaEmitter {
  explicit CudaEmitter(raw_ostream &os, bool declareVariablesAtTop);

  /// Emits attribute or returns failure.
  LogicalResult emitAttribute(Location loc, Attribute attr);

  /// Emits operation 'op' with/without training semicolon or returns failure.
  LogicalResult emitOperation(Operation &op, bool trailingSemicolon);

  /// Emits type 'type' or returns failure.
  LogicalResult emitType(Location loc, Type type);

  LogicalResult emitStruct(Location loc, LLVM::LLVMStructType sType);

  /// Emits array of types as a std::tuple of the emitted types.
  /// - emits void for an empty array;
  /// - emits the type of the only element for arrays of size one;
  /// - emits a std::tuple otherwise;
  LogicalResult emitTypes(Location loc, ArrayRef<Type> types);

  /// Emits array of types as a std::tuple of the emitted types independently of
  /// the array size.
  LogicalResult emitTupleType(Location loc, ArrayRef<Type> types);

  /// Emits an assignment for a variable which has been declared previously.
  LogicalResult emitVariableAssignment(OpResult result);

  /// Emits a variable declaration for a result of an operation.
  LogicalResult emitVariableDeclaration(OpResult result,
                                        bool trailingSemicolon);

  LogicalResult emitVariableDeclaration(Location loc, StringRef name, Type type,
                                        bool trailingSemicolon);

  /// Emits the variable declaration and assignment prefix for 'op'.
  /// - emits separate variable followed by std::tie for multi-valued operation;
  /// - emits single type followed by variable for single result;
  /// - emits nothing if no value produced by op;
  /// Emits final '=' operator where a type is produced. Returns failure if
  /// any result type could not be converted.
  LogicalResult emitAssignPrefix(Operation &op);

  /// Emits a label for the block.
  LogicalResult emitLabel(Block &block);

  /// Emits the operands and atttributes of the operation. All operands are
  /// emitted first and then all attributes in alphabetical order.
  LogicalResult emitOperandsAndAttributes(Operation &op,
                                          ArrayRef<StringRef> exclude = {});

  LogicalResult emitMemrefAccess(Location loc, Value base, ValueRange indices,
                                 const std::string &offset = "0");

  LogicalResult emitInitializeOps();
  void pushInitializeOps(Operation *op);
  bool inFunction;

  /// Emits the operands of the operation. All operands are emitted in order.
  LogicalResult emitOperands(Operation &op);

  /// Return the existing or a new name for a Value.
  StringRef getOrCreateName(Value val);

  /// Return the existing or a new label of a Block.
  StringRef getOrCreateName(Block &block);

  bool shouldMapToUnsigned(Type val);

  /// Whether to map an mlir integer to a unsigned integer in C++.
  bool shouldMapToUnsigned(IntegerType::SignednessSemantics val);

  /// RAII helper function to manage entering/exiting C++ scopes.
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

  /// Returns wether the Value is assigned to a C++ variable in the scope.
  bool hasValueInScope(Value val);

  // Returns whether a label is assigned to the block.
  bool hasBlockLabel(Block &block);

  /// Returns the output stream.
  raw_indented_ostream &ostream() { return os; };

  /// Returns if all variables for op results and basic block arguments need to
  /// be declared at the beginning of a function.
  bool shouldDeclareVariablesAtTop() { return declareVariablesAtTop; };

private:
  using ValueMapper = llvm::ScopedHashTable<Value, std::string>;
  using BlockMapper = llvm::ScopedHashTable<Block *, std::string>;

  /// Output stream to emit to.
  raw_indented_ostream os;
  std::queue<Operation *> initializeOps;

  /// Boolean to enforce that all variables for op results and block
  /// arguments are declared at the beginning of the function. This also
  /// includes results from ops located in nested regions.
  bool declareVariablesAtTop;

  /// Map from value to name of C++ variable that contain the name.
  ValueMapper valueMapper;

  /// Map from block to name of C++ label.
  BlockMapper blockMapper;

  /// The number of values in the current scope. This is used to declare the
  /// names of values in a scope.
  std::stack<int64_t> valueInScopeCount;
  std::stack<int64_t> labelInScopeCount;
};
} // namespace

static bool isMemrefStruct(LLVM::LLVMStructType sType) {
  if (sType.getBody().size() != 5 && sType.getBody().size() != 3) {
    return false;
  }
  if (!dyn_cast<LLVM::LLVMPointerType>(sType.getBody()[0])) {
    return false;
  }
  if (!dyn_cast<LLVM::LLVMPointerType>(sType.getBody()[1])) {
    return false;
  }
  if (!dyn_cast<IntegerType>(sType.getBody()[2])) {
    return false;
  }
  if (sType.getBody().size() > 3) {
    if (!dyn_cast<LLVM::LLVMArrayType>(sType.getBody()[3])) {
      return false;
    }
    if (!dyn_cast<LLVM::LLVMArrayType>(sType.getBody()[4])) {
      return false;
    }
  }
  return true;
}

static void printInt(CudaEmitter &emitter, const APInt &val, Type iType) {
  auto isUnsigned = emitter.shouldMapToUnsigned(iType);

  auto &os = emitter.ostream();
  if (val.getBitWidth() == 1) {
    if (val.getBoolValue())
      os << "true";
    else
      os << "false";
  } else {
    SmallString<8> postfix;
    if (auto idxType = iType.dyn_cast<IndexType>()) {
      postfix = "lu";
    } else if (auto itType = iType.dyn_cast<IntegerType>()) {
      if (itType.getWidth() == 64) {
        postfix = "ll";
      }
      if (isUnsigned) {
        postfix += "u";
      }
    }
    SmallString<128> strValue;
    val.toString(strValue, 10, !isUnsigned, false);
    os << strValue << postfix;
  }
};
static void printTupleAccess(raw_ostream &os, StringRef name,
                             const std::vector<int64_t> &pos) {
  for (auto i = pos.rbegin(); i != pos.rend(); i++) {
    os << "Get<" << *i << ">(";
  }
  os << name;
  for ([[maybe_unused]] auto i : pos) {
    os << ")";
  }
}
template <typename PrintFunc>
static LogicalResult printMakeTuple(CudaEmitter &emitter,
                                    llvm::ArrayRef<int64_t> shape,
                                    PrintFunc pf) {
  auto &os = emitter.ostream();
  std::vector<int64_t> pos(shape.size());
  auto emitAtDim = [&](auto self, int64_t dim) -> LogicalResult {
    os << "my_make_tuple(";
    for (int64_t i = 0; i < shape[dim]; i++) {
      pos[dim] = i;
      if (dim == shape.size() - 1) {
        if (failed(pf(emitter, pos))) {
          return failure();
        }
      } else {
        if (failed(self(self, dim + 1))) {
          return failure();
        }
      }
      if (i != shape[dim] - 1) {
        os << ",";
      }
    }
    os << ")";
    return success();
  };
  return emitAtDim(emitAtDim, 0);
}
template <typename Iterator, typename T, typename ValueFunc, typename PrintFunc>
static LogicalResult printMakeTuple(CudaEmitter &emitter,
                                    llvm::ArrayRef<int64_t> shape,
                                    Iterator begin, Iterator end, ValueFunc f,
                                    const T &init, PrintFunc pf) {
  auto v = init;
  return printMakeTuple(
      emitter, shape,
      [&v, &begin, &end, &f, &pf](auto &emitter, const std::vector<int64_t> &) {
        if (begin != end) {
          v = f(*begin);
          begin++;
        }
        return pf(emitter, v);
      });
}
static LogicalResult printConstantOp(CudaEmitter &emitter, Operation *operation,
                                     Attribute value) {
  OpResult result = operation->getResult(0);

  if (emitter.shouldDeclareVariablesAtTop()) {
    if (failed(emitter.emitVariableAssignment(result)))
      return failure();
    return emitter.emitAttribute(operation->getLoc(), value);
  }

  if (auto vType = dyn_cast<VectorType>(operation->getResult(0).getType())) {
    if (failed(emitter.emitAssignPrefix(*operation)))
      return failure();
    auto shape = vType.getShape();

    if (auto dense = dyn_cast<DenseFPElementsAttr>(value)) {
      auto fType = vType.getElementType();
      std::string typeNoter = "";
      if (fType.dyn_cast<Float32Type>()) {
        typeNoter = "(float)";
      } else if (fType.dyn_cast<Float64Type>()) {
        typeNoter = "(double)";
      } else if (fType.dyn_cast<Float16Type>()) {
        typeNoter = "(half)";
      } else if (fType.dyn_cast<BFloat16Type>()) {
        typeNoter = "(bfloat16)";
      } else {
        operation->emitOpError("Unsupported float type");
        return failure();
      }
      return printMakeTuple(
          emitter, shape, dense.begin(), dense.end(),
          [](const APFloat &i) { return i.convertToDouble(); }, 0.0,
          [&typeNoter](auto &emitter, const double &v) {
            if (std::isnan(v)) {
              emitter.ostream() << typeNoter << "NAN";
            } else if (std::isinf(v)) {
              emitter.ostream()
                  << typeNoter << ((v > 0) ? "INFINITY" : "-INFINITY");
            } else {
              emitter.ostream() << typeNoter << v;
            }
            return success();
          });
    } else if (auto dense = dyn_cast<DenseIntElementsAttr>(value)) {
      auto iType = dense.getElementType();
      return printMakeTuple(
          emitter, shape, dense.begin(), dense.end(),
          [](const APInt &i) { return i; }, APInt::getZero(64),
          [iType](auto &emitter, const APInt &v) {
            printInt(emitter, v, iType);
            return success();
          });
      return success();
    } else {
      return operation->emitOpError("Unknown Attr to init vector type");
    }
  }

  // Emit a variable declaration.
  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  return emitter.emitAttribute(operation->getLoc(), value);
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

static LogicalResult printOperation(CudaEmitter &emitter,
                                    LLVM::ReturnOp returnOp) {
  raw_ostream &os = emitter.ostream();
  os << "return";
  return success();
}

static LogicalResult printOperation(CudaEmitter &emitter, ModuleOp moduleOp) {
  CudaEmitter::Scope scope(emitter);
  emitter.ostream() << "#include \"codegen_header.cu\"\n";
  emitter.ostream() << "#include \"cmath\"\n";
  for (Operation &op : moduleOp) {
    if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/false)))
      return failure();
  }
  return success();
}

static LogicalResult printOperation(CudaEmitter &emitter,
                                    LLVM::LLVMFuncOp functionOp) {
  // We need to declare variables at top if the function has multiple blocks.
  if (!emitter.shouldDeclareVariablesAtTop() &&
      functionOp.getBlocks().size() > 1) {
    return functionOp.emitOpError(
        "with multiple blocks needs variables declared at top");
  }

  CudaEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();
  os << "extern\"C\" __global__ ";
  if (failed(emitter.emitTypes(functionOp.getLoc(),
                               functionOp.getFunctionType().getReturnTypes())))
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
  os.indent();
  if (failed(emitter.emitInitializeOps())) {
    return failure();
  }

  if (emitter.shouldDeclareVariablesAtTop()) {
    // Declare all variables that hold op results including those from nested
    // regions.
    WalkResult result =
        functionOp.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
          for (OpResult result : op->getResults()) {
            if (failed(emitter.emitVariableDeclaration(
                    result, /*trailingSemicolon=*/true))) {
              return WalkResult(
                  op->emitError("unable to declare result variable for op"));
            }
          }
          return WalkResult::advance();
        });
    if (result.wasInterrupted())
      return failure();
  }

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
      bool trailingSemicolon = !isa<scf::ForOp>(op);

      if (failed(emitter.emitOperation(
              op, /*trailingSemicolon=*/trailingSemicolon)))
        return failure();
    }
  }
  os.unindent() << "}\n";
  emitter.inFunction = false;
  return success();
}

CudaEmitter::CudaEmitter(raw_ostream &os, bool declareVariablesAtTop)
    : inFunction(false), os(os), declareVariablesAtTop(declareVariablesAtTop) {
  valueInScopeCount.push(0);
  labelInScopeCount.push(0);
}

/// Return the existing or a new name for a Value.
StringRef CudaEmitter::getOrCreateName(Value val) {
  // if (auto literal =
  // dyn_cast_if_present<emitc::LiteralOp>(val.getDefiningOp()))
  //   return literal.getValue();
  if (!valueMapper.count(val))
    valueMapper.insert(val, formatv("v{0}", ++valueInScopeCount.top()));
  return *valueMapper.begin(val);
}

/// Return the existing or a new label for a Block.
StringRef CudaEmitter::getOrCreateName(Block &block) {
  if (!blockMapper.count(&block))
    blockMapper.insert(&block, formatv("label{0}", ++labelInScopeCount.top()));
  return *blockMapper.begin(&block);
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

bool CudaEmitter::shouldMapToUnsigned(Type val) {
  if (auto iType = val.dyn_cast<IntegerType>()) {
    return shouldMapToUnsigned(iType.getSignedness());
  }
  if (auto iType = val.dyn_cast<IndexType>()) {
    return true;
  }
  if (auto vType = val.dyn_cast<VectorType>()) {
    return shouldMapToUnsigned(vType.getElementType());
  }
  return false;
}

bool CudaEmitter::hasValueInScope(Value val) { return valueMapper.count(val); }

bool CudaEmitter::hasBlockLabel(Block &block) {
  return blockMapper.count(&block);
}
void CudaEmitter::pushInitializeOps(Operation *op) { initializeOps.push(op); }

LogicalResult CudaEmitter::emitInitializeOps() {
  inFunction = true;
  while (!initializeOps.empty()) {
    if (failed(emitOperation(*initializeOps.front(), true))) {
      return failure();
    }
    initializeOps.pop();
  }
  return success();
}

LogicalResult CudaEmitter::emitAttribute(Location loc, Attribute attr) {

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
      printInt(*this, iAttr.getValue(), iType);
      return success();
    }
    if (auto iType = dyn_cast<IndexType>(iAttr.getType())) {
      printInt(*this, iAttr.getValue(), iType);
      return success();
    }
  }
  if (auto dense = dyn_cast<DenseIntElementsAttr>(attr)) {
    if (auto iType = dyn_cast<IntegerType>(
            cast<TensorType>(dense.getType()).getElementType())) {
      os << '{';
      interleaveComma(dense, os,
                      [&](const APInt &val) { printInt(*this, val, iType); });
      os << '}';
      return success();
    }
    if (auto iType = dyn_cast<IndexType>(
            cast<TensorType>(dense.getType()).getElementType())) {
      os << '{';
      interleaveComma(dense, os,
                      [&](const APInt &val) { printInt(*this, val, iType); });
      os << '}';
      return success();
    }
  }

  // Print opaque attributes.
  // if (auto oAttr = dyn_cast<emitc::OpaqueAttr>(attr)) {
  //   os << oAttr.getValue();
  //   return success();
  // }

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

LogicalResult CudaEmitter::emitOperands(Operation &op) {
  auto emitOperandName = [&](Value result) -> LogicalResult {
    // auto literalDef = dyn_cast_if_present<LiteralOp>(result.getDefiningOp());
    // if (!literalDef && !hasValueInScope(result))
    //   return op.emitOpError() << "operand value not in scope";
    os << getOrCreateName(result);
    return success();
  };
  return interleaveCommaWithError(op.getOperands(), os, emitOperandName);
}

LogicalResult
CudaEmitter::emitOperandsAndAttributes(Operation &op,
                                       ArrayRef<StringRef> exclude) {
  if (failed(emitOperands(op)))
    return failure();
  // Insert comma in between operands and non-filtered attributes if needed.
  if (op.getNumOperands() > 0) {
    for (NamedAttribute attr : op.getAttrs()) {
      if (!llvm::is_contained(exclude, attr.getName().strref())) {
        os << ", ";
        break;
      }
    }
  }
  // Emit attributes.
  auto emitNamedAttribute = [&](NamedAttribute attr) -> LogicalResult {
    if (llvm::is_contained(exclude, attr.getName().strref()))
      return success();
    os << "/* " << attr.getName().getValue() << " */";
    if (failed(emitAttribute(op.getLoc(), attr.getValue())))
      return failure();
    return success();
  };
  return interleaveCommaWithError(op.getAttrs(), os, emitNamedAttribute);
}

LogicalResult CudaEmitter::emitVariableAssignment(OpResult result) {
  if (!hasValueInScope(result)) {
    return result.getDefiningOp()->emitOpError(
        "result variable for the operation has not been declared");
  }
  os << getOrCreateName(result) << " = ";
  return success();
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

LogicalResult CudaEmitter::emitVariableDeclaration(Location loc, StringRef name,
                                                   Type type,
                                                   bool trailingSemicolon) {
  if (auto aType = dyn_cast<LLVM::LLVMArrayType>(type)) {
    if (failed(emitType(loc, aType.getElementType()))) {
      return failure();
    }
    os << " " << name << "[";
    os << aType.getNumElements() << "]";
  } else {
    if (failed(emitType(loc, type)))
      return failure();
    os << " " << name;
  }
  if (trailingSemicolon)
    os << ";\n";
  return success();
}

LogicalResult CudaEmitter::emitAssignPrefix(Operation &op) {
  switch (op.getNumResults()) {
  case 0:
    break;
  case 1: {
    OpResult result = op.getResult(0);
    if (shouldDeclareVariablesAtTop()) {
      if (failed(emitVariableAssignment(result)))
        return failure();
    } else {
      if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/false)))
        return failure();
      os << " = ";
    }
    break;
  }
  default:
    if (!shouldDeclareVariablesAtTop()) {
      for (OpResult result : op.getResults()) {
        if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/true)))
          return failure();
      }
    }
    os << "std::tie(";
    interleaveComma(op.getResults(), os,
                    [&](Value result) { os << getOrCreateName(result); });
    os << ") = ";
  }
  return success();
}

LogicalResult CudaEmitter::emitLabel(Block &block) {
  if (!hasBlockLabel(block))
    return block.getParentOp()->emitError("label for block not found");
  // FIXME: Add feature in `raw_indented_ostream` to ignore indent for block
  // label instead of using `getOStream`.
  os.getOStream() << getOrCreateName(block) << ":\n";
  return success();
}
LogicalResult CudaEmitter::emitMemrefAccess(Location loc, Value base,
                                            ValueRange indices,
                                            const std::string &offset) {
  auto type = base.getType().dyn_cast<MemRefType>();
  if (!type) {
    return emitError(loc, "call emitMemrefAccess with non-memref type value");
  }
  if (!type.hasRank()) {
    return emitError(loc, "can't access an unranked memref");
  }
  auto numDims = indices.size();
  if (numDims != type.getShape().size()) {
    return emitError(loc, "memref numDims not correspond to indices");
  }
  std::string mRefName = getOrCreateName(base).str();

  if (numDims == 0) {
    os << "((";
    if (failed(emitType(loc, type.getElementType()))) {
      return failure();
    }
    os << "*)" << mRefName << ".aligned)[" << offset << "]";
    return success();
  }

  std::vector<std::string> strideStrings(numDims);
  std::string offsetString;

  {
    auto itosMD = [](int64_t value, const std::string &dynamicName) {
      if (value != ShapedType::kDynamic) {
        return llvm::itostr(value);
      } else {
        return dynamicName;
      }
    };
    int64_t moffset = 0;
    llvm::SmallVector<int64_t> mstrides(numDims);
    if (succeeded(getStridesAndOffset(type, mstrides, moffset))) {
      offsetString = formatv("({0} + {1})", offset,
                             itosMD(moffset, formatv("{0}.offset", mRefName)));
      for (int64_t i = numDims - 1; i >= 0; i--) {
        // TODO: why the strides are always 0 ? use dynamic approach instead now
        //  llvm::errs() << llvm::itostr(mstrides[i]) + "\n";
        strideStrings[i] = formatv("{0}.strides[{1}]", mRefName, i);
        // itosMD(mstrides[i], formatv("{0}.strides[{1}]", mRefName, i));
      }
    } else {
      auto shape = type.getShape();
      std::string prefix = "(";
      int64_t staticDimSize = 1;
      for (int64_t i = numDims - 1; i >= 0; i--) {
        strideStrings[i] = prefix + llvm::itostr(staticDimSize) + ")";
        if (type.isDynamicDim(i)) {
          prefix += formatv("{0}.sizes[{1}]", mRefName, i).str() + " * ";
        } else {
          staticDimSize *= shape[i];
        }
      }
    }
  }

  os << "((";
  if (failed(emitType(loc, type.getElementType()))) {
    return failure();
  }
  os << "*)" << mRefName << ".aligned)[";
  for (int64_t i = numDims - 1; i >= 0; i--) {
    os << getOrCreateName(indices[i]);
    os << " * " << strideStrings[i];
    os << " + ";
  }
  os << offsetString;
  os << "]";
  return success();
}
LogicalResult CudaEmitter::emitType(Location loc, Type type) {
  if (auto vType = dyn_cast<LLVM::LLVMVoidType>(type)) {
    return (os << "void"), success();
  }
  if (auto sType = dyn_cast<LLVM::LLVMStructType>(type)) {
    return emitStruct(loc, sType);
  }
  if (auto pType = dyn_cast<LLVM::LLVMPointerType>(type)) {
    return (os << "void*"), success();
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
  if (auto vType = dyn_cast<VectorType>(type)) {
    auto shape = vType.getShape();
    auto eleType = vType.getElementType();
    auto *emitter = this;
    auto emitAtDim = [&shape, &loc, &eleType,
                      emitter](auto self, int64_t dim) -> LogicalResult {
      auto &os = emitter->ostream();
      os << "Tuple<";
      for (int64_t i = 0; i < shape[dim]; i++) {
        if (dim == shape.size() - 1) {
          if (failed(emitter->emitType(loc, eleType))) {
            return failure();
          }
        } else {
          if (failed(self(self, dim + 1))) {
            return failure();
          }
        }
        if (i != shape[dim] - 1) {
          os << ",";
        }
      }
      os << ">";
      return success();
    };
    if (failed(emitAtDim(emitAtDim, 0))) {
      return failure();
    }
    return success();
  }
  if (auto mType = dyn_cast<MemRefType>(type)) {
    if (!mType.hasRank()) {
      return emitError(loc, "can't emit unranked memref");
    }
    os << "MemRefDescriptor<";
    if (failed(emitType(loc, mType.getElementType()))) {
      return failure();
    }
    os << ", ";
    os << mType.getShape().size();
    os << ">";
    return success();
  }
  return emitError(loc, "cannot emit type ") << type;
}

LogicalResult CudaEmitter::emitStruct(Location loc,
                                      LLVM::LLVMStructType sType) {
  // if (isMemrefStruct(sType)) {
  //   os << "MemrefDescriptor<void,";
  //   auto bodySize = sType.getBody().size();
  //   if (bodySize == 3) {
  //     os << "0>";
  //   } else {
  //     os <<
  //     sType.getBody()[3].dyn_cast<LLVM::LLVMArrayType>().getNumElements(); os
  //     << ">";
  //   }
  // }
  os << "struct {"
     << "\n";
  int count = 0;
  for (auto type : sType.getBody()) {
    auto s = formatv("v{0}", count).str();
    if (failed(emitVariableDeclaration(loc, s, type, true))) {
      return failure();
    }
    count++;
  }
  os << "}";
  return success();
}

LogicalResult CudaEmitter::emitTypes(Location loc, ArrayRef<Type> types) {
  switch (types.size()) {
  case 0:
    os << "void";
    return success();
  case 1:
    return emitType(loc, types.front());
  default:
    return emitTupleType(loc, types);
  }
}

LogicalResult CudaEmitter::emitTupleType(Location loc, ArrayRef<Type> types) {
  os << "Tuple<";
  if (failed(interleaveCommaWithError(
          types, os, [&](Type type) { return emitType(loc, type); })))
    return failure();
  os << ">";
  return success();
}

static LogicalResult printOperation(CudaEmitter &emitter,
                                    arith::ConstantOp constantOp) {
  Operation *operation = constantOp.getOperation();
  Attribute value = constantOp.getValue();

  return printConstantOp(emitter, operation, value);
}
static LogicalResult printOperation(CudaEmitter &emitter, arith::MulIOp op) {
  Operation *operation = op.getOperation();
  return printBinaryOperation(emitter, operation, "*");
}
static LogicalResult printOperation(CudaEmitter &emitter, arith::AddIOp op) {
  Operation *operation = op.getOperation();
  return printBinaryOperation(emitter, operation, "+");
}
static LogicalResult printOperation(CudaEmitter &emitter, arith::CmpIOp op) {
  Operation *operation = op.getOperation();
  auto predicate = op.getPredicate();
  auto checkType = [&](Type t) -> LogicalResult {
    if (t.isIntOrIndex())
      return success();
    if (t.dyn_cast<VectorType>())
      return success();
    if (t.dyn_cast<TensorType>())
      return op->emitOpError("arith.cmpi of tensor type is not supported");
    return op->emitOpError("unknown arith.cmpi operand type");
  };
  Type lType = op.getLhs().getType();
  Type rType = op.getRhs().getType();
  if (failed(checkType(lType)))
    return failure();
  if (failed(checkType(rType)))
    return failure();

  bool opUnsigned = false;
  switch (predicate) {
  case arith::CmpIPredicate::eq:
    return printBinaryOperation(emitter, operation, "==");
  case arith::CmpIPredicate::ne:
    return printBinaryOperation(emitter, operation, "!=");
  case arith::CmpIPredicate::ult:
  case arith::CmpIPredicate::ule:
  case arith::CmpIPredicate::ugt:
  case arith::CmpIPredicate::uge:
    opUnsigned = true;
    break;
  default:;
  }

  bool lUnsigned = emitter.shouldMapToUnsigned(lType);
  bool rUnsigned = emitter.shouldMapToUnsigned(rType);
  raw_ostream &os = emitter.ostream();
  auto printSignedBinaryOperation =
      [&](const std::string &binaryOperator) -> LogicalResult {
    if (failed(emitter.emitAssignPrefix(*operation)))
      return failure();
    if (lUnsigned != opUnsigned) {
      if (opUnsigned) {
        os << "asUnSigned(" << emitter.getOrCreateName(op.getLhs()) << ")";
      } else {
        os << "asSigned(" << emitter.getOrCreateName(op.getLhs()) << ")";
      }
    } else {
      os << emitter.getOrCreateName(op.getLhs()) << ")";
    }
    os << " " << binaryOperator << " ";
    if (rUnsigned != opUnsigned) {
      if (opUnsigned) {
        os << "asUnSigned(" << emitter.getOrCreateName(op.getRhs()) << ")";
      } else {
        os << "asSigned(" << emitter.getOrCreateName(op.getRhs()) << ")";
      }
    } else {
      os << emitter.getOrCreateName(op.getRhs()) << ")";
    }
    return success();
  };

  switch (predicate) {
  case arith::CmpIPredicate::slt:
    return printSignedBinaryOperation("<");
  case arith::CmpIPredicate::sle:
    return printSignedBinaryOperation("<=");
  case arith::CmpIPredicate::sgt:
    return printSignedBinaryOperation(">");
  case arith::CmpIPredicate::sge:
    return printSignedBinaryOperation(">=");
  case arith::CmpIPredicate::ult:
    return printSignedBinaryOperation("<");
  case arith::CmpIPredicate::ule:
    return printSignedBinaryOperation("<=");
  case arith::CmpIPredicate::ugt:
    return printSignedBinaryOperation(">");
  case arith::CmpIPredicate::uge:
    return printSignedBinaryOperation(">=");
  default:
    return op->emitOpError("Unknown arith.cmpi signed/unsigned op");
  }
  llvm_unreachable("Unknown cmpi");
}
static LogicalResult printOperation(CudaEmitter &emitter, arith::CmpFOp op) {
  Operation *operation = op.getOperation();
  auto predicate = op.getPredicate();
  switch (predicate) {
  case arith::CmpFPredicate::OEQ:
    return printBinaryOperation(emitter, operation, "==");
  case arith::CmpFPredicate::ONE:
    return printBinaryOperation(emitter, operation, "!=");
  case arith::CmpFPredicate::OLT:
    return printBinaryOperation(emitter, operation, "<");
  case arith::CmpFPredicate::OLE:
    return printBinaryOperation(emitter, operation, "<=");
  case arith::CmpFPredicate::OGT:
    return printBinaryOperation(emitter, operation, ">");
  case arith::CmpFPredicate::OGE:
    return printBinaryOperation(emitter, operation, ">=");
  case arith::CmpFPredicate::ULT:
    return printBinaryOperation(emitter, operation, "<");
  case arith::CmpFPredicate::ULE:
    return printBinaryOperation(emitter, operation, "<=");
  case arith::CmpFPredicate::UGT:
    return printBinaryOperation(emitter, operation, ">");
  case arith::CmpFPredicate::UGE:
    return printBinaryOperation(emitter, operation, ">=");
  case arith::CmpFPredicate::UEQ:
    return printBinaryOperation(emitter, operation, "==");
  case arith::CmpFPredicate::UNE:
    return printBinaryOperation(emitter, operation, "!=");
  case arith::CmpFPredicate::ORD: {
    auto &os = emitter.ostream();
    if (failed(emitter.emitAssignPrefix(*operation))) {
      return failure();
    }
    os << "!isnan(" << emitter.getOrCreateName(op.getLhs()) << ")";
    os << "&&";
    os << "!isnan(" << emitter.getOrCreateName(op.getRhs()) << ")";
    return success();
  }
  case arith::CmpFPredicate::UNO: {
    auto &os = emitter.ostream();
    if (failed(emitter.emitAssignPrefix(*operation))) {
      return failure();
    }
    os << "isnan(" << emitter.getOrCreateName(op.getLhs()) << ")";
    os << "||";
    os << "isnan(" << emitter.getOrCreateName(op.getRhs()) << ")";
    return success();
  }
  case arith::CmpFPredicate::AlwaysFalse: {
    auto &os = emitter.ostream();
    if (failed(emitter.emitAssignPrefix(*operation))) {
      return failure();
    }
    os << "false";
    return success();
  }
  case arith::CmpFPredicate::AlwaysTrue: {
    auto &os = emitter.ostream();
    if (failed(emitter.emitAssignPrefix(*operation))) {
      return failure();
    }
    os << "true";
    return success();
  }
  }
  llvm_unreachable("Unknown cmpf");
}
static bool isBoolOperation(Operation &op) {
  IntegerType iType;
  if (auto lType = op.getOperand(0).getType().dyn_cast<VectorType>()) {
    iType = lType.getElementType().dyn_cast<IntegerType>();
  } else {
    iType = op.getOperand(0).getType().dyn_cast<IntegerType>();
  }
  if (!iType || (iType.getWidth() != 1))
    return false;
  if (auto lType = op.getOperand(1).getType().dyn_cast<VectorType>()) {
    iType = lType.getElementType().dyn_cast<IntegerType>();
  } else {
    iType = op.getOperand(1).getType().dyn_cast<IntegerType>();
  }
  if (!iType || (iType.getWidth() != 1))
    return false;
  return true;
}
static LogicalResult printOperation(CudaEmitter &emitter, arith::OrIOp op) {
  Operation *operation = op.getOperation();
  if (isBoolOperation(*operation))
    return printBinaryOperation(emitter, operation, "||");
  else
    return printBinaryOperation(emitter, operation, "|");
}
static LogicalResult printOperation(CudaEmitter &emitter, arith::XOrIOp op) {
  Operation *operation = op.getOperation();
  return printBinaryOperation(emitter, operation, "^");
}
static LogicalResult printOperation(CudaEmitter &emitter, arith::AndIOp op) {
  Operation *operation = op.getOperation();
  if (isBoolOperation(*operation))
    return printBinaryOperation(emitter, operation, "&&");
  else
    return printBinaryOperation(emitter, operation, "&");
}
static LogicalResult printOperation(CudaEmitter &emitter, arith::SelectOp op) {
  Operation *operation = op.getOperation();
  auto &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << "arithSelect(";
  os << emitter.getOrCreateName(operation->getOperand(0));
  os << " , ";
  os << emitter.getOrCreateName(operation->getOperand(1));
  os << " , ";
  os << emitter.getOrCreateName(operation->getOperand(2));
  os << ")";
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, arith::DivSIOp op) {
  Operation *operation = op.getOperation();
  return printBinaryOperation(emitter, operation, "/");
}
static LogicalResult printOperation(CudaEmitter &emitter, arith::SubIOp op) {
  Operation *operation = op.getOperation();
  return printBinaryOperation(emitter, operation, "-");
}
static LogicalResult printOperation(CudaEmitter &emitter, arith::AddFOp op) {
  Operation *operation = op.getOperation();
  return printBinaryOperation(emitter, operation, "+");
}
static LogicalResult printOperation(CudaEmitter &emitter, arith::SubFOp op) {
  Operation *operation = op.getOperation();
  return printBinaryOperation(emitter, operation, "-");
}
static LogicalResult printOperation(CudaEmitter &emitter, arith::MulFOp op) {
  Operation *operation = op.getOperation();
  return printBinaryOperation(emitter, operation, "*");
}
static LogicalResult printOperation(CudaEmitter &emitter, arith::DivFOp op) {
  Operation *operation = op.getOperation();
  return printBinaryOperation(emitter, operation, "/");
}
static LogicalResult printOperation(CudaEmitter &emitter, arith::MaxNumFOp op) {
  Operation *operation = op.getOperation();
  auto &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << emitter.getOrCreateName(op.getLhs());
  os << " > ";
  os << emitter.getOrCreateName(op.getRhs());
  os << " ? ";
  os << emitter.getOrCreateName(op.getLhs());
  os << " : ";
  os << emitter.getOrCreateName(op.getRhs());
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, arith::MinNumFOp op) {
  Operation *operation = op.getOperation();
  auto &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << emitter.getOrCreateName(op.getLhs());
  os << " < ";
  os << emitter.getOrCreateName(op.getRhs());
  os << " ? ";
  os << emitter.getOrCreateName(op.getLhs());
  os << " : ";
  os << emitter.getOrCreateName(op.getRhs());
  return success();
}
static LogicalResult printCastOperation(CudaEmitter &emitter,
                                        Operation &operation) {
  auto &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(operation)))
    return failure();
  if (failed(emitter.emitType(operation.getLoc(),
                              operation.getResult(0).getType())))
    return failure();
  os << "(" << emitter.getOrCreateName(operation.getOperand(0)) << ")";
  return success();
}
static LogicalResult printVecCastOperation(CudaEmitter &emitter,
                                           Operation &op) {
  auto &os = emitter.ostream();
  auto oType = op.getResult(0).getType();
  if (auto vType = oType.dyn_cast<VectorType>()) {
    oType = vType.getElementType();
  }
  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  os << "castTo<";
  if (failed(emitter.emitType(op.getLoc(), oType)))
    return failure();
  os << ">(" << emitter.getOrCreateName(op.getOperand(0)) << ")";
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter,
                                    arith::IndexCastUIOp op) {
  if (failed(printVecCastOperation(emitter, *op.getOperation()))) {
    return failure();
  }
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter,
                                    arith::IndexCastOp op) {
  if (failed(printVecCastOperation(emitter, *op.getOperation()))) {
    return failure();
  }
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, arith::BitcastOp op) {
  Operation *operation = op.getOperation();
  auto &os = emitter.ostream();
  auto outType = op.getOut().getType();
  auto inType = op.getIn().getType();
  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  if (outType == inType) {
    os << emitter.getOrCreateName(op.getIn());
    return success();
  }
  if (outType.isa<VectorType>()) {
    os << "bitcast<";
    if (failed(emitter.emitType(
            op->getLoc(), outType.dyn_cast<VectorType>().getElementType()))) {
      return failure();
    }
    os << ">";
    os << "(" << emitter.getOrCreateName(op.getIn()) << ")";
  } else {
    os << "*((";
    if (failed(emitter.emitType(op.getLoc(), outType)))
      return failure();
    os << "*)&";
    os << emitter.getOrCreateName(op.getIn()) << ")";
  }
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, arith::ShLIOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "<<");
}
static LogicalResult printOperation(CudaEmitter &emitter, arith::FPToSIOp op) {
  if (failed(printVecCastOperation(emitter, *op.getOperation()))) {
    return failure();
  }
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, arith::SIToFPOp op) {
  if (failed(printVecCastOperation(emitter, *op.getOperation()))) {
    return failure();
  }
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, arith::ShRUIOp op) {
  return printBinaryOperation(emitter, op.getOperation(), ">>");
}
static LogicalResult printOperation(CudaEmitter &emitter, math::FloorOp op) {
  Operation *operation = op.getOperation();
  auto &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << "floor(" << emitter.getOrCreateName(op.getOperand()) << ")";
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, math::FmaOp op) {
  Operation *operation = op.getOperation();
  auto &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << emitter.getOrCreateName(op.getA()) << " * ";
  os << emitter.getOrCreateName(op.getB()) << " + ";
  os << emitter.getOrCreateName(op.getC());
  return success();
}

static LogicalResult printOperation(CudaEmitter &emitter, math::RsqrtOp op) {
  Operation *operation = op.getOperation();
  auto &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << "rsqrt(" << emitter.getOrCreateName(op.getOperand()) << ")";
  return success();
}

static LogicalResult printOperation(CudaEmitter &emitter, math::AbsFOp op) {
  Operation *operation = op.getOperation();
  auto &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << "abs(" << emitter.getOrCreateName(op.getOperand()) << ")";
  return success();
}

static LogicalResult printOperation(CudaEmitter &emitter, gpu::ThreadIdOp op) {
  Operation *operation = op.getOperation();
  auto &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  switch (op.getDimension()) {
  case gpu::Dimension::x:
    os << "threadIdx.x";
    break;
  case gpu::Dimension::y:
    os << "threadIdx.y";
    break;
  case gpu::Dimension::z:
    os << "threadIdx.z";
    break;
  }
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, gpu::BarrierOp op) {
  auto &os = emitter.ostream();

  os << "__syncthreads()";
  return success();
}

static LogicalResult printMemrefDeclare(CudaEmitter &emitter, Location loc,
                                        OpResult value, StringRef allocated,
                                        StringRef aligned) {

  auto mType = value.getType().dyn_cast<MemRefType>();
  if (!mType) {
    return emitError(loc, "declare memref with non-memref type");
  }
  if (!mType.hasRank()) {
    return emitError(loc, "can't declare unranked memref");
  }
  if (!isStrided(mType)) {
    return emitError(loc, "directly delcare non-strided memref unimplemented");
  }
  auto [strides, offset] = getStridesAndOffset(mType);
  auto sizes = mType.getShape();
  auto &os = emitter.ostream();

  if (ShapedType::isDynamicShape(sizes) ||
      ShapedType::isDynamicShape(strides)) {
    return emitError(loc, "directly declare non-static memref unimplemented");
  }

  if (failed(emitter.emitType(value.getOwner()->getLoc(), value.getType())))
    return failure();
  os << " " << emitter.getOrCreateName(value);
  os << "{";
  os << allocated;
  os << ", " << aligned;
  os << ", " << offset;
  auto printShape = [&os](auto v) { return (os << v), success(); };
  {
    os << ", {";
    if (failed(interleaveCommaWithError(sizes, os, printShape))) {
      return failure();
    }
    os << "}";
  }
  {
    os << ", {";
    if (failed(interleaveCommaWithError(strides, os, printShape))) {
      return failure();
    }
    os << "}";
  }
  os << "}";
  return success();
}

static LogicalResult printOperation(CudaEmitter &emitter, memref::AllocOp op) {
  if (op->getNumOperands() != 0) {
    return emitError(op->getLoc(),
                     "memref.alloc must be static and have no operands!");
  }
  if (op->getNumResults() != 1) {
    return emitError(op->getLoc(), "memref.alloc must have only one result!");
  }
  auto &os = emitter.ostream();
  MemRefType mref = op.getResult().getType();
  if (!mref.hasStaticShape()) {
    return emitError(op->getLoc(), "memref.alloc must be static!");
  }
  if (iree_compiler::hasSharedMemoryAddressSpace(mref)) {
    os << "__shared__ ";
  } else {
    auto addrSpace =
        llvm::dyn_cast_if_present<gpu::AddressSpaceAttr>(mref.getMemorySpace());
    if (addrSpace && addrSpace.getValue() == gpu::AddressSpace::Global) {
      os << "__device__ static ";
    }
  }
  if (failed(emitter.emitType(op->getLoc(), mref.getElementType()))) {
    return failure();
  }
  StringRef resultName = emitter.getOrCreateName(op.getResult());
  std::string ptrName = resultName.str() + "_" + "ptr";
  os << " " << ptrName;
  auto shape = mref.getShape();
  auto shapeSize = std::accumulate(shape.begin(), shape.end(), 1ll,
                                   [](auto x, auto y) { return x * y; });
  os << "[" << shapeSize << "];\n";
  if (failed(printMemrefDeclare(emitter, op->getLoc(), op->getResult(0),
                                ptrName, ptrName))) {
    return failure();
  }
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, memref::GlobalOp op) {
  if (emitter.inFunction) {
    auto &os = emitter.ostream();
    auto mref = op.getType();
    if (!mref.hasStaticShape()) {
      return emitError(op->getLoc(), "memref.global must be static!");
    }
    if (op.getConstant()) {
      os << "__constant__ static ";
    }
    if (failed(emitter.emitType(op->getLoc(), mref.getElementType()))) {
      return failure();
    }
    os << " " << op.getSymName();
    auto shape = mref.getShape();
    auto shapeSize = std::accumulate(shape.begin(), shape.end(), 1ll,
                                     [](auto x, auto y) { return x * y; });
    os << "[" << shapeSize << "]";
    if (auto initer = op.getInitialValue()) {
      if (failed(emitter.emitAttribute(op->getLoc(), *initer))) {
        return failure();
      }
    }
  } else {
    emitter.pushInitializeOps(op.getOperation());
  }
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter,
                                    memref::GetGlobalOp op) {
  if (failed(printMemrefDeclare(emitter, op->getLoc(), op->getResult(0),
                                op.getName(), op.getName()))) {
    return failure();
  }
  return success();
}
static LogicalResult
printOperation(CudaEmitter &emitter,
               iree_compiler::IREE::HAL::InterfaceWorkgroupIDOp op) {
  Operation *operation = op.getOperation();
  auto &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  switch (op.getDimension().getZExtValue()) {
  case 0:
    os << "blockIdx.x";
    break;
  case 1:
    os << "blockIdx.y";
    break;
  case 2:
    os << "blockIdx.z";
    break;
  default:
    return failure();
  }
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter,
                                    LLVM::ConstantOp constantOp) {
  Operation *operation = constantOp.getOperation();
  Attribute value = constantOp.getValue();

  return printConstantOp(emitter, operation, value);
}
static LogicalResult printOperation(CudaEmitter &emitter, LLVM::UndefOp op) {
  if (failed(emitter.emitType(op->getLoc(), op->getOpResult(0).getType()))) {
    return failure();
  }
  auto &os = emitter.ostream();
  os << " " << emitter.getOrCreateName(op->getOpResult(0));
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, LLVM::MulOp op) {
  if (auto iType = op.getRes().getType()) {
    return printBinaryOperation(emitter, op.getOperation(), "*");
  }
  return op->emitError("llvm.mul of vector-like not supported");
}
static LogicalResult printOperation(CudaEmitter &emitter, LLVM::UDivOp op) {
  if (auto iType = op.getRes().getType()) {
    return printBinaryOperation(emitter, op.getOperation(), "/");
  }
  return op->emitError("llvm.udiv of vector-like not supported");
}
static LogicalResult printOperation(CudaEmitter &emitter,
                                    LLVM::InsertValueOp op) {
  auto &os = emitter.ostream();
  os << "auto " << emitter.getOrCreateName(op->getOpResult(0)) << " = ";
  os << emitter.getOrCreateName(op->getOperand(0)) << ";";
  os << emitter.getOrCreateName(op->getOpResult(0)) << ".v";
  auto position = op.getPosition();
  if (position.size() < 1) {
    return emitError(op->getLoc(), "Invalid Position");
  }
  os << position[0];
  if (position.size() > 1) {
    for (int i = 1; i < position.size(); i++) {
      os << "[" << position[i] << "]";
    }
  }
  os << " = " << emitter.getOrCreateName(op.getOperand(1));
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, vector::LoadOp op) {
  Operation *operation = op.getOperation();
  auto &os = emitter.ostream();
  auto vecType = op->getResult(0).getType().dyn_cast<VectorType>();
  if (!vecType || vecType.getShape().size() != 1)
    return failure();
  if (failed(emitter.emitAssignPrefix(*operation))) {
    return failure();
  }
  os << "my_make_tuple(";
  auto numEles = vecType.getShape()[0];
  for (long i = 0; i < numEles; i++) {
    if (failed(emitter.emitMemrefAccess(op->getLoc(), op.getBase(),
                                        op.getIndices(), std::to_string(i)))) {
      return failure();
    }
    if (i != numEles - 1)
      os << ",";
  }
  os << ")";
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, memref::LoadOp op) {
  Operation *operation = op.getOperation();
  if (failed(emitter.emitAssignPrefix(*operation))) {
    return failure();
  }
  if (failed(emitter.emitMemrefAccess(op.getLoc(), op.getMemRef(),
                                      op.getIndices()))) {
    return failure();
  }
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, memref::StoreOp op) {
  auto &os = emitter.ostream();
  if (failed(emitter.emitMemrefAccess(op.getLoc(), op.getMemRef(),
                                      op.getIndices()))) {
    return failure();
  }
  os << " = ";
  os << emitter.getOrCreateName(op.getValue());
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, vector::InsertOp op) {
  Operation *operation = op.getOperation();
  auto &os = emitter.ostream();
  if (failed(emitter.emitAssignPrefix(*operation))) {
    return failure();
  }
  os << emitter.getOrCreateName(op->getOperand(1)) << ";";
  if (op->getNumOperands() != 2) {
    return failure();
  }
  auto positions = op.getStaticPosition();
  for (auto pos = positions.rbegin(); pos != positions.rend(); pos++) {
    os << "Get<" << *pos << ">"
       << "(";
  }
  os << emitter.getOrCreateName(op->getResult(0));
  for (int i = 0; i < positions.size(); i++) {
    os << ")";
  }
  os << "=" << emitter.getOrCreateName(op->getOperand(0));
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter,
                                    vector::ExtractOp op) {
  Operation *operation = op.getOperation();
  auto &os = emitter.ostream();
  if (failed(emitter.emitAssignPrefix(*operation))) {
    return failure();
  }
  auto positions = op.getStaticPosition();
  for (auto pos = positions.rbegin(); pos != positions.rend(); pos++) {
    os << "Get<" << *pos << ">"
       << "(";
  }
  os << emitter.getOrCreateName(op->getOperand(0));
  for (int i = 0; i < positions.size(); i++) {
    os << ")";
  }
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, vector::SplatOp op) {
  Operation *operation = op.getOperation();
  auto vecType = op->getResult(0).getType().dyn_cast<VectorType>();
  if (failed(emitter.emitAssignPrefix(*operation))) {
    return failure();
  }
  auto shape = vecType.getShape();
  auto v = emitter.getOrCreateName(op->getOperand(0)).str();
  return printMakeTuple(emitter, shape,
                        [&v](auto &emitter, const std::vector<int64_t> &) {
                          emitter.ostream() << v;
                          return success();
                        });
}
static LogicalResult printOperation(CudaEmitter &emitter,
                                    vector::BroadcastOp op) {
  Operation *operation = op.getOperation();
  auto vecType = op.getResult().getType().dyn_cast<VectorType>();
  if (failed(emitter.emitAssignPrefix(*operation))) {
    return failure();
  }
  std::vector<int64_t> shapeArr;
  auto resultShape = vecType.getShape();
  auto dim = vecType.getShape().size() -
             op.getSource().getType().dyn_cast<VectorType>().getShape().size();
  for (auto i = 0lu; i < dim; i++) {
    shapeArr.push_back(resultShape[i]);
  }
  llvm::ArrayRef<int64_t> shape(shapeArr);
  auto v = emitter.getOrCreateName(op->getOperand(0)).str();
  return printMakeTuple(emitter, shape,
                        [&v](auto &emitter, const std::vector<int64_t> &) {
                          emitter.ostream() << v;
                          return success();
                        });
}
static LogicalResult printOperation(CudaEmitter &emitter, vector::GatherOp op) {
  Operation *operation = op.getOperation();
  auto vecType = op->getResult(0).getType().dyn_cast<VectorType>();
  auto shape = vecType.getShape();
  if (failed(emitter.emitAssignPrefix(*operation))) {
    return failure();
  }
  auto maskName = emitter.getOrCreateName(op.getMask());
  auto indexVecName = emitter.getOrCreateName(op.getIndexVec());
  auto passthruName = emitter.getOrCreateName(op.getPassThru());
  return printMakeTuple(
      emitter, shape, [&](auto &emitter, const std::vector<int64_t> &pos) {
        auto &os = emitter.ostream();
        printTupleAccess(emitter.ostream(), maskName, pos);
        os << "?";
        std::string temp;
        llvm::raw_string_ostream stemp(temp);
        printTupleAccess(stemp, indexVecName, pos);
        stemp.flush();
        if (failed(emitter.emitMemrefAccess(op->getLoc(), op.getBase(),
                                            op.getIndices(), stemp.str()))) {
          return failure();
        }
        os << ":";
        printTupleAccess(emitter.ostream(), passthruName, pos);
        return success();
      });
}
static LogicalResult printOperation(CudaEmitter &emitter,
                                    vector::ReductionOp op) {
  Operation *operation = op.getOperation();
  auto vecType = op.getVector().getType();
  auto shape = vecType.getShape();
  if (failed(emitter.emitAssignPrefix(*operation))) {
    return failure();
  }
  using mlir::vector::CombiningKind;
  std::string acc = emitter.getOrCreateName(op.getAcc()).str();
  std::function<void(std::string)> f = [&acc](std::string v) { acc = acc + v; };
  switch (op.getKind()) {
  case CombiningKind::ADD: {
    f = [&acc](std::string v) { acc = "(" + acc + "+" + v + ")"; };
    break;
  }
  case CombiningKind::AND: {
    f = [&acc](std::string v) { acc = "(" + acc + "&" + v + ")"; };
    break;
  }
  case CombiningKind::MUL: {
    f = [&acc](std::string v) { acc = "(" + acc + "*" + v + ")"; };
    break;
  }
  case CombiningKind::OR: {
    f = [&acc](std::string v) { acc = "(" + acc + "|" + v + ")"; };
    break;
  }
  case CombiningKind::XOR: {
    f = [&acc](std::string v) { acc = "(" + acc + "^" + v + ")"; };
    break;
  }
  case CombiningKind::MAXF: {
    f = [&acc](std::string v) { acc = "max(" + acc + "," + v + ")"; };
    break;
  }
  case CombiningKind::MAXIMUMF: {
    f = [&acc](std::string v) { acc = "maximum(" + acc + "," + v + ")"; };
    break;
  }
  case CombiningKind::MAXSI: {
    f = [&acc](std::string v) { acc = "max(" + acc + ",asSigned(" + v + "))"; };
    break;
  }
  case CombiningKind::MAXUI: {
    f = [&acc](std::string v) {
      acc = "max(" + acc + ",asUnsigned(" + v + "))";
    };
    break;
  }
  case CombiningKind::MINSI: {
    f = [&acc](std::string v) { acc = "min(" + acc + ",asSigned(" + v + "))"; };
    break;
  }
  case CombiningKind::MINUI: {
    f = [&acc](std::string v) {
      acc = "min(" + acc + ",asUnsigned(" + v + "))";
    };
    break;
  }
  case CombiningKind::MINIMUMF: {
    f = [&acc](std::string v) { acc = "minimum(" + acc + "," + v + ")"; };
    break;
  }
  case CombiningKind::MINF: {
    f = [&acc](std::string v) { acc = "min(" + acc + "," + v + ")"; };
    break;
  }
  }
  for (int64_t i = 0; i < shape[0]; i++) {
    f("Get<" + llvm::itostr(i) + ">(" +
      emitter.getOrCreateName(op.getVector()).str() + ")");
  }
  auto &os = emitter.ostream();
  os << acc;
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter,
                                    vector::ShuffleOp op) {
  Operation *operation = op.getOperation();
  auto vecType = op->getResult(0).getType().dyn_cast<VectorType>();
  if (vecType.getShape().empty()) {
    return op.emitOpError("ShuffleOp result can't be empty vector");
  }
  std::array<int64_t, 1> shapeArr{vecType.getShape()[0]};
  llvm::ArrayRef<int64_t> shape(shapeArr);
  if (failed(emitter.emitAssignPrefix(*operation))) {
    return failure();
  }
  auto v1Name = emitter.getOrCreateName(op.getV1());
  auto v2Name = emitter.getOrCreateName(op.getV2());
  auto mask = op.getMask();
  auto v1Size = op.getV1().getType().getShape()[0];
  return printMakeTuple(
      emitter, shape, [&](auto &emitter, const std::vector<int64_t> &pos) {
        auto &os = emitter.ostream();
        auto i = mask[pos[0]].dyn_cast<IntegerAttr>().getInt();
        if (i < v1Size) {
          printTupleAccess(os, v1Name, std::vector<int64_t>{i});
        } else {
          printTupleAccess(os, v2Name, std::vector<int64_t>{i - v1Size});
        }
        return success();
      });
}

static LogicalResult printOperation(CudaEmitter &emitter, vector::FMAOp op) {
  Operation *operation = op.getOperation();
  auto &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << emitter.getOrCreateName(op->getOperand(0)) << " * ";
  os << emitter.getOrCreateName(op->getOperand(1)) << " + ";
  os << emitter.getOrCreateName(op->getOperand(2));
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, vector::StoreOp op) {
  auto &os = emitter.ostream();
  auto vecType = op.getValueToStore().getType().dyn_cast<VectorType>();
  if (!vecType || vecType.getShape().size() != 1)
    return failure();

  auto numEles = vecType.getShape()[0];
  for (long i = 0; i < numEles; i++) {
    if (failed(emitter.emitMemrefAccess(op->getLoc(), op.getBase(),
                                        op.getIndices(), std::to_string(i)))) {
      return failure();
    }
    os << " = ";
    os << "Get<" << i << ">(" << emitter.getOrCreateName(op.getValueToStore());
    os << ")";
    if (i != numEles - 1) {
      os << ";";
    }
  }
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, scf::ForOp op) {
  auto &os = emitter.ostream();
  {
    int count = 0;
    for (auto arg : op.getRegionIterArgs()) {
      if (failed(emitter.emitType(op->getLoc(), arg.getType()))) {
        return failure();
      }
      os << " " << emitter.getOrCreateName(arg) << " = ";
      os << emitter.getOrCreateName(op.getInitArgs()[count]) << ";\n";
      count++;
    }
  }
  {
    CudaEmitter::Scope scope(emitter);
    os << "for(auto " << emitter.getOrCreateName(op.getInductionVar()) << " = ";
    os << emitter.getOrCreateName(op.getLowerBound()) << ";";
    os << emitter.getOrCreateName(op.getInductionVar()) << " < ";
    os << emitter.getOrCreateName(op.getUpperBound()) << ";";
    os << emitter.getOrCreateName(op.getInductionVar()) << " += ";
    os << emitter.getOrCreateName(op.getStep()) << "){\n";

    os.indent();
    if (emitter.shouldDeclareVariablesAtTop()) {
      // Declare all variables that hold op results including those from nested
      // regions.
      WalkResult result =
          op.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
            for (OpResult result : op->getResults()) {
              if (failed(emitter.emitVariableDeclaration(
                      result, /*trailingSemicolon=*/true))) {
                return WalkResult(
                    op->emitError("unable to declare result variable for op"));
              }
            }
            return WalkResult::advance();
          });
      if (result.wasInterrupted())
        return failure();
    }
    Region &forRegion = op.getRegion();

    auto regionOps = forRegion.getOps();

    for (auto it = regionOps.begin(); std::next(it) != regionOps.end(); ++it) {
      bool trailingSemicolon = !isa<scf::ForOp>(*it);

      if (failed(
              emitter.emitOperation(*it,
                                    /*trailingSemicolon=*/trailingSemicolon)))
        return failure();
    }
    for (long i = 0; i < op.getYieldedValues().size(); i++) {
      os << emitter.getOrCreateName(op.getRegionIterArg(i)) << " = ";
      os << emitter.getOrCreateName(op.getYieldedValues()[i]);
      if (i != op.getNumOperands() - 1) {
        os << ";\n";
      }
    }

    os.unindent() << "}";
  }
  {
    int count = 0;
    for (auto result : op->getResults()) {
      os << "\n";
      if (failed(emitter.emitVariableDeclaration(result, false))) {
        return failure();
      }
      os << " = ";
      os << emitter.getOrCreateName(op.getRegionIterArg(count)) << ";";
      count++;
    }
  }
  return success();
}

static LogicalResult printOperation(CudaEmitter &emitter, scf::IfOp op) {
  auto &os = emitter.ostream();
  for (auto arg : op->getResults()) {
    if (failed(emitter.emitType(op->getLoc(), arg.getType()))) {
      return failure();
    }
    os << " " << emitter.getOrCreateName(arg) << ";\n";
  }
  os << "if (" << emitter.getOrCreateName(op.getCondition()) << "){\n";

  {
    CudaEmitter::Scope scope(emitter);
    os.indent();
    Region &thenRegion = op.getThenRegion();
    auto regionOps = thenRegion.getOps();

    for (auto it = regionOps.begin(); std::next(it) != regionOps.end(); ++it) {
      bool trailingSemicolon = !isa<scf::ForOp>(*it);

      if (failed(
              emitter.emitOperation(*it,
                                    /*trailingSemicolon=*/trailingSemicolon)))
        return failure();
    }
    for (long i = 0; i < op.thenYield()->getNumOperands(); i++) {
      os << emitter.getOrCreateName(op.getResult(i)) << " = ";
      os << emitter.getOrCreateName(op.thenYield()->getOperand(i));
      if (i != op.thenYield()->getNumOperands() - 1) {
        os << ";\n";
      }
    }
  }
  os.unindent() << "}\nelse{\n";
  {
    CudaEmitter::Scope scope(emitter);
    os.indent();
    Region &elseRegion = op.getElseRegion();
    auto regionOps = elseRegion.getOps();

    for (auto it = regionOps.begin(); std::next(it) != regionOps.end(); ++it) {
      bool trailingSemicolon = !isa<scf::ForOp>(*it);

      if (failed(
              emitter.emitOperation(*it,
                                    /*trailingSemicolon=*/trailingSemicolon)))
        return failure();
    }
    for (long i = 0; i < op.elseYield()->getNumOperands(); i++) {
      os << emitter.getOrCreateName(op.getResult(i)) << " = ";
      os << emitter.getOrCreateName(op.elseYield()->getOperand(i));
      if (i != op.elseYield()->getNumOperands() - 1) {
        os << ";\n";
      }
    }
  }
  os.unindent() << "}";
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter, scf::YieldOp op) {
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter,
                                    memref::AssumeAlignmentOp op) {
  return success();
}
static LogicalResult printOperation(CudaEmitter &emitter,
                                    UnrealizedConversionCastOp op) {
  auto &os = emitter.ostream();
  Operation *operation = op.getOperation();
  auto inputType = op.getOperand(0).getType();

  if (auto sType = inputType.dyn_cast<LLVM::LLVMStructType>()) {
    if (!isMemrefStruct((sType))) {
      return op->emitOpError("Unknown UnrealizedConversionCastOp, "
                             "input struct is not memref struct");
    }
    auto mType = op->getResult(0).getType().dyn_cast<MemRefType>();
    if (!mType) {
      return op->emitOpError(
          "Unknown UnrealizedConversionCastOp, output is not memref");
    }
    if (!mType.hasRank()) {
      return op->emitOpError("can't cast to unranked memref");
    }
    if (failed(emitter.emitVariableDeclaration(op->getResult(0), false))) {
      return failure();
    }
    auto sName = emitter.getOrCreateName(op->getOperand(0));
    os << "{";
    {
      os << "(";
      if (failed(emitter.emitType(op.getLoc(), mType.getElementType()))) {
        return failure();
      }
      os << "*)";
      os << sName << ".v0";
    }
    {
      os << ", ";
      os << "(";
      if (failed(emitter.emitType(op.getLoc(), mType.getElementType()))) {
        return failure();
      }
      os << "*)";
      os << sName << ".v1";
    }
    os << ", " << sName << ".v2";
    if (sType.getBody().size() == 5) {
      os << ", {";
      for (long i = 0; i < mType.getShape().size(); i++) {
        os << sName << ".v3[" << i << "]";
        if (i != mType.getShape().size() - 1)
          os << ", ";
      }
      os << "}, {";
      for (long i = 0; i < mType.getShape().size(); i++) {
        os << sName << ".v4[" << i << "]";
        if (i != mType.getShape().size() - 1)
          os << ", ";
      }
      os << "}";
    }
    os << "}";
    return success();
  }

  if (auto iType = inputType.dyn_cast<IndexType>()) {
    if (failed(emitter.emitAssignPrefix(*operation))) {
      return failure();
    }
    os << emitter.getOrCreateName(op->getOperand(0));
    return success();
  }
  return op->emitError("Unknown UnrealizedConversionCastOp, unknown input");
}
LogicalResult CudaEmitter::emitOperation(Operation &op,
                                         bool trailingSemicolon) {
  // llvm::errs() << op.getName().getStringRef().str() + "\n";
  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          // Builtin ops.
          .Case<ModuleOp>([&](auto op) { return printOperation(*this, op); })
          .Case<arith::ConstantOp, arith::MulIOp, arith::AddIOp, arith::CmpIOp,
                arith::SelectOp, arith::DivSIOp, arith::SubIOp, arith::AddFOp,
                arith::SubFOp, arith::MulFOp, arith::DivFOp, arith::MaxNumFOp,
                arith::MinNumFOp, arith::CmpFOp, arith::OrIOp, arith::XOrIOp,
                arith::AndIOp, arith::IndexCastUIOp, arith::BitcastOp,
                arith::IndexCastOp, arith::ShLIOp, arith::FPToSIOp,
                arith::SIToFPOp, arith::ShRUIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<math::FloorOp, math::FmaOp, math::RsqrtOp, math::AbsFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<gpu::ThreadIdOp, gpu::BarrierOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<memref::AllocOp, memref::LoadOp, memref::StoreOp,
                memref::GlobalOp, memref::GetGlobalOp,
                memref::AssumeAlignmentOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<iree_compiler::IREE::HAL::InterfaceWorkgroupIDOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<LLVM::ConstantOp, LLVM::UndefOp, LLVM::InsertValueOp,
                LLVM::LLVMFuncOp, LLVM::ReturnOp, LLVM::MulOp, LLVM::UDivOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<vector::LoadOp, vector::InsertOp, vector::ExtractOp,
                vector::SplatOp, vector::FMAOp, vector::StoreOp,
                vector::BroadcastOp, vector::GatherOp, vector::ShuffleOp,
                vector::ReductionOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<scf::ForOp, scf::YieldOp, scf::IfOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<UnrealizedConversionCastOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Default([&](Operation *) {
            std::string s = "unable to find printer for op";
            s += op.getName().getStringRef().str();
            return op.emitOpError(s);
          });

  if (failed(status))
    return failure();

  os << (trailingSemicolon ? ";\n" : "\n");
  llvm::errs() << "success " + op.getName().getStringRef().str() + "\n";
  return success();
}
/// Translates the given module op to CUDA kernel code.
LogicalResult translateToCuda(Operation *op, raw_ostream &os) {
  CudaEmitter emitter(os, false);
  return emitter.emitOperation(*op, false);
}
} // namespace emitcuda
} // namespace mlir