// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/FormatVariadic.h"

#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/Transforms/Passes.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

/// Only retain those attributes that are not constructed by
/// `LLVMFuncOp::build`. If `filterArgAttrs` is set, also filter out argument
/// attributes.
static void filterFuncAttributes(ArrayRef<NamedAttribute> attrs,
                                 bool filterArgAndResAttrs,
                                 SmallVectorImpl<NamedAttribute> &result) {
  for (const auto &attr : attrs) {
    if (attr.getName() == SymbolTable::getSymbolAttrName() ||
        attr.getName() == FunctionOpInterface::getTypeAttrName() ||
        attr.getName() == "func.varargs" ||
        (filterArgAndResAttrs &&
         (attr.getName() == FunctionOpInterface::getArgDictAttrName() ||
          attr.getName() == FunctionOpInterface::getResultDictAttrName())))
      continue;
    result.push_back(attr);
  }
}

/// Helper function for wrapping all attributes into a single DictionaryAttr
static auto wrapAsStructAttrs(OpBuilder &b, ArrayAttr attrs) {
  return DictionaryAttr::get(
      b.getContext(),
      b.getNamedAttr(LLVM::LLVMDialect::getStructAttrsAttrName(), attrs));
}

/// Combines all result attributes into a single DictionaryAttr
/// and prepends to argument attrs.
/// This is intended to be used to format the attributes for a C wrapper
/// function when the result(s) is converted to the first function argument
/// (in the multiple return case, all returns get wrapped into a single
/// argument). The total number of argument attributes should be equal to
/// (number of function arguments) + 1.
static void
prependResAttrsToArgAttrs(OpBuilder &builder,
                          SmallVectorImpl<NamedAttribute> &attributes,
                          size_t numArguments) {
  auto allAttrs = SmallVector<Attribute>(
      numArguments + 1, DictionaryAttr::get(builder.getContext()));
  NamedAttribute *argAttrs = nullptr;
  for (auto *it = attributes.begin(); it != attributes.end();) {
    if (it->getName() == FunctionOpInterface::getArgDictAttrName()) {
      auto arrayAttrs = it->getValue().cast<ArrayAttr>();
      assert(arrayAttrs.size() == numArguments &&
             "Number of arg attrs and args should match");
      std::copy(arrayAttrs.begin(), arrayAttrs.end(), allAttrs.begin() + 1);
      argAttrs = it;
    } else if (it->getName() == FunctionOpInterface::getResultDictAttrName()) {
      auto arrayAttrs = it->getValue().cast<ArrayAttr>();
      assert(!arrayAttrs.empty() && "expected array to be non-empty");
      allAttrs[0] = (arrayAttrs.size() == 1)
                        ? arrayAttrs[0]
                        : wrapAsStructAttrs(builder, arrayAttrs);
      it = attributes.erase(it);
      continue;
    }
    it++;
  }

  auto newArgAttrs =
      builder.getNamedAttr(FunctionOpInterface::getArgDictAttrName(),
                           builder.getArrayAttr(allAttrs));
  if (!argAttrs) {
    attributes.emplace_back(newArgAttrs);
    return;
  }
  *argAttrs = newArgAttrs;
}

/// Converts the function type to a C-compatible format, in particular using
/// pointers to memref descriptors for arguments.
static std::pair<Type, std::pair<bool, bool>>
convertFunctionTypeCWrapper(OpBuilder &rewriter, LLVMTypeConverter typeConverter, FunctionType type) {
  SmallVector<Type, 4> inputs;

  bool noResults = type.getNumResults() == 0;
  Type resultType = noResults
                        ? LLVM::LLVMVoidType::get(rewriter.getContext())
                        : typeConverter.packFunctionResults(type.getResults());

  bool noInputs = type.getNumInputs() == 0;
  Type inputType = noInputs
                        ? LLVM::LLVMVoidType::get(rewriter.getContext())
                        : typeConverter.packFunctionResults(type.getInputs());

  if (auto structType = resultType.dyn_cast<LLVM::LLVMStructType>()) {
    // Struct types cannot be safely returned via C interface. Make this a
    // pointer argument, instead.
    inputs.push_back(LLVM::LLVMPointerType::get(structType));
    resultType = LLVM::LLVMVoidType::get(rewriter.getContext());
  }

  if (auto structType = inputType.dyn_cast<LLVM::LLVMStructType>()) {
    inputs.push_back(LLVM::LLVMPointerType::get(structType));
  }

  return {LLVM::LLVMFunctionType::get(resultType, inputs), {noResults, noInputs}};
}

/// Creates an auxiliary function with pointer-to-memref-descriptor-struct
/// arguments instead of unpacked arguments. This function can be called from C
/// by passing a pointer to a C struct corresponding to a memref descriptor.
/// Similarly, returned memrefs are passed via pointers to a C struct that is
/// passed as additional argument.
/// Internally, the auxiliary function unpacks the descriptor into individual
/// components and forwards them to `newFuncOp` and forwards the results to
/// the extra arguments.
static void wrapForExternalCallers(OpBuilder &rewriter, Location loc,
                                   LLVMTypeConverter &typeConverter,
                                   func::FuncOp funcOp,
                                   LLVM::LLVMFuncOp newFuncOp) {
  auto type = funcOp.getFunctionType();
  SmallVector<NamedAttribute, 4> attributes;
  filterFuncAttributes(funcOp->getAttrs(), /*filterArgAndResAttrs=*/false,
                       attributes);
  auto [wrapperFuncType, result_inputs] = convertFunctionTypeCWrapper(rewriter, typeConverter, type);
  funcOp->emitRemark() << wrapperFuncType;
  funcOp->emitRemark() << result_inputs.first << result_inputs.second;
/*
define void @_mlir_ciface_jit.f(ptr %0, ptr %1) {                  
  %3 = load { ptr, ptr, i64 }, ptr %1, align 8
  %4 = extractvalue { ptr, ptr, i64 } %3, 0
  %5 = extractvalue { ptr, ptr, i64 } %3, 1 
  %6 = extractvalue { ptr, ptr, i64 } %3, 2
  %7 = call { ptr, ptr, i64, [1 x i64], [1 x i64] } @jit.f(ptr %4, ptr %5, i64 %6)
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, ptr %0, align 8
  ret void                                  
}   

DEF F(ARG0, ARG1):
  STRUCT_OF_POINTERS_TO_MEMREFS = *ARG1
  FOR EACH FIELD IN STRUCT_OF_POINTER_TO_MEMREFS:
    MEMREF_PTR = EXTRACT_VALUE STRUCT_OF_POINTER_TO_MEMREFS FIELD
    MEMREF = *MEMREF_PTR
    FOR EACH FIELD IN MEMREF
       SCALAR = EXTRACT_VALUE MEMREF FIELD
  RESULT = CALL(SCALAR_1, SCALAR_2, ... SCALAR_n)
  *ARG0 = RESULT
*/
  return;
}

/// Creates an auxiliary function with pointer-to-memref-descriptor-struct
/// arguments instead of unpacked arguments. Creates a body for the (external)
/// `newFuncOp` that allocates a memref descriptor on stack, packs the
/// individual arguments into this descriptor and passes a pointer to it into
/// the auxiliary function. If the result of the function cannot be directly
/// returned, we write it to a special first argument that provides a pointer
/// to a corresponding struct. This auxiliary external function is now
/// compatible with functions defined in C using pointers to C structs
/// corresponding to a memref descriptor.
static void wrapExternalFunction(OpBuilder &builder, Location loc,
                                 LLVMTypeConverter &typeConverter,
                                 func::FuncOp funcOp,
                                 LLVM::LLVMFuncOp newFuncOp) {
  OpBuilder::InsertionGuard guard(builder);

  auto [wrapperType, resultIsNowArg] =
      typeConverter.convertFunctionTypeCWrapper(funcOp.getFunctionType());
  // This conversion can only fail if it could not convert one of the argument
  // types. But since it has been applied to a non-wrapper function before, it
  // should have failed earlier and not reach this point at all.
  assert(wrapperType && "unexpected type conversion failure");

  SmallVector<NamedAttribute, 4> attributes;
  filterFuncAttributes(funcOp->getAttrs(), /*filterArgAndResAttrs=*/false,
                       attributes);

  if (resultIsNowArg)
    prependResAttrsToArgAttrs(builder, attributes, funcOp.getNumArguments());
  // Create the auxiliary function.
  auto wrapperFunc = builder.create<LLVM::LLVMFuncOp>(
      loc, llvm::formatv("_mlir_ciface_{0}", funcOp.getName()).str(),
      wrapperType, LLVM::Linkage::External, /*dsoLocal*/ false,
      /*cconv*/ LLVM::CConv::C, attributes);

  builder.setInsertionPointToStart(newFuncOp.addEntryBlock());

  // Get a ValueRange containing arguments.
  FunctionType type = funcOp.getFunctionType();
  SmallVector<Value, 8> args;
  args.reserve(type.getNumInputs());
  ValueRange wrapperArgsRange(newFuncOp.getArguments());

  if (resultIsNowArg) {
    // Allocate the struct on the stack and pass the pointer.
    Type resultType =
        wrapperType.cast<LLVM::LLVMFunctionType>().getParamType(0);
    Value one = builder.create<LLVM::ConstantOp>(
        loc, typeConverter.convertType(builder.getIndexType()),
        builder.getIntegerAttr(builder.getIndexType(), 1));
    Value result = builder.create<LLVM::AllocaOp>(loc, resultType, one);
    args.push_back(result);
  }

  // Iterate over the inputs of the original function and pack values into
  // memref descriptors if the original type is a memref.
  for (auto &en : llvm::enumerate(type.getInputs())) {
    Value arg;
    int numToDrop = 1;
    auto memRefType = en.value().dyn_cast<MemRefType>();
    auto unrankedMemRefType = en.value().dyn_cast<UnrankedMemRefType>();
    if (memRefType || unrankedMemRefType) {
      numToDrop = memRefType
                      ? MemRefDescriptor::getNumUnpackedValues(memRefType)
                      : UnrankedMemRefDescriptor::getNumUnpackedValues();
      Value packed =
          memRefType
              ? MemRefDescriptor::pack(builder, loc, typeConverter, memRefType,
                                       wrapperArgsRange.take_front(numToDrop))
              : UnrankedMemRefDescriptor::pack(
                    builder, loc, typeConverter, unrankedMemRefType,
                    wrapperArgsRange.take_front(numToDrop));

      auto ptrTy = LLVM::LLVMPointerType::get(packed.getType());
      Value one = builder.create<LLVM::ConstantOp>(
          loc, typeConverter.convertType(builder.getIndexType()),
          builder.getIntegerAttr(builder.getIndexType(), 1));
      Value allocated =
          builder.create<LLVM::AllocaOp>(loc, ptrTy, one, /*alignment=*/0);
      builder.create<LLVM::StoreOp>(loc, packed, allocated);
      arg = allocated;
    } else {
      arg = wrapperArgsRange[0];
    }

    args.push_back(arg);
    wrapperArgsRange = wrapperArgsRange.drop_front(numToDrop);
  }
  assert(wrapperArgsRange.empty() && "did not map some of the arguments");

  auto call = builder.create<LLVM::CallOp>(loc, wrapperFunc, args);

  if (resultIsNowArg) {
    Value result = builder.create<LLVM::LoadOp>(loc, args.front());
    builder.create<LLVM::ReturnOp>(loc, result);
  } else {
    builder.create<LLVM::ReturnOp>(loc, call.getResults());
  }
}

struct FuncOpConversionBase : public ConvertOpToLLVMPattern<func::FuncOp> {
protected:
  using ConvertOpToLLVMPattern<func::FuncOp>::ConvertOpToLLVMPattern;

  // Convert input FuncOp to LLVMFuncOp by using the LLVMTypeConverter provided
  // to this legalization pattern.
  LLVM::LLVMFuncOp
  convertFuncOpToLLVMFuncOp(func::FuncOp funcOp,
                            ConversionPatternRewriter &rewriter) const {
    // Convert the original function arguments. They are converted using the
    // LLVMTypeConverter provided to this legalization pattern.
    auto varargsAttr = funcOp->getAttrOfType<BoolAttr>("func.varargs");
    TypeConverter::SignatureConversion result(funcOp.getNumArguments());
    auto llvmType = getTypeConverter()->convertFunctionSignature(
        funcOp.getFunctionType(), varargsAttr && varargsAttr.getValue(),
        result);
    if (!llvmType)
      return nullptr;

    // Propagate argument/result attributes to all converted arguments/result
    // obtained after converting a given original argument/result.
    SmallVector<NamedAttribute, 4> attributes;
    filterFuncAttributes(funcOp->getAttrs(), /*filterArgAndResAttrs=*/true,
                         attributes);
    if (ArrayAttr resAttrDicts = funcOp.getAllResultAttrs()) {
      assert(!resAttrDicts.empty() && "expected array to be non-empty");
      auto newResAttrDicts =
          (funcOp.getNumResults() == 1)
              ? resAttrDicts
              : rewriter.getArrayAttr(
                    {wrapAsStructAttrs(rewriter, resAttrDicts)});
      attributes.push_back(rewriter.getNamedAttr(
          FunctionOpInterface::getResultDictAttrName(), newResAttrDicts));
    }
    if (ArrayAttr argAttrDicts = funcOp.getAllArgAttrs()) {
      SmallVector<Attribute, 4> newArgAttrs(
          llvmType.cast<LLVM::LLVMFunctionType>().getNumParams());
      for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
        // Some LLVM IR attribute have a type attached to them. During FuncOp ->
        // LLVMFuncOp conversion these types may have changed. Account for that
        // change by converting attributes' types as well.
        SmallVector<NamedAttribute, 4> convertedAttrs;
        auto attrsDict = argAttrDicts[i].cast<DictionaryAttr>();
        convertedAttrs.reserve(attrsDict.size());
        for (const NamedAttribute &attr : attrsDict) {
          const auto convert = [&](const NamedAttribute &attr) {
            return TypeAttr::get(getTypeConverter()->convertType(
                attr.getValue().cast<TypeAttr>().getValue()));
          };
          if (attr.getName().getValue() ==
              LLVM::LLVMDialect::getByValAttrName()) {
            convertedAttrs.push_back(rewriter.getNamedAttr(
                LLVM::LLVMDialect::getByValAttrName(), convert(attr)));
          } else if (attr.getName().getValue() ==
                     LLVM::LLVMDialect::getByRefAttrName()) {
            convertedAttrs.push_back(rewriter.getNamedAttr(
                LLVM::LLVMDialect::getByRefAttrName(), convert(attr)));
          } else if (attr.getName().getValue() ==
                     LLVM::LLVMDialect::getStructRetAttrName()) {
            convertedAttrs.push_back(rewriter.getNamedAttr(
                LLVM::LLVMDialect::getStructRetAttrName(), convert(attr)));
          } else if (attr.getName().getValue() ==
                     LLVM::LLVMDialect::getInAllocaAttrName()) {
            convertedAttrs.push_back(rewriter.getNamedAttr(
                LLVM::LLVMDialect::getInAllocaAttrName(), convert(attr)));
          } else {
            convertedAttrs.push_back(attr);
          }
        }
        auto mapping = result.getInputMapping(i);
        assert(mapping && "unexpected deletion of function argument");
        for (size_t j = 0; j < mapping->size; ++j)
          newArgAttrs[mapping->inputNo + j] =
              DictionaryAttr::get(rewriter.getContext(), convertedAttrs);
      }
      attributes.push_back(
          rewriter.getNamedAttr(FunctionOpInterface::getArgDictAttrName(),
                                rewriter.getArrayAttr(newArgAttrs)));
    }
    for (const auto &pair : llvm::enumerate(attributes)) {
      if (pair.value().getName() == "llvm.linkage") {
        attributes.erase(attributes.begin() + pair.index());
        break;
      }
    }

    // Create an LLVM function, use external linkage by default until MLIR
    // functions have linkage.
    LLVM::Linkage linkage = LLVM::Linkage::External;
    if (funcOp->hasAttr("llvm.linkage")) {
      auto attr =
          funcOp->getAttr("llvm.linkage").dyn_cast<mlir::LLVM::LinkageAttr>();
      if (!attr) {
        funcOp->emitError()
            << "Contains llvm.linkage attribute not of type LLVM::LinkageAttr";
        return nullptr;
      }
      linkage = attr.getLinkage();
    }
    auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        funcOp.getLoc(), funcOp.getName(), llvmType, linkage,
        /*dsoLocal*/ false, /*cconv*/ LLVM::CConv::C, attributes);
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter,
                                           &result)))
      return nullptr;

    return newFuncOp;
  }
};

/// FuncOp legalization pattern that converts MemRef arguments to pointers to
/// MemRef descriptors (LLVM struct data types) containing all the MemRef type
/// information.
struct FuncOpConversion2 : public FuncOpConversionBase {
  FuncOpConversion2(LLVMTypeConverter &converter)
      : FuncOpConversionBase(converter) {}

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newFuncOp = convertFuncOpToLLVMFuncOp(funcOp, rewriter);
    if (!newFuncOp)
      return failure();

    if (funcOp->getAttrOfType<UnitAttr>(StringRef("llvm.emit_special_interface"))) {
      if (newFuncOp.isVarArg())
        return funcOp->emitError("C interface for variadic functions is not "
                                 "supported yet.");

      if (newFuncOp.isExternal())
        wrapExternalFunction(rewriter, funcOp.getLoc(), *getTypeConverter(),
                             funcOp, newFuncOp);
      else
      {
        wrapForExternalCallers(rewriter, funcOp.getLoc(), *getTypeConverter(),
                               funcOp, newFuncOp);
      }
    }

    rewriter.eraseOp(funcOp);
    return success();
  }
};

struct QIRTypeConverter : public LLVMTypeConverter {

    QIRTypeConverter(MLIRContext *ctx) : LLVMTypeConverter(ctx)
    {
        addConversion([&](QubitType type) { return convertQubitType(type); });
        addConversion([&](QuregType type) { return convertQuregType(type); });
        addConversion([&](ObservableType type) { return convertObservableType(type); });
        addConversion([&](ResultType type) { return convertResultType(type); });
    }

  private:
    Type convertQubitType(Type mlirType)
    {
        return LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getOpaque("Qubit", &getContext()));
    }

    Type convertQuregType(Type mlirType)
    {
        return LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getOpaque("Array", &getContext()));
    }

    Type convertObservableType(Type mlirType)
    {
        return this->convertType(IntegerType::get(&getContext(), 64));
    }

    Type convertResultType(Type mlirType)
    {
        return LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getOpaque("Result", &getContext()));
    }
};

struct QuantumConversionPass : public PassWrapper<QuantumConversionPass, OperationPass<ModuleOp>> {
    QuantumConversionPass() {}

    StringRef getArgument() const override { return "convert-quantum-to-llvm"; }

    StringRef getDescription() const override
    {
        return "Perform a dialect conversion from Quantum to LLVM (QIR).";
    }

    void getDependentDialects(DialectRegistry &registry) const override
    {
        registry.insert<LLVM::LLVMDialect>();
    }

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        QIRTypeConverter typeConverter(context);

        RewritePatternSet patterns(context);
        cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
        populateFuncToLLVMConversionPatterns(typeConverter, patterns);
	patterns.add<FuncOpConversion2>(typeConverter);
        populateQIRConversionPatterns(typeConverter, patterns);

        LLVMConversionTarget target(*context);
        target.addLegalOp<ModuleOp>();

        if (failed(applyFullConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace quantum

std::unique_ptr<Pass> createQuantumConversionPass()
{
    return std::make_unique<quantum::QuantumConversionPass>();
}

} // namespace catalyst
