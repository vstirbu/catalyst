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
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
static void filterFuncAttributes(ArrayRef<NamedAttribute> attrs, bool filterArgAndResAttrs,
                                 SmallVectorImpl<NamedAttribute> &result)
{
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
static auto wrapAsStructAttrs(OpBuilder &b, ArrayAttr attrs)
{
    return DictionaryAttr::get(b.getContext(),
                               b.getNamedAttr(LLVM::LLVMDialect::getStructAttrsAttrName(), attrs));
}

/// Converts the function type to a C-compatible format, in particular using
/// pointers to memref descriptors for arguments.
static std::pair<Type, std::pair<size_t, size_t>>
convertFunctionTypeCWrapper(OpBuilder &rewriter, LLVMTypeConverter typeConverter, FunctionType type)
{
    SmallVector<Type, 4> inputs;

    bool noResults = type.getNumResults() == 0;
    Type resultType = noResults ? LLVM::LLVMPointerType::get(rewriter.getContext())
                                : typeConverter.packFunctionResults(type.getResults());

    bool noInputs = type.getNumInputs() == 0;
    Type inputType = noInputs ? LLVM::LLVMPointerType::get(rewriter.getContext())
                              : typeConverter.packFunctionResults(type.getInputs());

    if (auto structType = resultType.dyn_cast<LLVM::LLVMStructType>()) {
        inputs.push_back(LLVM::LLVMPointerType::get(structType));
    }
    else {
        inputs.push_back(inputType);
    }

    if (auto structType = inputType.dyn_cast<LLVM::LLVMStructType>()) {
        inputs.push_back(LLVM::LLVMPointerType::get(structType));
    }
    else {
        inputs.push_back(inputType);
    }

    resultType = LLVM::LLVMVoidType::get(rewriter.getContext());
    return {LLVM::LLVMFunctionType::get(resultType, inputs),
            {type.getNumResults(), type.getNumInputs()}};
}

static void wrapForExternalCallers(OpBuilder &rewriter, Location loc,
                                   LLVMTypeConverter &typeConverter, func::FuncOp funcOp,
                                   LLVM::LLVMFuncOp newFuncOp)
{
    auto type = funcOp.getFunctionType();
    auto [wrapperFuncType, result_inputs] =
        convertFunctionTypeCWrapper(rewriter, typeConverter, type);
    auto [hasResults, hasInputs] = result_inputs;

    auto wrapperFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        loc, llvm::formatv("_mlir_ciface_{0}", funcOp.getName()).str(), wrapperFuncType,
        LLVM::Linkage::External, /*dsoLocal*/ false,
        /*cconv*/ LLVM::CConv::C);

    SmallVector<Value, 8> args;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(wrapperFuncOp.addEntryBlock());

    if (hasInputs) {
        Value arg = wrapperFuncOp.getArgument(1);

        Value structOfMemrefs = rewriter.create<LLVM::LoadOp>(loc, arg);
        Type aType = structOfMemrefs.getType();
        for (auto &en : llvm::enumerate(type.getInputs())) {
            Value memref =
                hasInputs == 1
                    ? structOfMemrefs
                    : rewriter.create<LLVM::ExtractValueOp>(loc, structOfMemrefs, en.index());
            if (auto memrefType = en.value().dyn_cast<MemRefType>()) {
                MemRefDescriptor::unpack(rewriter, loc, memref, memrefType, args);
                continue;
            }
            if (en.value().isa<UnrankedMemRefType>()) {
                UnrankedMemRefDescriptor::unpack(rewriter, loc, memref, args);
                continue;
            }
        }
    }

    auto call = rewriter.create<LLVM::CallOp>(loc, newFuncOp, args);

    if (hasResults) {
        rewriter.create<LLVM::StoreOp>(loc, call.getResult(), wrapperFuncOp.getArgument(0));
        rewriter.create<LLVM::ReturnOp>(loc, ValueRange{});
    }
    else {
        rewriter.create<LLVM::ReturnOp>(loc, call.getResults());
    }
}

struct FuncOpConversionBase : public ConvertOpToLLVMPattern<func::FuncOp> {
  protected:
    using ConvertOpToLLVMPattern<func::FuncOp>::ConvertOpToLLVMPattern;

    // Convert input FuncOp to LLVMFuncOp by using the LLVMTypeConverter provided
    // to this legalization pattern.
    LLVM::LLVMFuncOp convertFuncOpToLLVMFuncOp(func::FuncOp funcOp,
                                               ConversionPatternRewriter &rewriter) const
    {
        // Convert the original function arguments. They are converted using the
        // LLVMTypeConverter provided to this legalization pattern.
        auto varargsAttr = funcOp->getAttrOfType<BoolAttr>("func.varargs");
        TypeConverter::SignatureConversion result(funcOp.getNumArguments());
        auto llvmType = getTypeConverter()->convertFunctionSignature(
            funcOp.getFunctionType(), varargsAttr && varargsAttr.getValue(), result);
        if (!llvmType)
            return nullptr;

        // Propagate argument/result attributes to all converted arguments/result
        // obtained after converting a given original argument/result.
        SmallVector<NamedAttribute, 4> attributes;
        filterFuncAttributes(funcOp->getAttrs(), /*filterArgAndResAttrs=*/true, attributes);
        if (ArrayAttr resAttrDicts = funcOp.getAllResultAttrs()) {
            assert(!resAttrDicts.empty() && "expected array to be non-empty");
            auto newResAttrDicts =
                (funcOp.getNumResults() == 1)
                    ? resAttrDicts
                    : rewriter.getArrayAttr({wrapAsStructAttrs(rewriter, resAttrDicts)});
            attributes.push_back(rewriter.getNamedAttr(FunctionOpInterface::getResultDictAttrName(),
                                                       newResAttrDicts));
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
                    if (attr.getName().getValue() == LLVM::LLVMDialect::getByValAttrName()) {
                        convertedAttrs.push_back(rewriter.getNamedAttr(
                            LLVM::LLVMDialect::getByValAttrName(), convert(attr)));
                    }
                    else if (attr.getName().getValue() == LLVM::LLVMDialect::getByRefAttrName()) {
                        convertedAttrs.push_back(rewriter.getNamedAttr(
                            LLVM::LLVMDialect::getByRefAttrName(), convert(attr)));
                    }
                    else if (attr.getName().getValue() ==
                             LLVM::LLVMDialect::getStructRetAttrName()) {
                        convertedAttrs.push_back(rewriter.getNamedAttr(
                            LLVM::LLVMDialect::getStructRetAttrName(), convert(attr)));
                    }
                    else if (attr.getName().getValue() ==
                             LLVM::LLVMDialect::getInAllocaAttrName()) {
                        convertedAttrs.push_back(rewriter.getNamedAttr(
                            LLVM::LLVMDialect::getInAllocaAttrName(), convert(attr)));
                    }
                    else {
                        convertedAttrs.push_back(attr);
                    }
                }
                auto mapping = result.getInputMapping(i);
                assert(mapping && "unexpected deletion of function argument");
                for (size_t j = 0; j < mapping->size; ++j)
                    newArgAttrs[mapping->inputNo + j] =
                        DictionaryAttr::get(rewriter.getContext(), convertedAttrs);
            }
            attributes.push_back(rewriter.getNamedAttr(FunctionOpInterface::getArgDictAttrName(),
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
            auto attr = funcOp->getAttr("llvm.linkage").dyn_cast<mlir::LLVM::LinkageAttr>();
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
        rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(), newFuncOp.end());
        if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter, &result)))
            return nullptr;

        return newFuncOp;
    }
};

/// FuncOp legalization pattern that converts MemRef arguments to pointers to
/// MemRef descriptors (LLVM struct data types) containing all the MemRef type
/// information.
struct FuncCatalystCompatibleWrapper : public FuncOpConversionBase {
    FuncCatalystCompatibleWrapper(LLVMTypeConverter &converter) : FuncOpConversionBase(converter) {}

    LogicalResult matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        auto newFuncOp = convertFuncOpToLLVMFuncOp(funcOp, rewriter);
        if (!newFuncOp)
            return failure();

        if (funcOp->getAttrOfType<UnitAttr>(StringRef("llvm.emit_catalyst_wrapper"))) {
            if (newFuncOp.isVarArg())
                return funcOp->emitError("C interface for variadic functions is not "
                                         "supported yet.");

            if (!newFuncOp.isExternal())
                wrapForExternalCallers(rewriter, funcOp.getLoc(), *getTypeConverter(), funcOp,
                                       newFuncOp);
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
        // patterns.add<FuncCatalystCompatibleWrapper>(typeConverter);
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
