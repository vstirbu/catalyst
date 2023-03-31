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

#include "ClassicalJacobian.hpp"

#include <deque>
#include <string>
#include <vector>

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Utils/RemoveQuantumMeasurements.h"

namespace catalyst {
namespace gradient {

/// Generate a new mlir function that counts the (runtime) number of gate parameters.
///
/// This enables other functions like `genArgMapFunction` to allocate memory for vectors of gate
/// parameters without having to deal with dynamic memory management. The function works similarly
/// to `genArgMapFunction` by eliminating all quantum code and running the classical preprocessing,
/// but instead of storing gate parameters it merely counts them.
/// The impact on execution time is expected to be non-dominant, as the classical pre-processing is
/// already run multiple times, such as to differentiate the ArgMap and on every execution of
/// quantum function for the parameter-shift method. However, if this is inefficient in certain
/// use-cases, other approaches can be employed.
///
func::FuncOp genParamCountFunction(PatternRewriter &rewriter, Location loc, func::FuncOp callee)
{
    // Define the properties of the gate parameter counting version of the function to be
    // differentiated.
    std::string fnName = callee.getSymName().str() + ".pcount";
    FunctionType fnType =
        rewriter.getFunctionType(callee.getArgumentTypes(), rewriter.getIndexType());
    StringAttr visibility = rewriter.getStringAttr("private");

    func::FuncOp paramCountFn =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(callee, rewriter.getStringAttr(fnName));
    if (!paramCountFn) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);

        // First copy the original function as is, then we can replace all quantum ops by counting
        // their gate parameters instead.
        rewriter.setInsertionPointAfter(callee);
        paramCountFn = rewriter.create<func::FuncOp>(loc, fnName, fnType, visibility);
        rewriter.cloneRegionBefore(callee.getBody(), paramCountFn.getBody(), paramCountFn.end());

        // Store the counter in memory since we don't want to deal with returning the SSA value
        // for updated parameter counts from arbitrary regions/ops.
        rewriter.setInsertionPointToStart(&paramCountFn.getBody().front());
        MemRefType paramCountType = MemRefType::get({}, rewriter.getIndexType());
        Value paramCountBuffer = rewriter.create<memref::AllocaOp>(loc, paramCountType);
        Value cZero = rewriter.create<index::ConstantOp>(loc, 0);
        rewriter.create<memref::StoreOp>(loc, cZero, paramCountBuffer);

        // For each quantum gate add the number of parameters to the counter.
        paramCountFn.walk([&](quantum::CustomOp gate) {
            PatternRewriter::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPoint(gate);

            if (!gate.getParams().empty()) {
                Value currCount = rewriter.create<memref::LoadOp>(loc, paramCountBuffer);
                Value numParams = rewriter.create<index::ConstantOp>(loc, gate.getParams().size());
                Value newCount = rewriter.create<index::AddOp>(loc, currCount, numParams);
                rewriter.create<memref::StoreOp>(loc, newCount, paramCountBuffer);
            }

            rewriter.replaceOp(gate, gate.getInQubits());
        });

        // Replace any return statements from the original function with the parameter count.
        paramCountFn.walk([&](func::ReturnOp returnOp) {
            PatternRewriter::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPoint(returnOp);

            Value paramCount = rewriter.create<memref::LoadOp>(loc, paramCountBuffer);
            returnOp->setOperands(paramCount);
        });

        quantum::removeQuantumMeasurements(paramCountFn);
    }

    return paramCountFn;
}

/// Generate a new mlir function that maps qfunc arguments to gate parameters.
///
/// This enables to extract any classical preprocessing done inside the quantum function and compute
/// its jacobian separately in order to combine it with quantum-only gradients such as the
/// parameter-shift or adjoint method.
///
func::FuncOp genArgMapFunction(PatternRewriter &rewriter, Location loc, func::FuncOp callee)
{
    // Define the properties of the classical preprocessing function.
    std::string fnName = callee.getSymName().str() + ".argmap";
    std::vector<Type> fnArgTypes = callee.getArgumentTypes().vec();
    fnArgTypes.push_back(rewriter.getIndexType());
    RankedTensorType paramsVectorType =
        RankedTensorType::get({ShapedType::kDynamic}, rewriter.getF64Type());
    FunctionType fnType = rewriter.getFunctionType(fnArgTypes, paramsVectorType);
    StringAttr visibility = rewriter.getStringAttr("private");

    func::FuncOp argMapFn =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(callee, rewriter.getStringAttr(fnName));
    if (!argMapFn) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);

        // First copy the original function as is, then we can replace all quantum ops by collecting
        // their gate parameters in a memory buffer instead. The size of this vector is passed as an
        // input to the new function.
        argMapFn = rewriter.create<func::FuncOp>(loc, fnName, fnType, visibility);
        rewriter.cloneRegionBefore(callee.getBody(), argMapFn.getBody(), argMapFn.end());
        Value numParams = argMapFn.getBody().front().addArgument(rewriter.getIndexType(), loc);

        // Allocate the memory for the gate parameters collected at runtime.
        rewriter.setInsertionPointToStart(&argMapFn.getBody().front());
        MemRefType paramsBufferType =
            MemRefType::get({ShapedType::kDynamic}, rewriter.getF64Type());
        Value paramsBuffer = rewriter.create<memref::AllocOp>(loc, paramsBufferType, numParams);
        MemRefType paramsProcessedType = MemRefType::get({}, rewriter.getIndexType());
        Value paramsProcessed = rewriter.create<memref::AllocaOp>(loc, paramsProcessedType);
        Value cZero = rewriter.create<index::ConstantOp>(loc, 0);
        rewriter.create<memref::StoreOp>(loc, cZero, paramsProcessed);
        Value cOne = rewriter.create<index::ConstantOp>(loc, 1);

        // Insert gate parameters into the params buffer.
        argMapFn.walk([&](quantum::CustomOp gate) {
            PatternRewriter::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPoint(gate);

            if (!gate.getParams().empty()) {
                Value paramIdx = rewriter.create<memref::LoadOp>(loc, paramsProcessed);
                for (auto param : gate.getParams()) {
                    rewriter.create<memref::StoreOp>(loc, param, paramsBuffer, paramIdx);
                    paramIdx = rewriter.create<index::AddOp>(loc, paramIdx, cOne);
                }
                rewriter.create<memref::StoreOp>(loc, paramIdx, paramsProcessed);
            }

            rewriter.replaceOp(gate, gate.getInQubits());
        });

        // Replace any return statements from the original function with the params vector.
        argMapFn.walk([&](func::ReturnOp returnOp) {
            PatternRewriter::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPoint(returnOp);

            Value paramsVector = rewriter.create<bufferization::ToTensorOp>(loc, paramsBuffer);
            returnOp->setOperands(paramsVector);
        });

        quantum::removeQuantumMeasurements(argMapFn);
    }

    return argMapFn;
}

/// Generate an mlir function to wrap an existing function into a return-by-pointer style function.
///
/// .
///
func::FuncOp genEnzymeWrapperFunction(PatternRewriter &rewriter, Location loc, GradOp gradOp,
                                      func::FuncOp argMapFn)
{
    MLIRContext *ctx = rewriter.getContext();
    LLVMTypeConverter llvmTypeConverter(ctx);
    bufferization::BufferizeTypeConverter buffTypeConverter;

    // Define the properties of the enzyme wrapper function.
    std::string fnName = gradOp.getCallee().str() + ".enzyme_wrapper";
    SmallVector<Type> argTypes(argMapFn.getArgumentTypes().begin(),
                               argMapFn.getArgumentTypes().end());
    argTypes.insert(argTypes.end(), argMapFn.getResultTypes().begin(),
                    argMapFn.getResultTypes().end());

    SmallVector<Type> originalArgTypes, bufferizedArgTypes;
    for (auto argTypeIt = argTypes.begin(); argTypeIt < argTypes.end() - argMapFn.getNumResults();
         argTypeIt++) {
        originalArgTypes.push_back(*argTypeIt);
        if (argTypeIt->isa<TensorType>()) {
            Type buffArgType = buffTypeConverter.convertType(*argTypeIt);
            bufferizedArgTypes.push_back(buffArgType);
            Type llvmArgType = llvmTypeConverter.convertType(buffArgType);
            if (!llvmArgType)
                emitError(loc, "Could not convert argmap argument to LLVM type: ") << buffArgType;
            *argTypeIt = LLVM::LLVMPointerType::get(llvmArgType);
        }
        else {
            bufferizedArgTypes.push_back(*argTypeIt);
        }
    }
    SmallVector<Type> bufferizedResultTypes, llvmResultTypes;
    for (auto resTypeIt = argTypes.begin() + argMapFn.getNumArguments(); resTypeIt < argTypes.end();
         resTypeIt++) {
        Type buffResType = buffTypeConverter.convertType(*resTypeIt);
        bufferizedResultTypes.push_back(buffResType);
        Type llvmResType = llvmTypeConverter.convertType(buffResType);
        if (!llvmResType)
            emitError(loc, "Could not convert argmap result to LLVM type: ") << buffResType;
        llvmResultTypes.push_back(llvmResType);
        *resTypeIt = LLVM::LLVMPointerType::get(llvmResType);
    }

    FunctionType fnType = rewriter.getFunctionType(argTypes, {});
    StringAttr visibility = rewriter.getStringAttr("private");

    func::FuncOp enzymeFn =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(gradOp, rewriter.getStringAttr(fnName));
    if (!enzymeFn) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointAfter(argMapFn);

        enzymeFn = rewriter.create<func::FuncOp>(loc, fnName, fnType, visibility);
        Block *entryBlock = enzymeFn.addEntryBlock();
        rewriter.setInsertionPointToStart(entryBlock);

        SmallVector<Value> callArgs(enzymeFn.getArguments().begin(),
                                    enzymeFn.getArguments().end() - argMapFn.getNumResults());
        for (auto [arg, buffType] : llvm::zip(callArgs, bufferizedArgTypes)) {
            if (arg.getType().isa<LLVM::LLVMPointerType>()) {
                Value memrefStruct = rewriter.create<LLVM::LoadOp>(loc, arg);
                Value memref =
                    rewriter.create<UnrealizedConversionCastOp>(loc, buffType, memrefStruct)
                        .getResult(0);
                arg = rewriter.create<bufferization::ToTensorOp>(loc, memref);
            }
        }
        ValueRange results = rewriter.create<func::CallOp>(loc, argMapFn, callArgs).getResults();

        ValueRange resArgs = enzymeFn.getArguments().drop_front(argMapFn.getNumArguments());

        SmallVector<Value> tensorFreeResults;
        for (auto [result, memrefType] : llvm::zip(results, bufferizedResultTypes)) {
            if (result.getType().isa<TensorType>())
                result = rewriter.create<bufferization::ToMemrefOp>(loc, memrefType, result);
            tensorFreeResults.push_back(result);
        }

        ValueRange llvmResults =
            rewriter.create<UnrealizedConversionCastOp>(loc, llvmResultTypes, tensorFreeResults)
                .getResults();
        for (auto [result, resArg] : llvm::zip(llvmResults, resArgs)) {
            rewriter.create<LLVM::StoreOp>(loc, result, resArg);
        }

        rewriter.create<func::ReturnOp>(loc);
    }

    return enzymeFn;
}

/// Generate an mlir function to compute the classical Jacobian via Enzyme.
///
/// .
///
func::FuncOp genBackpropFunction(PatternRewriter &rewriter, Location loc, GradOp gradOp,
                                 func::FuncOp callee)
{
    MLIRContext *ctx = rewriter.getContext();
    LLVMTypeConverter llvmTypeConverter(ctx);
    bufferization::BufferizeTypeConverter buffTypeConverter;

    // Declare the special Enzyme autodiff function.
    std::string autodiffFnName = "__enzymne_autodiff";
    LLVM::LLVMFuncOp autodiffFn = SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(
        callee, rewriter.getStringAttr(autodiffFnName));
    if (!autodiffFn) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(callee->getParentOfType<mlir::ModuleOp>().getBody());

        LLVM::LLVMFunctionType fnType =
            LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {}, /*isVarArg=*/true);

        autodiffFn = rewriter.create<LLVM::LLVMFuncOp>(loc, autodiffFnName, fnType);
    }

    // Define the properties of the classical Jacobian function.
    std::stringstream uniquer;
    const std::vector<size_t> &diffArgIndices = gradOp.compDiffArgIndices();
    std::copy(diffArgIndices.begin(), diffArgIndices.end(), std::ostream_iterator<int>(uniquer));
    std::string fnName = callee.getName().str() + ".backprop" + uniquer.str();

    size_t numResultArgs = callee.getNumArguments() - gradOp.getNumOperands();
    TypeRange argTypes = gradOp.getOperandTypes();
    SmallVector<Type> resTypes; // (callee.getArgumentTypes().end() - numResultArgs,
                                // callee.getArgumentTypes().end());
    FunctionType fnType = rewriter.getFunctionType(argTypes, resTypes);
    StringAttr visibility = rewriter.getStringAttr("private");

    func::FuncOp gradFn =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(callee, rewriter.getStringAttr(fnName));
    if (!gradFn) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointAfter(callee);

        gradFn = rewriter.create<func::FuncOp>(loc, fnName, fnType, visibility);
        Block *entryBlock = gradFn.addEntryBlock();
        rewriter.setInsertionPointToStart(entryBlock);

        FunctionType calleeType = callee.getFunctionType();
        Value calleeValue =
            rewriter.create<func::ConstantOp>(loc, calleeType, callee.getName()).getResult();
        Type calleePtrType = llvmTypeConverter.convertType(calleeType);
        Value calleePtr =
            rewriter.create<UnrealizedConversionCastOp>(loc, calleePtrType, calleeValue)
                .getResult(0);

        SmallVector<Value> callArgs = {calleePtr};
        ValueRange gradFnArgs = gradFn.getArguments();
        for (auto [arg, targetType] :
             llvm::zip(gradFnArgs, callee.getArgumentTypes().drop_back(numResultArgs))) {
            if (auto tensorType = arg.getType().dyn_cast<TensorType>()) {
                // Assume we're dealing with our own converted pointer types for now.
                assert(targetType.isa<LLVM::LLVMPointerType>());
                Type structType = targetType.cast<LLVM::LLVMPointerType>().getElementType();

                Value memref = rewriter.create<bufferization::ToMemrefOp>(
                    loc, buffTypeConverter.convertType(tensorType), arg);
                Value memrefStruct =
                    rewriter.create<UnrealizedConversionCastOp>(loc, structType, memref)
                        .getResult(0);
                Value c1 = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);
                Value structPtr = rewriter.create<LLVM::AllocaOp>(loc, targetType, c1);
                rewriter.create<LLVM::StoreOp>(loc, memrefStruct, structPtr);
                arg = structPtr;
            }
            callArgs.push_back(arg);
        }

        rewriter.create<LLVM::CallOp>(loc, autodiffFn, callArgs);

        rewriter.create<func::ReturnOp>(loc);
    }

    return gradFn;
}

} // namespace gradient
} // namespace catalyst
