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

#include <algorithm>
#include <sstream>
#include <vector>

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "ClassicalJacobian.hpp"
#include "HybridGradient.hpp"

#include "Catalyst/Utils/CallGraph.h"
#include "Gradient/Utils/GradientShape.h"
#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Utils/RemoveQuantumMeasurements.h"

using namespace mlir;

namespace catalyst {
namespace gradient {

LogicalResult HybridGradientLowering::matchAndRewrite(GradOp op, PatternRewriter &rewriter) const
{
    Location loc = op.getLoc();
    func::FuncOp callee =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());
    rewriter.setInsertionPointAfter(callee);
    // Replace calls with the QNode with the split QNode in the callee.
    auto clonedCallee = cast<func::FuncOp>(rewriter.clone(*callee));
    std::string clonedCalleeName = (callee.getName() + ".cloned").str();
    clonedCallee.setName(clonedCalleeName);
    SmallPtrSet<Operation *, 4> qnodes;
    SymbolTableCollection symbolTable;
    auto isQNode = [](func::FuncOp funcOp) { return funcOp->hasAttr("qnode"); };
    if (isQNode(clonedCallee)) {
        qnodes.insert(callee);
    }
    else {
        traverseCallGraph(clonedCallee, symbolTable, [&qnodes, &isQNode](func::FuncOp funcOp) {
            if (isQNode(funcOp)) {
                qnodes.insert(funcOp);
            }
        });
    }

    for (Operation *qnodeOp : qnodes) {
        auto qnode = cast<func::FuncOp>(qnodeOp);

        // In order to allocate memory for various tensors relating to the number of gate parameters
        // at runtime we run a function that merely counts up for each gate parameter encountered.
        func::FuncOp paramCountFn = genParamCountFunction(rewriter, loc, qnode);
        func::FuncOp qnodeWithParams = genQNodeWithParams(rewriter, loc, qnode);
        func::FuncOp qnodeSplit = genSplitPreprocessed(rewriter, loc, qnode, qnodeWithParams);

        // This attribute tells downstream patterns that this QNode requires the registration of a
        // custom quantum gradient.
        qnode->setAttr("withparams", FlatSymbolRefAttr::get(qnodeWithParams));
        // Enzyme will fail if this function gets inlined.
        qnodeWithParams->setAttr("passthrough",
                                 rewriter.getArrayAttr(rewriter.getStringAttr("noinline")));

        // Replace calls to the original QNode with calls to the split QNode
        if (isQNode(clonedCallee)) {
            PatternRewriter::InsertionGuard insertionGuard(rewriter);
            rewriter.eraseBlock(&clonedCallee.getFunctionBody().front());
            Block *entryBlock = clonedCallee.addEntryBlock();

            rewriter.setInsertionPointToStart(entryBlock);
            Value paramCount =
                rewriter.create<func::CallOp>(loc, paramCountFn, clonedCallee.getArguments())
                    .getResult(0);
            SmallVector<Value> splitArgs{clonedCallee.getArguments()};
            splitArgs.push_back(paramCount);

            auto splitCall = rewriter.create<func::CallOp>(loc, qnodeSplit, splitArgs);
            rewriter.create<func::ReturnOp>(loc, splitCall.getResults());
        }
        else {
            traverseCallGraph(clonedCallee, symbolTable, [&](func::FuncOp funcOp) {
                funcOp.walk([&](func::CallOp callOp) {
                    if (callOp.getCallee() == qnode.getName()) {
                        PatternRewriter::InsertionGuard insertionGuard(rewriter);
                        rewriter.setInsertionPointToStart(&funcOp.getFunctionBody().front());
                        Value paramCount =
                            rewriter
                                .create<func::CallOp>(loc, paramCountFn, callOp.getArgOperands())
                                .getResult(0);
                        callOp.setCallee(qnodeSplit.getName());
                        callOp.getOperandsMutable().append(paramCount);
                    }
                });
            });
        }
    }

    rewriter.setInsertionPoint(op);
    std::vector<size_t> diffArgIndices = computeDiffArgIndices(op.getDiffArgIndices());
    SmallVector<Value> backpropResults{op.getNumResults()};
    // Iterate over the primal results
    for (const auto &[cotangentIdx, primalResult] :
         llvm::enumerate(clonedCallee.getResultTypes())) {
        // There is one Jacobian per distinct differential argument.
        SmallVector<Value> jacobians;
        for (unsigned argIdx = 0; argIdx < diffArgIndices.size(); argIdx++) {
            Type jacobianType =
                op.getResultTypes()[argIdx * clonedCallee.getNumResults() + cotangentIdx];
            jacobians.push_back(
                rewriter.create<tensor::EmptyOp>(loc, jacobianType, /*dynamicSizes=*/ValueRange{}));
        }

        auto primalTensorResultType = cast<RankedTensorType>(primalResult);
        assert(primalTensorResultType.hasStaticShape());

        ArrayRef<int64_t> shape = primalTensorResultType.getShape();
        // Compute the strides in reverse
        unsigned product = 1;
        SmallVector<unsigned> strides;
        for (int64_t dim = primalTensorResultType.getRank() - 1; dim >= 0; dim--) {
            strides.push_back(product);
            product *= shape[dim];
        }
        std::reverse(strides.begin(), strides.end());

        Value zero = rewriter.create<arith::ConstantOp>(
            loc, FloatAttr::get(primalTensorResultType.getElementType(), 0.0));
        Value one = rewriter.create<arith::ConstantOp>(
            loc, FloatAttr::get(primalTensorResultType.getElementType(), 1.0));
        Value zeroTensor = rewriter.create<tensor::EmptyOp>(loc, primalTensorResultType,
                                                            /*dynamicSizes=*/ValueRange{});
        zeroTensor = rewriter.create<linalg::FillOp>(loc, zero, zeroTensor).getResult(0);

        for (unsigned flatIdx = 0; flatIdx < primalTensorResultType.getNumElements(); flatIdx++) {
            // Unflatten the tensor indices
            SmallVector<Value> indices;
            for (int64_t dim = 0; dim < primalTensorResultType.getRank(); dim++) {
                indices.push_back(
                    rewriter.create<index::ConstantOp>(loc, flatIdx / strides[dim] % shape[dim]));
            }

            SmallVector<Value> cotangents;
            Value cotangent = rewriter.create<tensor::InsertOp>(loc, one, zeroTensor, indices);
            for (const auto &[resultIdx, primalResultType] :
                 llvm::enumerate(clonedCallee.getResultTypes())) {
                if (resultIdx == cotangentIdx) {
                    cotangents.push_back(cotangent);
                }
                else {
                    // Push back a zeroed-out cotangent
                    Value zeroTensor =
                        rewriter.create<tensor::EmptyOp>(loc, primalResultType, ValueRange{});
                    cotangents.push_back(
                        rewriter.create<linalg::FillOp>(loc, zero, zeroTensor).getResult(0));
                }
            }

            auto backpropOp = rewriter.create<gradient::BackpropOp>(
                loc, computeBackpropTypes(clonedCallee, diffArgIndices), clonedCallee.getName(),
                op.getArgOperands(),
                /*arg_shadows=*/ValueRange{}, /*primal results=*/ValueRange{}, cotangents,
                op.getDiffArgIndicesAttr());

            // Backprop gives a gradient of a single output entry w.r.t. all active inputs.
            for (const auto &[backpropIdx, jacobianSlice] :
                 llvm::enumerate(backpropOp.getResults())) {
                auto sliceType = cast<RankedTensorType>(jacobianSlice.getType());
                size_t sliceRank = sliceType.getRank();
                auto jacobianType = cast<RankedTensorType>(jacobians[backpropIdx].getType());
                size_t jacobianRank = jacobianType.getRank();
                if (sliceRank < jacobianRank) {
                    // Offsets are [...indices] + [0] * rank of backprop result
                    SmallVector<OpFoldResult> offsets;
                    offsets.append(indices.begin(), indices.end());
                    offsets.append(sliceRank, rewriter.getIndexAttr(0));

                    // Sizes are [1] * (jacobianRank - sliceRank) + [...sliceShape]
                    SmallVector<OpFoldResult> sizes;
                    sizes.append(jacobianRank - sliceRank, rewriter.getIndexAttr(1));
                    for (int64_t dim : sliceType.getShape()) {
                        sizes.push_back(rewriter.getIndexAttr(dim));
                    }

                    // Strides are [1] * jacobianRank
                    SmallVector<OpFoldResult> strides{jacobianRank, rewriter.getIndexAttr(1)};

                    jacobians[backpropIdx] = rewriter.create<tensor::InsertSliceOp>(
                        loc, jacobianSlice, jacobians[backpropIdx], offsets, sizes, strides);
                }
                else {
                    jacobians[backpropIdx] = jacobianSlice;
                }
                backpropResults[backpropIdx * clonedCallee.getNumResults() + cotangentIdx] =
                    jacobians[backpropIdx];
            }
        }
    }

    rewriter.replaceOp(op, backpropResults);
    return success();
}

func::FuncOp HybridGradientLowering::genQNodeWithParams(PatternRewriter &rewriter, Location loc,
                                                        func::FuncOp qnode)
{
    std::string fnName = (qnode.getName() + ".withparams").str();
    SmallVector<Type> fnArgTypes(qnode.getArgumentTypes());
    auto paramsTensorType = RankedTensorType::get({ShapedType::kDynamic}, rewriter.getF64Type());
    fnArgTypes.push_back(paramsTensorType);
    FunctionType fnType = rewriter.getFunctionType(fnArgTypes, qnode.getResultTypes());

    func::FuncOp modifiedCallee =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(qnode, rewriter.getStringAttr(fnName));
    if (modifiedCallee) {
        return modifiedCallee;
    }

    modifiedCallee = rewriter.create<func::FuncOp>(loc, fnName, fnType);
    modifiedCallee.setPrivate();
    rewriter.cloneRegionBefore(qnode.getBody(), modifiedCallee.getBody(), modifiedCallee.end());
    Block &entryBlock = modifiedCallee.getFunctionBody().front();
    BlockArgument paramsTensor = entryBlock.addArgument(paramsTensorType, loc);

    PatternRewriter::InsertionGuard insertionGuard(rewriter);
    rewriter.setInsertionPointToStart(&modifiedCallee.getFunctionBody().front());

    MemRefType paramsProcessedType = MemRefType::get({}, rewriter.getIndexType());
    Value paramCounter = rewriter.create<memref::AllocaOp>(loc, paramsProcessedType);
    Value cZero = rewriter.create<index::ConstantOp>(loc, 0);
    rewriter.create<memref::StoreOp>(loc, cZero, paramCounter);
    Value cOne = rewriter.create<index::ConstantOp>(loc, 1);

    auto loadThenIncrementCounter = [&](OpBuilder &builder, Value counter,
                                        Value paramTensor) -> Value {
        Value index = builder.create<memref::LoadOp>(loc, counter);
        Value nextIndex = builder.create<index::AddOp>(loc, index, cOne);
        builder.create<memref::StoreOp>(loc, nextIndex, counter);
        return builder.create<tensor::ExtractOp>(loc, paramTensor, index);
    };

    modifiedCallee.walk([&](Operation *op) {
        if (auto gateOp = dyn_cast<quantum::DifferentiableGate>(op)) {
            OpBuilder::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPoint(gateOp);

            ValueRange diffParams = gateOp.getDiffParams();
            SmallVector<Value> newParams{diffParams.size()};
            for (const auto [paramIdx, recomputedParam] : llvm::enumerate(diffParams)) {
                newParams[paramIdx] =
                    loadThenIncrementCounter(rewriter, paramCounter, paramsTensor);
            }
            MutableOperandRange range{gateOp, static_cast<unsigned>(gateOp.getDiffOperandIdx()),
                                      static_cast<unsigned>(diffParams.size())};
            range.assign(newParams);
        }
    });

    // This function is the point where we can remove the classical preprocessing as a later
    // optimization.
    return modifiedCallee;
}

/// Generate an mlir function to compute the full gradient of a quantum function.
///
/// With the parameter-shift method (and certain other methods) the gradient of a quantum function
/// is computed as two separate parts: the gradient of the classical pre-processing function for
/// gate parameters, termed "classical Jacobian", and the purely "quantum gradient" of a
/// differentiable output of a circuit. The two components can be combined to form the gradient of
/// the entire quantum function via tensor contraction along the gate parameter dimension.
///
func::FuncOp genFullGradFunction(PatternRewriter &rewriter, Location loc, GradOp gradOp,
                                 func::FuncOp paramCountFn, func::FuncOp argMapFn,
                                 func::FuncOp qGradFn, StringRef method)
{
    // Define the properties of the full gradient function.
    const std::vector<size_t> &diffArgIndices = computeDiffArgIndices(gradOp.getDiffArgIndices());
    std::stringstream uniquer;
    std::copy(diffArgIndices.begin(), diffArgIndices.end(), std::ostream_iterator<int>(uniquer));
    std::string fnName = gradOp.getCallee().str() + ".fullgrad" + uniquer.str() + method.str();
    FunctionType fnType =
        rewriter.getFunctionType(gradOp.getOperandTypes(), gradOp.getResultTypes());
    StringAttr visibility = rewriter.getStringAttr("private");

    func::FuncOp fullGradFn =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(gradOp, rewriter.getStringAttr(fnName));
    if (!fullGradFn) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointAfter(qGradFn);

        fullGradFn =
            rewriter.create<func::FuncOp>(loc, fnName, fnType, visibility, nullptr, nullptr);
        Block *entryBlock = fullGradFn.addEntryBlock();
        rewriter.setInsertionPointToStart(entryBlock);

        // Collect arguments and invoke the classical jacobian and quantum gradient functions.
        SmallVector<Value> callArgs(fullGradFn.getArguments());

        Value numParams = rewriter.create<func::CallOp>(loc, paramCountFn, callArgs).getResult(0);
        callArgs.push_back(numParams);
        ValueRange quantumGradients =
            rewriter.create<func::CallOp>(loc, qGradFn, callArgs).getResults();

        DenseIntElementsAttr diffArgIndicesAttr = gradOp.getDiffArgIndices().value_or(nullptr);

        auto resultsBackpropTypes = computeBackpropTypes(argMapFn, diffArgIndices);
        // Compute hybrid gradients via Enzyme
        std::vector<Value> hybridGradients;
        int j = 0;
        // Loop over the measurements
        for (Value quantumGradient : quantumGradients) {
            Type resultType = gradOp.getResult(j).getType();
            int64_t rankResult = 0;
            ArrayRef<int64_t> shapeResult;
            if (auto resultTensorType = dyn_cast<RankedTensorType>(resultType)) {
                rankResult = resultTensorType.getRank();
                shapeResult = resultTensorType.getShape();
            }
            j++;

            std::vector<BackpropOp> intermediateGradients;
            auto rank = quantumGradient.getType().cast<RankedTensorType>().getRank();

            if (rank > 1) {
                Value result = rewriter.create<tensor::EmptyOp>(loc, resultType, ValueRange{});
                std::vector<int64_t> sizes =
                    quantumGradient.getType().cast<RankedTensorType>().getShape();

                std::vector<std::vector<int64_t>> allOffsets;
                std::vector<int64_t> cutOffset(sizes.begin() + 1, sizes.end());

                std::vector<int64_t> currentOffset(cutOffset.size(), 0);

                int64_t totalOutcomes = 1;
                for (int64_t dim : cutOffset) {
                    totalOutcomes *= dim;
                }

                for (int64_t outcome = 0; outcome < totalOutcomes; outcome++) {
                    allOffsets.push_back(currentOffset);

                    for (int64_t i = cutOffset.size() - 1; i >= 0; i--) {
                        currentOffset[i]++;
                        if (currentOffset[i] < cutOffset[i]) {
                            break;
                        }
                        currentOffset[i] = 0;
                    }
                }

                std::vector<int64_t> strides(rank, 1);
                std::vector<Value> dynStrides = {};

                std::vector<Value> dynOffsets = {};

                std::vector<Value> dynSizes;

                for (size_t index = 0; index < sizes.size(); ++index) {
                    if (index == 0) {
                        Value idx = rewriter.create<index::ConstantOp>(loc, index);
                        Value dimSize = rewriter.create<tensor::DimOp>(loc, quantumGradient, idx);
                        dynSizes.push_back(dimSize);
                    }
                    else {
                        sizes[index] = 1;
                    }
                }
                for (auto offsetRight : allOffsets) {
                    std::vector<int64_t> offsets{0};
                    offsets.insert(offsets.end(), offsetRight.begin(), offsetRight.end());
                    auto rankReducedType =
                        tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
                            1, quantumGradient.getType().cast<RankedTensorType>(), offsets, sizes,
                            strides)
                            .cast<RankedTensorType>();
                    Value extractQuantumGradient = rewriter.create<tensor::ExtractSliceOp>(
                        loc, rankReducedType, quantumGradient, dynOffsets, dynSizes, dynStrides,
                        offsets, sizes, strides);
                    BackpropOp backpropOp = rewriter.create<BackpropOp>(
                        loc, resultsBackpropTypes, argMapFn.getName(), callArgs, ValueRange{},
                        ValueRange{}, extractQuantumGradient, diffArgIndicesAttr);

                    intermediateGradients.push_back(backpropOp);
                }
                for (size_t i = 0; i < resultsBackpropTypes.size(); i++) {
                    // strides
                    std::vector<int64_t> stridesSlice(rankResult, 1);

                    for (int64_t index = 0; index < totalOutcomes; index++) {
                        auto intermediateGradient = intermediateGradients[index];
                        Value gradient = intermediateGradient.getResult(i);

                        Type gradientType = gradient.getType();
                        if (auto gradientTensorType = dyn_cast<RankedTensorType>(gradientType)) {
                            int64_t rankGradient = gradientTensorType.getRank();
                            // sizes
                            std::vector<int64_t> sizesSlice{shapeResult};
                            for (int64_t sliceIndex = rankResult - 1; sliceIndex >= rankGradient;
                                 sliceIndex--) {
                                sizesSlice[sliceIndex] = 1;
                            }

                            // offset
                            auto offsetSlice = allOffsets[index];
                            for (int64_t offsetIndex = 0; offsetIndex < rankGradient;
                                 offsetIndex++) {
                                offsetSlice.insert(offsetSlice.begin(), 0);
                            }
                            result = rewriter.create<tensor::InsertSliceOp>(
                                loc, resultType, gradient, result, ValueRange{}, ValueRange{},
                                ValueRange{}, offsetSlice, sizesSlice, stridesSlice);
                        }
                        else {
                            assert(isa<FloatType>(gradient.getType()));
                            SmallVector<Value> insertIndices;
                            for (int64_t offset : allOffsets[index]) {
                                insertIndices.push_back(
                                    rewriter.create<index::ConstantOp>(loc, offset));
                            }
                            result = rewriter.create<tensor::InsertOp>(loc, gradient, result,
                                                                       insertIndices);
                        }
                    }
                    hybridGradients.push_back(result);
                }
            }
            else {
                // The quantum gradient is a rank 1 tensor
                BackpropOp backpropOp = rewriter.create<BackpropOp>(
                    loc, resultsBackpropTypes, argMapFn.getName(), callArgs, ValueRange{},
                    ValueRange{}, quantumGradient, diffArgIndicesAttr);
                for (OpResult result : backpropOp.getResults()) {
                    Value hybridGradient = result;
                    Type gradResultType = gradOp.getResult(result.getResultNumber()).getType();
                    if (gradResultType != result.getType()) {
                        // The backprop op produces a row of the Jacobian, which always has the same
                        // type as the differentiated argument. If the rank of the quantum gradient
                        // is 1, this implies the callee returns a rank-0 value (either a
                        // scalar or a tensor<scalar>). The Jacobian of a scalar -> scalar should be
                        // a scalar, but as a special case, the Jacobian of a scalar ->
                        // tensor<scalar> should be tensor<scalar>.
                        if (isa<RankedTensorType>(gradResultType) &&
                            isa<FloatType>(result.getType())) {
                            Value jacobian =
                                rewriter.create<tensor::EmptyOp>(loc, gradResultType, ValueRange{});
                            hybridGradient = rewriter.create<tensor::InsertOp>(
                                loc, result, jacobian, ValueRange{});
                        }

                        // We also support where the argument is a tensor<scalar> but the desired
                        // hybrid gradient is a scalar. This is less about mathematical precision
                        // and more about ergonomics.
                        if (isa<FloatType>(gradResultType) &&
                            isa<RankedTensorType>(result.getType())) {
                            hybridGradient =
                                rewriter.create<tensor::ExtractOp>(loc, result, ValueRange{});
                        }
                    }

                    hybridGradients.push_back(hybridGradient);
                }
            }
        }
        rewriter.create<func::ReturnOp>(loc, hybridGradients);
    }

    return fullGradFn;
}

} // namespace gradient
} // namespace catalyst
