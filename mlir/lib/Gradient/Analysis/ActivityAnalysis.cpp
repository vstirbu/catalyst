// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/Support/Debug.h"

#include "Gradient/Analysis/ActivityAnalysis.h"
#include "Gradient/IR/GradientOps.h"
#include "Gradient/Utils/CompDiffArgIndices.h"

#include <deque>

#define DEBUG_TYPE "activity-analysis"
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
#define ACTIVITY_DEBUG(X) LLVM_DEBUG(X)
#else
#define ACTIVITY_DEBUG(X)
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS

using namespace mlir;

namespace catalyst {

void topDownBFS(std::deque<Value> &frontier, DenseSet<Value> &visited)
{
    auto queueIfNotVisited = [&](Value node) {
        if (!visited.contains(node)) {
            visited.insert(node);
            frontier.push_back(node);
        }
    };
    visited.insert(frontier.begin(), frontier.end());

    while (!frontier.empty()) {
        Value v = frontier.front();
        frontier.pop_front();
        for (OpOperand &use : v.getUses()) {
            Operation *user = use.getOwner();
            if (isRegionReturnLike(user)) {
                Operation *parent = user->getParentOp();
                if (auto functionLike = dyn_cast_or_null<FunctionOpInterface>(parent)) {
                    // In top-down propagation, do nothing at function returns.
                }
                else if (auto branch = dyn_cast_or_null<RegionBranchOpInterface>(parent)) {
                    // ReturnLike operations propagate values to their parents
                    queueIfNotVisited(branch->getResult(use.getOperandNumber()));

                    // Loops propagate yielded values to their iteration args
                    size_t regionNumber = branch->getParentRegion()->getRegionNumber();
                    if (branch.isRepetitiveRegion(regionNumber)) {
                        queueIfNotVisited(
                            branch.getSuccessorEntryOperands(regionNumber)[use.getOperandNumber()]);
                    }
                }
            }
            else {
                for (OpResult result : user->getResults())
                    queueIfNotVisited(result);
            }
        }
    }
}

LogicalResult bottomUpBFS(std::deque<Value> &frontier, DenseSet<Value> &visited)
{
    auto queueIfNotVisited = [&](Value node) {
        if (!visited.contains(node)) {
            visited.insert(node);
            frontier.push_back(node);
        }
    };
    visited.insert(frontier.begin(), frontier.end());

    while (!frontier.empty()) {
        Value v = frontier.front();
        frontier.pop_front();

        if (Operation *parent = v.getDefiningOp()) {
            if (auto branch = dyn_cast<RegionBranchOpInterface>(parent)) {
                SmallVector<RegionSuccessor> successors;
                // llvm::None gets *all* successor regions.
                branch.getSuccessorRegions(llvm::None, successors);

                for (auto successor : successors) {
                    if (!successor.isParent()) {
                        if (!successor.getSuccessor()->hasOneBlock()) {
                            successor.getSuccessor()->getParentOp()->emitError()
                                << "Activity analysis only supports structured control flow (each "
                                   "region should have one block)";
                            return failure();
                        }

                        Operation *terminator = successor.getSuccessor()->front().getTerminator();

                        for (const auto &[operand, result] :
                             llvm::zip(terminator->getOperands(), branch->getResults())) {
                            if (result == v) {
                                queueIfNotVisited(operand);
                            }
                        }
                    }
                }
            }
            else {
                for (Value operand : parent->getOperands()) {
                    queueIfNotVisited(operand);
                }
            }
        }
    }
    return success();
}

ActivityAnalysis::ActivityAnalysis(Operation *moduleOp)
{
    // TODO(jacob): cache functions that we analyze, they might potentially call each other so it'd
    // be good to reuse results
    WalkResult result = moduleOp->walk([&](gradient::GradOp gradOp) {
        ACTIVITY_DEBUG(llvm::dbgs()
                       << "Running activity analysis on '@" << gradOp.getCallee() << "'\n");

        auto funcOp =
            SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(moduleOp, gradOp.getCalleeAttr());
        if (funcOp.isExternal()) {
            // We can't analyze a function without seeing its body
            return WalkResult::advance();
        }
        const std::vector<size_t> &diffArgIndices = compDiffArgIndices(gradOp.getDiffArgIndices());

        std::deque<Value> frontier;
        DenseSet<Value> topDownVisited;
        for (size_t index : diffArgIndices) {
            frontier.push_back(funcOp.getArgument(index));
        }
        topDownBFS(frontier, topDownVisited);

        frontier.clear();
        DenseSet<Value> bottomUpVisited;
        // TODO(jacob): Do we assume all outputs are active?
        funcOp.walk([&](func::ReturnOp returnOp) {
            frontier.insert(frontier.end(), returnOp.operand_begin(), returnOp.operand_end());
        });
        if (failed(bottomUpBFS(frontier, bottomUpVisited))) {
            return WalkResult::interrupt();
        }

        ValueSet invokedActiveValues;
        // Active values are the intersection of top-down and bottom-up active values.
        for (Value value : topDownVisited) {
            if (bottomUpVisited.contains(value)) {
                invokedActiveValues.insert(value);
            }
        }
        this->activeValues[gradOp] = invokedActiveValues;

        ACTIVITY_DEBUG(debugPrintValue(invokedActiveValues));
        return WalkResult::advance();
    });

    this->valid = !result.wasInterrupted();
}
} // namespace catalyst
