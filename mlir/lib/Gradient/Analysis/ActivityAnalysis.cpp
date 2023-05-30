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
#include "llvm/Support/raw_ostream.h"

#include "Gradient/Analysis/GradientAnalysis.h"
#include "Gradient/IR/GradientOps.h"
#include "Gradient/Utils/CompDiffArgIndices.h"

#include <deque>

using namespace mlir;

namespace catalyst {

using llvm::errs;
void topDownBFS(std::deque<Value> &frontier, DenseSet<Value> &visited)
{
    auto queueIfNotVisited = [&](Value node) {
        if (!visited.contains(node)) {
            visited.insert(node);
            frontier.push_back(node);
        }
    };
    while (!frontier.empty()) {
        Value v = frontier.front();
        frontier.pop_front();
        for (OpOperand &use : v.getUses()) {
            Operation *user = use.getOwner();
            errs() << "user: " << *user << " at " << user->getLoc() << "\n";
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

void bottomUpBFS(Operation *op) {}

LogicalResult runActivityAnalysis(Operation *top)
{
    DenseSet<StringAttr> visitedFuncs;
    top->walk([&](gradient::GradOp gradOp) {
        if (!visitedFuncs.contains(gradOp.getCalleeAttrName())) {
            visitedFuncs.insert(gradOp.getCalleeAttrName());
            auto funcOp =
                SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(top, gradOp.getCalleeAttr());
            const std::vector<size_t> &diffArgIndices =
                compDiffArgIndices(gradOp.getDiffArgIndices());

            std::deque<Value> frontier;
            DenseSet<Value> topDownVisited;
            for (size_t index : diffArgIndices) {
                frontier.push_back(funcOp.getArgument(index));
            }

            topDownBFS(frontier, topDownVisited);
        }
    });

    return success();
}
} // namespace catalyst
