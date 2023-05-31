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
#include "llvm/Support/Debug.h"

#include "Gradient/Analysis/GradientAnalysis.h"

using namespace mlir;

namespace catalyst {
const char *idKey = "activity.id";
Attribute ensureAttribute(Attribute attr)
{
    assert(attr && "value had no activity.id attribute");
    return attr;
}

void debugPrintValue(Value value)
{
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        Operation *op = blockArg.getParentRegion()->getParentOp();
        if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
            llvm::dbgs() << ensureAttribute(funcOp.getArgAttr(blockArg.getArgNumber(), idKey))
                         << "\n";
        }
        else {
            llvm::dbgs() << "PRINT ERROR: unhandled BlockArgument parent " << op->getName() << "\n";
        }
    }
    else {
        Operation *op = value.getDefiningOp();
        llvm::dbgs() << ensureAttribute(op->getAttr(idKey)) << "\n";
    }
}
} // namespace catalyst
