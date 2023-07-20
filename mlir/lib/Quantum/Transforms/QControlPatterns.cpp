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

#define DEBUG_TYPE "adjoint"

#include <algorithm>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst::quantum;

namespace {

struct QControlSingleOpRewritePattern : public mlir::OpRewritePattern<CtrlOp> {
    using mlir::OpRewritePattern<CtrlOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CtrlOp ctrl,
                                        mlir::PatternRewriter &rewriter) const override
    {
        SmallVector<Value> outputs;
        IRMapping mapping;
        for (auto &i : ctrl.getRegion().front()) {
            if (isa<QuantumDialect>(i.getDialect())) {
                LLVM_DEBUG(dbgs() << "quantum operation: " << i << "\n");
                if (YieldOp yield = dyn_cast<YieldOp>(i)) {
                    for (const auto &v : yield.getOperands())
                        outputs.push_back(mapping.lookup(v));
                }
                else if (QuantumGate gate = dyn_cast<QuantumGate>(i)) {
                    QuantumGate clone = dyn_cast<QuantumGate>(gate->clone(mapping));
                    clone.setCtrlArgs(ctrl.getCtrlQubits());
                    clone.setCtrlValues(ctrl.getCtrlValues());
                    rewriter.insert(clone);
                }
                else {
                    // TODO
                }
            }
            else {
                LLVM_DEBUG(dbgs() << "classical operation: " << i << "\n");
                rewriter.insert(i.clone(mapping));
            }
        }
        return failure();
    }
};

} // namespace

namespace catalyst {
namespace quantum {

void populateQControlPatterns(RewritePatternSet &patterns)
{
    patterns.add<QControlSingleOpRewritePattern>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
