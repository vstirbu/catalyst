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

#define DEBUG_TYPE "qcontrol"

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

CustomOp cloneWith(PatternRewriter &rewriter, CustomOp op, IRMapping &mapper, OperandRange ctrlQubits, OperandRange ctrlValues)
{
    std::vector<Value> inQubits;
    std::vector<Value> inParams;
    std::vector<Value> inCtrlQubits;
    std::vector<Value> inCtrlValues;
    std::vector<Type> resultTypes;
    std::vector<Type> controlTypes;
    {
        for (auto o: op.getInQubits())
            inQubits.push_back(mapper.lookupOrDefault(o));
        for (auto o: ctrlQubits)
            inCtrlQubits.push_back(mapper.lookupOrDefault(o));
        for (auto o: ctrlValues)
            inCtrlValues.push_back(mapper.lookupOrDefault(o));
        for (auto o: op.getParams())
            inParams.push_back(mapper.lookupOrDefault(o));
        for (auto o: op.getOutQubits())
            resultTypes.push_back(o.getType());
        for (auto o: ctrlQubits)
            controlTypes.push_back(o.getType());
    }

    LLVM_DEBUG(dbgs() << "nInQubits: " << inQubits.size() << "\n");
    LLVM_DEBUG(dbgs() << "nInCtrlQubits: " << inCtrlQubits.size() << "\n");
    LLVM_DEBUG(dbgs() << "nOutQubits: " << resultTypes.size() << "\n");
    LLVM_DEBUG(dbgs() << "nOutCtrlQubits: " << controlTypes.size() << "\n");

    /* OperationState state(op.getLoc(), op->getName()); */
    /* OpBuilder builder(op->getContext()); */
    auto newOp = rewriter.create<CustomOp>(op.getLoc(), resultTypes, controlTypes, inParams, inQubits,
        op.getGateName(), op.getAdjointAttr(), inCtrlQubits, inCtrlValues);
    /* CustomOp::build(builder, state, resultTypes, controlTypes, inParams, inQubits, */
    /*     op.getGateName(), op.getAdjointAttr(), inCtrlQubits, inCtrlValues); */
    /* CustomOp newOp = dyn_cast<CustomOp>(builder.create(state)); */

    /* mapper.map(op, newOp); */
    LLVM_DEBUG(dbgs() << "real outQubits: " << newOp.getOutQubits().size() << "\n");
    LLVM_DEBUG(dbgs() << "real ctrlQubits: " << newOp.getOutCtrlQubits().size() << "\n");
    LLVM_DEBUG(dbgs() << *newOp << "\n");
    LLVM_DEBUG(dbgs() << newOp.verify().succeeded() << "\n");
    LLVM_DEBUG(dbgs() << newOp->getAttrs().size() << "\n");

    for (const auto &[qN, qO] : zip(newOp.getOutQubits(), op.getOutQubits()))
        mapper.map(qO, qN);
    for (const auto &[qN, qC] : zip(newOp.getOutCtrlQubits(), ctrlQubits))
        mapper.map(qC, qN);

    return newOp;
}

struct QControlSingleOpRewritePattern : public mlir::OpRewritePattern<CtrlOp> {
    using mlir::OpRewritePattern<CtrlOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CtrlOp ctrl,
                                        mlir::PatternRewriter &rewriter) const override
    {
        SmallVector<Value> outputs;
        SmallVector<Value> ctrlWires = ctrl.getCtrlValues();
        IRMapping mapping;

        mapping.map(ctrl.getRegion().front().getArgument(0), ctrl.getInQreg());

        for (auto &i : ctrl.getRegion().front()) {
            if (isa<QuantumDialect>(i.getDialect())) {
                LLVM_DEBUG(dbgs() << "quantum operation: " << i << "\n");
                if (YieldOp yield = dyn_cast<YieldOp>(i)) {
                    for (const auto &v : yield.getOperands())
                        outputs.push_back(mapping.lookup(v));
                    for (const auto &v : ctrl.getCtrlQubits())
                        outputs.push_back(mapping.lookup(v));
                }
                else if (CustomOp custom = dyn_cast<CustomOp>(i)) {
                    cloneWith(rewriter, custom, mapping, ctrl.getCtrlQubits(), ctrl.getCtrlValues());
                    /* rewriter.insert(cloneWith(rewriter, custom, mapping, ctrl.getCtrlQubits(), ctrl.getCtrlValues())); */
                }
                /* else if (QuantumGate gate = dyn_cast<QuantumGate>(i)) { */
                /*     QuantumGate clone = dyn_cast<QuantumGate>(gate->clone(mapping)); */
                /*     clone.setCtrlArgs(ctrl.getCtrlQubits()); */
                /*     clone.setCtrlValues(ctrl.getCtrlValues()); */
                /*     rewriter.insert(clone); */
                /* } */
                else {
                    rewriter.insert(i.clone(mapping));
                }
            }
            else {
                LLVM_DEBUG(dbgs() << "classical operation: " << i << "\n");
                rewriter.insert(i.clone(mapping));
            }
        }

        LLVM_DEBUG(dbgs() << "replacing ctrl\n");
        rewriter.replaceOp(ctrl, outputs);
        return success();
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
