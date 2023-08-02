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
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst::quantum;

namespace {

void cloneModifyForOp(
    OpBuilder &builder,
    scf::ForOp forOp,
    IRMapping &mapper,
    OperandRange addCtrlQubits,
    OperandRange addCtrlValues)
{
    ssize_t numControl = addCtrlQubits.size();
    auto start = mapper.lookupOrDefault(forOp.getLowerBound());
    auto stop = mapper.lookupOrDefault(forOp.getUpperBound());
    auto step = mapper.lookupOrDefault(forOp.getStep());
    SmallVector<Value> argsInit;
    for (const auto &a : forOp.getIterOperands())
        argsInit.push_back(mapper.lookupOrDefault(a));
    for (const auto &a : addCtrlQubits)
        argsInit.push_back(mapper.lookupOrDefault(a));

    auto newFor = builder.create<scf::ForOp>(
        forOp.getLoc(), start, stop, step, argsInit,
        [&](OpBuilder &bodyBuilder, Location loc, Value iv, ValueRange iterArgs) {
            IRMapping bodyMapper(mapper);
            bodyMapper.map(iv, forOp.getRegion().front().getArgument(0));
            SmallVector<BlockArgument> args(forOp.getRegion().front().args_begin()+1,
                                            forOp.getRegion().front().args_end());
            for (const auto &[o, n] : zip(args,
                                          ValueRange(iterArgs.begin(),
                                                     iterArgs.end()+(-numControl)))) {
                bodyMapper.map(o, n);
            }
            for(auto &op : forOp.getRegion().front().without_terminator()) {
                bodyBuilder.insert(op.clone(bodyMapper));
            }

            SmallVector<Value> results;
            for(const auto &r: forOp.getRegion().front().getTerminator()->getOperands()) {
                results.push_back(bodyMapper.lookupOrDefault(r));
            }
            for(const auto &r: ValueRange(iterArgs.end()+(-numControl), iterArgs.end())) {
                results.push_back(bodyMapper.lookupOrDefault(r));
            }

            bodyBuilder.create<scf::YieldOp>(loc, results);
        });


    for (const auto &[o, n] : zip(forOp.getResults(), newFor.getResults())) {
        mapper.map(o, n);
    }
    for (const auto &[o, n] : zip(addCtrlQubits,
                                  ResultRange(newFor.getResults().end()+(-numControl),
                                              newFor.getResults().end()))) {
        mapper.map(o, n);
    }
}

scf::IfOp cloneModifyIfOp(
    scf::IfOp op,
    IRMapping &mapper,
    OperandRange addCtrlQubits,
    OperandRange addCtrlValues)
{
    // TODO
    return op;
}


CustomOp cloneModifyCustomOp(
    CustomOp op,
    IRMapping &mapper,
    OperandRange addCtrlQubits,
    OperandRange addCtrlValues)
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
        for (auto o: op.getParams())
            inParams.push_back(mapper.lookupOrDefault(o));
        for (auto o: op.getOutQubits())
            resultTypes.push_back(o.getType());

        for (auto o: op.getInCtrlQubits())
            inCtrlQubits.push_back(mapper.lookupOrDefault(o));
        for (auto o: addCtrlQubits)
            inCtrlQubits.push_back(mapper.lookupOrDefault(o));

        // FIXME: figure out how to use llvm::concat for joining iterators
        for (auto o: op.getInCtrlValues())
            inCtrlValues.push_back(mapper.lookupOrDefault(o));
        for (auto o: addCtrlValues)
            inCtrlValues.push_back(mapper.lookupOrDefault(o));

        // FIXME: figure out how to use llvm::concat for joining iterators
        for (auto o: op.getInCtrlQubits())
            controlTypes.push_back(o.getType());
        for (auto o: addCtrlQubits)
            controlTypes.push_back(o.getType());
    }

    OpBuilder builder(op->getContext());
    OperationState state(op->getLoc(), op->getName());
    CustomOp::build(builder, state, resultTypes, controlTypes, inParams, inQubits,
        op.getGateName(), op.getAdjointAttr(), inCtrlQubits, inCtrlValues);
    CustomOp newOp = dyn_cast<CustomOp>(builder.create(state));

    std::vector<Value> ctrlQubits;
    {
        // FIXME: figure out how to use llvm::concat for joining iterators
        for (auto o: op.getOutCtrlQubits())
            ctrlQubits.push_back(mapper.lookupOrDefault(o));
        for (auto o: addCtrlQubits)
            ctrlQubits.push_back(mapper.lookupOrDefault(o));
    }

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
                    rewriter.insert(cloneModifyCustomOp(custom, mapping,
                                                       ctrl.getCtrlQubits(),
                                                       ctrl.getCtrlValues()));
                }
                // FIXME: unify the `cloneModify*` functions and replace them with a QuantumGate
                // interface method call
                else {
                    rewriter.insert(i.clone(mapping));
                }
            }
            else if (isa<scf::SCFDialect>(i.getDialect())) {
                LLVM_DEBUG(dbgs() << "SCF (Hybrid) operation: " << i << "\n");

                if (scf::IfOp op = dyn_cast<scf::IfOp>(i)) {
                    rewriter.insert(cloneModifyIfOp(op, mapping,
                                                    ctrl.getCtrlQubits(),
                                                    ctrl.getCtrlValues()));
                }
                else if (scf::ForOp op = dyn_cast<scf::ForOp>(i)) {
                    cloneModifyForOp(rewriter, op, mapping,
                                     ctrl.getCtrlQubits(),
                                     ctrl.getCtrlValues());
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

void populateQControlLoweringPatterns(RewritePatternSet &patterns)
{
    patterns.add<QControlSingleOpRewritePattern>(patterns.getContext(), 1);
}

void populateQControlSymsubstPatterns(RewritePatternSet &patterns)
{
    patterns.add<QControlSingleOpRewritePattern>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
