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

// FIXME: We see no way of making this generic. So, we are going to rewrite the code using
// CustomOp's operands/results accessor helpers.
CustomOp cloneWith(CustomOp op, IRMapping &mapper, OperandRange ctrl_qubits)
{
    std::vector<Value> operands;
    std::vector<int32_t> ossV;
    {
        operands.reserve(op->getNumOperands());
        for (auto opValue : op->getOperands())
            operands.push_back(mapper.lookupOrDefault(opValue));

        Attribute ossAttr = op->getAttr(::llvm::StringRef("operand_segment_sizes"));
        ossV = ossAttr.cast<DenseI32ArrayAttr>().asArrayRef();
        ossV[ossV.size()-1] += ctrl_qubits.size();
        std::vector<Value> ctrl_qubits2;
        for (const auto &q: ctrl_qubits) {
            ctrl_qubits2.push_back(mapper.lookupOrDefault(q));
        }
        operands.insert(operands.end(), ctrl_qubits2.begin(), ctrl_qubits2.end());
    }

    std::vector<Type> resultTypes;
    std::vector<int32_t> rssV;
    {
        resultTypes.insert(resultTypes.end(), op->getResultTypes().begin(), op->getResultTypes().end());
        Attribute rssAttr = op->getAttr(::llvm::StringRef("result_segment_sizes"));
        rssV = rssAttr.cast<DenseI32ArrayAttr>().asArrayRef();
        rssV[rssV.size()-1] += ctrl_qubits.size();
        std::vector<Type> ctrl_types;
        for (const auto &q: ctrl_qubits) {
            ctrl_types.push_back(q.getType());
        }
        resultTypes.insert(resultTypes.end(), ctrl_types.begin(), ctrl_types.end());
    }

    NamedAttrList attrs = op->getAttrs();
    attrs.set(::llvm::StringRef("operand_segment_sizes"), DenseI32ArrayAttr::get(op->getContext(), ossV));
    attrs.set(::llvm::StringRef("result_segment_sizes"), DenseI32ArrayAttr::get(op->getContext(), rssV));

    SmallVector<Block*, 2> successors;
    {
        successors.reserve(op->getNumSuccessors());
        for (Block *successor : op->getSuccessors())
            successors.push_back(mapper.lookupOrDefault(successor));
    }

    CustomOp newOp = dyn_cast<CustomOp>(op->create(op->getLoc(), op->getName(), resultTypes,
        operands, attrs.getDictionary(op->getContext()), op->getPropertiesStorage(), successors,
        op->getNumRegions()));

    for (const auto &[qr, qs] : zip(newOp.getOutQubits(), op.getOutQubits()))
        mapper.map(qs, qr);
    for (const auto &[qr, qs] : zip(newOp.getOutCtrlQubits(), ctrl_qubits))
        mapper.map(qs, qr);

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
        for (auto &i : ctrl.getRegion().front()) {
            if (isa<QuantumDialect>(i.getDialect())) {
                LLVM_DEBUG(dbgs() << "quantum operation: " << i << "\n");
                if (YieldOp yield = dyn_cast<YieldOp>(i)) {
                    for (const auto &v : yield.getOperands())
                        outputs.push_back(mapping.lookup(v));
                }
                else if (CustomOp custom = dyn_cast<CustomOp>(i)) {
                    rewriter.insert(cloneWith(custom, mapping, ctrl.getCtrlQubits()));
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
