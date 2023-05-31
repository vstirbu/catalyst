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

#pragma once

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace catalyst {
namespace gradient {
class GradOp;
}

using ValueSet = mlir::DenseSet<mlir::Value>;

struct ActivityAnalysis {
    ActivityAnalysis(mlir::Operation *op);

    // There isn't a way to communicate a failure from the analysis manager.
    bool valid;
    /// Store the set of active values for each grad invocation. The set of active values can be
    /// different across grad invocations of the same function because different gradients can be
    /// requested.
    mlir::DenseMap<gradient::GradOp, ValueSet> activeValues;
};

void debugPrintValue(ValueSet value);

} // namespace catalyst
