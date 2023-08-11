# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Additional custom MLIR bindings not generated by us or JAX are provided here."""

import jaxlib.mlir.ir as ir
from jaxlib.mlir.dialects._ods_common import (
    get_op_result_or_value,
    get_op_results_or_values,
)


class TensorExtractOp(ir.OpView):
    OPERATION_NAME = "tensor.extract"

    _ODS_REGIONS = (0, True)

    def __init__(self, result, tensor, indices, *, loc=None, ip=None):
        operands = [get_op_result_or_value(tensor)]
        operands.extend(get_op_results_or_values(indices))
        super().__init__(
            self.build_generic(
                attributes={},
                results=[result],
                operands=operands,
                successors=None,
                regions=None,
                loc=loc,
                ip=ip,
            )
        )
