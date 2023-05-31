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

// RUN: quantum-opt %s --lower-gradients --debug-only=activity-analysis -o /dev/null 2>&1 | FileCheck %s

// CHECK-LABEL: Running activity analysis on '@noControlFlow'
// CHECK-NEXT: # of active values: 7
// CHECK-DAG: "x"
func.func @noControlFlow(%arg0: f64 {activity.id = "x"}) -> f64 {
    %c0_i64 = arith.constant 0 : i64
    quantum.device ["backend", "lightning.qubit"]
    %qreg = quantum.alloc(1) : !quantum.reg
    %qbit = quantum.extract %qreg[%c0_i64] : !quantum.reg -> !quantum.bit
    // CHECK-DAG: "x^2"
    %mul = arith.mulf %arg0, %arg0 {activity.id = "x^2"} : f64
    // CHECK-DAG: "rx^2"
    %rx = quantum.custom "RX"(%mul) %qbit {activity.id = "rx^2"} : !quantum.bit
    // CHECK-DAG: "qreg0"
    %qreg0 = quantum.insert %qreg[%c0_i64], %rx {activity.id = "qreg0"} : !quantum.reg, !quantum.bit
    // CHECK-DAG: "qbit0"
    %qbit0 = quantum.extract %qreg0[%c0_i64] {activity.id = "qbit0"} : !quantum.reg -> !quantum.bit
    // CHECK-DAG: "obs"
    %obs = quantum.namedobs %qbit0[3] {activity.id = "obs"} : !quantum.obs
    // CHECK-DAG: "expval"
    %expval = quantum.expval %obs {activity.id = "expval"} : f64
    quantum.dealloc %qreg : !quantum.reg
    return %expval : f64
}

// TODO: test cases
// - loops: loop around, iter_arg, free variable
// - both for and while loops
// - if statements: both branches, only one branch
// - requesting activity of more than one argument

func.func @gradCallNoControlFlow(%arg0: f64) -> f64 {
    %0 = gradient.grad "ps" @noControlFlow(%arg0) : (f64) -> f64
    func.return %0 : f64
}

// ---

// CHECK-LABEL: Running activity analysis on '@tensorTypes'
// CHECK-NEXT: # of active values: 2
// CHECK-DAG: "x"
func.func @tensorTypes(%arg0: tensor<3xf64> {activity.id = "x"}) -> tensor<3xf64> {
    // CHECK-DAG: "res"
    %res = math.sin %arg0 {activity.id = "res"} : tensor<3xf64>
    return %res : tensor<3xf64>
}

func.func @gradCallTensorTypes(%arg0: tensor<3xf64>) -> tensor<3x3xf64> {
    %0 = gradient.grad "ps" @tensorTypes(%arg0) : (tensor<3xf64>) -> tensor<3x3xf64>
    return %0 : tensor<3x3xf64>
}

// ---

// CHECK-LABEL: Running activity analysis on '@secondArg'
// CHECK-NEXT: # of active values: 1
// CHECK-NEXT: "y"
func.func @secondArg(%arg0: f64, %arg1: f64 {activity.id = "y"}) -> f64 {
    return %arg1 : f64
}

func.func @gradCallSecondArg(%arg0: f64, %arg1: f64) -> f64 {
    %0 = gradient.grad "ps" @secondArg(%arg0, %arg1) {diffArgIndices = dense<[1]> : tensor<1xindex>} : (f64, f64) -> f64
    return %0 : f64
}

// ---

// CHECK-LABEL: Running activity analysis on '@secondArgConst'
// CHECK-NEXT: # of active values: 0
func.func @secondArgConst(%arg0: f64, %arg1: f64 {activity.id = "y"}) -> f64 {
    return %arg0 : f64
}

func.func @gradCallSecondArgConst(%arg0: f64, %arg1: f64) -> f64 {
    %0 = gradient.grad "ps" @secondArgConst(%arg0, %arg1) {diffArgIndices = dense<[1]> : tensor<1xindex>} : (f64, f64) -> f64
    return %0 : f64
}
