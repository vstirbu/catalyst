from catalyst import qjit_ast
import pennylane as qml

n_qubits = 2

dev = qml.device("lightning.qubit", wires=n_qubits)


def test_basic():
    @qjit_ast
    @qml.qnode(dev)
    def simple():
        qml.CNOT((0, 1))
        qml.PauliX(0)
        qml.Rot(3.43, 44, 44.3, wires=(0,))
        return qml.expval(qml.PauliZ(0))

    simple()

    expected_snapshot = """
module {
  func.func @simple() -> f64 {
    %cst = arith.constant 4.430000e+01 : f64
    %cst_0 = arith.constant 4.400000e+01 : f64
    %cst_1 = arith.constant 3.430000e+00 : f64
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3:2 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit
    %4 = quantum.insert %0[ 0], %3#0 : !quantum.reg, !quantum.bit
    %5 = quantum.insert %4[ 1], %3#1 : !quantum.reg, !quantum.bit
    %6 = quantum.extract %5[ 0] : !quantum.reg -> !quantum.bit
    %7 = quantum.custom "PauliX"() %6 : !quantum.bit
    %8 = quantum.insert %5[ 0], %7 : !quantum.reg, !quantum.bit
    %9 = quantum.extract %8[ 0] : !quantum.reg -> !quantum.bit
    %10 = quantum.custom "Rot"(%cst_1, %cst_0, %cst) %9 : !quantum.bit
    %11 = quantum.insert %8[ 0], %10 : !quantum.reg, !quantum.bit
    %12 = quantum.extract %11[ 0] : !quantum.reg -> !quantum.bit
    %13 = quantum.namedobs %12[ PauliZ] : !quantum.obs
    %14 = quantum.expval %13 : f64
    return %14 : f64
  }
}
"""

    assert simple.mlir.strip() == expected_snapshot.strip()


def test_scalar_param():
    @qjit_ast
    @qml.qnode(dev)
    def scalar_param(x: float):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0))

    expected_snapshot = """
module {
  func.func @scalar_param(%arg0: f64) -> f64 {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.custom "RX"(%arg0) %1 : !quantum.bit
    %3 = quantum.insert %0[ 0], %2 : !quantum.reg, !quantum.bit
    %4 = quantum.extract %3[ 0] : !quantum.reg -> !quantum.bit
    %5 = quantum.namedobs %4[ PauliZ] : !quantum.obs
    %6 = quantum.expval %5 : f64
    return %6 : f64
  }
}
    """

    scalar_param(0.3)
    assert scalar_param.mlir.strip() == expected_snapshot.strip()


def test_if_else():
    @qjit_ast
    @qml.qnode(dev)
    def ifelse(x: float, n: int):
        if n % 2 == 0:
            qml.RX(x, wires=0)
        elif x > 4:
            qml.RZ(x - 2.3, 1)
        else:
            qml.RY(x * 2, 0)
        return qml.expval(qml.PauliZ(0))

    ifelse(5.4, 4)

    expected_snapshot = """
module {
  func.func @ifelse(%arg0: f64, %arg1: i64) -> f64 {
    %cst = arith.constant 2.000000e+00 : f64
    %cst_0 = arith.constant 4.000000e+00 : f64
    %cst_1 = arith.constant 2.300000e+00 : f64
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = arith.remsi %arg1, %c2_i64 : i64
    %2 = arith.cmpi eq, %1, %c0_i64 : i64
    %3 = scf.if %2 -> (!quantum.reg) {
      %7 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
      %8 = quantum.custom "RX"(%arg0) %7 : !quantum.bit
      %9 = quantum.insert %0[ 0], %8 : !quantum.reg, !quantum.bit
      scf.yield %9 : !quantum.reg
    } else {
      %7 = arith.cmpf ogt, %arg0, %cst_0 : f64
      %8 = scf.if %7 -> (!quantum.reg) {
        %9 = arith.subf %arg0, %cst_1 : f64
        %10 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
        %11 = quantum.custom "RZ"(%9) %10 : !quantum.bit
        %12 = quantum.insert %0[ 1], %11 : !quantum.reg, !quantum.bit
        scf.yield %12 : !quantum.reg
      } else {
        %9 = arith.mulf %arg0, %cst : f64
        %10 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
        %11 = quantum.custom "RY"(%9) %10 : !quantum.bit
        %12 = quantum.insert %0[ 0], %11 : !quantum.reg, !quantum.bit
        scf.yield %12 : !quantum.reg
      }
      scf.yield %8 : !quantum.reg
    }
    %4 = quantum.extract %3[ 0] : !quantum.reg -> !quantum.bit
    %5 = quantum.namedobs %4[ PauliZ] : !quantum.obs
    %6 = quantum.expval %5 : f64
    return %6 : f64
  }
}
    """
    assert ifelse.mlir.strip() == expected_snapshot.strip()

def test_range_for_loop():
    @qjit_ast
    @qml.qnode(dev)
    def range_for(x: float, n: int):
        for i in range(n):
            qml.RX(x * i, wires=i)
        return qml.expval(qml.PauliZ(0))

    range_for(5.4, 4)
