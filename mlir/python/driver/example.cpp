#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace py = pybind11;

// llvm::ModulePass *createEnzymePass(bool PostOpt = false);

int add(int i, int j)
{
    // auto *enzymePass = createEnzymePass();
    llvm::errs() << "using llvm errs\n";
    return i + j;
}

PYBIND11_MODULE(example_python, m)
{
    m.doc() = "pybind11 example plugin";
    m.def("add", &add, "A function that adds two integers");
}
