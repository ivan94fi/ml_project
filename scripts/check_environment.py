# flake8: noqa
# pylint: disable=all
print("Testing if the project can be imported successfully ...")
try:
    import ml_project

    print("ml_project correctly installed")
except ModuleNotFoundError:
    print(
        "ERROR: cannot import project as python module. Install with 'make install[-dev]'"
    )
    import sys

    sys.exit(1)

print("Testing pytorch-CUDA interaction ...")

import torch

if torch.cuda.is_available():
    print("CUDA is available")
else:
    raise Exception("ERROR: CUDA is not available to PyTorch")

print("Trying to do a matrix multiplication in gpu ...")
a_cpu = torch.rand(20, 20)
b_cpu = torch.rand(20, 20)
a = a_cpu.cuda()
b = b_cpu.cuda()

try:
    # multiply in gpu
    c = a @ b

    # copy result in cpu and test if it is accessible
    _ = c.cpu()[5:10, :]

    # assert that gpu multiply gives same result as cpu multiply
    assert torch.allclose(c.cpu(), a_cpu @ b_cpu)
except Exception as e:
    import sys
    import traceback

    traceback.print_exc()
    print()
    print(
        "ERROR: CUDA is available but the sample multiplication in gpu raised an error"
    )
    sys.exit(1)

print("... done")

print()
print("All checks passed")
