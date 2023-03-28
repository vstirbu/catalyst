from hypothesis import given
from catalyst.synthesis.hypothesis import *
from catalyst.synthesis.pprint import pprint


@given(x=conds())
def test_ptint_conditionals(x):
    pprint(x)

@given(x=forloops())
def test_ptint_forloops(x):
    pprint(x)

@given(x=whileloops())
def test_ptint_whileloops(x):
    pprint(x)
