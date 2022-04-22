import math

import numpy as np
import pyproj

from helmert import Helmert

np.set_printoptions(precision=4)

def assert_vector_closeness(A: np.array, B: np.array, errmsg: str):
    for i in range(3):
        if not math.isclose(A[i], B[i]):
            raise SystemExit(errmsg)

# Parameters for transformation 1
H1 = Helmert(
    x =    23.041,
    y =    0.041,
    z =   -0.049,
    s =   -0.00158,
    rx =   0.000891,
    ry =   0.00539,
    rz =  -0.008772,
)

# Parameters for transformation 2
H2 = Helmert(
    x =   0.054,
    y =  -0.014,
    z =   0.292,
    s =   0.00242,
    rx =  0.000521,
    ry =  0.00923,
    rz = -0.004592,
)


# Inputer coordinate (12.0E,55.0N)
Pa = np.array([3586469.6568, 762327.6588, 5201383.5231])

# Two consecutive helmerts
Pb = H1.transform(Pa)
Pc = H2.transform(Pb)

# Combined helmert parameters
H3 = H1+H2
Pd = H3.transform(Pa)


# Verify that the results are the same, if they are the math is correct
assert_vector_closeness(Pc, Pd, "Math verification failed!")    

# Now check results against PROJ
pipeline = f"""
    +proj=pipeline +step
    +proj=helmert +x={H1.x} +y={H1.y} +z={H1.z}
                  +rx={H1.rx} +ry={H1.ry} +rz={H1.rz}
                  +s={H1.s}  +convention=position_vector
                  +step
    +proj=helmert +x={H2.x} +y={H2.y} +z={H2.z}
                  +rx={H2.rx} +ry={H2.ry} +rz={H2.rz}
                  +s={H2.s} +convention=position_vector
"""
helmert = pyproj.transformer.Transformer.from_pipeline(pipeline)
Pe = np.array(helmert.transform(Pa[0], Pa[1], Pa[2]))

# Verify that the results are the same, if they are my math is the same as PROJ's
assert_vector_closeness(Pd, Pe, "PROJ verification failed!")


# Are Helmert transformations commutative?
H4 = H1+H2+H3
H5 = H3+H2+H1

Pf = H4.transform(Pa)
Pg = H5.transform(Pa)
# Verify that Helmerts commute(?)
assert_vector_closeness(Pf, Pg, "Commutative property verification failed!")


# Test inverse transform
Ph = H1.transform(Pa)
Pi = H1.transform(Ph, inverse=True)

# Verify that the inverse transform works
assert_vector_closeness(Pa, Pi, "Inverse property verification failed!")


print("Test succeeded!")
print()
print(f"{Pc=}")
print(f"{Pd=}")
print(f"{Pe=}")
print()
print(f"{Pf=}")
print(f"{Pg=}")
print()
print(f"{Pa=}, start")
print(f"{Ph=}, frem")
print(f"{Pi=}, tilbage")
