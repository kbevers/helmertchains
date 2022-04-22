import numpy as np
import pyproj

def build_rot_matrix(rx: float, ry: float, rz: float):
    """Construct rotation matrix with correctly scaled rotation parameters."""
    arcsec2rad: float = lambda arcsec: np.deg2rad(arcsec) / 3600.0

    rx =  arcsec2rad(rx)
    ry =  arcsec2rad(ry)
    rz =  arcsec2rad(rz)

    return np.array([
        [1,    -rz,  ry],
        [ rz,    1, -rx],
        [-ry,  rx,    1],
    ])

# Parameters for transformation 1
x1 =    0.041
y1 =    0.041
z1 =   -0.049
s1 =   -0.00158
rx1 =   0.000891
ry1 =   0.00539
rz1 =  -0.008772

# Parameters for transformation 2
x2 =   0.054
y2 =  -0.014
z2 =   0.292
s2 =   0.00242
rx2 =  0.000521
ry2 =  0.00923
rz2 = -0.004592

# Set up scalars, vectors and matrices
c1 = 1 + s1*1e-6
c2 = 1 + s2*1e-6

T1 = np.array([x1, y1, z1])
T2 = np.array([x2, y2, z2])

R1 = build_rot_matrix(rx1, ry1, rz1)
R2 = build_rot_matrix(rx2, ry2, rz2)


# Inputer coordinate (12.0E,55.0N)
Pa = np.array([3586469.6568, 762327.6588, 5201383.5231])

# Two consecutive helmerts
Pb = T1 + c1*R1.dot(Pa)
Pc = T2 + c2*R2.dot(Pb)

# Combined helmert parameters
c3 = c1*c2
T3 = T2 + c2*R2.dot(T1)
R3 = R2@R1

Pd = T3 + c3*R3.dot(Pa)

# Verify that the results are the same, if they are the math is correct
if not np.isclose(Pc, Pd).all():
    raise SystemExit("Math verification failed!")

# Now check results against PROJ
pipeline = f"""
    +proj=pipeline +step
    +proj=helmert +x={x1} +y={y1} +z={z1}
                  +rx={rx1} +ry={ry1} +rz={rz1}
                  +s={s1}  +convention=position_vector
                  +step
    +proj=helmert +x={x2} +y={y2} +z={z2}
                  +rx={rx2} +ry={ry2} +rz={rz2}
                  +s={s2} +convention=position_vector
"""
helmert = pyproj.transformer.Transformer.from_pipeline(pipeline)
Pe = helmert.transform(Pa[0], Pa[1], Pa[2])

# Verify that the results are the same, if they are my math is the same as PROJ's
if not np.isclose(Pd, Pe).all():
    raise SystemExit("PROJ verification failed!")

print("Test succeeded!")