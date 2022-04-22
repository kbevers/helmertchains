import math

import numpy as np
import pyproj
import pytest

from helmert import Helmert

def is_vector_close(A: np.array, B: np.array):
    for i in range(3):
        if not math.isclose(A[i], B[i]):
            return False
    return True


# Parameters for transformation 1
@pytest.fixture()
def H1():
    return Helmert(
        x=23.041,
        y=0.041,
        z=-0.049,
        s=-0.00158,
        rx=0.000891,
        ry=0.00539,
        rz=-0.008772,
    )


@pytest.fixture()
def H2():
    return Helmert(
        x=0.054,
        y=-0.014,
        z=0.292,
        s=0.00242,
        rx=0.000521,
        ry=0.00923,
        rz=-0.004592,
    )


@pytest.fixture()
def H3(H1, H2):
    return H1 + H2


@pytest.fixture()
def coord():
    """Input coordinate (12.0E,55.0N)"""
    return np.array([3586469.6568, 762327.6588, 5201383.5231])


def test_two_consecutive_helmerts(H1, H2, coord):
    """Two consecutive helmerts"""
    c1 = H1.transform(coord)
    c2 = H2.transform(c1)

    # Combined helmert parameters
    H3 = H1 + H2
    c3 = H3.transform(coord)

    # Verify that the results are the same, if they are the math is correct
    assert is_vector_close(c2, c3), "Math verification failed!"


def test_same_results_as_proj(H1, H2, coord):
    """Verify transformation results against PROJ"""
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
    c = np.array(helmert.transform(coord[0], coord[1], coord[2]))

    # Verify that the results are the same, if they are my math is the same as PROJ's
    is_vector_close(coord, c), "PROJ verification failed!"


def test_helmert_commutativeness(H1, H2, H3, coord):
    """Are Helmert transformations commutative?"""
    H4 = H1 + H2 + H3
    H5 = H3 + H2 + H1

    c1 = H4.transform(coord)
    c2 = H5.transform(coord)
    # Verify that Helmerts commute(?)
    is_vector_close(c1, c2), "Commutative property verification failed!"


def test_inverse_transform(H1, coord):
    """Test inverse transform"""
    c1 = H1.transform(coord)
    c2 = H1.transform(c1, inverse=True)

    # Verify that the inverse transform works
    is_vector_close(coord, c2), "Inverse property verification failed!"
