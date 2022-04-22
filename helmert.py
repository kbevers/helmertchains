"""
Fun with Helmert transformations.
"""
from enum import Enum

import numpy as np


class Convention(Enum):
    POSITION_VECTOR = 1
    COORDINATE_FRAME = 2


class Helmert:
    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        s: float = 0.0,
        rx: float = 0.0,
        ry: float = 0.0,
        rz: float = 0.0,
        convention: Convention = Convention.POSITION_VECTOR,
    ) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.s = s
        self.rx = rx
        self.ry = ry
        self.rz = rz
        self.convention = convention

        self.c = 1 + self.s * 1e-6

        self.T = np.array([self.x, self.y, self.z])
        self.R = self._build_rot_matrix(self.rx, self.ry, self.rz)

    def __add__(self: "Helmert", H2: "Helmert") -> "Helmert":
        # determine scaling, translations and rotations
        H1 = self

        c = H1.c * H2.c
        T = H2.T + H2.c * H2.R.dot(H1.T)
        R = H2.R @ H1.R

        # decompose the above
        s = (c - 1) * 1e6
        x = T[0]
        y = T[1]
        z = T[2]

        rad2arcsec: float = lambda rad: np.rad2deg(rad * 3600.0)
        rx = rad2arcsec(R[2][1])
        ry = rad2arcsec(R[0][2])
        rz = rad2arcsec(R[1][0])

        return Helmert(x, y, z, s, rx, ry, rz)

    def _build_rot_matrix(self, rx: float, ry: float, rz: float):
        """Construct rotation matrix with correctly scaled parameters."""
        arcsec2rad: float = lambda arcsec: np.deg2rad(arcsec) / 3600.0

        rx = arcsec2rad(rx)
        ry = arcsec2rad(ry)
        rz = arcsec2rad(rz)

        R = np.array(
            [
                [1, -rz, ry],
                [rz, 1, -rx],
                [-ry, rx, 1],
            ]
        )

        if self.convention == Convention.POSITION_VECTOR:
            return R

        # If it's not position vector convention is must be coordinate frame
        # which we get by transposing the rotation matrix
        return R.T

    def transform(self, P: np.array, inverse: bool = False) -> np.array:
        """
        Transform a cartesian coordinate.
        """
        if inverse:
            return -self.T + 1 / self.c * self.R.T.dot(P)

        return self.T + self.c * self.R.dot(P)
