"""
Fun with Helmert transformations.
"""
from enum import Enum

from numpy import array, rad2deg, deg2rad


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
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        ds: float = 0.0,
        drx: float = 0.0,
        dry: float = 0.0,
        drz: float = 0.0,
        ref_epoch: float = 0.0,
        convention: Convention = Convention.POSITION_VECTOR,
    ) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.s = s
        self.rx = rx
        self.ry = ry
        self.rz = rz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.ds = ds
        self.drx = drx
        self.dry = dry
        self.drz = drz
        self.ref_epoch = ref_epoch
        self.convention = convention

        self.c = 1 + self.s * 1e-6
        self.dc = self.ds * 1e-6

        self.T = array([self.x, self.y, self.z])
        self.dT = array([self.dx, self.dy, self.dz])
        self.R = self._build_rot_matrix(self.rx, self.ry, self.rz)
        self.dR = self._build_drot_matrix(self.drx, self.dry, self.drz)

    def __repr__(self) -> "Helmert":
        return(f"Helmert(x={self.x}, y={self.y}, z={self.z}, s={self.s}, "
               f"dx={self.dx}, dy={self.dy}, dz={self.dz}, ds={self.ds}, "
               f"rx={self.rx}, ry={self.ry}, rz={self.rz}, "
               f"drx={self.drx}, dry={self.dry}, drz={self.drz}, "
               f"ref_epoch={self.ref_epoch}, convention={self.convention})")
    
    def __str__(self) -> str:
        return(f" Tx: {self.x:10.7f} m        Ty: {self.y:10.7f} m       Tz: {self.z:10.7f} m\n"
               f"dTx: {self.dx:10.7f} m/yr   dTy: {self.dy:10.7f} m/yr   dTz: {self.dz:10.7f} m/yr\n"
               f" Rx: {self.rx:10.7f} arcsec      Ry: {self.ry:10.7f} arcsec      Rz: {self.rz:10.7f} arcsec\n"
               f"dRx: {self.drx:10.7f} arcsec/yr dRy: {self.dry:10.7f} arcsec/yr dRz: {self.drz:10.7f} arcsec/yr\n"
               f"s: {self.s:11.9f} ppm     ds: {self.ds:11.9f} ppm/yr\n"
               f"ref epoch:{self.ref_epoch:.3f} yr     Convention: {self.convention}")

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

        rad2arcsec: float = lambda rad: rad2deg(rad * 3600.0)
        rx = rad2arcsec(R[2][1])
        ry = rad2arcsec(R[0][2])
        rz = rad2arcsec(R[1][0])
        
        # determine scaling rate, translation rates, and rotation rates
        dc = H2.dc * H1.c + H2.c * H1.dc
        dT = H2.dT + (H2.dc * H2.R + H2.c * H2.dR).dot(H1.T) + (H2.c * H2.R).dot(H1.dT)
        dR = H2.dR @ H1.R + H2.R @ H1.dR

        # decompose the above
        ds = dc * 1e6
        dx = dT[0]
        dy = dT[1]
        dz = dT[2]

        drx = rad2arcsec(dR[2][1])
        dry = rad2arcsec(dR[0][2])
        drz = rad2arcsec(dR[1][0])

        return Helmert(x, y, z, s, rx, ry, rz, dx, dy, dz, ds, drx, dry, drz)

    def _build_rot_matrix(self, rx: float, ry: float, rz: float):
        """Construct rotation matrix with correctly scaled parameters."""
        arcsec2rad: float = lambda arcsec: deg2rad(arcsec) / 3600.0

        rx = arcsec2rad(rx)
        ry = arcsec2rad(ry)
        rz = arcsec2rad(rz)

        R = array(
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

    def _build_drot_matrix(self, drx: float, dry: float, drz: float):
        """Construct rotation matrix with correctly scaled parameters."""
        arcsec2rad: float = lambda arcsec: deg2rad(arcsec) / 3600.0

        drx = arcsec2rad(drx)
        dry = arcsec2rad(dry)
        drz = arcsec2rad(drz)

        dR = array(
            [
                [0, -drz, dry],
                [drz, 0, -drx],
                [-dry, drx, 0],
            ]
        )

        if self.convention == Convention.POSITION_VECTOR:
            return dR

        # If it's not position vector convention is must be coordinate frame
        # which we get by transposing the rotation matrix
        return dR.T
    
    def transform(self, P: array, inverse: bool = False) -> array:
        """
        Transform a cartesian coordinate.
        """
        P_out = P
        if inverse:
            P_out[:3] = -self.T + 1 / self.c * self.R.T.dot(P[:3])

        P_out[:3] = self.T + self.c * self.R.dot(P[:3])
        return P_out