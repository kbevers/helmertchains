"""Fun with Helmert transformations."""

from enum import Enum
from numpy import array, rad2deg, deg2rad
from math import isclose


class Observation:
    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
        epoch: float = 0.0,
        stn_vel: bool = False,
    ) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.epoch = epoch
        self.stn_vel = stn_vel

    def position(self, pos: array = None) -> array:
        if pos is None:
            return array([self.x, self.y, self.z])
        else:
            self.x = pos[0]
            self.y = pos[1]
            self.z = pos[2]
            return array([self.x, self.y, self.z])

    def velocity(self, vel: array = None) -> array:
        if vel is None:
            return array([self.vx, self.vy, self.vz])
        else:
            self.vx = vel[0]
            self.vy = vel[1]
            self.vz = vel[2]
            self.stn_vel = True
            return array([self.vx, self.vy, self.vz])

    def observation(self, obs: array = None) -> array:
        if obs is None:
            return array([self.x, self.y, self.z, self.epoch])
        else:
            self.x = obs[0]
            self.y = obs[1]
            self.z = obs[2]
            self.epoch = obs[3]
            return array([self.x, self.y, self.z, self.epoch])

    def __repr__(self) -> str:
        return (
            f"Observation(x={self.x}, y={self.y}, z={self.z}, "
            f"vx=(self.vx), vy=(self.vy), vz=(self.vz), "
            f"epoch={self.epoch}, stn_vel={self.stn_vel})"
        )

    def __str__(self) -> str:
        string = f"X={self.x:13.5f} m    Y={self.y:13.5f} m    Z={self.z:13.5f} m  "
        string += f"Epoch: {self.epoch:8.3f} yr  "
        if self.stn_vel:
            string += f"\nVX={self.vx:12.5f} m/yr VY={self.vy:12.5f} m/yr "
            string += f"VZ={self.vz:12.5f} m/yr  "
        return string

    def __add__(self: "Observation", obs2: "Observation") -> "Observation":
        obs1 = self
        x = obs1.x + obs2.x
        y = obs1.y + obs2.y
        z = obs1.z + obs2.z
        epoch = obs1.epoch + obs2.epoch
        return Observation(x, y, z, epoch)

    def __sub__(self: "Observation", obs2: "Observation") -> "Observation":
        obs1 = self
        x = obs1.x - obs2.x
        y = obs1.y - obs2.y
        z = obs1.z - obs2.z
        epoch = obs1.epoch - obs2.epoch
        return Observation(x, y, z, epoch)


class Convention(Enum):
    """Conventions to choose between for the Helmert transform."""

    POSITION_VECTOR = 1
    COORDINATE_FRAME = 2


class Helmert:
    """
    14 parameter Helmert transformation class.

    Class to express 14 parameter Helmert transformations and chain them together to
    concatenate multiple 14 parameter Helmert transformations into a single 14 parameter
    Helmert transformation.
    """

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
        vel_model: bool = False,
        convention: Convention = Convention.POSITION_VECTOR,
    ) -> None:
        """
        Create an instance of the Helmert class.

        Parameters
        ----------
        x : float, optional
            Translation in meters along the X axis. The default is 0.0.
        y : float, optional
            Translation in meters along the Y axis. The default is 0.0.
        z : float, optional
            Translation in meters along the Z axis. The default is 0.0.
        s : float, optional
            Scaling of the system in parts per million. The default is 0.0.
        rx : float, optional
            Rotation around the X axis in arcseconds. The default is 0.0.
        ry : float, optional
            Rotation around the Y axis in arcseconds. The default is 0.0.
        rz : float, optional
            Rotation around the Z axis in arcseconds. The default is 0.0.
        dx : float, optional
            Translationrate in meters per year along the X axis. The default is 0.0.
        dy : float, optional
            Translationrate in meters per year along the Y axis. The default is 0.0.
        dz : float, optional
            Translationrate in meters per year along the Z axis. The default is 0.0.
        ds : float, optional
            Scalingrate of the system in parts per millionper year. The default is 0.0.
        drx : float, optional
            Rotationrate around the X axis in arcseconds per year. The default is 0.0.
        dry : float, optional
            Rotationrate around the Y axis in arcseconds per year. The default is 0.0.
        drz : float, optional
            Rotationrate around the Z axis in arcseconds per year. The default is 0.0.
        ref_epoch : float, optional
            Reference epoch for the transformation. The default is 0.0.
        vel_model : bool, optional
            Indicates that the transformation is a velocity model and i.e. a
            transformation in time e.g. a plate motion model. The default is false.
        convention : Convention, optional
            The default is Convention.POSITION_VECTOR.

        Returns
        -------
        None

        """
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
        self.vel_model = vel_model
        self.convention = convention

        self.c = 1 + self.s * 1e-6
        self.dc = self.ds * 1e-6

        self.T = array([self.x, self.y, self.z])
        self.dT = array([self.dx, self.dy, self.dz])
        self.R = self._build_rot_matrix(self.rx, self.ry, self.rz)
        self.dR = self._build_drot_matrix(self.drx, self.dry, self.drz)

    def __repr__(self) -> str:
        """
        Representation of the Helmert object.

        Returns
        -------
        str
            Human readeable version of the Helmert object.

        """
        return (
            f"Helmert(x={self.x}, y={self.y}, z={self.z}, s={self.s}, "
            f"dx={self.dx}, dy={self.dy}, dz={self.dz}, ds={self.ds}, "
            f"rx={self.rx}, ry={self.ry}, rz={self.rz}, "
            f"drx={self.drx}, dry={self.dry}, drz={self.drz}, "
            f"ref_epoch={self.ref_epoch}, vel_model={self.vel_model}, "
            f"convention={self.convention})"
        )

    def __str__(self) -> str:
        """
        Pretty printed representation with units of the Helmert object.

        Returns
        -------
        str
            Print the Helmert object elemets in a nicely formated way with fairly clear
            indication of elements and their units.

        """
        return (
            f" Tx: {self.x:10.7f} m          "
            f" Ty: {self.y:10.7f} m          "
            f" Tz: {self.z:10.7f} m\n"
            f"dTx: {self.dx:10.7f} m/yr       "
            f"dTy: {self.dy:10.7f} m/yr       "
            f"dTz: {self.dz:10.7f} m/yr\n"
            f" Rx: {self.rx:10.7f} arcsec     "
            f" Ry: {self.ry:10.7f} arcsec     "
            f" Rz: {self.rz:10.7f} arcsec\n"
            f"dRx: {self.drx:10.7f} arcsec/yr  "
            f"dRy: {self.dry:10.7f} arcsec/yr  "
            f"dRz: {self.drz:10.7f} arcsec/yr\n"
            f"  s:  {self.s:11.9f} ppm      "
            f" ds:  {self.ds:11.9f} ppm/yr\n"
            f"Ref epoch: {self.ref_epoch:.3f} yr  "
            f"Velocity model: {self.vel_model}  "
            f"Convention: {self.convention}"
        )

    def __add__(self: "Helmert", H2: "Helmert") -> "Helmert":
        """
        Chain two Helmert transformations together: H3 = H1 + H2.

        Parameters
        ----------
        H2 : "Helmert"
            Helmert transformation to chained together with the Helmert transformation
            before the +.

        Raises
        ------
        UserWarning
            Warning to indicate that the two Helmert transformations to be chained
            together does not have the same reference epoch. Use the set_target_epoch
            function to express one of the Helmert transformations at the other Helmert
            transformations reference epoch.

        Returns
        -------
        "Helmert"
            The combined Helmert transformation.

        """
        H1 = self
        vel_model = False

        if not isclose(H1.ref_epoch, H2.ref_epoch, abs_tol=2e-3):
            raise UserWarning(
                "Helmert transformations can only be chained if they have the same "
                "reference epoch (i.e. within one day). Your epochs differ by: "
                f"{H1.ref_epoch:.3f} - {H2.ref_epoch:.3f} = "
                f"{H1.ref_epoch - H2.ref_epoch:.3f} years or "
                f"~{(H1.ref_epoch - H2.ref_epoch) * 365.0:.1f} days.\n"
                "Consider using set_target_epoch to shift one of the Helmert "
                "transformations to the reference epoch of the other Helmert."
            ) from None

        # determine scaling, translations and rotations
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

        if H1.vel_model or H2.vel_model:
            vel_model = True
            if not H1.vel_model:
                H1.dx = 0.0
                H1.dy = 0.0
                H1.dz = 0.0
                H1.drx = 0.0
                H1.dry = 0.0
                H1.drz = 0.0
                H1.dc = 0.0
                H1.dT = array([H1.dx, H1.dy, H1.dz])
                H1.dR = self._build_drot_matrix(H1.drx, H1.dry, H1.drz)
            if not H2.vel_model:
                H2.dx = 0.0
                H2.dy = 0.0
                H2.dz = 0.0
                H2.drx = 0.0
                H2.dry = 0.0
                H2.drz = 0.0
                H2.dc = 0.0
                H2.dT = array([H2.dx, H2.dy, H2.dz])
                H2.dR = self._build_drot_matrix(H2.drx, H2.dry, H2.drz)

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

        return Helmert(
            x,
            y,
            z,
            s,
            rx,
            ry,
            rz,
            dx,
            dy,
            dz,
            ds,
            drx,
            dry,
            drz,
            H1.ref_epoch,
            vel_model,
            self.convention,
        )

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

    def set_target_epoch(self, target_epoch: float) -> "Helmert":
        """
        Express the Helmert transformation at a new reference epoch.

        Parameters
        ----------
        target_epoch : float
            Epoch to express the Helmert transformation at.

        Returns
        -------
        "Helmert"
            Helmert transformation object expressed at target_epoch, i.e.
            ref_epoch=target_epoch.

        """
        dt = target_epoch - self.ref_epoch
        self.x = self.x + self.dx * dt
        self.y = self.y + self.dy * dt
        self.z = self.z + self.dz * dt
        self.s = self.s + self.ds * dt
        self.rx = self.rx + self.drx * dt
        self.ry = self.ry + self.dry * dt
        self.rz = self.rz + self.drz * dt
        self.ref_epoch = target_epoch

        self.c = 1 + self.s * 1e-6
        self.dc = self.ds * 1e-6

        self.T = array([self.x, self.y, self.z])
        self.dT = array([self.dx, self.dy, self.dz])
        self.R = self._build_rot_matrix(self.rx, self.ry, self.rz)
        self.dR = self._build_drot_matrix(self.drx, self.dry, self.drz)
        return self

    def inverse(self) -> "Helmert":
        """
        Change the sign on all elements in the Helmert object.

        Returns
        -------
        "Helmert"
            Invers Helmert object.

        """
        self.x = -self.x
        self.y = -self.y
        self.z = -self.z
        self.s = -self.s
        self.rx = -self.rx
        self.ry = -self.ry
        self.rz = -self.rz
        self.dx = -self.dx
        self.dy = -self.dy
        self.dz = -self.dz
        self.ds = -self.ds
        self.drx = -self.drx
        self.dry = -self.dry
        self.drz = -self.drz

        self.c = 1 + self.s * 1e-6
        self.dc = self.ds * 1e-6

        self.T = array([self.x, self.y, self.z])
        self.dT = array([self.dx, self.dy, self.dz])
        self.R = self._build_rot_matrix(self.rx, self.ry, self.rz)
        self.dR = self._build_drot_matrix(self.drx, self.dry, self.drz)
        return self

    #    def transform(self, P: array, inverse: bool = False) -> array:
    def transform(self, posObs: "Observation", inverse: bool = False) -> "Observation":
        """Transform a cartesian coordinate."""
        posObs_out = Observation()
        dt = posObs.epoch - self.ref_epoch
        if self.vel_model:
            posObs_out.epoch = self.ref_epoch
        else:
            posObs_out.epoch = posObs.epoch
        if inverse:
            posObs_out.position(
                -self.T
                - self.dT * dt
                + 1 / (self.c + self.dc * dt) * self.R.T.dot(posObs.position())
            )
            if posObs.stn_vel:
                posObs_out.stn_vel = True
                posObs_out.velocity(
                    posObs.velocity()
                    - self.dT
                    - self.dc * posObs.position()
                    - self.dR.dot(posObs.position())
                )
        else:
            posObs_out.position(
                self.T
                + self.dT * dt
                + (self.c + self.dc * dt)
                * (self.R + self.dR * dt).dot(posObs.position())
            )
            if posObs.stn_vel:
                posObs_out.stn_vel = True
                posObs_out.velocity(
                    posObs.velocity()
                    + self.dT
                    + self.dc * posObs.position()
                    + self.dR.dot(posObs.position())
                )
        return posObs_out
