#!/usr/bin/env python3
"""Create an Isaac Sim Franka FR3 + Robotiq 2F-85 environment."""

from __future__ import annotations

import argparse
import math
import os
from typing import Iterable, Optional

import numpy as np
from isaacsim import SimulationApp


def _parse_vec3(value: str) -> tuple[float, float, float]:
    parts = [float(v.strip()) for v in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated numbers.")
    return parts[0], parts[1], parts[2]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a Franka FR3 + Robotiq 2F-85 USD scene.")
    parser.add_argument("--headless", action="store_true", help="No UI.")
    parser.add_argument("--asset-root", default=None, help="Asset root.")
    parser.add_argument("--franka-usd", default=None, help="Franka USD.")
    parser.add_argument("--robotiq-usd", default=None, help="Robotiq USD.")
    parser.add_argument("--franka-prim", default="/World/Franka", help="Franka prim.")
    parser.add_argument("--robotiq-prim", default="/World/Robotiq2F85", help="Robotiq prim.")
    parser.add_argument("--franka-ee-name", default="", help="Franka EE link.")
    parser.add_argument("--robotiq-base-name", default="", help="Robotiq base link.")
    parser.add_argument("--adapter-xyz", type=_parse_vec3, default=(0.0, 0.0, 0.0), help="EE offset xyz.")
    parser.add_argument("--adapter-rpy-deg", type=_parse_vec3, default=(0.0, 0.0, 0.0), help="EE offset rpy deg.")
    parser.add_argument("--keep-franka-hand", action="store_true", help="Keep stock hand.")
    parser.add_argument("--save-usd", default=os.path.abspath("franka_robotiq_2f85_env.usd"), help="Output USD.")
    parser.add_argument("--play", action="store_true", help="Run sim steps.")
    parser.add_argument("--stay-open", action="store_true", help="Keep app open.")
    parser.add_argument("--close-after-save", action="store_true", help="Close after save.")
    parser.add_argument("--merge-gripper-articulation", action="store_true", default=True, help="Remove gripper articulation root.")
    parser.add_argument("--steps", type=int, default=600, help="Step count.")
    return parser


class Franka2F85:
    """Isaac Sim environment for a Franka FR3 with a Robotiq 2F-85 gripper."""

    FRANKA_USD_CANDIDATES = [
        "Isaac/Robots/FrankaRobotics/FrankaFR3/fr3.usd",
        "Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        "Isaac/Robots/Franka/franka.usd",
        "Robots/FrankaRobotics/FrankaFR3/fr3.usd",
        "Robots/FrankaRobotics/FrankaPanda/franka.usd",
        "Robots/Franka/franka.usd",
        "FrankaRobotics/FrankaFR3/fr3.usd",
        "FrankaRobotics/FrankaPanda/franka.usd",
        "Franka/franka.usd",
    ]
    ROBOTIQ_USD_CANDIDATES = [
        "Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_base.usd",
        "Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd",
        "Robots/Robotiq/2F-85/Robotiq_2F_85_base.usd",
        "Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd",
        "Robotiq/2F-85/Robotiq_2F_85_base.usd",
        "Robotiq/2F-85/Robotiq_2F_85_edit.usd",
        "Isaac/Samples/Rigging/Gripper/Robotiq 2F-85/Robotiq_2F_85_base.usd",
        "Samples/Rigging/Gripper/Robotiq 2F-85/Robotiq_2F_85_base.usd",
    ]

    def __init__(
        self,
        headless: bool = False,
        asset_root: Optional[str] = None,
        franka_usd: Optional[str] = None,
        robotiq_usd: Optional[str] = None,
        franka_prim: str = "/World/Franka",
        robotiq_prim: str = "/World/Robotiq2F85",
        franka_ee_name: str = "",
        robotiq_base_name: str = "",
        adapter_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
        adapter_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        keep_franka_hand: bool = False,
        save_usd: str = os.path.abspath("franka_robotiq_2f85_env.usd"),
        merge_gripper_articulation: bool = True,
        play: bool = False,
        steps: int = 600,
    ) -> None:
        self.headless = headless
        self.asset_root = asset_root
        self.franka_usd = franka_usd
        self.robotiq_usd = robotiq_usd
        self.franka_prim_path = franka_prim
        self.robotiq_prim_path = robotiq_prim
        self.franka_ee_name = franka_ee_name
        self.robotiq_base_name = robotiq_base_name
        self.adapter_xyz = adapter_xyz
        self.adapter_rpy_deg = adapter_rpy_deg
        self.keep_franka_hand = keep_franka_hand
        self.save_usd = save_usd
        self.merge_gripper_articulation = merge_gripper_articulation
        self.sim_app = SimulationApp({"headless": self.headless})

        import omni.client as omni_client
        import omni.physics.tensors as omni_physics_tensors
        import omni.timeline as omni_timeline
        import omni.usd as omni_usd
        from isaacsim.core.api.physics_context import PhysicsContext
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.utils.stage import get_current_stage_id
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics

        self.Gf = Gf
        self.PhysxSchema = PhysxSchema
        self.Sdf = Sdf
        self.UsdGeom = UsdGeom
        self.UsdLux = UsdLux
        self.UsdPhysics = UsdPhysics
        self.PhysicsContext = PhysicsContext
        self.SingleArticulation = SingleArticulation
        self.SimulationManager = SimulationManager
        self.ArticulationAction = ArticulationAction
        self.add_reference_to_stage = add_reference_to_stage
        self.get_current_stage_id = get_current_stage_id
        self.get_assets_root_path = get_assets_root_path
        self.omni_client = omni_client
        self.omni_physics_tensors = omni_physics_tensors
        self.omni_timeline = omni_timeline
        self.omni_usd = omni_usd

        self.stage = None
        self.ee_prim = None
        self.robotiq_base_prim = None
        self.franka = None
        self.robotiq = None
        self._running_forever = False
        self.franka_usd, self.robotiq_usd, self.saved_path = self.create_scene()
        self._setup_articulation_handles()

        if play:
            self.step(steps=steps, play_timeline=True)

    def _path_exists(self, path: str) -> bool:
        if path.startswith(("omniverse://", "http://", "https://")):
            result, _ = self.omni_client.stat(path)
            return result == self.omni_client.Result.OK
        return os.path.exists(path)

    def _first_existing_path(self, paths: Iterable[str], label: str) -> str:
        checked = []
        for path in paths:
            checked.append(path)
            if self._path_exists(path):
                print("path =", path)
                return path

        joined = "\n  - ".join(checked)
        raise FileNotFoundError(f"Could not find {label} USD. Checked:\n  - {joined}")

    def _resolve_asset_root(self) -> str:
        if self.asset_root:
            return self.asset_root.rstrip("/")

        root = self.get_assets_root_path()
        if root:
            return root.rstrip("/")

        raise RuntimeError("Could not resolve Isaac Sim asset root.")

    def _resolve_robot_usds(self) -> tuple[str, str]:
        asset_root = self._resolve_asset_root()
        root = asset_root.rstrip("/")
        franka_paths = [f"{root}/{path.lstrip('/')}" for path in self.FRANKA_USD_CANDIDATES]
        robotiq_paths = [f"{root}/{path.lstrip('/')}" for path in self.ROBOTIQ_USD_CANDIDATES]
        franka_usd = self.franka_usd or self._first_existing_path(franka_paths, "Franka")
        robotiq_usd = self.robotiq_usd or self._first_existing_path(robotiq_paths, "Robotiq 2F-85")
        return franka_usd, robotiq_usd

    def _quat_from_rpy_deg(self, rpy_deg: tuple[float, float, float]):
        roll, pitch, yaw = (math.radians(value) for value in rpy_deg)
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return self.Gf.Quatf(float(w), self.Gf.Vec3f(float(x), float(y), float(z)))

    def _matrix_from_pose(self, xyz: tuple[float, float, float], quat):
        q = self.Gf.Quatd(
            float(quat.GetReal()),
            self.Gf.Vec3d(*[float(value) for value in quat.GetImaginary()]),
        )
        matrix = self.Gf.Matrix4d(1.0)
        matrix.SetRotate(q)
        matrix.SetTranslateOnly(self.Gf.Vec3d(*xyz))
        return matrix

    def _find_child_by_name(self, root_path: str, preferred_names: Iterable[str]):
        root = self.stage.GetPrimAtPath(root_path)
        if not root:
            return None

        names = [name for name in preferred_names if name]
        for name in names:
            for prim in root.GetAllChildren():
                if prim.GetName() == name:
                    return prim

        for prim in root.GetAllChildren():
            if prim.GetName() in names:
                return prim

        for prim in root.GetAllChildren():
            for child in prim.GetAllChildren():
                if child.GetName() in names:
                    return child

        for prim in self.stage.Traverse():
            if str(prim.GetPath()).startswith(root_path + "/") and prim.GetName() in names:
                return prim
        return None

    def _find_first_rigid_body(self, root_path: str):
        for prim in self.stage.Traverse():
            is_child = str(prim.GetPath()).startswith(root_path + "/")
            if is_child and prim.HasAPI(self.UsdPhysics.RigidBodyAPI):
                return prim
        return None

    def _count_rigid_bodies_under(self, root_path: str) -> int:
        count = 0
        for prim in self.stage.Traverse():
            if str(prim.GetPath()).startswith(root_path + "/") and prim.HasAPI(self.UsdPhysics.RigidBodyAPI):
                count += 1
        return count

    def _find_articulation_root_path(self, root_path: str) -> Optional[str]:
        root = self.stage.GetPrimAtPath(root_path)
        if root and root.HasAPI(self.UsdPhysics.ArticulationRootAPI) and self._count_rigid_bodies_under(root_path) > 0:
            return root_path

        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            has_articulation = prim.HasAPI(self.UsdPhysics.ArticulationRootAPI)
            has_rigid_bodies = self._count_rigid_bodies_under(path) > 0
            if path.startswith(root_path + "/") and has_articulation and has_rigid_bodies:
                return path
        return None

    def print_prim_debug_info(self, root_path: str = "/World") -> None:
        print(f"Prim debug under {root_path}:")
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith(root_path):
                continue
            tags = []
            if prim.HasAPI(self.UsdPhysics.ArticulationRootAPI):
                tags.append("ArticulationRoot")
            if prim.HasAPI(self.UsdPhysics.RigidBodyAPI):
                tags.append("RigidBody")
            type_name = prim.GetTypeName()
            if "Joint" in type_name:
                tags.append(type_name)
            if tags:
                print(f"  {path}: {', '.join(tags)}")

    def print_usd_validation_report(self, root_path: str = "/World") -> None:
        """Print the physics structure that matters for joint-level control."""
        articulation_roots = []
        rigid_bodies = []
        joints = []

        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith(root_path):
                continue

            if prim.HasAPI(self.UsdPhysics.ArticulationRootAPI):
                articulation_roots.append(path)
            if prim.HasAPI(self.UsdPhysics.RigidBodyAPI):
                rigid_bodies.append(path)

            type_name = prim.GetTypeName()
            if "Joint" in type_name:
                body0_rel = prim.GetRelationship("physics:body0")
                body1_rel = prim.GetRelationship("physics:body1")
                body0 = [str(target) for target in body0_rel.GetTargets()] if body0_rel else []
                body1 = [str(target) for target in body1_rel.GetTargets()] if body1_rel else []
                applied_schemas = [str(schema) for schema in prim.GetAppliedSchemas()]
                has_drive = any(schema.startswith("PhysicsDriveAPI") for schema in applied_schemas)
                joints.append((path, type_name, body0, body1, has_drive))

        print(f"USD validation under {root_path}:")
        print(f"  Articulation roots ({len(articulation_roots)}):")
        for path in articulation_roots:
            print(f"    {path}")

        print(f"  Rigid bodies ({len(rigid_bodies)}):")
        for path in rigid_bodies:
            print(f"    {path}")

        print(f"  Physics joints ({len(joints)}):")
        for path, type_name, body0, body1, has_drive in joints:
            drive_text = "drive" if has_drive else "no drive"
            print(f"    {path}: {type_name}, {drive_text}")
            print(f"      body0={body0 or '[]'}")
            print(f"      body1={body1 or '[]'}")

    def _detect_robotiq_base(self):
        candidate_names = [
            self.robotiq_base_name,
            "robotiq_arg2f_base_link",
            "robotiq_85_base_link",
            "Robotiq_2F_85_base_link",
            "base_link",
            "robotiq_base",
        ]
        prim = self._find_child_by_name(self.robotiq_prim_path, candidate_names)
        return prim or self._find_first_rigid_body(self.robotiq_prim_path)

    def _detect_franka_ee(self):
        candidate_names = [
            self.franka_ee_name,
            "fr3_link8",
            "panda_link8",
            "fr3_hand",
            "panda_hand",
            "hand",
            "flange",
        ]
        return self._find_child_by_name(self.franka_prim_path, candidate_names)

    def _hide_default_franka_hand(self) -> None:
        hand_names = {
            "fr3_hand",
            "fr3_leftfinger",
            "fr3_rightfinger",
            "panda_hand",
            "panda_leftfinger",
            "panda_rightfinger",
        }
        for prim in self.stage.Traverse():
            if not str(prim.GetPath()).startswith(self.franka_prim_path + "/"):
                continue
            if prim.GetName() not in hand_names:
                continue

            stack = [prim]
            while stack:
                subprim = stack.pop()
                self.UsdGeom.Imageable(subprim).MakeInvisible()
                collision_attr = subprim.GetAttribute("physics:collisionEnabled")
                if collision_attr:
                    collision_attr.Set(False)
                stack.extend(list(subprim.GetAllChildren()))

    def _remove_nested_articulation_roots(self) -> None:
        for prim in self.stage.Traverse():
            if not str(prim.GetPath()).startswith(self.robotiq_prim_path + "/"):
                continue
            if prim.HasAPI(self.UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(self.UsdPhysics.ArticulationRootAPI)
            if prim.HasAPI(self.PhysxSchema.PhysxArticulationAPI):
                prim.RemoveAPI(self.PhysxSchema.PhysxArticulationAPI)

    def _define_fixed_joint(self) -> None:
        quat = self._quat_from_rpy_deg(self.adapter_rpy_deg)
        joint = self.UsdPhysics.FixedJoint.Define(self.stage, "/World/Franka_Robotiq_FixedJoint")
        joint.CreateBody0Rel().SetTargets([self.Sdf.Path(str(self.ee_prim.GetPath()))])
        joint.CreateBody1Rel().SetTargets([self.Sdf.Path(str(self.robotiq_base_prim.GetPath()))])
        joint.CreateLocalPos0Attr().Set(self.Gf.Vec3f(*self.adapter_xyz))
        joint.CreateLocalRot0Attr().Set(quat)
        joint.CreateLocalPos1Attr().Set(self.Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(self.Gf.Quatf(1.0, self.Gf.Vec3f(0.0, 0.0, 0.0)))

    def _create_basic_scene(self) -> None:
        self.UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        self.UsdGeom.SetStageUpAxis(self.stage, self.UsdGeom.Tokens.z)
        self.UsdPhysics.Scene.Define(self.stage, "/World/PhysicsScene")
        self.PhysicsContext(prim_path="/World/PhysicsScene")

        ground = self.UsdGeom.Cube.Define(self.stage, "/World/Ground")
        ground.CreateSizeAttr(1.0)
        ground_matrix = self._matrix_from_pose(
            (0.0, 0.0, -0.025),
            self._quat_from_rpy_deg((0.0, 0.0, 0.0)),
        )
        xformable = self.UsdGeom.Xformable(ground.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp().Set(ground_matrix)
        ground.AddScaleOp().Set(self.Gf.Vec3f(4.0, 4.0, 0.05))
        self.UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

        light = self.UsdLux.DomeLight.Define(self.stage, "/World/DomeLight")
        light.CreateIntensityAttr(700.0)

        camera = self.UsdGeom.Camera.Define(self.stage, "/World/Camera")
        camera_matrix = self.Gf.Matrix4d(1.0)
        camera_matrix.SetRotate(
            self.Gf.Rotation(self.Gf.Vec3d(0, 0, 1), 38)
            * self.Gf.Rotation(self.Gf.Vec3d(1, 0, 0), 62)
        )
        camera_matrix.SetTranslateOnly(self.Gf.Vec3d(1.8, -2.2, 1.4))
        xformable = self.UsdGeom.Xformable(camera.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp().Set(camera_matrix)
        camera.CreateFocalLengthAttr(28.0)

    def create_scene(self) -> tuple[str, str, str]:
        ctx = self.omni_usd.get_context()
        ctx.new_stage()
        self.stage = ctx.get_stage()
        self.UsdGeom.Xform.Define(self.stage, "/World")
        self.stage.SetDefaultPrim(self.stage.GetPrimAtPath("/World"))
        self._create_basic_scene()

        franka_usd, robotiq_usd = self._resolve_robot_usds()
        self.add_reference_to_stage(usd_path=franka_usd, prim_path=self.franka_prim_path)
        self.add_reference_to_stage(usd_path=robotiq_usd, prim_path=self.robotiq_prim_path)

        if not self.keep_franka_hand:
            self._hide_default_franka_hand()

        self.ee_prim = self._detect_franka_ee()
        if self.ee_prim is None:
            raise RuntimeError("Could not find a Franka end-effector link.")

        self.robotiq_base_prim = self._detect_robotiq_base()
        if self.robotiq_base_prim is None:
            raise RuntimeError("Could not find a Robotiq base rigid body.")

        if self.merge_gripper_articulation:
            self._remove_nested_articulation_roots()

        cache = self.UsdGeom.XformCache()
        ee_world = cache.GetLocalToWorldTransform(self.ee_prim)
        base_world = cache.GetLocalToWorldTransform(self.robotiq_base_prim)
        root_world = cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.robotiq_prim_path))
        root_to_base = base_world * root_world.GetInverse()

        adapter_quat = self._quat_from_rpy_deg(self.adapter_rpy_deg)
        target_base_world = ee_world * self._matrix_from_pose(self.adapter_xyz, adapter_quat)
        target_root_world = target_base_world * root_to_base.GetInverse()
        xformable = self.UsdGeom.Xformable(self.stage.GetPrimAtPath(self.robotiq_prim_path))
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp().Set(target_root_world)

        self._define_fixed_joint()
        self.stage.GetRootLayer().Export(self.save_usd)
        return franka_usd, robotiq_usd, self.save_usd

    def _setup_articulation_handles(self) -> None:
        timeline = self.omni_timeline.get_timeline_interface()
        timeline.play()
        for _ in range(5):
            self.sim_app.update()

        if self.SimulationManager.get_physics_sim_view() is None:
            self.SimulationManager.initialize_physics()
            self.sim_app.update()

        physics_sim_view = self.SimulationManager.get_physics_sim_view()
        if physics_sim_view is None:
            physics_sim_view = self.omni_physics_tensors.create_simulation_view(
                self.SimulationManager.get_backend(),
                stage_id=self.get_current_stage_id(),
            )
            physics_sim_view.set_subspace_roots("/")

        franka_root_path = self._find_articulation_root_path(self.franka_prim_path)
        if franka_root_path is None:
            raise RuntimeError(f"Could not find Franka articulation root under {self.franka_prim_path}.")

        self.franka = self.SingleArticulation(prim_path=franka_root_path, name="franka")
        self.franka.initialize(physics_sim_view=physics_sim_view)

        if self.merge_gripper_articulation:
            print("Robotiq is controlled through the Franka articulation.")
        else:
            print("Robotiq separate articulation handle is disabled for this USD.")

    def print_joint_info(self) -> None:
        print("Franka joints:")
        for index, name in enumerate(self.franka.dof_names):
            print(f"  {index}: {name}")

        if self.robotiq is None:
            print("Robotiq joints: use matching Franka articulation DOFs")
            for index, name in enumerate(self.franka.dof_names):
                lowered = name.lower()
                if "finger" in lowered or "robotiq" in lowered or "knuckle" in lowered:
                    print(f"  {index}: {name}")
            return

        print("Robotiq joints:")
        for index, name in enumerate(self.robotiq.dof_names):
            print(f"  {index}: {name}")

    def get_joint_names(self) -> list[str]:
        return list(self.franka.dof_names)

    def _joint_indices(self, articulation, joint_names: Optional[Iterable[str]]) -> Optional[np.ndarray]:
        if joint_names is None:
            return None
        return np.array([articulation.get_dof_index(name) for name in joint_names], dtype=np.int32)

    def _robotiq_joint_names_from_franka(self) -> list[str]:
        names = []
        for name in self.franka.dof_names:
            lowered = name.lower()
            if "finger" in lowered or "robotiq" in lowered or "knuckle" in lowered:
                names.append(name)
        return names

    def get_robotiq_joint_names(self) -> list[str]:
        if self.robotiq is not None:
            return list(self.robotiq.dof_names)
        return self._robotiq_joint_names_from_franka()

    def set_joint_positions_by_name(self, positions_by_name: dict[str, float], steps: int = 120) -> None:
        unknown_names = [name for name in positions_by_name if name not in self.franka.dof_names]
        if unknown_names:
            available = "\n  - ".join(self.franka.dof_names)
            unknown = ", ".join(unknown_names)
            raise ValueError(f"Unknown joint name(s): {unknown}\nAvailable joints:\n  - {available}")

        joint_names = list(positions_by_name.keys())
        positions = [positions_by_name[name] for name in joint_names]
        self.set_franka_joint_positions(positions, joint_names=joint_names, steps=steps)

    def set_franka_joint_positions(
        self,
        positions: Iterable[float],
        joint_names: Optional[Iterable[str]] = None,
        steps: int = 120,
    ) -> None:
        joint_indices = self._joint_indices(self.franka, joint_names)
        action = self.ArticulationAction(
            joint_positions=np.array(list(positions), dtype=np.float32),
            joint_indices=joint_indices,
        )
        self.franka.apply_action(action)
        if self._running_forever:
            return
        self.step(steps=steps, play_timeline=True)

    def set_robotiq_joint_positions(
        self,
        positions: Iterable[float],
        joint_names: Optional[Iterable[str]] = None,
        steps: int = 120,
    ) -> None:
        if self.robotiq is None:
            joint_names = list(joint_names) if joint_names is not None else self._robotiq_joint_names_from_franka()
            if not joint_names:
                raise RuntimeError("Could not find Robotiq DOFs in the Franka articulation. Run env.print_joint_info().")
            self.set_franka_joint_positions(positions, joint_names=joint_names, steps=steps)
            return

        joint_indices = self._joint_indices(self.robotiq, joint_names)
        action = self.ArticulationAction(
            joint_positions=np.array(list(positions), dtype=np.float32),
            joint_indices=joint_indices,
        )
        self.robotiq.apply_action(action)
        if self._running_forever:
            return
        self.step(steps=steps, play_timeline=True)

    def step(self, steps: int = 1, play_timeline: bool = False) -> None:
        timeline = None
        if play_timeline:
            timeline = self.omni_timeline.get_timeline_interface()
            timeline.play()
        for _ in range(steps):
            self.sim_app.update()
        if timeline is not None:
            timeline.stop()

    def run_forever(self) -> None:
        if self.sim_app is None:
            return
        print("Keeping Isaac Sim open. Press Ctrl+C in this terminal to stop.")
        self._running_forever = True
        while self.sim_app.is_running():
            self.sim_app.update()

    def close(self) -> None:
        if self.sim_app is None:
            return
        self.sim_app.close()
        self.sim_app = None


def main() -> int:
    args = _build_arg_parser().parse_args()
    env = Franka2F85(
        headless=args.headless,
        asset_root=args.asset_root,
        franka_usd=args.franka_usd,
        robotiq_usd=args.robotiq_usd,
        franka_prim=args.franka_prim,
        robotiq_prim=args.robotiq_prim,
        franka_ee_name=args.franka_ee_name,
        robotiq_base_name=args.robotiq_base_name,
        adapter_xyz=args.adapter_xyz,
        adapter_rpy_deg=args.adapter_rpy_deg,
        keep_franka_hand=args.keep_franka_hand,
        save_usd=args.save_usd,
        merge_gripper_articulation=args.merge_gripper_articulation,
        play=args.play,
        steps=args.steps,
    )

    print("Created Franka + Robotiq 2F-85 scene")
    print(f"  Franka USD:  {env.franka_usd}")
    print(f"  Robotiq USD: {env.robotiq_usd}")
    print(f"  Saved USD:   {env.saved_path}")

    should_stay_open = args.stay_open or (not args.headless and not args.close_after_save)
    if should_stay_open:
        env.run_forever()
    elif not args.play:
        env.step()

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
