# Franka + Robotiq 2F-85 Isaac Sim Environment

This directory contains a standalone Isaac Sim script that composes a Franka arm
with a Robotiq 2F-85 gripper and saves the result as a USD scene.

## Run

Use Isaac Sim's bundled Python:

```bash
/path/to/isaac-sim/python.sh create_franka_robotiq_env.py
```

The default output is:

```text
franka_robotiq_2f85_env.usd
```

To create the USD and step the simulation:

```bash
/path/to/isaac-sim/python.sh create_franka_robotiq_env.py --play
```

## Common Options

If Isaac Sim cannot find the assets automatically, pass explicit paths:

```bash
/path/to/isaac-sim/python.sh create_franka_robotiq_env.py \
  --franka-usd /path/to/franka.usd \
  --robotiq-usd /path/to/Robotiq_2F_85_edit.usd
```

If the gripper pose needs alignment, tune the fixed adapter transform:

```bash
/path/to/isaac-sim/python.sh create_franka_robotiq_env.py \
  --adapter-xyz 0,0,0.04 \
  --adapter-rpy-deg 0,0,90
```

The script attaches the Robotiq base link to `panda_link8` by default. If your
Franka USD uses a different wrist or hand link, override it:

```bash
/path/to/isaac-sim/python.sh create_franka_robotiq_env.py --franka-ee-name panda_hand
```

By default the stock Franka hand is hidden and its collisions are disabled so the
Robotiq gripper is the visible end-effector. To keep the original hand:

```bash
/path/to/isaac-sim/python.sh create_franka_robotiq_env.py --keep-franka-hand
```

## Notes

- The script supports newer Isaac Sim asset paths such as
  `Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd` and Robotiq's
  `Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd`.
- For precise hardware-style mounting, adjust `--adapter-xyz` and
  `--adapter-rpy-deg` to match your adapter plate CAD or flange convention.
- If you later move this into Isaac Lab, keep the generated USD as the robot
  asset and build the task environment around that asset.
