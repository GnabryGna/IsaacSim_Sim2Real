#!/usr/bin/env python3
"""Training entry point placeholder."""

from franka_robotiq_env import Franka2F85

env = None


def main():
    global env
    env = Franka2F85()
    env.print_joint_info()
    env.print_usd_validation_report()
    env.print_prim_debug_info(env.robotiq_prim_path)
    env.run_forever()


if __name__ == "__main__":
    main()
