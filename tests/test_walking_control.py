from __future__ import annotations

import unittest
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.build import command_manifest


namespace = {"np": np, "math": math, "ControllerInitialState": object, "StepResult": object}
exec(compile(Path("policy.py").read_text(), "policy.py", "exec"), namespace, namespace)
MicroduckPolicy = namespace["MicroduckPolicy"]


class WalkingControlTests(unittest.TestCase):
    def test_manifest_exposes_one_two_axis_joystick(self):
        commands = command_manifest({"mode": "walking"})
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0]["id"], "walking-control")
        self.assertEqual(commands[0]["kind"], "joystick2d")
        self.assertEqual(commands[0]["axisRanges"], [[-1.0, 1.0], [-0.2, 0.25]])

    def test_joystick_maps_turn_and_forward_to_the_policy_command_block(self):
        policy = MicroduckPolicy()
        policy.mode = "walking"
        command = policy.command_block(SimpleNamespace(commands={"walking-control": [0.6, 0.2]}))
        np.testing.assert_allclose(command[[0, 2]], [0.2, 0.6])

    def test_walking_command_is_clamped_to_trained_ranges(self):
        policy = MicroduckPolicy()
        policy.mode = "walking"
        command = policy.command_block(SimpleNamespace(commands={"walking-control": [-4, 4]}))
        np.testing.assert_allclose(command[[0, 2]], [0.25, -1.0])


if __name__ == "__main__":
    unittest.main()
