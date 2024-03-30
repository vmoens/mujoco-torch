# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for mujoco_torch logger."""

import logging

from absl.testing import absltest

import mujoco_torch
from mujoco_torch._src.log import logger


class LoggerTest(absltest.TestCase):
    def test_logger_name(self):
        self.assertEqual(logger.name, "mujoco_torch")

    def test_logger_has_handler(self):
        self.assertTrue(logger.hasHandlers())
        self.assertIsInstance(logger.handlers[0], logging.StreamHandler)

    def test_logger_does_not_propagate(self):
        self.assertFalse(logger.propagate)

    def test_logger_default_level(self):
        self.assertEqual(logger.level, logging.INFO)

    def test_public_export(self):
        self.assertIs(mujoco_torch.mujoco_logger, logger)


if __name__ == "__main__":
    absltest.main()
