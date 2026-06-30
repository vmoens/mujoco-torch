"""Tests for MuJoCo version-dependent enum compatibility."""

import mujoco

from mujoco_torch._src import types


def _assert_enum_values(enum_cls, enum_type, expected):
    assert set(enum_cls.__members__) == set(expected)
    for name, member_name in expected.items():
        assert int(enum_cls.__members__[name]) == int(getattr(enum_type, member_name))


def test_disable_bit_members_match_installed_mujoco():
    expected = {
        "CONSTRAINT": "mjDSBL_CONSTRAINT",
        "EQUALITY": "mjDSBL_EQUALITY",
        "FRICTIONLOSS": "mjDSBL_FRICTIONLOSS",
        "LIMIT": "mjDSBL_LIMIT",
        "CONTACT": "mjDSBL_CONTACT",
        "SPRING": "mjDSBL_SPRING",
        "DAMPER": "mjDSBL_DAMPER",
        "GRAVITY": "mjDSBL_GRAVITY",
        "CLAMPCTRL": "mjDSBL_CLAMPCTRL",
        "WARMSTART": "mjDSBL_WARMSTART",
        "ACTUATION": "mjDSBL_ACTUATION",
        "REFSAFE": "mjDSBL_REFSAFE",
        "SENSOR": "mjDSBL_SENSOR",
        "EULERDAMP": "mjDSBL_EULERDAMP",
        "FILTERPARENT": "mjDSBL_FILTERPARENT",
    }
    if not hasattr(mujoco.mjtDisableBit, "mjDSBL_SPRING"):
        expected["SPRING"] = "mjDSBL_PASSIVE"
        expected["DAMPER"] = "mjDSBL_PASSIVE"
    if hasattr(mujoco.mjtDisableBit, "mjDSBL_MULTICCD"):
        expected["MULTICCD"] = "mjDSBL_MULTICCD"

    _assert_enum_values(types.DisableBit, mujoco.mjtDisableBit, expected)


def test_enable_bit_members_match_installed_mujoco():
    expected = {"INVDISCRETE": "mjENBL_INVDISCRETE"}
    if hasattr(mujoco.mjtEnableBit, "mjENBL_MULTICCD"):
        expected["MULTICCD"] = "mjENBL_MULTICCD"
    if hasattr(mujoco.mjtEnableBit, "mjENBL_SLEEP"):
        expected["SLEEP"] = "mjENBL_SLEEP"

    _assert_enum_values(types.EnableBit, mujoco.mjtEnableBit, expected)


def test_sensor_type_members_match_installed_mujoco():
    expected = {
        "MAGNETOMETER": "mjSENS_MAGNETOMETER",
        "CAMPROJECTION": "mjSENS_CAMPROJECTION",
        "RANGEFINDER": "mjSENS_RANGEFINDER",
        "JOINTPOS": "mjSENS_JOINTPOS",
        "TENDONPOS": "mjSENS_TENDONPOS",
        "ACTUATORPOS": "mjSENS_ACTUATORPOS",
        "BALLQUAT": "mjSENS_BALLQUAT",
        "FRAMEPOS": "mjSENS_FRAMEPOS",
        "FRAMEXAXIS": "mjSENS_FRAMEXAXIS",
        "FRAMEYAXIS": "mjSENS_FRAMEYAXIS",
        "FRAMEZAXIS": "mjSENS_FRAMEZAXIS",
        "FRAMEQUAT": "mjSENS_FRAMEQUAT",
        "SUBTREECOM": "mjSENS_SUBTREECOM",
        "CLOCK": "mjSENS_CLOCK",
        "VELOCIMETER": "mjSENS_VELOCIMETER",
        "GYRO": "mjSENS_GYRO",
        "JOINTVEL": "mjSENS_JOINTVEL",
        "TENDONVEL": "mjSENS_TENDONVEL",
        "ACTUATORVEL": "mjSENS_ACTUATORVEL",
        "BALLANGVEL": "mjSENS_BALLANGVEL",
        "FRAMELINVEL": "mjSENS_FRAMELINVEL",
        "FRAMEANGVEL": "mjSENS_FRAMEANGVEL",
        "SUBTREELINVEL": "mjSENS_SUBTREELINVEL",
        "SUBTREEANGMOM": "mjSENS_SUBTREEANGMOM",
        "TOUCH": "mjSENS_TOUCH",
        "ACCELEROMETER": "mjSENS_ACCELEROMETER",
        "FORCE": "mjSENS_FORCE",
        "TORQUE": "mjSENS_TORQUE",
        "ACTUATORFRC": "mjSENS_ACTUATORFRC",
        "JOINTACTFRC": "mjSENS_JOINTACTFRC",
        "FRAMELINACC": "mjSENS_FRAMELINACC",
        "FRAMEANGACC": "mjSENS_FRAMEANGACC",
    }
    if hasattr(mujoco.mjtSensor, "mjSENS_CONTACT"):
        expected["CONTACT"] = "mjSENS_CONTACT"
    if hasattr(mujoco.mjtSensor, "mjSENS_TENDONACTFRC"):
        expected["TENDONACTFRC"] = "mjSENS_TENDONACTFRC"

    _assert_enum_values(types.SensorType, mujoco.mjtSensor, expected)
