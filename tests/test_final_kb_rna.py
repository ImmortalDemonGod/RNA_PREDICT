"""
test_final_kb_rna.py

A comprehensive unittest-based test suite for final_kb_rna.py, aiming for
high coverage. Tests are logically grouped by function or functionality,
use descriptive method names, include docstrings, use fixtures to reduce
redundancy, incorporate robust assertions, leverage hypothesis for a range
of generated inputs, and include a minimal demonstration of mocking (though
this module has no real external dependencies to mock).

Run this file with:
    python -m unittest test_final_kb_rna.py
"""

import math
import unittest
from unittest.mock import patch
from hypothesis import given, strategies as st
from hypothesis import settings, HealthCheck

import rna_predict.pipeline.stageC.mp_nerf.final_kb_rna as final_kb_rna


# A custom strategy for angles in degrees. We'll avoid extremes beyond +/- 1e6 for speed.
angle_strat = st.floats(min_value=-1e6, max_value=1e6, allow_infinity=False, allow_nan=False)

# A custom strategy for valid sugar puckers. We'll test some valid strings plus random text for edge cases.
pucker_strat = st.sampled_from(["C3'-endo", "C2'-endo"])


class TestAngleConversions(unittest.TestCase):
    """
    Tests for deg_to_rad and rad_to_deg, including property-based round-trip checks.
    """
    def setUp(self) -> None:
        """Setup runs before each test."""
        self.sample_angle_deg = 180.0
        self.sample_angle_rad = math.pi

    def test_deg_to_rad_known_value(self) -> None:
        """
        Verify deg_to_rad returns a known correct value for 180 degrees (pi radians).
        """
        result = final_kb_rna.deg_to_rad(self.sample_angle_deg)
        self.assertAlmostEqual(result, math.pi, places=7)

    def test_rad_to_deg_known_value(self) -> None:
        """
        Verify rad_to_deg returns a known correct value for pi radians (180 degrees).
        """
        result = final_kb_rna.rad_to_deg(self.sample_angle_rad)
        self.assertAlmostEqual(result, 180.0, places=7)

    @given(angle=angle_strat)
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_round_trip_deg_to_rad_to_deg(self, angle: float) -> None:
        """
        Round-trip test: converting degrees -> radians -> degrees should yield the original,
        within floating-point tolerance.
        """
        rad = final_kb_rna.deg_to_rad(angle)
        deg = final_kb_rna.rad_to_deg(rad)
        # Float rounding can cause slight differences, so use approx assertion
        self.assertAlmostEqual(angle, deg, places=7)

    @given(angle=angle_strat)
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_round_trip_rad_to_deg_to_rad(self, angle: float) -> None:
        """
        Round-trip test: converting radians -> degrees -> radians should yield the original,
        within floating-point tolerance.
        """
        deg = final_kb_rna.rad_to_deg(angle)
        rad = final_kb_rna.deg_to_rad(deg)
        self.assertAlmostEqual(angle, rad, places=7)


class TestBondLengths(unittest.TestCase):
    """
    Tests for get_bond_length function, covering both 'C3'-endo' and 'C2'-endo' states,
    as well as error handling for unknown sugar puckers.
    """
    def setUp(self) -> None:
        """Common test data for bond lengths."""
        self.valid_pair = "C1'-C2'"
        self.invalid_pair = "FOO-BAR"
        self.expected_length_c3 = 1.528  # from RNA_BOND_LENGTHS_C3_ENDO
        self.expected_length_c2 = 1.526  # from RNA_BOND_LENGTHS_C2_ENDO

    def test_get_bond_length_c3_endo_known_pair(self) -> None:
        """
        Ensure get_bond_length returns the expected length for a known C3'-endo bond pair.
        """
        length = final_kb_rna.get_bond_length(self.valid_pair, sugar_pucker="C3'-endo")
        self.assertIsNotNone(length)
        self.assertAlmostEqual(length, self.expected_length_c3, places=3)

    def test_get_bond_length_c2_endo_known_pair(self) -> None:
        """
        Ensure get_bond_length returns the expected length for a known C2'-endo bond pair.
        """
        length = final_kb_rna.get_bond_length(self.valid_pair, sugar_pucker="C2'-endo")
        self.assertIsNotNone(length)
        self.assertAlmostEqual(length, self.expected_length_c2, places=3)

    def test_get_bond_length_unknown_pair(self) -> None:
        """
        get_bond_length should return None for an unrecognized bond pair.
        """
        length = final_kb_rna.get_bond_length(self.invalid_pair, sugar_pucker="C3'-endo")
        self.assertIsNone(length)

    def test_get_bond_length_unknown_sugar_pucker(self) -> None:
        """
        get_bond_length should raise a ValueError for unknown sugar_pucker states.
        """
        with self.assertRaises(ValueError):
            final_kb_rna.get_bond_length(self.valid_pair, sugar_pucker="UNKNOWN")


class TestBondAngles(unittest.TestCase):
    """
    Tests for get_bond_angle function, including coverage for degrees vs. radians,
    as well as unknown sugar pucker states.
    """
    def setUp(self) -> None:
        self.triplet = "C1'-C2'-C3'"
        self.expected_angle_deg = 101.5  # from RNA_BOND_ANGLES_C3_ENDO
        self.expected_angle_rad = final_kb_rna.deg_to_rad(101.5)

    def test_get_bond_angle_c3_endo_degrees(self) -> None:
        """
        Verify a known bond angle is returned in degrees for C3'-endo.
        """
        angle_deg = final_kb_rna.get_bond_angle(self.triplet, sugar_pucker="C3'-endo", degrees=True)
        self.assertAlmostEqual(angle_deg, self.expected_angle_deg, places=3)

    def test_get_bond_angle_c3_endo_radians(self) -> None:
        """
        Verify a known bond angle is returned in radians for C3'-endo.
        """
        angle_rad = final_kb_rna.get_bond_angle(self.triplet, sugar_pucker="C3'-endo", degrees=False)
        self.assertAlmostEqual(angle_rad, self.expected_angle_rad, places=5)

    def test_get_bond_angle_unknown_triplet(self) -> None:
        """
        get_bond_angle should return None for an unrecognized bond angle triplet.
        """
        angle = final_kb_rna.get_bond_angle("X-Y-Z", sugar_pucker="C3'-endo", degrees=True)
        self.assertIsNone(angle)

    def test_get_bond_angle_unknown_sugar_pucker(self) -> None:
        """
        get_bond_angle should raise ValueError for unknown sugar_pucker states.
        """
        with self.assertRaises(ValueError):
            final_kb_rna.get_bond_angle(self.triplet, sugar_pucker="UNKNOWN", degrees=True)


class TestBackboneTorsion(unittest.TestCase):
    """
    Tests for get_backbone_torsion function, verifying known torsion names,
    units (degrees/radians), and unknown torsion handling.
    """
    def test_get_backbone_torsion_known_alpha_deg(self) -> None:
        """
        Check alpha torsion in degrees.
        """
        alpha_deg = final_kb_rna.get_backbone_torsion("alpha", degrees=True)
        self.assertAlmostEqual(alpha_deg, 300.0, places=3)

    def test_get_backbone_torsion_unknown_name(self) -> None:
        """
        get_backbone_torsion should return None if the torsion name is not recognized.
        """
        torsion = final_kb_rna.get_backbone_torsion("foo")
        self.assertIsNone(torsion)

    def test_get_backbone_torsion_zeta_radians(self) -> None:
        """
        Validate conversion of zeta torsion to radians.
        """
        zeta_deg = 290.0
        zeta_rad_expected = final_kb_rna.deg_to_rad(zeta_deg)
        zeta_rad = final_kb_rna.get_backbone_torsion("zeta", degrees=False)
        self.assertAlmostEqual(zeta_rad, zeta_rad_expected, places=6)


class TestSugarPuckerTorsions(unittest.TestCase):
    """
    Tests for get_sugar_pucker_torsions function.
    """
    def test_get_sugar_pucker_torsions_c3_endo(self) -> None:
        """
        Confirm dictionary keys for known pucker C3'-endo.
        """
        torsions = final_kb_rna.get_sugar_pucker_torsions("C3'-endo")
        self.assertIn("nu0", torsions)
        self.assertIn("nu1", torsions)
        self.assertIn("nu2", torsions)
        self.assertIn("nu3", torsions)
        self.assertIn("nu4", torsions)

    def test_get_sugar_pucker_torsions_c2_endo(self) -> None:
        """
        Confirm dictionary structure for C2'-endo.
        """
        torsions = final_kb_rna.get_sugar_pucker_torsions("C2'-endo")
        # We know "nu0" should be present, though not all angles are fully enumerated
        self.assertIn("nu0", torsions)

    def test_get_sugar_pucker_torsions_unknown_pucker(self) -> None:
        """
        Should return an empty dict for unrecognized pucker strings.
        """
        torsions = final_kb_rna.get_sugar_pucker_torsions("FOO")
        self.assertEqual(torsions, {})


class TestBaseGeometry(unittest.TestCase):
    """
    Tests for get_base_geometry function (A, G, C, U).
    """
    def test_get_base_geometry_adenine(self) -> None:
        """
        Check if base geometry for A is returned and has expected sub-keys.
        """
        data = final_kb_rna.get_base_geometry("A")
        self.assertIn("bond_lengths", data)
        self.assertIn("bond_angles_deg", data)
        self.assertAlmostEqual(data["bond_lengths"]["N9-C4"], 1.374, places=3)

    def test_get_base_geometry_unknown_base(self) -> None:
        """
        get_base_geometry should return an empty dict for unrecognized bases.
        """
        data = final_kb_rna.get_base_geometry("Z")
        self.assertEqual(data, {})


class TestConnectivity(unittest.TestCase):
    """
    Tests for get_connectivity function.
    """
    def test_get_connectivity_backbone(self) -> None:
        """
        Ensure backbone connectivity is not empty.
        """
        conn = final_kb_rna.get_connectivity("backbone")
        self.assertTrue(len(conn) > 0)
        self.assertIn(("P", "O5'"), conn)

    def test_get_connectivity_unknown_fragment(self) -> None:
        """
        get_connectivity should return an empty list for unrecognized fragments.
        """
        conn = final_kb_rna.get_connectivity("FOO")
        self.assertEqual(conn, [])

    def test_mock_example(self) -> None:
        """
        Demonstration of minimal mocking usage.
        We mock deg_to_rad inside get_bond_angle to illustrate patch usage.
        Although it's not particularly necessary in real usage, this shows
        the approach for external dependencies or side effects.
        """
        with patch('final_kb_rna.deg_to_rad', return_value=123.456) as mock_converter:
            # Now calling get_bond_angle in radians should yield the patched result for a known triplet
            angle = final_kb_rna.get_bond_angle("C1'-C2'-C3'", sugar_pucker="C3'-endo", degrees=False)
            self.assertEqual(angle, 123.456)
            mock_converter.assert_called_once()


if __name__ == "__main__":
    unittest.main()