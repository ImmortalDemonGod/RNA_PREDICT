import unittest
from rna_predict.scripts.hypot_test_gen import fix_leading_zeros

class TestFixLeadingZeros(unittest.TestCase):
    def test_positive_number(self):
        # "007" should be fixed to "7"
        self.assertEqual(fix_leading_zeros("007"), "7")
    
    def test_negative_number(self):
        # "-0123" should be fixed to "-123"
        self.assertEqual(fix_leading_zeros("-0123"), "-123")
    
    def test_mixed_text(self):
        # Multiple numbers in a string should be fixed correctly.
        input_text = "Value: 0042, Error: -0007 and 000"
        expected = "Value: 42, Error: -7 and 0"
        self.assertEqual(fix_leading_zeros(input_text), expected)
    
    def test_already_fixed(self):
        # Numbers without leading zeros remain unchanged.
        self.assertEqual(fix_leading_zeros("123"), "123")
    
    def test_zero_only(self):
        # A lone "0" should not be changed.
        self.assertEqual(fix_leading_zeros("0"), "0")
    
    def test_number_in_string_literal(self):
        # Even inside a string literal the numbers get fixed.
        input_text = 'print("Error 007 occurred")'
        expected = 'print("Error 7 occurred")'
        self.assertEqual(fix_leading_zeros(input_text), expected)

if __name__ == '__main__':
    unittest.main()