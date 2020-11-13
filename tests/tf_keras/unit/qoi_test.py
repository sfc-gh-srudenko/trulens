import os
os.environ['NETLENS_BACKEND'] = 'tf.keras'

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from unittest import TestCase, main

from tests.unit.qoi_test_base import QoiTestBase


class QoiTest(QoiTestBase, TestCase):
    pass


if __name__ == '__main__':
    main()
