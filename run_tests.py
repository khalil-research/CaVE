import unittest
import warnings

if __name__ == "__main__":
    # iIgnore all warnings
    warnings.filterwarnings("ignore")

    # create a test loader
    loader = unittest.TestLoader()

    # discover all test cases
    start_dir = "test"
    suite = loader.discover(start_dir)

    # run the test cases
    runner = unittest.TextTestRunner()
    runner.run(suite)
