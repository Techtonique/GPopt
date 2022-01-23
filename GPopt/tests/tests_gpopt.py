"""GPOpt tests."""

import numpy as np
import unittest as ut
import GPopt as gp
import matplotlib.pyplot as plt

# in /Users/moudiki/Documents/Python_Packages/GPopt/GPopt
# > nose2 --with-coverage

# Tests on Branin function


class TestBranin(ut.TestCase):
    def test_base_and_more_iter(self):

        # braninsc
        def braninsc(x):
            x1_bar = 15 * x[0] - 5
            x2_bar = 15 * x[1]
            term1 = (
                x2_bar
                - (5.1 / (4 * np.pi ** 2)) * x1_bar ** 2
                + (5 / np.pi) * x1_bar
                - 6
            ) ** 2
            term2 = 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1_bar)

            return (term1 + term2 - 44.81) / 51.95

        gp_opt1 = gp.GPOpt(
            objective_func=braninsc,
            lower_bound=np.array([-5, 0]),
            upper_bound=np.array([10, 15]),
            n_choices=10000,
            n_init=10,
            n_iter=50,
        )

        print("\n")
        print("----- Test base -----\n")
        print("\n")

        res1 = gp_opt1.optimize(verbose=1)

        print("\n")
        print("----- Test n_more_iter -----\n")
        print("\n")

        res2 = gp_opt1.optimize(n_more_iter=25, verbose=2)

        self.assertTrue(np.allclose(res1[0][0], 3.20333734))
        self.assertTrue(np.allclose(res1[1], -0.61975891324380084))
        self.assertTrue(np.allclose(res2[0][0], -0.718994140625))
        self.assertTrue(np.allclose(res2[1], -1.0467104543674774))

    def test_early_stopping(self):

        # braninsc
        def braninsc(x):
            x1_bar = 15 * x[0] - 5
            x2_bar = 15 * x[1]
            term1 = (
                x2_bar
                - (5.1 / (4 * np.pi ** 2)) * x1_bar ** 2
                + (5 / np.pi) * x1_bar
                - 6
            ) ** 2
            term2 = 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1_bar)

            return (term1 + term2 - 44.81) / 51.95

        print("\n")
        print("----- Test early stopping #1 -----\n")
        print("\n")

        gp_opt1 = gp.GPOpt(
            objective_func=braninsc,
            lower_bound=np.array([-5, 0]),
            upper_bound=np.array([10, 15]),
            n_choices=10000,
            n_init=10,
            n_iter=200,
        )
        gp_opt1.optimize(verbose=2, abs_tol=1e-4)

        # -------

        print("\n")
        print("----- Test early stopping #2 -----\n")
        print("\n")

        gp_opt2 = gp.GPOpt(
            objective_func=braninsc,
            lower_bound=np.array([-5, 0]),
            upper_bound=np.array([10, 15]),
            n_choices=10000,
            n_init=10,
            n_iter=30,
        )
        gp_opt2.optimize()

        gp_opt2.optimize(verbose=1, n_more_iter=100, abs_tol=1e-4)

        # -------

        self.assertTrue(np.allclose(gp_opt1.n_iter, 74))
        self.assertTrue(np.allclose(gp_opt2.n_iter, 74))

    def test_save(self):

        # braninsc
        def braninsc(x):
            x1_bar = 15 * x[0] - 5
            x2_bar = 15 * x[1]
            term1 = (
                x2_bar
                - (5.1 / (4 * np.pi ** 2)) * x1_bar ** 2
                + (5 / np.pi) * x1_bar
                - 6
            ) ** 2
            term2 = 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1_bar)

            return (term1 + term2 - 44.81) / 51.95

        print("\n")
        print("----- Test saving -----\n")
        print("\n")

        path = input("Enter the path for saving data \n")

        # with saving and loading
        gp_opt1 = gp.GPOpt(
            objective_func=braninsc,
            lower_bound=np.array([-5, 0]),
            upper_bound=np.array([10, 15]),
            n_choices=10000,
            n_init=10,
            n_iter=25,
            save=path,
        )
        gp_opt1.optimize(verbose=1)
        gp_opt1.close_shelve()

        gp_optload = gp.GPOpt(
            objective_func=braninsc,
            lower_bound=np.array([-5, 0]),
            upper_bound=np.array([10, 15]),
        )

        gp_optload.load(path=path)
        gp_optload.optimize(verbose=1, n_more_iter=190, abs_tol=1e-4)

        self.assertTrue(np.allclose(gp_optload.n_iter, 74))
