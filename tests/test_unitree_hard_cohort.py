import unittest
import numpy as np

from unitree_hard_cohort import Geometry


def cells(boxes):
    return np.concatenate([np.array([(x, y) for x in np.arange(x0, x1+0.01, .1)
                                    for y in np.arange(y0, y1+0.01, .1)])
                           for x0, x1, y0, y1 in boxes])


class HardCohortTests(unittest.TestCase):
    def test_two_blockers_and_feasible_detour(self):
        g = Geometry(cells([(-1.3, -.7, -1.6, 1.6), (.7, 1.3, -1.6, 1.6)]), [6, 6], .1)
        audit = g.audit([-3, 0], [3, 0])
        self.assertEqual(audit['physical_blocking_components'], 2)
        self.assertGreaterEqual(audit['clearance_grid_route_ratio'], 1.25)
        route = np.array(audit['reference_route'])
        samples = np.concatenate([a + np.linspace(0, 1, 20)[:, None]*(b-a) for a, b in zip(route[:-1], route[1:])])
        self.assertGreaterEqual(g.tree.query(samples)[0].min()-g.cell_margin, .6)

    def test_straight_shortcut_rejected(self):
        g = Geometry(cells([(-1, 1, 2, 3)]), [6, 6], .1)
        with self.assertRaises(ValueError):
            g.audit([-3, 0], [3, 0])

    def test_inflated_near_miss_is_not_physical_blockage(self):
        g = Geometry(cells([(-1.3, -.7, .3, 1), (.7, 1.3, .3, 1)]), [6, 6], .1)
        self.assertEqual(g.blocking_components([-3, 0], [3, 0]), [])

    def test_unreachable_rejected(self):
        g = Geometry(cells([(-1.3, -.7, -6, 6), (.7, 1.3, -6, 6)]), [6, 6], .1)
        with self.assertRaises(ValueError):
            g.audit([-3, 0], [3, 0])

    def test_single_blocker_rejected(self):
        g = Geometry(cells([(-1, 1, -2, 2)]), [6, 6], .1)
        with self.assertRaises(ValueError):
            g.audit([-3, 0], [3, 0])

    def test_close_spawn_rejected(self):
        g = Geometry(cells([(-1.3, -.7, -1.6, 1.6), (.7, 1.3, -1.6, 1.6)]), [6, 6], .1)
        with self.assertRaises(ValueError):
            g.audit([-1.5, 0], [3, 0])


if __name__ == '__main__':
    unittest.main()
