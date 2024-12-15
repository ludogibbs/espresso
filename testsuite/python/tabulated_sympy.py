#
# Copyright (C) 2013-2022 The ESPResSo project
#
# This file is part of ESPResSo.
#
# ESPResSo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ESPResSo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import unittest as ut
import unittest_decorators as utx
import espressomd
import espressomd.interactions
import numpy as np
from sympy import symbols, sympify

class TabulatedTest(ut.TestCase):
    system = espressomd.System(box_l=3 * [10.])
    system.time_step = 0.01
    system.cell_system.skin = 0.4

    def setUp(self):
        self.min_ = 1.
        self.max_ = 2.
        self.eps_ = 1.
        self.sig_ = 2.
        self.steps_ = 100
        self.force = [-4*self.eps_*(12/r*(self.sig_/r)**12-6/r*(self.sig_/r)**6) for r in np.linspace(self.min_,self.max_,self.steps_)]
        self.energy = [4*self.eps_*((self.sig_/r)**12-(self.sig_/r)**6) for r in np.linspace(self.min_,self.max_,self.steps_)]
        self.system.part.add(type=0, pos=[5., 5., 5.0])
        self.system.part.add(type=0, pos=[5., 5., 5.5])

    def tearDown(self):
        self.system.part.clear()

    def check(self):
        p0, p1 = self.system.part.all()
        # Below cutoff
        np.testing.assert_allclose(np.copy(self.system.part.all().f), 0.0)

        for i,z in enumerate(np.linspace(0, self.max_ - self.min_, self.steps_)):
            if z >= self.max_ - self.min_:
                continue
            p1.pos = [5., 5., 6. + z]
            self.system.integrator.run(0)
            np.testing.assert_allclose(
                np.copy(p0.f), [0., 0.,self.force[i]])
            np.testing.assert_allclose(np.copy(p0.f), -np.copy(p1.f))
            self.assertAlmostEqual(
                self.system.analysis.energy()['total'], self.energy[i])

    @utx.skipIfMissingFeatures("TABULATED")
    def test_tabulated_sympy(self):
        self.system.non_bonded_inter[0, 0].tabulated.set_params(
            min=self.min_, max=self.max_,steps=self.steps_,sig=self.sig_,eps=self.eps_, f="4*eps*((sig/r)**12-(sig/r)**6)")
        self.assertEqual(
            self.system.non_bonded_inter[0, 0].tabulated.cutoff, self.max_)
        self.check()

if __name__ == "__main__":
    ut.main()