/*
  Copyright (C) 2017 The ESPResSo project
    Max-Planck-Institute for Polymer Research, Theory Group

  This file is part of ESPResSo.

  ESPResSo is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  ESPResSo is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/** \file MpiCallbacks_test.cpp Unit tests for the MpiCallbacks class.
 *
*/

#include <boost/mpi.hpp>

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_MODULE ParticleCache test
#define BOOST_TEST_ALTERNATIVE_INIT_API
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "core/ParticleCache.hpp"
#include "utils/List.hpp"

#include "mock/Particle.hpp"

class Particle : public Testing::Particle {
public:
  Particle() = default;
  Particle(int id) : Testing::Particle(id) {}

  IntList bl;
};

struct Particles {
  std::vector<Particle> parts;

  std::vector<Particle> &particles() { return parts; }
};

BOOST_AUTO_TEST_CASE(update) {
  Particles local_parts;

  auto const rank = Communication::mpiCallbacks().comm().rank();
  auto const size = Communication::mpiCallbacks().comm().size();
  auto const n_part = 10000;

  local_parts.parts.reserve(n_part);

  for (int i = 0; i < n_part; i++) {
    local_parts.parts.emplace_back(rank * n_part + i);
  }

  ParticleCache<Particles> part_cfg{local_parts};

  if (rank == 0) {
    BOOST_CHECK(part_cfg.size() == size * n_part);

    for (int i = 0; i < size * n_part; i++) {
      BOOST_CHECK(i == part_cfg[i].identity());
    }

    Communication::mpiCallbacks().abort_loop();
  } else
    Communication::mpiCallbacks().loop();
}

BOOST_AUTO_TEST_CASE(update_with_bonds) {
  auto const bond_lengths = std::array<int, 6>{1, 2, 4, 9, 21, 0};

  Particles local_parts;

  auto const rank = Communication::mpiCallbacks().comm().rank();
  auto const size = Communication::mpiCallbacks().comm().size();
  auto const n_part = 1234;

  local_parts.parts.reserve(n_part);

  for (int i = 0; i < n_part; i++) {
    auto const id = rank * n_part + i;
    local_parts.parts.emplace_back(id);
    auto const bond_length = bond_lengths[id % bond_lengths.size()];
    auto &part = local_parts.parts.back();
    part.bl.e = nullptr;
    part.bl.max = 0;
    part.bl.resize(bond_length);
    part.bl.n = bond_length;
    std::fill(part.bl.begin(), part.bl.end(), id);
  }

  ParticleCache<Particles> part_cfg{local_parts};
  if (rank == 0) {
    part_cfg.update_bonds();

    for (int i = 0; i < size * n_part; i++) {
      /* Check that the length is set correctly */
      BOOST_CHECK(part_cfg[i].bl.size() ==
                  bond_lengths[part_cfg[i].identity() % bond_lengths.size()]);
      /* Check that the content was copied correctly. */
      BOOST_CHECK(std::all_of(part_cfg[i].bl.begin(), part_cfg[i].bl.end(),
                              [&i](int j) { return j == i; }));
    }
    Communication::mpiCallbacks().abort_loop();
  } else {
    Communication::mpiCallbacks().loop();
  }
}

BOOST_AUTO_TEST_CASE(iterators) {
  Particles local_parts;

  auto const rank = Communication::mpiCallbacks().comm().rank();
  auto const size = Communication::mpiCallbacks().comm().size();
  auto const n_part = 1000;

  local_parts.parts.reserve(n_part);

  for (int i = 0; i < n_part; i++) {
    local_parts.parts.emplace_back(rank * n_part + i);
  }

  ParticleCache<Particles> part_cfg{local_parts};

  if (rank == 0) {
    BOOST_CHECK(part_cfg.size() == size * n_part);

    std::vector<int> id_counts(size * n_part, 0);
    for (auto &p : part_cfg) {
      id_counts[p.identity()]++;
    }

    /* Every id should have been visitied exactly once... */
    BOOST_CHECK(std::all_of(id_counts.begin(), id_counts.end(),
                            [](int count) { return count == 1; }));
    /* and in the correct order. */
    BOOST_CHECK(std::is_sorted(part_cfg.begin(), part_cfg.end(),
                               [](Particle const &a, Particle const &b) {
                                 return a.identity() < b.identity();
                               }));
    Communication::mpiCallbacks().abort_loop();
  } else {
    Communication::mpiCallbacks().loop();
  }
}

int main(int argc, char **argv) {
  boost::mpi::environment mpi_env(argc, argv);

  boost::unit_test::unit_test_main(init_unit_test, argc, argv);
}
