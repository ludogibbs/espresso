/*
 * Copyright (C) 2022-2023 The ESPResSo project
 *
 * This file is part of ESPResSo.
 *
 * ESPResSo is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ESPResSo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <walberla_bridge/LatticeWalberla.hpp>
#include <walberla_bridge/electrokinetics/PoissonSolver/PoissonSolver.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

template <class EKSpecies> class EKContainer {
  using container_type = std::vector<std::shared_ptr<EKSpecies>>;

public:
  using value_type = typename container_type::value_type;
  using iterator = typename container_type::iterator;
  using const_iterator = typename container_type::const_iterator;

private:
  double m_tau;
  std::shared_ptr<walberla::PoissonSolver> m_poisson_solver;
  container_type m_ekcontainer;

  bool lattice_equal(LatticeWalberla const &lhs,
                     LatticeWalberla const &rhs) const {
    return (lhs.get_ghost_layers() == rhs.get_ghost_layers()) and
           (lhs.get_grid_dimensions() == rhs.get_grid_dimensions());
  }

  void sanity_checks(std::shared_ptr<EKSpecies> const &new_ek_species) const {
    if (not lattice_equal(new_ek_species->get_lattice(),
                          m_poisson_solver->get_lattice())) {
      throw std::runtime_error("EKSpecies lattice incompatible with existing "
                               "Poisson solver lattice");
    }
  }

  void sanity_checks(
      std::shared_ptr<walberla::PoissonSolver> const &new_ek_solver) const {
    if (not m_ekcontainer.empty()) {
      auto const &old_ek_species = m_ekcontainer.front();
      if (not lattice_equal(new_ek_solver->get_lattice(),
                            old_ek_species->get_lattice())) {
        throw std::runtime_error("Poisson solver lattice incompatible with "
                                 "existing EKSpecies lattice");
      }
    }
  }

public:
  EKContainer(double tau, std::shared_ptr<walberla::PoissonSolver> solver)
      : m_tau{tau}, m_poisson_solver{std::move(solver)}, m_ekcontainer{} {}

  bool contains(std::shared_ptr<EKSpecies> const &ek_species) const noexcept {
    return std::ranges::find(m_ekcontainer, ek_species) != m_ekcontainer.end();
  }

  void add(std::shared_ptr<EKSpecies> const &ek_species) {
    assert(not contains(ek_species));
    sanity_checks(ek_species);
    m_ekcontainer.emplace_back(ek_species);
  }

  void remove(std::shared_ptr<EKSpecies> const &ek_species) {
    assert(contains(ek_species));
    std::erase(m_ekcontainer, ek_species);
  }

  iterator begin() noexcept { return m_ekcontainer.begin(); }
  iterator end() noexcept { return m_ekcontainer.end(); }
  const_iterator begin() const noexcept { return m_ekcontainer.begin(); }
  const_iterator end() const noexcept { return m_ekcontainer.end(); }
  [[nodiscard]] bool empty() const noexcept { return m_ekcontainer.empty(); }

  void
  set_poisson_solver(std::shared_ptr<walberla::PoissonSolver> const &solver) {
    assert(solver != nullptr);
    sanity_checks(solver);
    m_poisson_solver = solver;
  }

  [[nodiscard]] double get_tau() const noexcept { return m_tau; }

  void set_tau(double tau) noexcept { m_tau = tau; }

  void reset_charge() const { m_poisson_solver->reset_charge_field(); }

  void add_charge(std::size_t const id, double valency,
                  bool is_double_precision) const {
    m_poisson_solver->add_charge_to_field(id, valency, is_double_precision);
  }

  void solve_poisson() const { m_poisson_solver->solve(); }

  [[nodiscard]] std::size_t get_potential_field_id() const {
    return m_poisson_solver->get_potential_field_id();
  }

  LatticeWalberla const &get_lattice() const noexcept {
    return m_poisson_solver->get_lattice();
  }
};
