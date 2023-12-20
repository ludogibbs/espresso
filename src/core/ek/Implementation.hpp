/*
 * Copyright (C) 2023 The ESPResSo project
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

#include "config/config.hpp"

#include "ek/Solver.hpp"

#include "ek/EKNone.hpp"
#include "ek/EKWalberla.hpp"

#include <memory>
#include <optional>
#include <variant>

namespace EK {

using DiffusionAdvectionReactionActor = std::variant<
#ifdef WALBERLA
    std::shared_ptr<EK::EKWalberla>,
#endif
    std::shared_ptr<EK::EKNone>>;

struct Solver::Implementation {
  /// @brief Main diffusion-advection-reaction solver.
  std::optional<DiffusionAdvectionReactionActor> solver;

  Implementation() : solver{} {}
};

} // namespace EK