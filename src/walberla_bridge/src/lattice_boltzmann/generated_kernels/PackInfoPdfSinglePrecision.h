//======================================================================================================================
//
//  This file is part of waLBerla. waLBerla is free software: you can
//  redistribute it and/or modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation, either version 3 of
//  the License, or (at your option) any later version.
//
//  waLBerla is distributed in the hope that it will be useful, but WITHOUT
//  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
//  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
//  for more details.
//
//  You should have received a copy of the GNU General Public License along
//  with waLBerla (see COPYING.txt). If not, see <http://www.gnu.org/licenses/>.
//
//! \\file PackInfoPdfSinglePrecision.h
//! \\author pystencils
//======================================================================================================================

// kernel generated with pystencils v1.3.3, lbmpy v1.3.3,
// lbmpy_walberla/pystencils_walberla from waLBerla commit
// b0842e1a493ce19ef1bbb8d2cf382fc343970a7f

#pragma once
#include "communication/UniformPackInfo.h"
#include "core/DataTypes.h"
#include "core/cell/CellInterval.h"
#include "domain_decomposition/IBlock.h"
#include "field/GhostLayerField.h"
#include "stencil/Directions.h"

#define FUNC_PREFIX

#ifdef __GNUC__
#define RESTRICT __restrict__
#elif _MSC_VER
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

namespace walberla {
namespace pystencils {

class PackInfoPdfSinglePrecision
    : public ::walberla::communication::UniformPackInfo {
public:
  PackInfoPdfSinglePrecision(BlockDataID pdfsID_) : pdfsID(pdfsID_){};
  virtual ~PackInfoPdfSinglePrecision() {}

  bool constantDataExchange() const { return true; }
  bool threadsafeReceiving() const { return true; }

  void unpackData(IBlock *receiver, stencil::Direction dir,
                  mpi::RecvBuffer &buffer) {
    const auto dataSize = size(dir, receiver);
    unpack(dir, buffer.skip(dataSize + sizeof(float)), receiver);
  }

  void communicateLocal(const IBlock *sender, IBlock *receiver,
                        stencil::Direction dir) {
    mpi::SendBuffer sBuffer;
    packData(sender, dir, sBuffer);
    mpi::RecvBuffer rBuffer(sBuffer);
    unpackData(receiver, stencil::inverseDir[dir], rBuffer);
  }

  void packDataImpl(const IBlock *sender, stencil::Direction dir,
                    mpi::SendBuffer &outBuffer) const {
    const auto dataSize = size(dir, sender);
    pack(dir, outBuffer.forward(dataSize + sizeof(float)),
         const_cast<IBlock *>(sender));
  }

  void pack(stencil::Direction dir, unsigned char *buffer, IBlock *block) const;
  void unpack(stencil::Direction dir, unsigned char *buffer,
              IBlock *block) const;
  uint_t size(stencil::Direction dir, const IBlock *block) const;

private:
  BlockDataID pdfsID;
};

} // namespace pystencils
} // namespace walberla