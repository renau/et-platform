/*-------------------------------------------------------------------------
* Copyright (c) 2025 Ainekko, Co.
* SPDX-License-Identifier: Apache-2.0
*-------------------------------------------------------------------------*/

#include "emu_defines.h"
#include "system.h"
#include "memory/sysreg_region.h"
#include "memory/dense_region.h"

namespace bemu {

void MainMemory::reset()
{
    size_t pos = 0;

    // TODO: These are all simple memory, will have to be implemented as special types
    regions[pos++].reset(new DenseRegion<dram_base, 16_MiB>());
    regions[pos++].reset(new DenseRegion<bootrom_base, 8_KiB, false>());
    regions[pos++].reset(new DenseRegion<sram_base, 8_KiB>());
    regions[pos++].reset(new DenseRegion<erbreg_base, 64_KiB>());
    regions[pos++].reset(new SysregRegion<sysreg_base, 4_GiB>());
}

}
