/*-------------------------------------------------------------------------
* Copyright (c) 2025 Ainekko, Co.
* SPDX-License-Identifier: Apache-2.0
*-------------------------------------------------------------------------*/

#include "emu_defines.h"

#if EMU_ERBIUM
#include "memory/erbium/main_memory.cpp"
#elif EMU_ETSOC1
#include "memory/etsoc1/main_memory.cpp"
#else
#error "Architecture not supported."
#endif
