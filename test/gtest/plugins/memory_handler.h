/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef __MEMORY_HANDLER_H
#define __MEMORY_HANDLER_H

#include "backend/backend_aux.h"
#include "absl/log/check.h"

class MemoryHandler {
private:
    nixl_mem_t memType_;
    void *addr_;
    size_t len_;
    int devId_;

public:
    MemoryHandler(nixl_mem_t memType, int devId) : memType_(memType), devId_(devId) {}

    void
    allocate(size_t len) {
        switch (memType_) {
            case DRAM_SEG:
                addr_ = new char[len];
                break;
            case OBJ_SEG:
                addr_ = nullptr;
                break;
            default:
                CHECK(false) << "Unsupported memory type!";
                break;
        }
        len_ = len;
    }

    void
    deallocate() {
        switch (memType_) {
            case DRAM_SEG:
                delete[] static_cast<char *>(addr_);
                break;
            case OBJ_SEG:
                break;
            default:
                CHECK(false) << "Unsupported memory type!";
                break;
        }
    }

    void
    set(char byte) {
        switch (memType_) {
            case DRAM_SEG:
                for (size_t i = 0; i < len_; i++)
                    ((char *)addr_)[i] = byte + i;
                break;
            case OBJ_SEG:
                break;
            default:
                CHECK(false) << "Unsupported memory type!";
                break;
        }
    }
    
    void
    populateBlobDesc(nixlBlobDesc *desc) {
        switch (memType_) {
            case DRAM_SEG:
                break;
            case OBJ_SEG:
                desc->metaInfo = "test-obj-key";
                break;
            default:
                CHECK(false) << "Unsupported memory type!";
                break;
        }
        desc->addr = reinterpret_cast<uintptr_t>(addr_);
        desc->len = len_;
        desc->devId = devId_;
    }

    void
    populateMetaDesc(nixlMetaDesc *desc, nixlBackendMD *&md) {
        switch (memType_) {
            case DRAM_SEG:
                break;
            case OBJ_SEG:
                break;
            default:
                CHECK(false) << "Unsupported memory type!";
                break;
        }
        desc->addr = reinterpret_cast<uintptr_t>(addr_);
        desc->len = len_;
        desc->devId = devId_;
        desc->metadataP = md;
    }
};

#endif // __MEMORY_HANDLER_H