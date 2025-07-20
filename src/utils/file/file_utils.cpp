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
#include "file_utils.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>
#include <cstring>
#include <iostream>

nixl_status_t queryFileInfo(const std::string& filename, nixl_query_resp_t& resp) {
    // Initialize response
    resp.accessible = false;
    resp.info.clear();

    if (filename.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    // Check if file exists using stat
    struct stat stat_buf;
    bool file_exists = (stat(filename.c_str(), &stat_buf) == 0);

    resp.accessible = file_exists;

    // Add additional file information if file exists
    if (file_exists) {
        resp.info["size"] = std::to_string(stat_buf.st_size);
        resp.info["mode"] = std::to_string(stat_buf.st_mode);
        resp.info["mtime"] = std::to_string(stat_buf.st_mtime);
    }

    return NIXL_SUCCESS;
}

nixl_status_t queryFileInfoList(const std::vector<std::string>& filenames, std::vector<nixl_query_resp_t>& resp) {
    resp.clear();
    resp.reserve(filenames.size());

    for (const auto& filename : filenames) {
        nixl_query_resp_t query_resp;
        nixl_status_t status = queryFileInfo(filename, query_resp);

        if (status != NIXL_SUCCESS) {
            // If one file fails, mark it as inaccessible but continue
            query_resp.accessible = false;
            query_resp.info.clear();
        }

        resp.push_back(query_resp);
    }

    return NIXL_SUCCESS;
}
