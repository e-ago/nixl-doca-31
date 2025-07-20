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

#include <iostream>
#include <fstream>
#include <cassert>
#include <sys/stat.h>
#include <unistd.h>
#include <filesystem>
#include "utils/file/file_utils.h"

int main() {
    // Create temporary test files
    std::string test_file1 = "/tmp/nixl_test_file_1.txt";
    std::string test_file2 = "/tmp/nixl_test_file_2.txt";
    std::string non_existent_file = "/tmp/nixl_test_nonexistent_file.txt";

    // Create test file 1
    {
        std::ofstream test_file(test_file1);
        test_file << "Test content for file 1" << std::endl;
        test_file.close();
    }

    // Create test file 2
    {
        std::ofstream test_file(test_file2);
        test_file << "Test content for file 2" << std::endl;
        test_file.close();
    }

    // Test 1: queryFileInfo with existing file
    {
        nixl_query_resp_t resp;
        nixl_status_t status = queryFileInfo(test_file1, resp);

        assert(status == NIXL_SUCCESS);
        assert(resp.accessible == true);
        assert(resp.info.find("size") != resp.info.end());
        assert(resp.info.find("mode") != resp.info.end());
        assert(resp.info.find("mtime") != resp.info.end());

        std::cout << "✓ queryFileInfo with existing file test passed" << std::endl;
    }

    // Test 2: queryFileInfo with non-existent file
    {
        nixl_query_resp_t resp;
        nixl_status_t status = queryFileInfo(non_existent_file, resp);

        assert(status == NIXL_SUCCESS);
        assert(resp.accessible == false);
        assert(resp.info.empty());

        std::cout << "✓ queryFileInfo with non-existent file test passed" << std::endl;
    }

    // Test 3: queryFileInfo with empty filename
    {
        nixl_query_resp_t resp;
        nixl_status_t status = queryFileInfo("", resp);

        assert(status == NIXL_ERR_INVALID_PARAM);

        std::cout << "✓ queryFileInfo with empty filename test passed" << std::endl;
    }

    // Test 4: queryFileInfoList with multiple existing files
    {
        std::vector<std::string> filenames = {test_file1, test_file2};
        std::vector<nixl_query_resp_t> resp;
        nixl_status_t status = queryFileInfoList(filenames, resp);

        assert(status == NIXL_SUCCESS);
        assert(resp.size() == 2);
        assert(resp[0].accessible == true);
        assert(resp[1].accessible == true);
        assert(resp[0].info.find("size") != resp[0].info.end());
        assert(resp[1].info.find("size") != resp[1].info.end());

        std::cout << "✓ queryFileInfoList with existing files test passed" << std::endl;
    }

    // Test 5: queryFileInfoList with mixed existing and non-existent files
    {
        std::vector<std::string> filenames = {test_file1, non_existent_file, test_file2};
        std::vector<nixl_query_resp_t> resp;
        nixl_status_t status = queryFileInfoList(filenames, resp);

        assert(status == NIXL_SUCCESS);
        assert(resp.size() == 3);
        assert(resp[0].accessible == true);   // test_file1 exists
        assert(resp[1].accessible == false);  // non_existent_file doesn't exist
        assert(resp[2].accessible == true);   // test_file2 exists
        assert(resp[0].info.find("size") != resp[0].info.end());
        assert(resp[1].info.empty());         // No info for non-existent file
        assert(resp[2].info.find("size") != resp[2].info.end());

        std::cout << "✓ queryFileInfoList with mixed files test passed" << std::endl;
    }

    // Test 6: queryFileInfoList with empty vector
    {
        std::vector<std::string> filenames;
        std::vector<nixl_query_resp_t> resp;
        nixl_status_t status = queryFileInfoList(filenames, resp);

        assert(status == NIXL_SUCCESS);
        assert(resp.empty());

        std::cout << "✓ queryFileInfoList with empty vector test passed" << std::endl;
    }

    // Test 7: queryFileInfoList with empty filenames
    {
        std::vector<std::string> filenames = {"", "", ""};
        std::vector<nixl_query_resp_t> resp;
        nixl_status_t status = queryFileInfoList(filenames, resp);

        assert(status == NIXL_SUCCESS);
        assert(resp.size() == 3);
        assert(resp[0].accessible == false);
        assert(resp[1].accessible == false);
        assert(resp[2].accessible == false);

        std::cout << "✓ queryFileInfoList with empty filenames test passed" << std::endl;
    }

    // Clean up
    std::filesystem::remove(test_file1);
    std::filesystem::remove(test_file2);

    std::cout << "All file utils tests passed!" << std::endl;
    return 0;
}
