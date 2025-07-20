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

#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>
#include "utils/file/file_utils.h"

class QueryMemTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temporary test files
        test_file1 = "/tmp/test_query_mem_1.txt";
        test_file2 = "/tmp/test_query_mem_2.txt";
        non_existent_file = "/tmp/non_existent_file.txt";

        // Create test file 1
        {
            std::ofstream file(test_file1);
            file << "Test content for file 1" << std::endl;
            file.close();
        }

        // Create test file 2
        {
            std::ofstream file(test_file2);
            file << "Test content for file 2" << std::endl;
            file.close();
        }
    }

    void TearDown() override {
        // Clean up test files
        std::filesystem::remove(test_file1);
        std::filesystem::remove(test_file2);
    }

    std::string test_file1;
    std::string test_file2;
    std::string non_existent_file;
};


