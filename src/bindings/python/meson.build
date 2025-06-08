# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

pybind_dep = dependency('pybind11')

nixl_dep = declare_dependency(link_with: nixl_lib, include_directories: nixl_inc_dirs)

py = import('python').find_installation('python3', pure: false)

# Create Python extension modules
bindings_ext = py.extension_module('_bindings',
           'nixl_bindings.cpp',
           subdir: ('nixl'),
           dependencies: [nixl_dep, serdes_interface, pybind_dep],
           include_directories: [nixl_inc_dirs, utils_inc_dirs],
           install: true)

utils_ext = py.extension_module('_utils',
           'nixl_utils.cpp',
           subdir: ('nixl'),
           dependencies: [pybind_dep],
           install: true)

# Install Python package files
py.install_sources(['__init__.py'], subdir: 'nixl')

# Create wheel target
wheel_dir = meson.current_build_dir() / 'wheel'
wheel_name = 'nixl-0.3.0-py3-none-any.whl'

custom_target('wheel',
    output: wheel_name,
    command: [
        py.path(),
        '-m', 'pip',
        'wheel',
        '--no-deps',
        '--no-index',
        '--wheel-dir', wheel_dir,
        meson.current_source_dir()
    ],
    depends: [bindings_ext, utils_ext],
    install: true,
    install_dir: py.get_install_dir()
)
