# File Utils and QueryMem API Implementation

This directory contains the implementation of file utilities for NIXL file backends and the QueryMem API.

## Overview

The implementation provides:

1. **QueryMem API**: Implementation of the QueryMem API for file backends that uses `nixl_reg_dlist_t` with filenames in `metaInfo` and uses `stat` to check file existence.

2. **File Utils (`file_utils.h` and `file_utils.cpp`)**: Utility functions for file operations including file query functions.

## QueryMem API Implementation

The QueryMem API has been implemented for all file backends:

- **POSIX Backend** (`src/plugins/posix/`)
- **HF3FS Backend** (`src/plugins/hf3fs/`)
- **GDS MT Backend** (`src/plugins/gds_mt/`)
- **CUDA GDS Backend** (`src/plugins/cuda_gds/`)

### How it works:

1. **Input**: Takes a `nixl_reg_dlist_t` containing file descriptors with filenames in the `metaInfo` field
2. **Processing**:
   - Uses `extractMetadata()` method from `nixlDescList` class to extract filenames from descriptors
   - Calls `queryFileInfoList()` to check file existence using `stat`
   - Strips any prefixes (RO:, RW:, WR:) before checking file existence
3. **Output**: Returns a vector of `nixl_query_resp_t` structures containing:
   - `accessible`: Boolean indicating if file exists
   - `info`: Additional file information (size, mode, mtime) if file exists

### Usage Example:

```cpp
// Create registration descriptor list with filenames in metaInfo
nixl_reg_dlist_t descs(FILE_SEG, false);
descs.addDesc(nixlBlobDesc(0, 0, 0, "/path/to/file1.txt"));
descs.addDesc(nixlBlobDesc(0, 0, 0, "/path/to/file2.txt"));
descs.addDesc(nixlBlobDesc(0, 0, 0, "/path/to/file3.txt"));

// Query file status using the plugin's queryMem method
std::vector<nixl_query_resp_t> resp;
nixl_status_t status = plugin->queryMem(descs, resp);

// Check results
for (const auto& result : resp) {
    if (result.accessible) {
        std::cout << "File exists, size: " << result.info["size"] << std::endl;
    } else {
        std::cout << "File does not exist" << std::endl;
    }
}
```

## File Utils Functions

### `queryFileInfo`
- **Purpose**: Query file information for a single file
- **Parameters**:
  - `filename`: The filename to query
  - `resp`: Output response structure
- **Returns**: NIXL_SUCCESS on success, error code otherwise

### `queryFileInfoList`
- **Purpose**: Query file information for multiple files
- **Parameters**:
  - `filenames`: Vector of filenames to query
  - `resp`: Output response vector
- **Returns**: NIXL_SUCCESS on success, error code otherwise

## Architecture

The current architecture separates concerns:

1. **Descriptor Operations**: The `nixlDescList` class provides `extractMetadata()` method to extract metadata from descriptors
2. **File Operations**: The `file_utils` provides generic file query functions (`queryFileInfo`, `queryFileInfoList`)
3. **Plugin Integration**: Each plugin directly uses `extractMetadata()` and `queryFileInfoList()` without intermediate layers
4. **File Descriptor Management**: Plugins currently use `devId` directly as file descriptor

This approach eliminates the need for the `file_query_helper` layer and provides better separation of concerns.

## Building

The file utils are built as a shared library (`libfile_utils.so`) and linked with all file backends. The build system has been updated to include the file utils dependency in all relevant backend meson.build files.



## Testing

Test files are provided:
- `test/unit/utils/file/test_file_utils.cpp`: Tests the file utils functions
- `test/python/test_query_mem.py`: Python tests for QueryMem API functionality

## Dependencies

- Standard C++ libraries
- POSIX system calls (`stat`, `open`, `close`)
- NIXL common library for logging
