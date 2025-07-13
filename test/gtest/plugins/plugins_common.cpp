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

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <absl/strings/str_format.h>

#include "nixl.h"
#include "common/nixl_log.h"
#include "plugin_manager.h"
#include "backend_engine.h"
#include "plugins_common.h"

namespace plugins_common {

nixl_status_t
SetupBackendTestFixture::backendAllocReg(nixlBackendEngine *engine,
                                        nixl_mem_t mem_type,
                                        size_t len,
                                        std::unique_ptr<MemoryHandler>& mem_handler,
                                        nixlBackendMD *&md,
                                        int dev_id) {
    nixlBlobDesc desc;
    nixl_status_t ret;

    mem_handler = std::make_unique<MemoryHandler>(mem_type, dev_id);

    try {
        mem_handler->allocate(len);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to allocate memory: " << e.what();
        return NIXL_ERR_BACKEND;
    }

    mem_handler->populateBlobDesc(&desc);

    NIXL_INFO << "Registering memory type " << mem_handler->getMemType() << " with length " << len << " and device ID " << dev_id;

    ret = engine->registerMem(desc, mem_handler->getMemType(), md);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to register memory: " << ret;
        return ret;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
SetupBackendTestFixture::backendDeregDealloc(nixlBackendEngine *engine,
                                             std::unique_ptr<MemoryHandler>& mem_handler,
                                             nixlBackendMD *&md,
                                             int dev_id) {
    nixl_status_t ret;

    NIXL_INFO << "Deregistering memory type " << mem_handler->getMemType() << " with device ID " << dev_id;

    ret = engine->deregisterMem(md);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to deregister memory: " << ret;
        return ret;
    }

    mem_handler->deallocate();

    return NIXL_SUCCESS;
}

void
SetupBackendTestFixture::ResetLocalBuf() {
    localMemHandler_->reset();
}

bool
SetupBackendTestFixture::CheckLocalBuf() {
    return localMemHandler_->check(LOCAL_BUF_BYTE);
}

void
SetupBackendTestFixture::populateDescList(nixl_meta_dlist_t &descs,
                                          std::unique_ptr<MemoryHandler>& mem_handler,
                                          nixlBackendMD *&md,
                                          int dev_id) {
    nixlMetaDesc req;
    mem_handler->populateMetaDesc(&req, md);
    descs.addDesc(req);
}

void
SetupBackendTestFixture::SetUp() {
    if (backend_engine_->getInitErr()) {
        CHECK(false) << "Failed to initialize backend engine";
    }

    localAgent_ = "Agent1";
    remoteAgent_ = "Agent2";

    isSetup_ = true;
}

bool
SetupBackendTestFixture::VerifyConnInfo() {
    std::string conn_info;
    nixl_status_t ret;

    ret = backend_engine_->getConnInfo(conn_info);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to get connection info";
        return false;
    }

    ret = xferBackendEngine_->getConnInfo(conn_info);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to get remote connection info";
        return false;
    }

    ret = backend_engine_->loadRemoteConnInfo(xferAgent_, conn_info);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to load remote connection info";
        return false;
    }

    return true;
}

void
SetupBackendTestFixture::SetupNotifs(std::string msg) {
    optionalXferArgs_.notifMsg = msg;
    optionalXferArgs_.hasNotif = true;
}

bool
SetupBackendTestFixture::PrepXferMem(nixl_mem_t local_mem_type, nixl_mem_t xfer_mem_type, bool is_remote) {
    nixl_status_t ret;

    ret = backendAllocReg(backend_engine_.get(), local_mem_type, BUF_SIZE, localMemHandler_, localMD_, localDevId_);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to register local memory";
        return false;
    }

    ret = backendAllocReg(xferBackendEngine_, xfer_mem_type, BUF_SIZE, xferMemHandler_, xferMD_, xferDevId_);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to register xfer memory";
        return false;
    }

    if (is_remote) {
        nixlBlobDesc info;
        xferMemHandler_->populateBlobDesc(&info);
        ret = backend_engine_->getPublicData(xferMD_, info.metaInfo);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to get meta info";
            return false;
        }
        if (info.metaInfo.size() == 0) {
            NIXL_ERROR << "Failed to get meta info";
            return false;
        }

        ret = backend_engine_->loadRemoteMD(info, xfer_mem_type, xferAgent_, xferLoadedMem_);
    } else {
        ret = backend_engine_->loadLocalMD(xferMD_, xferLoadedMem_);
    }
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to load MD from " << xferAgent_;
        return false;
    }

    reqSrcDescs_ = std::make_unique<nixl_meta_dlist_t>(local_mem_type);
    reqDstDescs_ = std::make_unique<nixl_meta_dlist_t>(xfer_mem_type);
    populateDescList(*reqSrcDescs_, localMemHandler_, localMD_, localDevId_);
    populateDescList(*reqDstDescs_, xferMemHandler_, xferLoadedMem_, xferDevId_);

    localMemHandler_->set(LOCAL_BUF_BYTE);
    xferMemHandler_->set(XFER_BUF_BYTE);

    return true;
}

bool
SetupBackendTestFixture::TestXfer(nixl_xfer_op_t op) {
    nixlBackendReqH *handle_;
    nixl_status_t ret;

    ret = backend_engine_->prepXfer(
            op, *reqSrcDescs_, *reqDstDescs_, xferAgent_, handle_, &optionalXferArgs_);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to prepare transfer";
        return false;
    }

    ret = backend_engine_->postXfer(
            op, *reqSrcDescs_, *reqDstDescs_, xferAgent_, handle_, &optionalXferArgs_);
    if (ret != NIXL_SUCCESS && ret != NIXL_IN_PROG) {
        NIXL_ERROR << "Failed to post transfer";
        return false;
    }

    auto end_time = absl::Now() + absl::Seconds(3);

    NIXL_INFO << "\t\tWaiting for transfer to complete...";

    while (ret == NIXL_IN_PROG && absl::Now() < end_time) {
        ret = backend_engine_->checkXfer(handle_);
        if (ret != NIXL_SUCCESS && ret != NIXL_IN_PROG) {
            NIXL_ERROR << "Transfer check failed";
            return false;
        }

        if (xferBackendEngine_->supportsProgTh()) {
            xferBackendEngine_->progress();
        }
    }

    NIXL_INFO << "\nTransfer complete";

    ret = backend_engine_->releaseReqH(handle_);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to release transfer handle";
        return false;
    }

    return true;
}

bool
SetupBackendTestFixture::VerifyNotifs(std::string &msg) {
    notif_list_t target_notifs;
    int num_notifs = 0;
    nixl_status_t ret;

    NIXL_INFO << "\t\tChecking notification flow: ";

    auto end_time = absl::Now() + absl::Seconds(3);

    while (num_notifs == 0 && absl::Now() < end_time) {
        ret = xferBackendEngine_->getNotifs(target_notifs);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to get notifications";
            return false;
        }
        num_notifs = target_notifs.size();
        if (backend_engine_->supportsProgTh()) {
            backend_engine_->progress();
        }
    }

    NIXL_INFO << "\nNotification transfer complete";

    if (num_notifs != 1) {
        NIXL_ERROR << "Expected 1 notification, got " << num_notifs;
        return false;
    }

    if (target_notifs.front().first != localAgent_) {
        NIXL_ERROR << "Expected notification from " << localAgent_ << ", got "
                  << target_notifs.front().first;
        return false;
    }
    if (target_notifs.front().second != msg) {
        NIXL_ERROR << "Expected notification message " << msg << ", got "
                  << target_notifs.front().second;
        return false;
    }

    NIXL_INFO << "OK\n"
              << "message: " << target_notifs.front().second << " from "
              << target_notifs.front().first;

    return true;
}

bool
SetupBackendTestFixture::VerifyXfer() {
    if (backend_engine_->supportsNotif()) {
        if (!VerifyNotifs(optionalXferArgs_.notifMsg)) {
            NIXL_ERROR << "Failed in notifications verification";
            return false;
        }

        optionalXferArgs_.notifMsg = "";
        optionalXferArgs_.hasNotif = false;
    }

    return true;
}

bool
SetupBackendTestFixture::TeardownXfer() {
    nixl_status_t ret;

    ret = backend_engine_->unloadMD(xferLoadedMem_);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to unload MD";
        return false;
    }

    ret = backend_engine_->disconnect(xferAgent_);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to disconnect";
        return false;
    }

    ret = backendDeregDealloc(xferBackendEngine_, xferMemHandler_, xferMD_, xferDevId_);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to deallocate xfer memory";
        return false;
    }

    ret = backendDeregDealloc(backend_engine_.get(), localMemHandler_, localMD_, localDevId_);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to deallocate local memory";
        return false;
    }

    return true;
}

bool
SetupBackendTestFixture::SetupLocalXfer(nixl_mem_t local_mem_type, nixl_mem_t xfer_mem_type) {
    CHECK(backend_engine_->supportsLocal()) << "Backend engine does not support local transfers";

    xferBackendEngine_ = backend_engine_.get();
    xferAgent_ = localAgent_;
    localDevId_ = 0;
    xferDevId_ = 0;

    if (xferBackendEngine_->supportsNotif()) {
        SetupNotifs("Test");
    }

    if (!PrepXferMem(local_mem_type, xfer_mem_type, false /* local xfer */)) {
        NIXL_ERROR << "Failed to prepare transfer";
        return false;
    }

    return true;
}

bool
SetupBackendTestFixture::TestLocalXfer(nixl_xfer_op_t op) {
    if (!TestXfer(op)) {
        NIXL_ERROR << "Failed to test transfer";
        return false;
    }

    if (!VerifyXfer()) {
        NIXL_ERROR << "Failed in transfer verification";
        return false;
    }

    return true;
}

bool
SetupBackendTestFixture::SetupRemoteXfer(nixl_mem_t local_mem_type, nixl_mem_t xfer_mem_type) {
    CHECK(backend_engine_->supportsRemote()) << "Backend engine does not support remote transfers";

    xferBackendEngine_ = remote_backend_engine_.get();
    xferAgent_ = remoteAgent_;
    localDevId_ = 0;
    xferDevId_ = 1;

    if (!VerifyConnInfo()) {
        NIXL_ERROR << "Failed to verify connection info";
        return false;
    }

    if (backend_engine_->supportsNotif()) {
        SetupNotifs("Test");
    }

    if (!PrepXferMem(local_mem_type, xfer_mem_type, true /* remote xfer */)) {
        NIXL_ERROR << "Failed to prepare transfer";
        return false;
    }

    return true;
}

bool
SetupBackendTestFixture::TestRemoteXfer(nixl_xfer_op_t op) {
    if (!TestXfer(op)) {
        NIXL_ERROR << "Failed to test transfer";
        return false;
    }

    if (!VerifyXfer()) {
        NIXL_ERROR << "Failed in transfer verification";
        return false;
    }

    return true;
}

bool
SetupBackendTestFixture::TestGenNotif(std::string msg) {
    nixl_status_t ret;

    ret = backend_engine_->genNotif(remoteAgent_, msg);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to generate notification";
        return false;
    }

    if (!VerifyNotifs(msg)) {
        NIXL_ERROR << "Failed in notification verification";
        return false;
    }

    return true;
}

} // namespace plugins_common
