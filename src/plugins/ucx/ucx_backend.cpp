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

#include "ucx_backend.h"
#include "common/nixl_log.h"
#include "serdes/serdes.h"
#include "common/nixl_log.h"

#include <optional>
#include <limits>
#include <list>
#include <string.h>
#include <unistd.h>
#include "absl/strings/numbers.h"
#include <asio.hpp>

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <cufile.h>

#endif

namespace {
    void moveNotifList(notif_list_t &src, notif_list_t &tgt)
    {
        if (src.size() > 0) {
            std::move(src.begin(), src.end(), std::back_inserter(tgt));
            src.clear();
        }
    }
}

/****************************************
 * CUDA related code
 *****************************************/

class nixlUcxCudaCtx {
public:
#ifdef HAVE_CUDA
    CUcontext pthrCudaCtx;
    int myDevId;

    nixlUcxCudaCtx() {
        pthrCudaCtx = NULL;
        myDevId = -1;
    }
#endif
    void cudaResetCtxPtr();
    int cudaUpdateCtxPtr(void *address, int expected_dev, bool &was_updated);
    int cudaSetCtx();
};

class nixlUcxCudaDevicePrimaryCtx {
#ifndef HAVE_CUDA
public:
    bool push() { return false; }
    void pop() {};
#else
    static constexpr int defaultCudaDeviceOrdinal = 0;
    int m_ordinal{defaultCudaDeviceOrdinal};
    CUdevice m_device{CU_DEVICE_INVALID};
    CUcontext m_context{nullptr};
public:

    bool push() {
        CUcontext context;

        const auto res = cuCtxGetCurrent(&context);
        if (res != CUDA_SUCCESS || context != nullptr) {
            return false;
        }

        if (m_context == nullptr) {
            CUresult res = cuDeviceGet(&m_device, m_ordinal);
            if (res != CUDA_SUCCESS) {
                return false;
            }

            res = cuDevicePrimaryCtxRetain(&m_context, m_device);
            if (res != CUDA_SUCCESS) {
                m_context = nullptr;
                return false;
            }
        }

        return cuCtxPushCurrent(m_context) == CUDA_SUCCESS;
    }

    void pop() {
        cuCtxPopCurrent(nullptr);
    }

    ~nixlUcxCudaDevicePrimaryCtx() {
        if (m_context != nullptr) {
            cuDevicePrimaryCtxRelease(m_device);
        }
    }
#endif
};

class nixlUcxCudaCtxGuard {
    nixlUcxCudaDevicePrimaryCtxPtr m_primary;
public:
    nixlUcxCudaCtxGuard(nixl_mem_t nixl_mem,
                        nixlUcxCudaDevicePrimaryCtxPtr primary) {
        if (nixl_mem == VRAM_SEG && primary && primary->push()) {
            m_primary = primary;
        }
    }
    ~nixlUcxCudaCtxGuard() {
        if (m_primary) {
            m_primary->pop();
        }
    }
};

#ifdef HAVE_CUDA

static int cudaQueryAddr(void *address, bool &is_dev,
                         CUdevice &dev, CUcontext &ctx)
{
    CUmemorytype mem_type = CU_MEMORYTYPE_HOST;
    uint32_t is_managed = 0;
#define NUM_ATTRS 4
    CUpointer_attribute attr_type[NUM_ATTRS];
    void *attr_data[NUM_ATTRS];
    CUresult result;

    attr_type[0] = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
    attr_data[0] = &mem_type;
    attr_type[1] = CU_POINTER_ATTRIBUTE_IS_MANAGED;
    attr_data[1] = &is_managed;
    attr_type[2] = CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;
    attr_data[2] = &dev;
    attr_type[3] = CU_POINTER_ATTRIBUTE_CONTEXT;
    attr_data[3] = &ctx;

    result = cuPointerGetAttributes(4, attr_type, attr_data, (CUdeviceptr)address);

    is_dev = (mem_type == CU_MEMORYTYPE_DEVICE);

    return (CUDA_SUCCESS != result);
}

int nixlUcxCudaCtx::cudaUpdateCtxPtr(void *address, int expected_dev, bool &was_updated)
{
    bool is_dev;
    CUdevice dev;
    CUcontext ctx;
    int ret;

    was_updated = false;

    /* TODO: proper error codes and log outputs through this method */
    if (expected_dev == -1)
        return -1;

    // incorrect dev id from first registration
    if (myDevId != -1 && expected_dev != myDevId)
        return -1;

    ret = cudaQueryAddr(address, is_dev, dev, ctx);
    if (ret) {
        return ret;
    }

    if (!is_dev) {
        return 0;
    }

    if (dev != expected_dev) {
        // User provided address that does not match dev_id
        return -1;
    }

    if (pthrCudaCtx) {
        // Context was already set previously, and does not match new context
        if (pthrCudaCtx != ctx) {
            return -1;
        }
        return 0;
    }

    pthrCudaCtx = ctx;
    was_updated = true;
    myDevId = expected_dev;

    return 0;
}

int nixlUcxCudaCtx::cudaSetCtx()
{
    CUresult result;
    if (NULL == pthrCudaCtx) {
        return 0;
    }

    result = cuCtxSetCurrent(pthrCudaCtx);

    return (CUDA_SUCCESS == result);
}

#else

int nixlUcxCudaCtx::cudaUpdateCtxPtr(void *address, int expected_dev, bool &was_updated)
{
    was_updated = false;
    return 0;
}

int nixlUcxCudaCtx::cudaSetCtx() {
    return 0;
}

#endif


void nixlUcxEngine::vramInitCtx()
{
    cudaCtx = std::make_unique<nixlUcxCudaCtx>();
}

int nixlUcxEngine::vramUpdateCtx(void *address, uint64_t  devId, bool &restart_reqd)
{
    int ret;
    bool was_updated;

    restart_reqd = false;

    if(!cuda_addr_wa) {
        // Nothing to do
        return 0;
    }

    ret = cudaCtx->cudaUpdateCtxPtr(address, devId, was_updated);
    if (ret) {
        return ret;
    }

    restart_reqd = was_updated;

    return 0;
}

int nixlUcxEngine::vramApplyCtx()
{
    if(!cuda_addr_wa) {
        // Nothing to do
        return 0;
    }

    return cudaCtx->cudaSetCtx();
}

void nixlUcxEngine::vramFiniCtx()
{
    cudaCtx.reset();
}

/****************************************
 * UCX request management
*****************************************/


class nixlUcxIntReq : public nixlLinkElem<nixlUcxIntReq> {
    private:
        int _completed;
    public:
        std::unique_ptr<std::string> amBuffer;

        nixlUcxIntReq() : nixlLinkElem() {
            _completed = 0;
        }

        bool is_complete() const { return _completed; }
        void completed() { _completed = 1; }
};

static void _internalRequestInit(void *request)
{
    /* Initialize request in-place (aka "placement new")*/
    new(request) nixlUcxIntReq;
}

static void _internalRequestFini(void *request)
{
    /* Finalize request */
    nixlUcxIntReq *req = (nixlUcxIntReq*)request;
    req->~nixlUcxIntReq();
}


static void _internalRequestReset(nixlUcxIntReq *req) {
    _internalRequestFini((void *)req);
    _internalRequestInit((void *)req);
}

/****************************************
 * Backend request management
*****************************************/

class nixlUcxBackendH : public nixlBackendReqH {
protected:
    // TODO: use std::vector here for a single allocation and cache friendly
    // traversal
    nixlUcxIntReq head;
    nixlUcxWorker *worker;
    size_t worker_id;
    size_t num_chunks;
    std::atomic<bool> is_completed;

    // Notification to be sent after completion of all requests
    struct Notif {
        std::string agent;
        nixl_blob_t payload;
        Notif(const std::string& remote_agent, const nixl_blob_t& msg)
            : agent(remote_agent), payload(msg) {}
    };
    std::optional<Notif> notif;

public:
    // TODO: make this configurable?
    static constexpr uint64_t MAX_PROGRESS = 1000;

    nixlUcxBackendH() :
        worker(nullptr), worker_id(UINT64_MAX), num_chunks(1), is_completed(true) {}

    nixlUcxBackendH(nixlUcxWorker *worker_, size_t worker_id_) :
        worker(worker_), worker_id(worker_id_), num_chunks(1), is_completed(true) {}

    nixlUcxBackendH(nixlUcxBackendH &&other) noexcept :
        head(std::move(other.head)),
        worker(other.worker),
        worker_id(other.worker_id),
        num_chunks(other.num_chunks),
        is_completed(other.is_completed.load()),
        notif(std::move(other.notif)) {
    }

    auto& notification() {
        return notif;
    }

    size_t getNumChunks() const {
        return num_chunks;
    }

    void start(nixlUcxWorker *worker_, size_t worker_id_) {
        worker = worker_;
        worker_id = worker_id_;
        is_completed.store(false);
    }

    void complete() {
        worker = nullptr;
        worker_id = UINT64_MAX;
        is_completed.store(true);
    }

    bool isCompleted() const {
        return is_completed.load();
    }

    void append(nixlUcxIntReq *req) {
        head.link(req);
    }

    virtual nixl_status_t release() {
        nixlUcxIntReq *req = head.next();

        if (!req) {
            return NIXL_SUCCESS;
        }

        // TODO: Error log: uncompleted requests found! Cancelling ...
        while(req) {
            nixlUcxIntReq *cur = req;
            bool done = cur->is_complete();
            req = cur->unlink();
            if (!done) {
                // TODO: Need process this properly.
                // it may not be enough to cancel UCX request
                worker->reqCancel((nixlUcxReq)cur);
            }
            _internalRequestReset(cur);
            worker->reqRelease((nixlUcxReq)cur);
        }
        return NIXL_SUCCESS;
    }

    virtual nixl_status_t status(uint64_t maxProgress = MAX_PROGRESS) {
        nixlUcxIntReq *req = head.next();
        nixl_status_t out_ret = NIXL_SUCCESS;

        if (NULL == req) {
            /* No pending transmissions */
            return NIXL_SUCCESS;
        }

        /* Maximum progress */
        for (uint64_t i = 0; i < maxProgress && worker->progress(); i++);

        /* Go over all request updating their status */
        while (req) {
            nixl_status_t ret;
            if (!req->is_complete()) {
                ret = ucx_status_to_nixl(ucp_request_check_status((nixlUcxReq)req));
                switch (ret) {
                    case NIXL_SUCCESS:
                        /* Mark as completed */
                        req->completed();
                        break;
                    case NIXL_IN_PROG:
                        out_ret = NIXL_IN_PROG;
                        break;
                    default:
                        /* Any other ret value is ERR and will be returned */
                        return ret;
                }
            }
            req = req->next();
        }

        /* Remove completed requests keeping the first one as
        request representative */
        req = head.unlink();
        while (req) {
            nixlUcxIntReq *next_req = req->unlink();
            if (req->is_complete()) {
                _internalRequestReset(req);
                worker->reqRelease((nixlUcxReq)req);
            } else {
                /* Enqueue back */
                append(req);
            }
            req = next_req;
        }

        return out_ret;
    }

    size_t getWorkerId() const {
        return worker_id;
    }
};

/****************************************
 * Progress thread management
*****************************************/

nixlUcxThreadEngine::nixlUcxThreadEngine(const nixlBackendInitParams &init_params)
    : nixlUcxEngine(init_params) {
    if (!nixlUcxMtLevelIsSupported(nixl_ucx_mt_t::WORKER)) {
        NIXL_ERROR << "UCX library does not support multi-threading";
        this->initErr = true;
        return;
    }
    if (pipe(pthrControlPipe) < 0) {
        NIXL_PERROR << "Couldn't create progress thread control pipe";
        this->initErr = true;
        return;
    }

    // This will ensure that the resulting delay is at least 1ms and fits into int in order for
    // it to be compatible with poll()
    pthrDelay = std::chrono::ceil<std::chrono::milliseconds>(
        std::chrono::microseconds(init_params.pthrDelay < std::numeric_limits<int>::max() ?
                                  init_params.pthrDelay : std::numeric_limits<int>::max()));

    for (auto &uw: uws) {
        pollFds.push_back({uw->getEfd(), POLLIN, 0});
    }
    pollFds.push_back({pthrControlPipe[0], POLLIN, 0});

    progressThreadStart();
}

nixlUcxThreadEngine::~nixlUcxThreadEngine() {
    progressThreadStop();
    close(pthrControlPipe[0]);
    close(pthrControlPipe[1]);
}

void nixlUcxThreadEngine::progressFunc()
{
    using namespace nixlTime;

    nixlUcxEngine::vramApplyCtx();

    {
        std::unique_lock<std::mutex> lock(pthrActiveLock);
        pthrActive = true;
    }
    pthrActiveCV.notify_one();

    // Set timeout event so that the main loop would progress all workers on first iteration
    bool timeout = true;
    bool pthrStop = false;
    while (!pthrStop) {
        for (size_t wid = 0; wid < pollFds.size() - 1; wid++) {
            if (!(pollFds[wid].revents & POLLIN) && !timeout)
                continue;
            pollFds[wid].revents = 0;

            bool made_progress = false;
            nixl_status_t status;
            const auto &uw = uws[wid];
            do {
                while (uw->progress())
                    made_progress = true;

                status = uw->arm();
            } while (status == NIXL_IN_PROG);
            NIXL_ASSERT(status == NIXL_SUCCESS) << ", status: " << status;

            if (made_progress && !wid)
                notifProgress();
        }
        timeout = false;

        int ret;
        while ((ret = poll(pollFds.data(), pollFds.size(), pthrDelay.count())) < 0)
            NIXL_PTRACE << "Call to poll() was interrupted, retrying";

        if (!ret) {
            timeout = true;
        } else if (pollFds.back().revents & POLLIN) {
            pollFds.back().revents = 0;

            char signal;
            int ret = read(pollFds.back().fd, &signal, sizeof(signal));
            if (ret < 0)
                NIXL_PERROR << "read() on control pipe failed";

            pthrStop = true;
        }
    }
}

void nixlUcxThreadEngine::progressThreadStart()
{
    {
        std::unique_lock<std::mutex> lock(pthrActiveLock);
        pthrActive = false;
    }

    pthr = std::thread(&nixlUcxThreadEngine::progressFunc, this);

    std::unique_lock<std::mutex> lock(pthrActiveLock);
    pthrActiveCV.wait(lock, [&]{ return pthrActive; });
}

void nixlUcxThreadEngine::progressThreadStop()
{
    const char signal = 'X';
    int ret = write(pthrControlPipe[1], &signal, sizeof(signal));
    if (ret < 0)
        NIXL_PERROR << "write to progress thread control pipe failed";
    pthr.join();
}

int nixlUcxThreadEngine::vramApplyCtx()
{
    progressThreadStop();
    progressThreadStart();
    return nixlUcxEngine::vramApplyCtx();
}

void nixlUcxThreadEngine::appendNotif(std::string remote_name, std::string msg)
{
    if (isProgressThread()) {
        /* Append to the private list to allow batching */
        notifPthrPriv.push_back(std::make_pair(std::move(remote_name), std::move(msg)));
    } else {
        nixlUcxEngine::appendNotif(std::move(remote_name), std::move(msg));
    }
}

void nixlUcxThreadEngine::notifProgressCombineHelper(notif_list_t &src, notif_list_t &tgt)
{
    const std::lock_guard<std::mutex> lock(notifMtx);
    moveNotifList(src, tgt);
}

void nixlUcxThreadEngine::notifProgress()
{
    notifProgressCombineHelper(notifPthrPriv, notifPthr);
}

nixl_status_t nixlUcxThreadEngine::getNotifs(notif_list_t &notif_list)
{
    if (!notif_list.empty())
        return NIXL_ERR_INVALID_PARAM;

    moveNotifList(notifMainList, notif_list);
    notifProgressCombineHelper(notifPthr, notif_list);
    return NIXL_SUCCESS;
}

/****************************************
 * Threadpool engine
*****************************************/

class nixlUcxCompositeBackendH : public nixlUcxBackendH {
    public:
        nixlUcxCompositeBackendH(nixlUcxWorker *worker, size_t workerId,
                                 size_t chunkSize, size_t numChunks) :
            nixlUcxBackendH(worker, workerId),
            m_chunkSize(chunkSize) {
            num_chunks = numChunks;
            m_chunks.resize(numChunks);
        }

        nixlUcxBackendH* getChunk(size_t idx) {
            return &m_chunks[idx];
        }

        size_t getChunkSize() const {
            return m_chunkSize;
        }

        nixl_status_t release() override {
            // TODO: release all chunks
            return nixlUcxBackendH::release();
        }

        nixl_status_t status(uint64_t maxProgress = MAX_PROGRESS) override {
            // Progress only shared worker
            nixl_status_t ret = nixlUcxBackendH::status(maxProgress);
            if (ret == NIXL_SUCCESS) {
                bool allCompleted = true;
                // TODO: iterate only incomplete chunks
                for (auto &chunk : m_chunks) {
                    if (!chunk.isCompleted()) {
                        allCompleted = false;
                        break;
                    }
                }
                if (!allCompleted) {
                    ret = NIXL_IN_PROG;
                }
            }
            return ret;
        }

    private:
        std::vector<nixlUcxBackendH> m_chunks;
        size_t m_chunkSize;
};

class nixlUcxThreadContext {
    public:
        nixlUcxThreadContext(asio::io_context &io, nixlUcxWorker &worker,
                             size_t workerId) :
            m_io(io), m_worker(worker), m_workerId(workerId),
            m_ev(m_io, m_worker.getEfd()) {}

    void operator()() {
        nixlUcxThreadContext::tlsCtx = this;

        // keep io_context alive even if queue is temporarily empty
        auto guard = asio::make_work_guard(m_io);

        // TODO: try single polling thread
        auto workerEvents = [&](auto&& self)->void {
            nixl_status_t status;
            do {
                while (m_worker.progress());
                status = m_worker.arm();
            } while (status == NIXL_IN_PROG);

            uint64_t buf;
            m_ev.async_read_some(asio::buffer(&buf, sizeof(buf)),
                [&, self](const asio::error_code& ec, size_t bytes_read) {
                    if (!ec) {
                        self(self);
                    }
                });
        };
        workerEvents(workerEvents);

        while (!m_io.stopped()) {
            // If there are pending requests, poll task queue in order to prioritize
            // new post requests over pending requests
            if (!m_requests.empty()) {
                m_io.poll_one();
            } else {
                // Otherwise blocking wait for new requests
                m_io.run_one();
            }

            if (m_requests.empty()) {
                for (size_t i = 0; i < nixlUcxBackendH::MAX_PROGRESS && m_worker.progress(); i++);
                continue;
            }

            size_t maxProgress = nixlUcxBackendH::MAX_PROGRESS / m_requests.size();
            for (auto it = m_requests.begin(); it != m_requests.end();) {
                if ((*it)->status(maxProgress) == NIXL_SUCCESS) {
                    (*it)->complete();
                    it = m_requests.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }

    nixlUcxWorker& getWorker() const {
        return m_worker;
    }

    size_t getWorkerId() const {
        return m_workerId;
    }

    static nixlUcxThreadContext& getCtx() {
        return *tlsCtx;
    }

    void addRequest(nixlUcxBackendH *handle) {
        m_requests.push_back(handle);
    }

    private:
        asio::io_context &m_io;
        nixlUcxWorker &m_worker;
        size_t m_workerId;
        asio::posix::stream_descriptor m_ev;
        // TODO: make it intrusive
        std::list<nixlUcxBackendH *> m_requests;

        static thread_local nixlUcxThreadContext* tlsCtx;
};

thread_local nixlUcxThreadContext* nixlUcxThreadContext::tlsCtx = nullptr;

nixlUcxThreadPoolEngine::nixlUcxThreadPoolEngine(const nixlBackendInitParams &init_params)
    : nixlUcxEngine(init_params) {
    if (init_params.numThreads >= uws.size()) {
        throw std::invalid_argument("Number of threads is greater than number of workers");
    }

    m_io = std::make_unique<asio::io_context>();

    m_numDedicatedWorkers = std::min(init_params.numThreads, uws.size() - 1);
    m_threadContexts.reserve(m_numDedicatedWorkers);
    m_numSharedWorkers = uws.size() - m_numDedicatedWorkers;

    for (size_t i = 0; i < m_numDedicatedWorkers; i++) {
        // Shared workers come first in the array
        size_t workerId = i;
        m_threadContexts.emplace_back(*m_io, *getWorker(workerId), workerId);
        m_threads.emplace_back(std::ref(m_threadContexts.back()));
    }
}

nixlUcxThreadPoolEngine::~nixlUcxThreadPoolEngine() {
    m_io->stop();
    for (auto &thread : m_threads) {
        thread.join();
    }

    /* Explicitly clear UCX resources because this class injects event descriptor
     * into the UCX worker and this descriptor must be valid when worker is
     * destroyed. */
    remoteConnMap.clear();
    uws.clear();
    uc.reset();
}

nixl_status_t
nixlUcxThreadPoolEngine::prepXfer(const nixl_xfer_op_t &operation,
                                  const nixl_meta_dlist_t &local,
                                  const nixl_meta_dlist_t &remote,
                                  const std::string &remote_agent,
                                  nixlBackendReqH* &handle,
                                  const nixl_opt_b_args_t* opt_args) const {
    // TODO: find the best split strategy based on batchSize and numThreads
    // Maybe make it configurable
    const char *minEnv = getenv("MIN_CHUNK_SIZE");
    const char *maxEnv = getenv("MAX_CHUNK_SIZE");
    size_t MIN_CHUNK_SIZE = (minEnv != NULL) ? atoi(minEnv) : 256;
    size_t MAX_CHUNK_SIZE = (maxEnv != NULL) ? atoi(maxEnv) : 2048;

    size_t batchSize = local.descCount();
    if (batchSize <= MIN_CHUNK_SIZE) {
        return nixlUcxEngine::prepXfer(operation, local, remote, remote_agent, handle, opt_args);
    }

    size_t chunkSize = std::clamp(batchSize / m_numDedicatedWorkers,
                                  MIN_CHUNK_SIZE, MAX_CHUNK_SIZE);
    size_t numChunks = (batchSize + chunkSize - 1) / chunkSize;

    size_t workerId = getWorkerId();
    handle = new nixlUcxCompositeBackendH(getWorker(workerId).get(), workerId,
                                          chunkSize, numChunks);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxThreadPoolEngine::sendXferRange(const nixl_xfer_op_t &operation,
                                  const nixl_meta_dlist_t &local,
                                  const nixl_meta_dlist_t &remote,
                                  const std::string &remote_agent,
                                  nixlBackendReqH *handle,
                                  size_t start_idx, size_t end_idx) const {
    nixlUcxBackendH *intHandle = (nixlUcxBackendH *)handle;
    if (intHandle->getNumChunks() == 1) {
        return nixlUcxEngine::sendXferRange(operation, local, remote, remote_agent,
                                            handle, start_idx, end_idx);
    }

    nixlUcxCompositeBackendH *compHandle = (nixlUcxCompositeBackendH *)intHandle;
    size_t chunkSize = compHandle->getChunkSize();

    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    std::atomic<size_t> remaining{compHandle->getNumChunks()};
    std::atomic<nixl_status_t> status{NIXL_SUCCESS};

    for (size_t i = 0; i < compHandle->getNumChunks(); i++) {
        m_io->post([&, i]() {
            nixlUcxThreadContext& ctx = nixlUcxThreadContext::getCtx();
            nixlUcxBackendH *chunkHandle = compHandle->getChunk(i);
            chunkHandle->start(&ctx.getWorker(), ctx.getWorkerId());

            size_t startIdx = i * chunkSize;
            size_t endIdx = std::min(startIdx + chunkSize, (size_t)local.descCount());
            nixl_status_t ret;
            ret = nixlUcxEngine::sendXferRange(operation, local, remote, remote_agent,
                                               chunkHandle, startIdx, endIdx);
            if (ret != NIXL_SUCCESS) {
                // TODO: test error handling
                status.store(ret);
                chunkHandle->release();
                chunkHandle->complete();
            } else {
                ctx.addRequest(chunkHandle);
            }

            if (remaining.fetch_sub(1) == 1) {
                promise.set_value();
            }
        });
    }

    future.wait();
    return status.load();
}

int nixlUcxThreadPoolEngine::vramApplyCtx() {
    // TODO: apply ctx to all dedicated workers
    return nixlUcxEngine::vramApplyCtx();
}

/****************************************
 * Constructor/Destructor
*****************************************/

std::unique_ptr<nixlUcxEngine>
nixlUcxEngine::create(const nixlBackendInitParams &init_params)
{
    nixlUcxEngine *engine;
    switch (init_params.numThreads) {
        case 0:
            engine = new nixlUcxEngine(init_params);
            break;
        case 1:
            engine = new nixlUcxThreadEngine(init_params);
            break;
        default:
            engine = new nixlUcxThreadPoolEngine(init_params);
            break;
    }
    return std::unique_ptr<nixlUcxEngine>(engine);
}

nixlUcxEngine::nixlUcxEngine (const nixlBackendInitParams& init_params)
: nixlBackendEngine (&init_params) {
    unsigned long numWorkers;
    std::vector<std::string> devs; /* Empty vector */
    nixl_b_params_t* custom_params = init_params.customParams;

    if (custom_params->count("device_list")!=0)
        devs = str_split((*custom_params)["device_list"], ", ");

    const auto num_workers_iter = custom_params->find("num_workers");
    if (num_workers_iter == custom_params->end() || !absl::SimpleAtoi(num_workers_iter->second, &numWorkers))
        numWorkers = 1;

    if (numWorkers < init_params.numThreads) {
        numWorkers = init_params.numThreads + 1;
    }

    ucp_err_handling_mode_t err_handling_mode;
    const auto err_handling_mode_it =
        custom_params->find(std::string(nixl_ucx_err_handling_param_name));
    if (err_handling_mode_it == custom_params->end()) {
        err_handling_mode = UCP_ERR_HANDLING_MODE_NONE;
    } else {
        try {
            err_handling_mode = ucx_err_mode_from_string(err_handling_mode_it->second);
        }
        catch (const std::invalid_argument &e) {
            NIXL_ERROR << e.what();
            initErr = true;
            return;
        }
    }

    uc = std::make_unique<nixlUcxContext>(devs,
                                          sizeof(nixlUcxIntReq),
                                          _internalRequestInit,
                                          _internalRequestFini,
                                          init_params.numThreads > 0,
                                          numWorkers,
                                          init_params.syncMode);

    for (unsigned int i = 0; i < numWorkers; i++)
        uws.emplace_back(std::make_unique<nixlUcxWorker>(*uc, err_handling_mode));

    const auto &uw = uws.front();
    workerAddr = uw->epAddr();

    if (workerAddr.empty()) {
        NIXL_ERROR << "Failed to get UCX worker address";
        initErr = true;
        return;
    }

    uw->regAmCallback(CONN_CHECK, connectionCheckAmCb, this);
    uw->regAmCallback(DISCONNECT, connectionTermAmCb, this);
    uw->regAmCallback(NOTIF_STR, notifAmCb, this);

    // Temp fixup
    if (getenv("NIXL_DISABLE_CUDA_ADDR_WA")) {
        NIXL_INFO << "disabling CUDA address workaround";
        cuda_addr_wa = false;
    } else {
        cuda_addr_wa = true;
    }

    m_cudaPrimaryCtx = std::make_shared<nixlUcxCudaDevicePrimaryCtx>();
    vramInitCtx();
}

nixl_mem_list_t nixlUcxEngine::getSupportedMems () const {
    nixl_mem_list_t mems;
    mems.push_back(DRAM_SEG);
    mems.push_back(VRAM_SEG);
    return mems;
}

// Through parent destructor the unregister will be called.
nixlUcxEngine::~nixlUcxEngine () {
    // per registered memory deregisters it, which removes the corresponding metadata too
    // parent destructor takes care of the desc list
    // For remote metadata, they should be removed here
    if (this->initErr) {
        // Nothing to do
        return;
    }

    vramFiniCtx();
}

/****************************************
 * Connection management
*****************************************/

nixl_status_t nixlUcxEngine::checkConn(const std::string &remote_agent) {
    return remoteConnMap.count(remote_agent) ? NIXL_SUCCESS : NIXL_ERR_NOT_FOUND;
}

nixl_status_t nixlUcxEngine::endConn(const std::string &remote_agent) {

    auto search = remoteConnMap.find(remote_agent);

    if(search == remoteConnMap.end()) {
        return NIXL_ERR_NOT_FOUND;
    }

    //thread safety?
    remoteConnMap.erase(search);

    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::getConnInfo(std::string &str) const {
    str = workerAddr;
    return NIXL_SUCCESS;
}

ucs_status_t
nixlUcxEngine::connectionCheckAmCb(void *arg, const void *header,
                                   size_t header_length, void *data,
                                   size_t length,
                                   const ucp_am_recv_param_t *param)
{
    std::string remote_agent( (char*) data, length);
    nixlUcxEngine* engine = (nixlUcxEngine*) arg;

    NIXL_ASSERT(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));
    NIXL_ASSERT(header_length == 0) << "header_length " << header_length;

    if(engine->checkConn(remote_agent)) {
        NIXL_ERROR << "Received connect AM from agent we don't recognize: " << remote_agent;
        return UCS_OK;
    }

    return UCS_OK;
}

ucs_status_t
nixlUcxEngine::connectionTermAmCb (void *arg, const void *header,
                                   size_t header_length, void *data,
                                   size_t length,
                                   const ucp_am_recv_param_t *param)
{
    std::string remote_agent( (char*) data, length);

    NIXL_ASSERT(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));
    NIXL_ASSERT(header_length == 0) << "header_length " << header_length;

/*
    // TODO: research UCX connection logic and fix.
    nixlUcxEngine* engine = (nixlUcxEngine*) arg;
    if(NIXL_SUCCESS != engine->endConn(remote_agent)) {
        //TODO: received connect AM from agent we don't recognize
        return UCS_ERR_INVALID_PARAM;
    }
*/
    return UCS_OK;
}

nixl_status_t nixlUcxEngine::connect(const std::string &remote_agent) {
    if(remote_agent == localAgent) {
        return loadRemoteConnInfo(remote_agent, workerAddr);
    }
    const auto search = remoteConnMap.find(remote_agent);

    if(search == remoteConnMap.end()) {
        return NIXL_ERR_NOT_FOUND;
    }

    bool error = false;
    nixl_status_t ret = NIXL_SUCCESS;
    std::vector<nixlUcxReq> reqs;
    for (size_t i = 0; i < uws.size(); i++) {
        reqs.emplace_back();
        ret = search->second->getEp(i)->sendAm(CONN_CHECK, NULL, 0,
                                               (void*) localAgent.data(), localAgent.size(),
                                               UCP_AM_SEND_FLAG_EAGER, reqs.back());
        if(ret < 0) {
            error = true;
            break;
        }
    }

    //wait for AM to send
    ret = NIXL_IN_PROG;
    for (size_t i = 0; i < reqs.size(); i++)
        while(ret == NIXL_IN_PROG)
            ret = getWorker(i)->test(reqs[i]);

    return error ? NIXL_ERR_BACKEND : NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::disconnect(const std::string &remote_agent) {
    if (remote_agent != localAgent) {
        auto search = remoteConnMap.find(remote_agent);

        if(search == remoteConnMap.end()) {
            return NIXL_ERR_NOT_FOUND;
        }

        nixl_status_t ret = NIXL_SUCCESS;
        for (size_t i = 0; i < uws.size(); i++) {
            if (search->second->getEp(i)->checkTxState() == NIXL_SUCCESS) {
                nixlUcxReq req;
                ret = search->second->getEp(i)->sendAm(DISCONNECT, NULL, 0,
                                                       (void*) localAgent.data(), localAgent.size(),
                                                       UCP_AM_SEND_FLAG_EAGER, req);
                //don't care
                if (ret == NIXL_IN_PROG)
                    getWorker(i)->reqRelease(req);
            }
        }
    }

    endConn(remote_agent);

    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::loadRemoteConnInfo (const std::string &remote_agent,
                                                 const std::string &remote_conn_info)
{
    size_t size = remote_conn_info.size();
    std::vector<char> addr(size);

    if(remoteConnMap.count(remote_agent)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlSerDes::_stringToBytes(addr.data(), remote_conn_info, size);
    std::shared_ptr<nixlUcxConnection> conn = std::make_shared<nixlUcxConnection>();
    bool error = false;
    for (auto &uw: uws) {
        auto result = uw->connect(addr.data(), size);
        if (!result.ok()) {
            error = true;
            break;
        }
        conn->eps.push_back(std::move(*result));
    }

    if (error)
        return NIXL_ERR_BACKEND;

    conn->remoteAgent = remote_agent;

    remoteConnMap.insert({remote_agent, conn});

    return NIXL_SUCCESS;
}

/****************************************
 * Memory management
*****************************************/
nixl_status_t nixlUcxEngine::registerMem (const nixlBlobDesc &mem,
                                          const nixl_mem_t &nixl_mem,
                                          nixlBackendMD* &out)
{
    auto priv = std::make_unique<nixlUcxPrivateMetadata>();

    if (nixl_mem == VRAM_SEG) {
        bool need_restart;
        if (vramUpdateCtx((void*)mem.addr, mem.devId, need_restart)) {
            return NIXL_ERR_NOT_SUPPORTED;
            //TODO Add to logging
        }
        if (need_restart) {
            // set the ctx for main thread
            vramApplyCtx();
        }
    }

    // TODO: Add nixl_mem check?
    const int ret = uc->memReg((void*) mem.addr, mem.len, priv->mem, nixl_mem);
    if (ret) {
        return NIXL_ERR_BACKEND;
    }
    priv->rkeyStr = uc->packRkey(priv->mem);

    if (priv->rkeyStr.empty()) {
        return NIXL_ERR_BACKEND;
    }
    out = priv.release();
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::deregisterMem (nixlBackendMD* meta)
{
    nixlUcxPrivateMetadata *priv = (nixlUcxPrivateMetadata*) meta;
    uc->memDereg(priv->mem);
    delete priv;
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::getPublicData (const nixlBackendMD* meta,
                                            std::string &str) const {
    const nixlUcxPrivateMetadata *priv = (nixlUcxPrivateMetadata*) meta;
    str = priv->get();
    return NIXL_SUCCESS;
}


// To be cleaned up
nixl_status_t
nixlUcxEngine::internalMDHelper (const nixl_blob_t &blob,
                                 const std::string &agent,
                                 nixlBackendMD* &output) {
    try {
        auto md = std::make_unique<nixlUcxPublicMetadata>();
        size_t size = blob.size();

        auto search = remoteConnMap.find(agent);

        if (search == remoteConnMap.end()) {
            // TODO: err: remote connection not found
            return NIXL_ERR_NOT_FOUND;
        }
        md->conn = search->second;

        std::vector<char> addr(size);
        nixlSerDes::_stringToBytes(addr.data(), blob, size);

        for (size_t wid = 0; wid < uws.size(); wid++) {
            md->addRkey(*md->conn->getEp(wid), addr.data());
        }

        output = (nixlBackendMD *)md.release();

        return NIXL_SUCCESS;
    }
    catch (const std::runtime_error &e) {
        NIXL_ERROR << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlUcxEngine::loadLocalMD (nixlBackendMD* input,
                            nixlBackendMD* &output)
{
    nixlUcxPrivateMetadata* input_md = (nixlUcxPrivateMetadata*) input;
    return internalMDHelper(input_md->rkeyStr, localAgent, output);
}

// To be cleaned up
nixl_status_t nixlUcxEngine::loadRemoteMD (const nixlBlobDesc &input,
                                           const nixl_mem_t &nixl_mem,
                                           const std::string &remote_agent,
                                           nixlBackendMD* &output)
{
    // Set CUDA context of first device, UCX will anyways detect proper device when sending
    nixlUcxCudaCtxGuard guard(nixl_mem, m_cudaPrimaryCtx);
    return internalMDHelper(input.metaInfo, remote_agent, output);
}

nixl_status_t nixlUcxEngine::unloadMD (nixlBackendMD* input) {

    nixlUcxPublicMetadata *md = (nixlUcxPublicMetadata*) input; //typecast?
    delete md;

    return NIXL_SUCCESS;
}

/****************************************
 * Data movement
*****************************************/

static nixl_status_t _retHelper(nixl_status_t ret,  nixlUcxBackendH *hndl, nixlUcxReq &req)
{
    /* if transfer wasn't immediately completed */
    switch(ret) {
        case NIXL_IN_PROG:
            // TODO: this cast does not look safe
            // We need to allocate a vector of nixlUcxIntReq and set nixlUcxReqt
            hndl->append((nixlUcxIntReq*)req);
        case NIXL_SUCCESS:
            // Nothing to do
            break;
        default:
            // Error. Release all previously initiated ops and exit:
            hndl->release();
            return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::prepXfer (const nixl_xfer_op_t &operation,
                                       const nixl_meta_dlist_t &local,
                                       const nixl_meta_dlist_t &remote,
                                       const std::string &remote_agent,
                                       nixlBackendReqH* &handle,
                                       const nixl_opt_b_args_t* opt_args) const
{
    if (local.descCount() == 0 || remote.descCount() == 0) {
        NIXL_ERROR << "Local or remote descriptor list is empty";
        return NIXL_ERR_INVALID_PARAM;
    }

    /* TODO: try to get from a pool first */
    size_t workerId = getWorkerId();
    handle = new nixlUcxBackendH(getWorker(workerId).get(), workerId);
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::estimateXferCost (const nixl_xfer_op_t &operation,
                                               const nixl_meta_dlist_t &local,
                                               const nixl_meta_dlist_t &remote,
                                               const std::string &remote_agent,
                                               nixlBackendReqH* const &handle,
                                               std::chrono::microseconds &duration,
                                               std::chrono::microseconds &err_margin,
                                               nixl_cost_t &method,
                                               const nixl_opt_args_t* opt_args) const
{
    nixlUcxBackendH *intHandle = (nixlUcxBackendH *)handle;
    size_t workerId = intHandle->getWorkerId();

    if (local.descCount() != remote.descCount()) {
        NIXL_ERROR << "Local (" << local.descCount() << ") and remote (" << remote.descCount()
                   << ") descriptor lists differ in size for cost estimation";
        return NIXL_ERR_MISMATCH;
    }

    duration = std::chrono::microseconds(0);
    err_margin = std::chrono::microseconds(0);

    if (local.descCount() == 0) {
        // Nothing to do, use a default value
        method = nixl_cost_t::ANALYTICAL_BACKEND;
        return NIXL_SUCCESS;
    }

    for (int i = 0; i < local.descCount(); i++) {
        size_t lsize = local[i].len;
        size_t rsize = remote[i].len;

        nixlUcxPrivateMetadata *lmd = static_cast<nixlUcxPrivateMetadata*>(local[i].metadataP);
        nixlUcxPublicMetadata *rmd = static_cast<nixlUcxPublicMetadata*>(remote[i].metadataP);

        NIXL_ASSERT(lmd && rmd) << "No metadata found in descriptor lists at index " << i << " during cost estimation";
        NIXL_ASSERT(lsize == rsize) << "Local size (" << lsize << ") != Remote size (" << rsize
                                    << ") at index " << i << " during cost estimation";

        std::chrono::microseconds msg_duration;
        std::chrono::microseconds msg_err_margin;
        nixl_cost_t msg_method;
        nixl_status_t ret = rmd->conn->getEp(workerId)->estimateCost(lsize, msg_duration, msg_err_margin, msg_method);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Worker failed to estimate cost for segment " << i << " status: " << ret;
            return ret;
        }

        duration += msg_duration;
        err_margin += msg_err_margin;
        method = msg_method;
    }

    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::sendXferRange(const nixl_xfer_op_t &operation,
                                          const nixl_meta_dlist_t &local,
                                          const nixl_meta_dlist_t &remote,
                                          const std::string &remote_agent,
                                          nixlBackendReqH *handle,
                                          size_t start_idx, size_t end_idx) const
{
    nixlUcxBackendH *intHandle = (nixlUcxBackendH *)handle;
    nixlUcxPrivateMetadata *lmd;
    nixlUcxPublicMetadata *rmd;
    nixl_status_t ret;
    nixlUcxReq req;
    size_t workerId = intHandle->getWorkerId();

    for (size_t i = start_idx; i < end_idx; i++) {
        void *laddr = (void*) local[i].addr;
        size_t lsize = local[i].len;
        uint64_t raddr = (uint64_t) remote[i].addr;
        size_t rsize = remote[i].len;

        lmd = (nixlUcxPrivateMetadata*) local[i].metadataP;
        rmd = (nixlUcxPublicMetadata*) remote[i].metadataP;
        auto &ep = rmd->conn->getEp(workerId);

        if (lsize != rsize) {
            return NIXL_ERR_INVALID_PARAM;
        }

        switch (operation) {
        case NIXL_READ:
            ret = ep->read(raddr, rmd->getRkey(workerId), laddr, lmd->mem, lsize, req);
            break;
        case NIXL_WRITE:
            ret = ep->write(laddr, lmd->mem, raddr, rmd->getRkey(workerId), lsize, req);
            break;
        default:
            return NIXL_ERR_INVALID_PARAM;
        }

        if (_retHelper(ret, intHandle, req)) {
            return ret;
        }
    }

    /*
     * Flush keeps intHandle non-empty until the operation is actually
     * completed, which can happen after local requests completion.
     */
    // TODO: should we flush all distinct endpoints?
    rmd = (nixlUcxPublicMetadata*) remote[0].metadataP;
    ret = rmd->conn->getEp(workerId)->flushEp(req);
    if (_retHelper(ret, intHandle, req)) {
        return ret;
    }

    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::postXfer (const nixl_xfer_op_t &operation,
                                       const nixl_meta_dlist_t &local,
                                       const nixl_meta_dlist_t &remote,
                                       const std::string &remote_agent,
                                       nixlBackendReqH* &handle,
                                       const nixl_opt_b_args_t* opt_args) const
{
    size_t lcnt = local.descCount();
    size_t rcnt = remote.descCount();
    nixlUcxBackendH *intHandle = (nixlUcxBackendH *)handle;
    nixl_status_t ret;

    if (lcnt != rcnt) {
        NIXL_ERROR << "Local (" << lcnt << ") and remote (" << rcnt
                   << ") descriptor lists differ in size";
        return NIXL_ERR_INVALID_PARAM;
    }

    // TODO: assert that handle is empty/completed, as we can't post request before completion

    ret = sendXferRange(operation, local, remote, remote_agent, handle, 0, lcnt);
    if (ret != NIXL_SUCCESS) {
        return ret;
    }

    ret = intHandle->status();
    if (opt_args && opt_args->hasNotif) {
        if (ret == NIXL_SUCCESS) {
            nixlUcxReq req;
            ret = notifSendPriv(remote_agent, opt_args->notifMsg, req,
                                intHandle->getWorkerId());
            if (_retHelper(ret, intHandle, req)) {
                return ret;
            }

            ret = intHandle->status();
        } else if (ret == NIXL_IN_PROG) {
            intHandle->notification().emplace(remote_agent, opt_args->notifMsg);
        }
    }

    return ret;
}

nixl_status_t nixlUcxEngine::checkXfer (nixlBackendReqH* handle) const
{
    nixlUcxBackendH *intHandle = (nixlUcxBackendH *)handle;
    size_t workerId = intHandle->getWorkerId();

    nixl_status_t status = intHandle->status();
    auto& notif = intHandle->notification();
    if (status == NIXL_SUCCESS && notif.has_value()) {
        nixlUcxReq req;
        status = notifSendPriv(notif->agent, notif->payload, req, workerId);
        notif.reset();
        if (_retHelper(status, intHandle, req)) {
            return status;
        }

        status = intHandle->status();
    }

    return status;
}

nixl_status_t nixlUcxEngine::releaseReqH(nixlBackendReqH* handle) const
{
    nixlUcxBackendH *intHandle = (nixlUcxBackendH *)handle;
    nixl_status_t status = intHandle->release();

    /* TODO: return to a pool instead. */
    delete intHandle;

    return status;
}

int nixlUcxEngine::progress() {
    // TODO: add listen for connection handling if necessary
    int ret = 0;
    for (auto &uw: uws)
        ret += uw->progress();
    return ret;
}

/****************************************
 * Notifications
*****************************************/

//agent will provide cached msg
nixl_status_t nixlUcxEngine::notifSendPriv(const std::string &remote_agent,
                                           const std::string &msg,
                                           nixlUcxReq &req,
                                           size_t worker_id) const
{
    nixlSerDes ser_des;
    nixl_status_t ret;

    auto search = remoteConnMap.find(remote_agent);

    if(search == remoteConnMap.end()) {
        //TODO: err: remote connection not found
        return NIXL_ERR_NOT_FOUND;
    }

    ser_des.addStr("name", localAgent);
    ser_des.addStr("msg", msg);
    // TODO: replace with mpool for performance

    auto buffer = std::make_unique<std::string>(std::move(ser_des.exportStr()));
    ret = search->second->getEp(worker_id)->sendAm(NOTIF_STR, NULL, 0,
                                                   (void*)buffer->data(), buffer->size(),
                                                   UCP_AM_SEND_FLAG_EAGER, req);

    if (ret == NIXL_IN_PROG) {
        nixlUcxIntReq* nReq = (nixlUcxIntReq*)req;
        nReq->amBuffer = std::move(buffer);
    }
    return ret;
}

void nixlUcxEngine::appendNotif(std::string remote_name, std::string msg)
{
    notifMainList.push_back(std::make_pair(std::move(remote_name), std::move(msg)));
}

ucs_status_t
nixlUcxEngine::notifAmCb(void *arg, const void *header,
                         size_t header_length, void *data,
                         size_t length,
                         const ucp_am_recv_param_t *param)
{
    nixlSerDes ser_des;

    std::string ser_str( (char*) data, length);
    nixlUcxEngine* engine = (nixlUcxEngine*) arg;

    // send_am should be forcing EAGER protocol
    NIXL_ASSERT(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));
    NIXL_ASSERT(header_length == 0) << "header_length " << header_length;

    ser_des.importStr(ser_str);
    std::string remote_name = ser_des.getStr("name");
    std::string msg = ser_des.getStr("msg");

    engine->appendNotif(std::move(remote_name), std::move(msg));
    return UCS_OK;
}

nixl_status_t nixlUcxEngine::getNotifs(notif_list_t &notif_list)
{
    if (!notif_list.empty())
        return NIXL_ERR_INVALID_PARAM;

    while(progress());
    moveNotifList(notifMainList, notif_list);
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::genNotif(const std::string &remote_agent, const std::string &msg) const
{
    nixl_status_t ret;
    nixlUcxReq req;
    size_t wid = getWorkerId();

    ret = notifSendPriv(remote_agent, msg, req, wid);

    switch(ret) {
    case NIXL_IN_PROG:
        /* do not track the request */
        getWorker(wid)->reqRelease(req);
    case NIXL_SUCCESS:
        break;
    default:
        /* error case */
        return ret;
    }
    return NIXL_SUCCESS;
}
