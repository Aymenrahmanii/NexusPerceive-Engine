#include "nexus/common/memory_pool.hpp"
#include "nexus/common/logger.hpp"
#include <cuda_runtime.h>
#include <sstream>

namespace nexus {

HostPinnedBufferManager::HostPinnedBufferManager(size_t buffer_size, size_t pool_capacity)
    : buffer_size_(buffer_size), pool_capacity_(pool_capacity) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < pool_capacity_; ++i) {
        void* ptr = nullptr;
#ifdef __CUDACC__
        cudaError_t err = cudaHostAlloc(&ptr, buffer_size_, cudaHostAllocMapped | cudaHostAllocWriteCombined);
        if (err != cudaSuccess) {
            std::stringstream ss;
            ss << "Failed to allocate pinned host memory: " << cudaGetErrorString(err);
            NEXUS_LOG_ERROR(ss.str());
            throw std::runtime_error(ss.str());
        }
#else
        ptr = std::malloc(buffer_size_);
#endif
        free_buffers_.push_back(ptr);
        allocated_master_list_.push_back(ptr);
    }
    
    std::stringstream ss;
    ss << "HostPinnedBufferManager initialized with " << pool_capacity_ 
       << " buffers of size " << (buffer_size_ / 1024.0 / 1024.0) << " MB each.";
    NEXUS_LOG_INFO(ss.str());
}

HostPinnedBufferManager::~HostPinnedBufferManager() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (void* ptr : allocated_master_list_) {
#ifdef __CUDACC__
        cudaFreeHost(ptr);
#else
        std::free(ptr);
#endif
    }
    free_buffers_.clear();
    allocated_master_list_.clear();
}

void* HostPinnedBufferManager::allocate() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (free_buffers_.empty()) {
        NEXUS_LOG_WARN("HostPinnedBufferManager pool exhausted, falling back to dynamic allocation.");
        void* ptr = nullptr;
#ifdef __CUDACC__
        cudaHostAlloc(&ptr, buffer_size_, cudaHostAllocMapped);
#else
        ptr = std::malloc(buffer_size_);
#endif
        allocated_master_list_.push_back(ptr);
        return ptr;
    }
    void* ptr = free_buffers_.back();
    free_buffers_.pop_back();
    return ptr;
}

void HostPinnedBufferManager::deallocate(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex_);
    free_buffers_.push_back(ptr);
}

} // namespace nexus
