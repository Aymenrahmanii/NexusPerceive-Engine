#ifndef NEXUS_COMMON_MEMORY_POOL_HPP
#define NEXUS_COMMON_MEMORY_POOL_HPP

#include <vector>
#include <mutex>
#include <cstddef>
#include <stdexcept>

namespace nexus {

class HostPinnedBufferManager {
public:
    HostPinnedBufferManager(size_t buffer_size, size_t pool_capacity);
    ~HostPinnedBufferManager();

    void* allocate();
    void deallocate(void* ptr);

    size_t getBufferSize() const { return buffer_size_; }
    size_t getPoolCapacity() const { return pool_capacity_; }

private:
    size_t buffer_size_;
    size_t pool_capacity_;
    std::mutex mutex_;
    std::vector<void*> free_buffers_;
    std::vector<void*> allocated_master_list_;
};

} // namespace nexus

#endif // NEXUS_COMMON_MEMORY_POOL_HPP
