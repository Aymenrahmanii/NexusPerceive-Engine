#include <gtest/gtest.h>
#include "nexus/common/memory_pool.hpp"

TEST(MemoryPoolTest, AllocationAndDeallocation) {
    size_t buf_size = 1024 * 1024; // 1 MB
    size_t capacity = 5;

    nexus::HostPinnedBufferManager pool(buf_size, capacity);
    EXPECT_EQ(pool.getBufferSize(), buf_size);
    EXPECT_EQ(pool.getPoolCapacity(), capacity);

    void* buf1 = pool.allocate();
    EXPECT_NE(buf1, nullptr);

    void* buf2 = pool.allocate();
    EXPECT_NE(buf2, nullptr);
    EXPECT_NE(buf1, buf2);

    pool.deallocate(buf1);
    pool.deallocate(buf2);
}
