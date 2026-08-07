#ifndef NEXUS_PIPELINE_FRAME_STREAMER_HPP
#define NEXUS_PIPELINE_FRAME_STREAMER_HPP

#include <string>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <opencv2/opencv.hpp>
#include "nexus/common/types.hpp"

namespace nexus {

class FrameStreamer {
public:
    explicit FrameStreamer(const std::string& stream_uri);
    ~FrameStreamer();

    bool open();
    bool getNextFrame(cv::Mat& frame, int64_t& frame_id, uint64_t& timestamp_us);
    void close();

    bool isOpened() const { return is_open_; }
    int getWidth() const { return width_; }
    int getHeight() const { return height_; }
    double getFPS() const { return fps_; }

private:
    void captureLoop();

    std::string stream_uri_;
    cv::VideoCapture cap_;
    int width_ = 0;
    int height_ = 0;
    double fps_ = 30.0;

    std::atomic<bool> is_running_{false};
    std::atomic<bool> is_open_{false};
    std::thread capture_thread_;

    std::mutex queue_mutex_;
    struct FrameData {
        cv::Mat frame;
        int64_t frame_id;
        uint64_t timestamp_us;
    };
    std::queue<FrameData> frame_queue_;
    size_t max_queue_size_ = 5;
    int64_t current_frame_id_ = 0;
};

} // namespace nexus

#endif // NEXUS_PIPELINE_FRAME_STREAMER_HPP
