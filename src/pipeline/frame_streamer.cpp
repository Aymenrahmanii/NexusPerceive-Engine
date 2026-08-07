#include "nexus/pipeline/frame_streamer.hpp"
#include "nexus/common/logger.hpp"
#include <chrono>

namespace nexus {

FrameStreamer::FrameStreamer(const std::string& stream_uri)
    : stream_uri_(stream_uri) {}

FrameStreamer::~FrameStreamer() {
    close();
}

bool FrameStreamer::open() {
    NEXUS_LOG_INFO("Opening real-time video stream: " + stream_uri_);

    // Check if input is a camera index (e.g. "0", "1") or RTSP/file URI
    if (stream_uri_.find_first_not_of("0123456789") == std::string::npos) {
        int cam_idx = std::stoi(stream_uri_);
        cap_.open(cam_idx, cv::CAP_ANY);
    } else {
        cap_.open(stream_uri_);
    }

    if (!cap_.isOpened()) {
        NEXUS_LOG_WARN("Could not open stream: " + stream_uri_ + ". Initializing fallback video generator.");
        width_ = 1920;
        height_ = 1080;
        fps_ = 30.0;
        is_open_ = true;
        return true;
    }

    width_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    height_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    fps_ = cap_.get(cv::CAP_PROP_FPS);
    if (fps_ <= 0) fps_ = 30.0;

    is_open_ = true;
    is_running_ = true;
    capture_thread_ = std::thread(&FrameStreamer::captureLoop, this);

    NEXUS_LOG_INFO("Successfully opened stream (" + std::to_string(width_) + "x" + 
                   std::to_string(height_) + " @ " + std::to_string(fps_) + " FPS).");
    return true;
}

void FrameStreamer::captureLoop() {
    while (is_running_) {
        cv::Mat frame;
        if (!cap_.read(frame) || frame.empty()) {
            // Loop video files if end reached
            cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
            if (!cap_.read(frame) || frame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
        }

        auto now = std::chrono::high_resolution_clock::now();
        uint64_t timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count();

        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (frame_queue_.size() >= max_queue_size_) {
            frame_queue_.pop(); // Drop oldest frame to ensure zero-latency real-time processing
        }
        frame_queue_.push({frame, ++current_frame_id_, timestamp_us});
    }
}

bool FrameStreamer::getNextFrame(cv::Mat& frame, int64_t& frame_id, uint64_t& timestamp_us) {
    if (!is_open_) return false;

    if (capture_thread_.joinable()) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (frame_queue_.empty()) return false;

        auto data = frame_queue_.front();
        frame_queue_.pop();
        frame = data.frame;
        frame_id = data.frame_id;
        timestamp_us = data.timestamp_us;
        return true;
    } else {
        // Fallback synthetic frame generation
        frame = cv::Mat(1080, 1920, CV_8UC3, cv::Scalar(114, 114, 114));
        frame_id = ++current_frame_id_;
        auto now = std::chrono::high_resolution_clock::now();
        timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
        return true;
    }
}

void FrameStreamer::close() {
    is_running_ = false;
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
    if (cap_.isOpened()) {
        cap_.release();
    }
    is_open_ = false;
}

} // namespace nexus
