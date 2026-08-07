#ifndef NEXUS_PIPELINE_TRACKER_HPP
#define NEXUS_PIPELINE_TRACKER_HPP

#include <vector>
#include <memory>
#include "nexus/common/types.hpp"

namespace nexus {

struct Tracklet {
    int track_id;
    BoundingBox box;
    int age = 0;
    int hits = 0;
    int time_since_update = 0;
    bool is_confirmed = false;
    std::vector<std::pair<float, float>> history_points;
};

class ByteTrackManager {
public:
    explicit ByteTrackManager(int max_age = 30, float track_thresh = 0.5f);
    ~ByteTrackManager() = default;

    std::vector<Tracklet> update(const std::vector<BoundingBox>& detections);

private:
    float computeIoU(const BoundingBox& a, const BoundingBox& b);

    int next_id_ = 1;
    int max_age_;
    float track_thresh_;
    std::vector<Tracklet> tracked_stracks_;
};

} // namespace nexus

#endif // NEXUS_PIPELINE_TRACKER_HPP
