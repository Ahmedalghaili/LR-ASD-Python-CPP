#pragma once

#include "VideoAudioBuffer.hpp"
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

class LRASDEngine;

struct LRASDPrediction
{
    int trackId;
    float logit;
    bool speaking;
    cv::Rect faceBox;
};

class LRASDWorker
{
public:
    LRASDWorker();
    ~LRASDWorker();
    LRASDWorker(const LRASDWorker &) = delete;
    LRASDWorker &operator=(const LRASDWorker &) = delete;

    bool start(const std::string &modelDirectory);
    void submit(VABuffer snapshot);
    std::vector<LRASDPrediction> latestPredictions() const;
    void stop();

private:
    void run();
    std::unique_ptr<LRASDEngine> engine_;
    std::thread thread_;
    std::atomic_bool running_{false};
    std::mutex inputMutex_;
    std::condition_variable inputReady_;
    std::optional<VABuffer> pendingInput_;
    mutable std::mutex predictionMutex_;
    std::vector<LRASDPrediction> latestPredictions_;
};
