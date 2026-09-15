#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct SCRFDBBox
{
    float x1, y1, x2, y2, confidence;
};

// SCRFD face detector used by LRASDWorker. The preprocessing, output decoding,
// and NMS match the validated experimental C++ implementation.
class SCRFDDetector
{
public:
    SCRFDDetector(Ort::Env &env, const std::string &modelPath,
                  int width = 480, int height = 288,
                  float scoreThreshold = 0.5f, float nmsThreshold = 0.4f)
        : width_(width), height_(height), scoreThreshold_(scoreThreshold),
          nmsThreshold_(nmsThreshold)
    {
        if (width_ <= 0 || height_ <= 0 || width_ % 32 != 0 || height_ % 32 != 0)
            throw std::runtime_error("SCRFD dimensions must be positive multiples of 32");

        options_.SetIntraOpNumThreads(1);
        options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef LR_ASD_USE_CUDA
        OrtCUDAProviderOptions cuda{};
        cuda.device_id = 0;
        cuda.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
        cuda.do_copy_in_default_stream = 1;
        options_.AppendExecutionProvider_CUDA(cuda);
#endif
        session_ = std::make_unique<Ort::Session>(env, modelPath.c_str(), options_);

        Ort::AllocatorWithDefaultOptions allocator;
        inputName_ = session_->GetInputNameAllocated(0, allocator).get();
        for (size_t i = 0; i < session_->GetOutputCount(); ++i)
            outputNames_.emplace_back(session_->GetOutputNameAllocated(i, allocator).get());
        if (outputNames_.size() != 6 && outputNames_.size() != 9)
            throw std::runtime_error("Expected SCRFD outputs for strides 8/16/32");
    }

    std::vector<SCRFDBBox> detect(const cv::Mat &frame)
    {
        if (frame.empty()) return {};

        const double imageRatio = double(frame.rows) / frame.cols;
        int resizedWidth, resizedHeight;
        if (imageRatio > double(height_) / width_) {
            resizedHeight = height_;
            resizedWidth = int(resizedHeight / imageRatio);
        } else {
            resizedWidth = width_;
            resizedHeight = int(resizedWidth * imageRatio);
        }
        resizedWidth = std::max(1, resizedWidth);
        resizedHeight = std::max(1, resizedHeight);

        const float scale = float(resizedHeight) / frame.rows;
        cv::Mat resized, padded(height_, width_, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::resize(frame, resized, {resizedWidth, resizedHeight});
        resized.copyTo(padded(cv::Rect(0, 0, resizedWidth, resizedHeight)));

        std::vector<float> inputData(size_t(3) * height_ * width_);
        for (int y = 0; y < height_; ++y) {
            const auto *row = padded.ptr<cv::Vec3b>(y);
            for (int x = 0; x < width_; ++x) {
                for (int c = 0; c < 3; ++c)
                    inputData[size_t(c) * height_ * width_ + size_t(y) * width_ + x] =
                        (row[x][2 - c] - 127.5f) / 128.0f;
            }
        }

        std::array<int64_t, 4> shape{1, 3, height_, width_};
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input = Ort::Value::CreateTensor<float>(memory, inputData.data(), inputData.size(),
                                                     shape.data(), shape.size());
        const char *inputNames[] = {inputName_.c_str()};
        std::vector<const char *> outputNames;
        for (size_t i = 0; i < 6; ++i) outputNames.push_back(outputNames_[i].c_str());
        auto outputs = session_->Run(Ort::RunOptions{nullptr}, inputNames, &input, 1,
                                     outputNames.data(), outputNames.size());

        std::vector<SCRFDBBox> boxes;
        for (int level = 0; level < 3; ++level) {
            const int stride = 8 << level;
            const int columns = width_ / stride;
            const size_t count = size_t(height_ / stride) * columns * 2;
            if (outputs[level].GetTensorTypeAndShapeInfo().GetElementCount() != count ||
                outputs[level + 3].GetTensorTypeAndShapeInfo().GetElementCount() != count * 4)
                throw std::runtime_error("Unexpected SCRFD output shape");

            const float *scores = outputs[level].GetTensorData<float>();
            const float *distances = outputs[level + 3].GetTensorData<float>();
            for (size_t i = 0; i < count; ++i) {
                if (scores[i] < scoreThreshold_) continue;
                const float cx = float((i / 2) % columns * stride);
                const float cy = float((i / 2) / columns * stride);
                const float *d = distances + i * 4;
                SCRFDBBox box{
                    (cx - d[0] * stride) / scale,
                    (cy - d[1] * stride) / scale,
                    (cx + d[2] * stride) / scale,
                    (cy + d[3] * stride) / scale,
                    scores[i]};
                if (std::isfinite(box.x1 + box.y1 + box.x2 + box.y2 + box.confidence) &&
                    box.x2 > box.x1 && box.y2 > box.y1)
                    boxes.push_back(box);
            }
        }

        std::sort(boxes.begin(), boxes.end(), [](const SCRFDBBox &a, const SCRFDBBox &b) {
            return a.confidence > b.confidence;
        });
        std::vector<SCRFDBBox> kept;
        for (const auto &box : boxes) {
            bool suppressed = false;
            for (const auto &old : kept) {
                const float area = (box.x2 - box.x1 + 1) * (box.y2 - box.y1 + 1);
                const float oldArea = (old.x2 - old.x1 + 1) * (old.y2 - old.y1 + 1);
                const float intersection =
                    std::max(0.0f, std::min(box.x2, old.x2) - std::max(box.x1, old.x1) + 1) *
                    std::max(0.0f, std::min(box.y2, old.y2) - std::max(box.y1, old.y1) + 1);
                if (intersection / (area + oldArea - intersection) > nmsThreshold_) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed) kept.push_back(box);
        }
        return kept;
    }

private:
    int width_, height_;
    float scoreThreshold_, nmsThreshold_;
    Ort::SessionOptions options_;
    std::unique_ptr<Ort::Session> session_;
    std::string inputName_;
    std::vector<std::string> outputNames_;
};
