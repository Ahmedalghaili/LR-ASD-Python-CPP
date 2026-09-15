#pragma once
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <array>

// Box and overlap are provided by the original LR-ASD pipeline.
class Scrfd {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "scrfd"};
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    std::string input_name;
    std::vector<std::string> output_names;
    int width, height;
    float threshold, nms_threshold;
public:
    Scrfd(const char* path, int w, int h, float score, float nms, bool cuda)
        : width(w), height(h), threshold(score), nms_threshold(nms) {
        if (w <= 0 || h <= 0 || w % 32 || h % 32)
            throw std::runtime_error("SCRFD input dimensions must be positive multiples of 32");
        options.SetIntraOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (cuda) {
            OrtCUDAProviderOptions provider{};
            provider.device_id = 0;
            provider.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
            provider.do_copy_in_default_stream = 1;
            options.AppendExecutionProvider_CUDA(provider);
        }
        session = std::make_unique<Ort::Session>(env, path, options);
        Ort::AllocatorWithDefaultOptions allocator;
        input_name = session->GetInputNameAllocated(0, allocator).get();
        for (size_t i = 0; i < session->GetOutputCount(); ++i)
            output_names.emplace_back(session->GetOutputNameAllocated(i, allocator).get());
        if (output_names.size() != 6 && output_names.size() != 9)
            throw std::runtime_error("Expected SCRFD with strides 8/16/32 and two anchors");
    }
    std::vector<Box> operator()(const cv::Mat& frame) {
        const double image_ratio = double(frame.rows) / frame.cols;
        int rw, rh;
        if (image_ratio > double(height) / width) {
            rh = height; rw = int(rh / image_ratio);
        } else {
            rw = width; rh = int(rw * image_ratio);
        }
        // Match InsightFace: top-left letterbox, scale derived from resized height.
        float scale = float(rh) / frame.rows;
        cv::Mat resized, padded(height, width, CV_8UC3, cv::Scalar(0,0,0));
        cv::resize(frame, resized, {rw,rh});
        resized.copyTo(padded(cv::Rect(0,0,rw,rh)));
        std::vector<float> data(3 * height * width);
        for (int y=0; y<height; ++y) {
            const auto* row = padded.ptr<cv::Vec3b>(y);
            for (int x=0; x<width; ++x)
                for (int c=0; c<3; ++c)
                    data[c*height*width+y*width+x]=(row[x][2-c]-127.5f)/128.f;
        }
        std::array<int64_t,4> shape{1,3,height,width};
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input = Ort::Value::CreateTensor<float>(memory, data.data(), data.size(), shape.data(), shape.size());
        const char* input_names[] = {input_name.c_str()};
        std::vector<const char*> names;
        // Keypoints are not consumed by LR-ASD.
        for (size_t i=0; i<6; ++i) names.push_back(output_names[i].c_str());
        auto outputs = session->Run(Ort::RunOptions{nullptr}, input_names, &input, 1, names.data(), names.size());
        std::vector<Box> boxes;
        for (int level=0; level<3; ++level) {
            int stride = 8 << level, columns = width/stride;
            size_t count = size_t(height/stride)*columns*2;
            if (outputs[level].GetTensorTypeAndShapeInfo().GetElementCount()!=count ||
                outputs[level+3].GetTensorTypeAndShapeInfo().GetElementCount()!=count*4)
                throw std::runtime_error("Unexpected SCRFD output shape");
            const float* scores = outputs[level].GetTensorData<float>();
            const float* distances = outputs[level+3].GetTensorData<float>();
            for (size_t i=0; i<count; ++i) {
                if (scores[i]<threshold) continue;
                float cx=(i/2 % columns)*stride, cy=(i/2 / columns)*stride;
                const float* d=distances+i*4;
                Box b{(cx-d[0]*stride)/scale,(cy-d[1]*stride)/scale,
                      (cx+d[2]*stride)/scale,(cy+d[3]*stride)/scale,scores[i]};
                if (std::isfinite(b.x1+b.y1+b.x2+b.y2+b.score) && b.x2>b.x1 && b.y2>b.y1)
                    boxes.push_back(b);
            }
        }
        std::sort(boxes.begin(),boxes.end(),[](const Box&a,const Box&b){return a.score>b.score;});
        std::vector<Box> kept;
        for (const auto& b: boxes) {
            bool suppress=false;
            for (const auto& k: kept) {
                // Inclusive-coordinate IoU matches the reference SCRFD NMS.
                float area=(b.x2-b.x1+1)*(b.y2-b.y1+1), other=(k.x2-k.x1+1)*(k.y2-k.y1+1);
                float inter=std::max(0.f,std::min(b.x2,k.x2)-std::max(b.x1,k.x1)+1)*
                            std::max(0.f,std::min(b.y2,k.y2)-std::max(b.y1,k.y1)+1);
                if(inter/(area+other-inter)>nms_threshold){suppress=true;break;}
            }
            if(!suppress) kept.push_back(b);
        }
        return kept;
    }
};
