// Speed test of the two production detectors on Zenbo camera frames.
// SCRFD: the server's SCRFDDetector.hpp, included unchanged.
// S3FD: LRASDWorker.cpp's detect() body and session options, copied verbatim.
// Detectors are interleaved in alternating blocks so both see the same GPU load.
#define LR_ASD_USE_CUDA
#include "SCRFDDetector.hpp"
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>

struct Box { float x1, y1, x2, y2, confidence; };
static float iou(const Box &a, const Box &b)
{
    const float x1 = std::max(a.x1, b.x1), y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2), y2 = std::min(a.y2, b.y2);
    const float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const float area = (a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - intersection;
    return area > 0 ? intersection / area : 0;
}

static Ort::SessionOptions makeOptions()
{
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetIntraOpNumThreads(1);
    OrtCUDAProviderOptions cuda{}; options.AppendExecutionProvider_CUDA(cuda);
    return options;
}

static std::vector<Box> s3fdDetect(Ort::Session *detector_, const cv::Mat &frame)
{
    cv::Mat resized; cv::resize(frame,resized,{480,270});
    std::vector<float> data(3*270*480); const float mean[]={123,117,104};
    for (int c=0;c<3;++c) for(int y=0;y<270;++y) for(int x=0;x<480;++x)
        data[c*270*480+y*480+x]=resized.at<cv::Vec3b>(y,x)[c]-mean[c];
    std::array<int64_t,4> shape{1,3,270,480};
    auto mem=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    auto tensor=Ort::Value::CreateTensor<float>(mem,data.data(),data.size(),shape.data(),shape.size());
    const char *inputs[]={"image"}, *outputs[]={"loc","conf"};
    auto values=detector_->Run(Ort::RunOptions{nullptr},inputs,&tensor,1,outputs,2);
    const float *loc=values[0].GetTensorData<float>(), *conf=values[1].GetTensorData<float>();
    const int steps[]={4,8,16,32,64,128}, mins[]={16,32,64,128,256,512};
    int heights[6], widths[6]; heights[0]=270/4; widths[0]=480/4;
    for(int z=1;z<6;++z) { if(z==3){heights[z]=heights[z-1]/2;widths[z]=widths[z-1]/2;}
        else {heights[z]=(heights[z-1]+1)/2;widths[z]=(widths[z-1]+1)/2;} }
    std::vector<Box> boxes; int k=0;
    for(int z=0;z<6;++z) for(int y=0;y<heights[z];++y) for(int x=0;x<widths[z];++x,++k) {
        if(conf[k*2+1]<0.9f) continue;
        float cx=(x+.5f)*steps[z]/480, cy=(y+.5f)*steps[z]/270;
        float w=mins[z]/480.0f, h=mins[z]/270.0f;
        cx+=loc[k*4]*.1f*w; cy+=loc[k*4+1]*.1f*h;
        w*=std::exp(loc[k*4+2]*.2f); h*=std::exp(loc[k*4+3]*.2f);
        boxes.push_back({(cx-w/2)*frame.cols,(cy-h/2)*frame.rows,
                         (cx+w/2)*frame.cols,(cy+h/2)*frame.rows,conf[k*2+1]});
    }
    std::sort(boxes.begin(),boxes.end(),[](const Box&a,const Box&b){return a.confidence>b.confidence;});
    std::vector<Box> kept;
    for(const Box &box:boxes) { bool add=true; for(const Box &old:kept) if(iou(box,old)>.1f){add=false;break;} if(add)kept.push_back(box); }
    return kept;
}

int main(int argc, char **argv)
{
    const std::string dir = argv[1], s3fdPath = argv[2], scrfdPath = argv[3], csv = argv[4];
    const int rounds = argc > 5 ? std::atoi(argv[5]) : 20;
    std::vector<cv::Mat> frames;
    for (auto &e : std::filesystem::directory_iterator(dir)) {
        auto n = e.path().filename().string();
        if (n.find("face") == std::string::npos && n.find("outFrame") == std::string::npos)
            frames.push_back(cv::imread(e.path().string()));
    }
    std::printf("frames %zu (%dx%d)\n", frames.size(), frames[0].cols, frames[0].rows);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bench");
    auto opts = makeOptions();
    Ort::Session s3fd(env, s3fdPath.c_str(), opts);
    SCRFDDetector scrfd(env, scrfdPath);

    using Clock = std::chrono::steady_clock;
    auto timeOne = [&](int which, const cv::Mat &f) {
        auto t = Clock::now();
        size_t n = which == 0 ? s3fdDetect(&s3fd, f).size() : scrfd.detect(f).size();
        (void)n;
        return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
    };
    for (int i = 0; i < 50; ++i) { timeOne(0, frames[i % frames.size()]); timeOne(1, frames[i % frames.size()]); }

    std::ofstream out(csv);
    out << "round,detector,frame,ms\n";
    const int block = 50;
    for (int r = 0; r < rounds; ++r) {
        const int first = r % 2;  // alternate which detector goes first
        for (int k = 0; k < 2; ++k) {
            const int which = k == 0 ? first : 1 - first;
            for (int i = 0; i < block; ++i) {
                const int idx = (r * block + i) % frames.size();
                out << r << ',' << (which == 0 ? "s3fd" : "scrfd") << ',' << idx << ','
                    << timeOne(which, frames[idx]) << '\n';
            }
        }
    }
    std::printf("done\n");
}
