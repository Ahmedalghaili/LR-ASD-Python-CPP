#include "LRASDWorker.hpp"
#include "SCRFDDetector.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <vector>

using Clock = std::chrono::steady_clock;

namespace {
struct Box { float x1, y1, x2, y2, confidence; };
float iou(const Box &a, const Box &b)
{
    const float x1 = std::max(a.x1, b.x1), y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2), y2 = std::min(a.y2, b.y2);
    const float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const float area = (a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - intersection;
    return area > 0 ? intersection / area : 0;
}

double hzToMel(double hz) { return 2595.0 * std::log10(1.0 + hz / 700.0); }
double melToHz(double mel) { return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0); }

// Matches python_speech_features.mfcc: 16 kHz, 25 ms window, 10 ms step,
// 512 FFT, 26 filters, 13 cepstra, pre-emphasis 0.97 and lifter 22.
std::vector<float> mfcc(const std::vector<float> &input)
{
    constexpr int sampleRate=16000, win=400, step=160, nfft=512, filters=26, cepstra=13;
    std::vector<float> x(input.size(), 0.0f);
    if (!input.empty()) x[0]=input[0];
    for (size_t i=1; i<input.size(); ++i) x[i]=input[i]-0.97f*input[i-1];
    const int frameCount = input.size() <= win ? 1 : 1 + static_cast<int>(std::ceil((input.size()-win)/double(step)));
    x.resize((frameCount-1)*step+win, 0.0f);
    std::vector<int> bins(filters+2);
    const double low=hzToMel(0), high=hzToMel(sampleRate/2.0);
    for (int i=0; i<filters+2; ++i)
        bins[i]=static_cast<int>(std::floor((nfft+1)*melToHz(low+(high-low)*i/(filters+1))/sampleRate));
    std::vector<float> output(frameCount*cepstra);
    const double pi=std::acos(-1.0), eps=std::numeric_limits<double>::epsilon();
    for (int n=0; n<frameCount; ++n) {
        std::vector<double> power(nfft/2+1), bank(filters);
        double energy=0;
        for (int k=0; k<=nfft/2; ++k) {
            double re=0, im=0;
            for (int j=0; j<win; ++j) {
                const double value=x[n*step+j];
                re += value*std::cos(2*pi*k*j/nfft);
                im -= value*std::sin(2*pi*k*j/nfft);
            }
            power[k]=(re*re+im*im)/nfft; energy += power[k];
        }
        for (int m=1; m<=filters; ++m) {
            for (int k=bins[m-1]; k<bins[m]; ++k)
                bank[m-1] += power[k]*(k-bins[m-1])/double(bins[m]-bins[m-1]);
            for (int k=bins[m]; k<bins[m+1]; ++k)
                bank[m-1] += power[k]*(bins[m+1]-k)/double(bins[m+1]-bins[m]);
            bank[m-1]=std::log(std::max(bank[m-1], eps));
        }
        for (int c=0; c<cepstra; ++c) {
            double value=0;
            for (int m=0; m<filters; ++m)
                value += bank[m]*std::cos(pi*c*(m+0.5)/filters);
            value *= std::sqrt(2.0/filters);
            if (c==0) value /= std::sqrt(2.0);
            output[n*cepstra+c]=value*(1+11*std::sin(pi*c/22.0));
        }
        output[n*cepstra]=std::log(std::max(energy, eps));
    }
    return output;
}

std::vector<float> cropFace(const cv::Mat &frame, const Box &box)
{
    const float size=std::max(box.x2-box.x1, box.y2-box.y1);
    const float cx=(box.x1+box.x2)/2, cy=(box.y1+box.y2)/2, half=size*0.55f;
    int x1=std::floor(cx-half), y1=std::floor(cy-half);
    int x2=std::ceil(cx+half), y2=std::ceil(cy+half);
    cv::Mat padded;
    const int left=std::max(0,-x1), top=std::max(0,-y1);
    const int right=std::max(0,x2-frame.cols), bottom=std::max(0,y2-frame.rows);
    cv::copyMakeBorder(frame,padded,top,bottom,left,right,cv::BORDER_CONSTANT,{110,110,110});
    x1+=left; x2+=left; y1+=top; y2+=top;
    cv::Mat gray;
    cv::cvtColor(padded(cv::Rect(x1,y1,x2-x1,y2-y1)),gray,cv::COLOR_BGR2GRAY);
    cv::resize(gray,gray,{112,112});
    std::vector<float> result(112*112);
    for (int y=0; y<112; ++y)
        for (int x=0; x<112; ++x) result[y*112+x]=gray.at<unsigned char>(y,x);
    return result;
}

std::string modelPath(const std::string &dir, const char *name)
{ return dir + (dir.empty() || dir.back()=='/' ? "" : "/") + name; }
}

class LRASDEngine
{
    struct Track { Box box{}; std::deque<std::vector<float>> faces; float score=0; int missed=0; };
public:
    explicit LRASDEngine(const std::string &directory)
        : env_(ORT_LOGGING_LEVEL_WARNING,"LR-ASD"), options_(makeOptions()),
          asd_(env_,modelPath(directory,"lr_asd.onnx").c_str(),options_)
    {
        const char *configuredDetector = std::getenv("LR_ASD_DETECTOR");
        detectorName_ = configuredDetector && *configuredDetector ? configuredDetector : "s3fd";
        if (detectorName_ == "scrfd") {
            const char *configuredModel = std::getenv("LR_ASD_SCRFD_MODEL");
            const std::string model = configuredModel && *configuredModel
                ? configuredModel
                : modelPath(directory,"scrfd_2.5g_bnkps.dynamic.onnx");
            scrfd_ = std::make_unique<SCRFDDetector>(env_, model);
            std::cout << "LR-ASD face detector: SCRFD (" << model << ")\n";
        } else if (detectorName_ == "s3fd") {
            const char *configuredModel = std::getenv("LR_ASD_S3FD_MODEL");
            const std::string model = configuredModel && *configuredModel
                ? configuredModel
                : modelPath(directory,"s3fd_270x480.onnx");
            detector_ = std::make_unique<Ort::Session>(env_, model.c_str(), options_);
            std::cout << "LR-ASD face detector: S3FD (" << model << ")\n";
        } else {
            throw std::runtime_error("LR_ASD_DETECTOR must be 's3fd' or 'scrfd'");
        }
    }

    std::vector<LRASDPrediction> process(const std::vector<cv::Mat> &frames,
                                         const std::vector<float> &audio)
    {
        const auto started=Clock::now();
        for (const cv::Mat &frame : frames) updateTracks(frame);
        std::vector<LRASDPrediction> predictions;
        if (audio.size()<3440 || tracks_.empty()) return predictions;
        for (auto &[id, track] : tracks_) {
            const int count=std::min<int>(25,track.faces.size());
            if (count<5) continue;
            const size_t needed=count*640+240;
            if (audio.size()<needed) continue;
            std::vector<float> segment(audio.end()-needed,audio.end());
            std::vector<float> features=mfcc(segment), visual;
            if (features.size()!=static_cast<size_t>(count*4*13)) continue;
            visual.reserve(count*112*112);
            for (int i=track.faces.size()-count; i<static_cast<int>(track.faces.size()); ++i)
                visual.insert(visual.end(),track.faces[i].begin(),track.faces[i].end());
            std::array<int64_t,3> audioShape{1,count*4,13};
            std::array<int64_t,4> videoShape{1,count,112,112};
            auto mem=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
            std::array<Ort::Value,2> inputs{
                Ort::Value::CreateTensor<float>(mem,features.data(),features.size(),audioShape.data(),audioShape.size()),
                Ort::Value::CreateTensor<float>(mem,visual.data(),visual.size(),videoShape.data(),videoShape.size())};
            const char *inputNames[]={"audio","visual"}, *outputNames[]={"logits"};
            auto output=asd_.Run(Ort::RunOptions{nullptr},inputNames,inputs.data(),2,outputNames,1);
            const float *logits=output[0].GetTensorData<float>();
            track.score=logits[(count-1)*2+1];
            const int x1=std::max(0,static_cast<int>(std::lround(track.box.x1)));
            const int y1=std::max(0,static_cast<int>(std::lround(track.box.y1)));
            const int x2=std::max(x1+1,static_cast<int>(std::lround(track.box.x2)));
            const int y2=std::max(y1+1,static_cast<int>(std::lround(track.box.y2)));
            predictions.push_back({id,track.score,track.score>=0,
                                   cv::Rect(x1,y1,x2-x1,y2-y1)});
            std::cout << "{\"component\":\"LR-ASD\",\"track\":" << id
                      << ",\"class1_logit\":" << std::fixed << std::setprecision(3) << track.score
                      << ",\"speaking\":" << (track.score>=0 ? "true" : "false")
                      << ",\"window_frames\":" << count << "}" << std::endl;
        }
        ++calls_;
        latencyMs_ += std::chrono::duration<double,std::milli>(Clock::now()-started).count();
        if (calls_%50==0)
            std::cout << "{\"component\":\"LR-ASD-performance\",\"mean_update_ms\":"
                      << latencyMs_/calls_ << ",\"updates\":" << calls_ << "}" << std::endl;
        return predictions;
    }

private:
    static Ort::SessionOptions makeOptions()
    {
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.SetIntraOpNumThreads(1);
#ifdef LR_ASD_USE_CUDA
        try { OrtCUDAProviderOptions cuda{}; options.AppendExecutionProvider_CUDA(cuda); std::cout << "LR-ASD execution provider: CUDA\n"; }
        catch (const Ort::Exception &e) { std::cerr << "LR-ASD CUDA unavailable, using CPU: " << e.what() << '\n'; }
#endif
        return options;
    }

    std::vector<Box> detect(const cv::Mat &frame)
    {
        if (scrfd_) {
            std::vector<Box> boxes;
            for (const auto &box : scrfd_->detect(frame))
                boxes.push_back({box.x1, box.y1, box.x2, box.y2, box.confidence});
            return boxes;
        }

        cv::Mat resized,rgb; cv::resize(frame,resized,{480,270}); cv::cvtColor(resized,rgb,cv::COLOR_BGR2RGB);
        std::vector<float> data(3*270*480); const float mean[]={123,117,104};
        for (int c=0;c<3;++c) for(int y=0;y<270;++y) for(int x=0;x<480;++x)
            data[c*270*480+y*480+x]=rgb.at<cv::Vec3b>(y,x)[c]-mean[c];
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

    void updateTracks(const cv::Mat &frame)
    {
        auto boxes=detect(frame); std::vector<int> available,assigned;
        for(auto &[id,track]:tracks_) available.push_back(id);
        for(const Box &box:boxes) {
            int id=-1; float best=0;
            for(int candidate:available) { float value=iou(box,tracks_[candidate].box); if(value>best){best=value;id=candidate;} }
            if(best<.3f){id=nextTrack_++;tracks_[id]=Track{};}
            else available.erase(std::find(available.begin(),available.end(),id));
            auto &track=tracks_[id]; track.box=box; track.faces.push_back(cropFace(frame,box));
            if(track.faces.size()>25) track.faces.pop_front();
            track.missed=0;
            assigned.push_back(id);
        }
        for(auto it=tracks_.begin();it!=tracks_.end();) {
            if(std::find(assigned.begin(),assigned.end(),it->first)==assigned.end() && ++it->second.missed>10) it=tracks_.erase(it);
            else ++it;
        }
    }

    Ort::Env env_; Ort::SessionOptions options_; Ort::Session asd_;
    std::unique_ptr<Ort::Session> detector_;
    std::unique_ptr<SCRFDDetector> scrfd_;
    std::string detectorName_;
    std::map<int,Track> tracks_; int nextTrack_=0; std::uint64_t calls_=0; double latencyMs_=0;
};

LRASDWorker::LRASDWorker() = default;
LRASDWorker::~LRASDWorker() { stop(); }

bool LRASDWorker::start(const std::string &modelDirectory)
{
    try { engine_=std::make_unique<LRASDEngine>(modelDirectory); }
    catch(const std::exception &e) { std::cerr << "LR-ASD initialization failed: " << e.what() << '\n'; return false; }
    running_=true; thread_=std::thread(&LRASDWorker::run,this); return true;
}

void LRASDWorker::stop()
{
    running_=false;
    inputReady_.notify_all();
    if(thread_.joinable()) thread_.join();
    engine_.reset();
}

void LRASDWorker::submit(VABuffer snapshot)
{
    if(!running_) return;
    {
        std::lock_guard lock(inputMutex_);
        pendingInput_=std::move(snapshot); // keep only the newest live window
    }
    inputReady_.notify_one();
}

std::vector<LRASDPrediction> LRASDWorker::latestPredictions() const
{
    std::lock_guard lock(predictionMutex_);
    return latestPredictions_;
}

void LRASDWorker::run()
{
    const unsigned char *lastFrameData=nullptr;
    while(running_) {
        VABuffer snapshot;
        {
            std::unique_lock lock(inputMutex_);
            inputReady_.wait(lock,[this]{return !running_ || pendingInput_.has_value();});
            if(!running_) break;
            snapshot=std::move(*pendingInput_);
            pendingInput_.reset();
        }
        if(snapshot.vecframes.empty()) continue;
        std::vector<cv::Mat> allFrames,frames;
        while(!snapshot.vecframes.empty()) { allFrames.push_back(snapshot.vecframes.front()); snapshot.vecframes.pop(); }
        auto previous=std::find_if(allFrames.begin(),allFrames.end(),[lastFrameData](const cv::Mat &f){return f.data==lastFrameData;});
        if(previous!=allFrames.end()) frames.assign(std::next(previous),allFrames.end());
        else frames=allFrames;
        lastFrameData=allFrames.back().data;
        if(frames.empty()) continue;
        if(frames.size()>5) frames.erase(frames.begin(),frames.end()-5);
        std::vector<float> audio; audio.reserve(snapshot.AudioSignal.size());
        // scipy.io.wavfile.read supplies PCM16 magnitudes directly to the
        // official Python MFCC implementation; preserve that scale here.
        while(!snapshot.AudioSignal.empty()) {
            audio.push_back(static_cast<float>(snapshot.AudioSignal.front()));
            snapshot.AudioSignal.pop();
        }
        try {
            auto predictions=engine_->process(frames,audio);
            std::lock_guard lock(predictionMutex_);
            latestPredictions_=std::move(predictions);
        }
        catch(const std::exception &e) { std::cerr << "LR-ASD update failed: " << e.what() << '\n'; }
    }
}
