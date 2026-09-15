// Reuse the original detector, crop, tracking data, MFCC and LR-ASD helpers.
// Only the experimental executable gets the new detector and instrumentation.
#define main original_main
#include "../../../cpp/src/main.cpp"
#undef main
#include "scrfd.hpp"
#include <functional>

struct Options {
    std::string detector="s3fd", device="cuda", model, asd, video, audio, output;
    int width=640,height=640,warmup=10,max_frames=0;
    float score=.5f,nms=.4f;
};
Options parse(int argc,char**argv) {
    Options o;
    std::map<std::string,std::string> values;
    for(int i=1;i<argc;i+=2) {
        if(i+1>=argc) throw std::runtime_error("Options require values");
        values[argv[i]]=argv[i+1];
    }
    auto take=[&](const std::string& key,const std::string& fallback){
        auto it=values.find(key); if(it==values.end())return fallback;
        auto value=it->second;values.erase(it);return value;
    };
    o.detector=take("--detector",o.detector);o.device=take("--device",o.device);
    o.model=take("--model","");o.asd=take("--asd","");o.video=take("--video","");
    o.audio=take("--audio","");o.output=take("--output","");
    o.width=std::stoi(take("--width","640"));o.height=std::stoi(take("--height","640"));
    o.warmup=std::stoi(take("--warmup","10"));o.max_frames=std::stoi(take("--max-frames","0"));
    o.score=std::stof(take("--score","0.5"));o.nms=std::stof(take("--nms","0.4"));
    if(!values.empty())throw std::runtime_error("Unknown option: "+values.begin()->first);
    if(o.model.empty()||o.asd.empty()||o.video.empty()||o.audio.empty()||o.output.empty())
        throw std::runtime_error("Required: --detector s3fd|scrfd --model PATH --asd PATH --video PATH --audio WAV --output MP4 [--device cuda|cpu]");
    if((o.detector!="s3fd"&&o.detector!="scrfd")||(o.device!="cuda"&&o.device!="cpu")||o.warmup<0||o.max_frames<0||!(o.score>0&&o.score<=1)||!(o.nms>0&&o.nms<=1))
        throw std::runtime_error("Invalid detector, device, threshold or frame count");
    return o;
}
double ms(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
void sync_gpu(){if(runtime_device.is_cuda())torch::cuda::synchronize();}
double mean(const std::vector<double>& v){return v.empty()?0:std::accumulate(v.begin(),v.end(),0.)/v.size();}
double p95(std::vector<double> v){if(v.empty())return 0;std::sort(v.begin(),v.end());return v[size_t(.95*(v.size()-1))];}

int main(int argc,char**argv) try {
    const auto o=parse(argc,argv);
    torch::NoGradGuard ng;
    if(o.device=="cuda"&&!torch::cuda::is_available())throw std::runtime_error("CUDA unavailable");
    runtime_device=torch::Device(o.device=="cuda"?torch::kCUDA:torch::kCPU);
    auto asd=torch::jit::load(o.asd,runtime_device);asd.eval();
    std::unique_ptr<Scrfd> scrfd;
    torch::jit::script::Module s3fd;
    if(o.detector=="scrfd")scrfd=std::make_unique<Scrfd>(o.model.c_str(),o.width,o.height,o.score,o.nms,runtime_device.is_cuda());
    else {s3fd=torch::jit::load(o.model,runtime_device);s3fd.eval();}
    auto detector=[&](const cv::Mat& frame){return scrfd?(*scrfd)(frame):detect(s3fd,frame);};
    auto w=read_wav(o.audio);if(w.rate!=16000)throw std::runtime_error("Expected 16 kHz audio");
    auto af=mfcc(w.x,w.rate);
    cv::VideoCapture cap(o.video);
    if(!cap.isOpened())throw std::runtime_error("Cannot open video");
    const double sfps=cap.get(cv::CAP_PROP_FPS);
    const int width=cap.get(cv::CAP_PROP_FRAME_WIDTH),height=cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    if(sfps<=0)throw std::runtime_error("Invalid source FPS");
    cv::Mat first;if(!cap.read(first))throw std::runtime_error("Empty video");
    for(int i=0;i<o.warmup;++i){detector(first);asd.forward({torch::zeros({1,100,13},runtime_device),torch::zeros({1,25,112,112},runtime_device)});}
    sync_gpu();cap.release();cap.open(o.video);
    cv::VideoWriter out(o.output,cv::VideoWriter::fourcc('m','p','4','v'),25,{width,height});
    if(!out.isOpened())throw std::runtime_error("Cannot create output video");
    std::ofstream csv(o.output+".csv"),timing(o.output+".timing.csv");
    if(!csv||!timing)throw std::runtime_error("Cannot create CSV");
    csv<<"frame,time_s,track,score,speaking,x1,y1,x2,y2\n";
    timing<<"frame,detector_ms,asd_ms,total_ms,faces\n";
    std::map<int,Track>tracks;int next=0,fi=0,si=-1;size_t face_count=0;
    cv::Mat frame;std::vector<double>det_lat,asd_lat,frame_lat;
    auto begin=Clock::now();
    while(!o.max_frames||fi<o.max_frames){
        auto frame_start=Clock::now();
        int want=std::lround(fi*sfps/25.0);
        while(si<want){if(!cap.read(frame)){frame.release();break;}++si;}
        if(frame.empty())break;
        auto td=Clock::now();auto boxes=detector(frame);sync_gpu();double det_ms=ms(td);det_lat.push_back(det_ms);
        std::vector<int>assigned,available;for(auto&[id,t]:tracks)available.push_back(id);
        for(auto&b:boxes){
            int id=-1;float best=0;
            for(int q:available){float z=overlap(b,tracks[q].box);if(z>best){best=z;id=q;}}
            if(best<.3){id=next++;tracks[id]=Track();}else available.erase(std::find(available.begin(),available.end(),id));
            auto&t=tracks[id];t.box=b;t.faces.push_back(crop112(frame,b));if(t.faces.size()>25)t.faces.pop_front();
            t.missed=0;assigned.push_back(id);
        }
        for(auto it=tracks.begin();it!=tracks.end();){if(std::find(assigned.begin(),assigned.end(),it->first)==assigned.end()&&++it->second.missed>10)it=tracks.erase(it);else ++it;}
        double asd_ms=0;
        if(fi%5==0){
            int ae=std::min<int>(af.size()/13,(fi+1)*4);
            for(int id:assigned){
                auto&t=tracks[id];int n=std::min<int>(t.faces.size(),ae/4);if(n<5)continue;
                std::vector<float>vv;for(int j=t.faces.size()-n;j<(int)t.faces.size();j++)vv.insert(vv.end(),t.faces[j].begin(),t.faces[j].end());
                auto at=torch::from_blob(af.data()+(ae-n*4)*13,{1,n*4,13},torch::kFloat32).clone().to(runtime_device);
                auto vt=torch::from_blob(vv.data(),{1,n,112,112},torch::kFloat32).clone().to(runtime_device);
                sync_gpu();auto ts=Clock::now();auto z=asd.forward({at,vt}).toTensor();sync_gpu();double elapsed=ms(ts);
                asd_lat.push_back(elapsed);asd_ms+=elapsed;t.score=z[-1][1].item<float>();
            }
        }
        face_count+=assigned.size();
        for(int id:assigned){
            auto&t=tracks[id];auto b=t.box;cv::Scalar color=t.score>=0?cv::Scalar(0,255,0):cv::Scalar(0,0,255);
            cv::rectangle(frame,{(int)b.x1,(int)b.y1},{(int)b.x2,(int)b.y2},color,4);
            cv::putText(frame,std::to_string(t.score).substr(0,4),{(int)b.x1,std::max(25,(int)b.y1-5)},cv::FONT_HERSHEY_SIMPLEX,.8,color,2);
            csv<<fi<<','<<fi/25.0<<','<<id<<','<<t.score<<','<<(t.score>=0)<<','<<b.x1<<','<<b.y1<<','<<b.x2<<','<<b.y2<<'\n';
        }
        out.write(frame);frame_lat.push_back(ms(frame_start));
        timing<<fi<<','<<det_ms<<','<<asd_ms<<','<<frame_lat.back()<<','<<assigned.size()<<'\n';++fi;
    }
    out.release();csv.close();timing.close();double wall=ms(begin)/1000;
    std::cout<<std::setprecision(8)<<"{\"detector\":\""<<o.detector<<"\",\"device\":\""<<o.device
      <<"\",\"frames\":"<<fi<<",\"wall_s\":"<<wall<<",\"pipeline_fps\":"<<fi/wall
      <<",\"detector_mean_ms\":"<<mean(det_lat)<<",\"detector_p95_ms\":"<<p95(det_lat)
      <<",\"frame_p95_ms\":"<<p95(frame_lat)<<",\"tracks_created\":"<<next<<",\"face_observations\":"<<face_count
      <<",\"asd_calls\":"<<asd_lat.size()<<",\"asd_mean_ms\":"<<mean(asd_lat)<<"}\n";
    return fi?0:1;
} catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
