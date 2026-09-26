#include <torch/script.h>
#include <torch/cuda.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <map>
#include <deque>
#include <string>
#include <thread>
#include <vector>

using Clock=std::chrono::steady_clock;
static torch::Device runtime_device(torch::kCPU);
struct Wav { int rate=0; std::vector<float> x; };

static uint32_t u32(std::ifstream&f){uint32_t v;f.read((char*)&v,4);return v;}
static uint16_t u16(std::ifstream&f){uint16_t v;f.read((char*)&v,2);return v;}
Wav read_wav(const std::string&p){
 std::ifstream f(p,std::ios::binary); char id[4]; f.read(id,4); u32(f); f.read(id,4);
 int fmt=0,ch=0,bits=0,rate=0; std::vector<char> raw;
 while(f.read(id,4)){auto n=u32(f); std::string s(id,4); if(s=="fmt "){fmt=u16(f);ch=u16(f);rate=u32(f);u32(f);u16(f);bits=u16(f);if(n>16)f.seekg(n-16,std::ios::cur);}else if(s=="data"){raw.resize(n);f.read(raw.data(),n);}else f.seekg(n+(n&1),std::ios::cur);}
 if(fmt!=1||bits!=16||ch<1)throw std::runtime_error("Only PCM16 WAV supported: "+p);
 Wav w;w.rate=rate;auto*q=(int16_t*)raw.data();size_t frames=raw.size()/2/ch;w.x.resize(frames);
 for(size_t i=0;i<frames;i++){float sum=0;for(int c=0;c<ch;c++)sum+=q[i*ch+c];w.x[i]=sum/ch;}return w;
}
double hzmel(double h){return 2595*std::log10(1+h/700);} double melhz(double m){return 700*(std::pow(10,m/2595)-1);}
std::vector<float> mfcc(const std::vector<float>&in,int sr){
 const int win=std::lround(.025*sr),step=std::lround(.010*sr),nfft=512,nfilt=26,ncep=13;
 std::vector<float>x(in.size());if(!in.empty())x[0]=in[0];for(size_t i=1;i<in.size();i++)x[i]=in[i]-.97f*in[i-1];
 int nf=in.size()<=size_t(win)?1:1+int(std::ceil((in.size()-win)/double(step)));size_t padded=(nf-1)*step+win;x.resize(padded);
 std::vector<int>bin(nfilt+2);double lo=hzmel(0),hi=hzmel(sr/2.0);for(int i=0;i<nfilt+2;i++)bin[i]=std::floor((nfft+1)*melhz(lo+(hi-lo)*i/(nfilt+1))/sr);
 std::vector<float>out(nf*ncep); const double pi=std::acos(-1.0);
 for(int n=0;n<nf;n++){
  std::vector<double>powv(nfft/2+1);double energy=0;
  for(int k=0;k<=nfft/2;k++){double re=0,im=0;for(int j=0;j<win;j++){double v=x[n*step+j];re+=v*std::cos(2*pi*k*j/nfft);im-=v*std::sin(2*pi*k*j/nfft);}powv[k]=(re*re+im*im)/nfft;energy+=powv[k];}
  energy=std::max(energy,std::numeric_limits<double>::epsilon());std::vector<double>fb(nfilt);
  for(int m=1;m<=nfilt;m++){for(int k=bin[m-1];k<bin[m];k++)fb[m-1]+=powv[k]*(k-bin[m-1])/double(bin[m]-bin[m-1]);for(int k=bin[m];k<bin[m+1];k++)fb[m-1]+=powv[k]*(bin[m+1]-k)/double(bin[m+1]-bin[m]);fb[m-1]=std::log(std::max(fb[m-1],std::numeric_limits<double>::epsilon()));}
  for(int c=0;c<ncep;c++){double z=0;for(int m=0;m<nfilt;m++)z+=fb[m]*std::cos(pi*c*(m+.5)/nfilt);z*=std::sqrt(2.0/nfilt);if(c==0)z/=std::sqrt(2.0);z*=1+11*std::sin(pi*c/22.0);out[n*ncep+c]=z;}out[n*ncep]=std::log(energy);
 }
 return out;
}
std::vector<float> load_faces(const std::string&p,int&count){cv::VideoCapture cap(p);cv::Mat f,g;std::vector<float>v;count=0;while(cap.read(f)){cv::cvtColor(f,g,cv::COLOR_BGR2GRAY);cv::resize(g,g,{224,224});g=g(cv::Rect(56,56,112,112));v.reserve(v.size()+12544);for(int y=0;y<112;y++)for(int x=0;x<112;x++)v.push_back(g.at<uint8_t>(y,x));count++;}return v;}
std::vector<float> infer(torch::jit::script::Module&m,const std::vector<float>&a,const std::vector<float>&v,int frames,std::vector<double>&lat){
 const std::vector<int>durs={1,1,1,2,2,2,3,3,4,5,6};std::vector<std::vector<float>>all;
 auto opt=torch::TensorOptions().dtype(torch::kFloat32).device(runtime_device);
  for(int d:durs){std::vector<float>s;for(int st=0;st<frames;st+=d*25){int vn=std::min(d*25,frames-st),an=vn*4;auto at=torch::from_blob((void*)(a.data()+st*4*13),{1,an,13},torch::kFloat32).clone().to(opt);auto vt=torch::from_blob((void*)(v.data()+st*112*112),{1,vn,112,112},torch::kFloat32).clone().to(opt);auto t=Clock::now();auto o=m.forward({at,vt}).toTensor();lat.push_back(std::chrono::duration<double,std::milli>(Clock::now()-t).count());auto cpu=o.to(torch::kCPU).contiguous();auto q=cpu.accessor<float,2>();for(int i=0;i<vn;i++)s.push_back(q[i][1]);}all.push_back(std::move(s));}
 std::vector<float>r(frames);for(int i=0;i<frames;i++){double z=0;for(auto&s:all)z+=s[i];r[i]=std::round(z/all.size()*10)/10;}return r;
}
struct Box{float x1,y1,x2,y2,score;};
float overlap(const Box&a,const Box&b){float x1=std::max(a.x1,b.x1),y1=std::max(a.y1,b.y1),x2=std::min(a.x2,b.x2),y2=std::min(a.y2,b.y2);float in=std::max(0.f,x2-x1)*std::max(0.f,y2-y1);float u=(a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-in;return u>0?in/u:0;}
std::vector<Box> detect(torch::jit::script::Module&net,const cv::Mat&frame){
 int dw=std::lround(frame.cols*.25),dh=std::lround(frame.rows*.25);cv::Mat small,rgb;cv::resize(frame,small,{dw,dh});cv::cvtColor(small,rgb,cv::COLOR_BGR2RGB);std::vector<float>buf(3*dh*dw);float mean[3]={123,117,104};for(int c=0;c<3;c++)for(int y=0;y<dh;y++)for(int x=0;x<dw;x++)buf[c*dh*dw+y*dw+x]=rgb.at<cv::Vec3b>(y,x)[c]-mean[c];
 auto t=torch::from_blob(buf.data(),{1,3,dh,dw},torch::kFloat32).clone().to(runtime_device);auto tup=net.forward({t}).toTuple();auto loc=tup->elements()[0].toTensor().to(torch::kCPU).contiguous();auto conf=tup->elements()[1].toTensor().to(torch::kCPU).contiguous();auto l=loc.accessor<float,3>();auto c=conf.accessor<float,3>();int fh[6],fw[6],steps[6]={4,8,16,32,64,128},mins[6]={16,32,64,128,256,512};fh[0]=dh/4;fw[0]=dw/4;for(int z=1;z<6;z++){if(z==3){fh[z]=fh[z-1]/2;fw[z]=fw[z-1]/2;}else{fh[z]=(fh[z-1]+1)/2;fw[z]=(fw[z-1]+1)/2;}}std::vector<Box>b;int k=0;
 for(int z=0;z<6;z++)for(int y=0;y<fh[z];y++)for(int x=0;x<fw[z];x++,k++){
  if(c[0][k][1]<.9)continue;
  float cx=(x+.5f)*steps[z]/dw,cy=(y+.5f)*steps[z]/dh,pw=mins[z]/float(dw),ph=mins[z]/float(dh);
  cx+=l[0][k][0]*.1f*pw;cy+=l[0][k][1]*.1f*ph;
  pw*=std::exp(l[0][k][2]*.2f);ph*=std::exp(l[0][k][3]*.2f);
  Box q{(cx-pw/2)*frame.cols,(cy-ph/2)*frame.rows,(cx+pw/2)*frame.cols,(cy+ph/2)*frame.rows,c[0][k][1]};
  if(!std::isfinite(q.x1)||!std::isfinite(q.y1)||!std::isfinite(q.x2)||!std::isfinite(q.y2))continue;
  q.x1=std::clamp(q.x1,0.f,float(frame.cols-1));q.y1=std::clamp(q.y1,0.f,float(frame.rows-1));
  q.x2=std::clamp(q.x2,0.f,float(frame.cols));q.y2=std::clamp(q.y2,0.f,float(frame.rows));
  if(q.x2>q.x1+1.f&&q.y2>q.y1+1.f)b.push_back(q);
 }
 std::sort(b.begin(),b.end(),[](auto&a,auto&d){return a.score>d.score;});std::vector<Box>keep;for(auto&q:b){bool ok=true;for(auto&r:keep)if(overlap(q,r)>.1){ok=false;break;}if(ok)keep.push_back(q);}return keep;
}
std::vector<float> crop112(const cv::Mat&frame,const Box&b){float size=std::max(b.x2-b.x1,b.y2-b.y1),cx=(b.x1+b.x2)/2,cy=(b.y1+b.y2)/2,h=size*.55f;int x1=std::floor(cx-h),y1=std::floor(cy-h),x2=std::ceil(cx+h),y2=std::ceil(cy+h);cv::Mat pad;int pl=std::max(0,-x1),pt=std::max(0,-y1),pr=std::max(0,x2-frame.cols),pb=std::max(0,y2-frame.rows);cv::copyMakeBorder(frame,pad,pt,pb,pl,pr,cv::BORDER_CONSTANT,{110,110,110});x1+=pl;x2+=pl;y1+=pt;y2+=pt;cv::Mat g;cv::cvtColor(pad(cv::Rect(x1,y1,x2-x1,y2-y1)),g,cv::COLOR_BGR2GRAY);cv::resize(g,g,{112,112});std::vector<float>v(12544);for(int y=0;y<112;y++)for(int x=0;x<112;x++)v[y*112+x]=g.at<uint8_t>(y,x);return v;}
struct Track{Box box;std::deque<std::vector<float>>faces;float score=0;int missed=0;};
int video_main(int argc,char**argv){if(argc<7){std::cerr<<"usage: lr_asd --video ASD_MODEL S3FD_MODEL VIDEO AUDIO_WAV OUTPUT_MP4\n";return 2;}torch::NoGradGuard ng;auto asd=torch::jit::load(argv[2],runtime_device);auto fd=torch::jit::load(argv[3],runtime_device);asd.eval();fd.eval();auto w=read_wav(argv[5]);if(w.rate!=16000)throw std::runtime_error("audio must be 16 kHz mono");auto af=mfcc(w.x,w.rate);cv::VideoCapture cap(argv[4]);double sfps=cap.get(cv::CAP_PROP_FPS);int width=cap.get(cv::CAP_PROP_FRAME_WIDTH),height=cap.get(cv::CAP_PROP_FRAME_HEIGHT);cv::VideoWriter out(argv[6],cv::VideoWriter::fourcc('m','p','4','v'),25,{width,height});std::map<int,Track>tracks;int next=0,fi=0,si=-1;cv::Mat frame;std::ofstream csv(std::string(argv[6])+".csv");csv<<"frame,time_s,track,score,speaking,x1,y1,x2,y2\n";std::vector<double>lat;auto begin=Clock::now();
 while(true){int want=std::lround(fi*sfps/25.0);while(si<want){if(!cap.read(frame)){frame.release();break;}si++;}if(frame.empty())break;auto boxes=detect(fd,frame);std::vector<int>assigned;std::vector<int>available;for(auto&[id,t]:tracks)available.push_back(id);for(auto&b:boxes){int id=-1;float best=0;for(int q:available){float z=overlap(b,tracks[q].box);if(z>best){best=z;id=q;}}if(best<.3){id=next++;tracks[id]=Track();}else available.erase(std::find(available.begin(),available.end(),id));auto&t=tracks[id];t.box=b;t.faces.push_back(crop112(frame,b));if(t.faces.size()>25)t.faces.pop_front();t.missed=0;assigned.push_back(id);}for(auto it=tracks.begin();it!=tracks.end();){if(std::find(assigned.begin(),assigned.end(),it->first)==assigned.end()&&++it->second.missed>10)it=tracks.erase(it);else ++it;}
  if(fi%5==0){int ae=std::min<int>(af.size()/13,(fi+1)*4);for(int id:assigned){auto&t=tracks[id];int n=std::min<int>(t.faces.size(),ae/4);if(n<5)continue;std::vector<float>vv;for(int j=t.faces.size()-n;j<(int)t.faces.size();j++)vv.insert(vv.end(),t.faces[j].begin(),t.faces[j].end());auto at=torch::from_blob(af.data()+(ae-n*4)*13,{1,n*4,13},torch::kFloat32).clone().to(runtime_device);auto vt=torch::from_blob(vv.data(),{1,n,112,112},torch::kFloat32).clone().to(runtime_device);if(runtime_device.is_cuda())torch::cuda::synchronize();auto ts=Clock::now();auto z=asd.forward({at,vt}).toTensor();if(runtime_device.is_cuda())torch::cuda::synchronize();lat.push_back(std::chrono::duration<double,std::milli>(Clock::now()-ts).count());t.score=z[-1][1].item<float>();}}
  for(int id:assigned){auto&t=tracks[id];auto b=t.box;cv::Scalar color=t.score>=0?cv::Scalar(0,255,0):cv::Scalar(0,0,255);cv::rectangle(frame,{(int)b.x1,(int)b.y1},{(int)b.x2,(int)b.y2},color,4);cv::putText(frame,std::to_string(t.score).substr(0,4),{(int)b.x1,std::max(25,(int)b.y1-5)},cv::FONT_HERSHEY_SIMPLEX,.8,color,2);csv<<fi<<','<<fi/25.0<<','<<id<<','<<t.score<<','<<(t.score>=0)<<','<<b.x1<<','<<b.y1<<','<<b.x2<<','<<b.y2<<'\n';}out.write(frame);fi++;const char*pace=std::getenv("LR_ASD_PACE");if(pace&&std::string(pace)=="1"){double delay=fi/25.0-std::chrono::duration<double>(Clock::now()-begin).count();if(delay>0)std::this_thread::sleep_for(std::chrono::duration<double>(delay));}}
 double wall=std::chrono::duration<double>(Clock::now()-begin).count();double lm=lat.empty()?0:std::accumulate(lat.begin(),lat.end(),0.0)/lat.size();std::cout<<"{\"device\":\""<<(runtime_device.is_cuda()?"CUDA":"CPU")<<"\",\"frames\":"<<fi<<",\"wall_s\":"<<wall<<",\"pipeline_fps\":"<<fi/wall<<",\"tracks_created\":"<<next<<",\"asd_calls\":"<<lat.size()<<",\"asd_latency_mean_ms\":"<<lm<<"}\n";return 0;}
int main(int argc,char**argv){
 const char* requested=std::getenv("LR_ASD_DEVICE");if(requested&&std::string(requested)=="cuda"){if(!torch::cuda::is_available()){std::cerr<<"CUDA requested but unavailable\n";return 1;}runtime_device=torch::Device(torch::kCUDA);}
 if(argc>1&&std::string(argv[1])=="--video")try{return video_main(argc,argv);}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
 if(argc<5){std::cerr<<"usage: lr_asd MODEL FACE_AVI AUDIO_WAV OUTPUT_CSV [TRACK]\n";return 2;}int track=argc>5?std::stoi(argv[5]):0;
 try{torch::NoGradGuard ng;torch::jit::script::Module model=torch::jit::load(argv[1],torch::kCPU);model.eval();auto w=read_wav(argv[3]);if(w.rate!=16000)throw std::runtime_error("audio must be 16 kHz");auto a=mfcc(w.x,w.rate);int vf=0;auto v=load_faces(argv[2],vf);int frames=std::min(vf,int(a.size()/13/4));a.resize(frames*4*13);v.resize(frames*112*112);std::vector<double>lat;auto start=Clock::now();auto score=infer(model,a,v,frames,lat);double wall=std::chrono::duration<double>(Clock::now()-start).count();std::ofstream f(argv[4]);f<<"track,frame,time_s,official_class1_logit,speaking\n";for(int i=0;i<frames;i++)f<<track<<','<<i<<','<<std::fixed<<std::setprecision(2)<<i/25.0<<','<<std::setprecision(1)<<score[i]<<','<<(score[i]>=0)<<'\n';std::sort(lat.begin(),lat.end());double mean=std::accumulate(lat.begin(),lat.end(),0.0)/lat.size();std::cout<<"{\"device\":\"CPU\",\"frames\":"<<frames<<",\"wall_s\":"<<wall<<",\"effective_fps\":"<<frames/wall<<",\"calls\":"<<lat.size()<<",\"latency_mean_ms\":"<<mean<<",\"latency_median_ms\":"<<lat[lat.size()/2]<<",\"latency_p95_ms\":"<<lat[size_t(.95*(lat.size()-1))]<<"}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}return 0;
}
