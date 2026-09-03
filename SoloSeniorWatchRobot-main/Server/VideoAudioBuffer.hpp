#ifndef __VIDEOAUDIOBUFFER_HPP__
#define __VIDEOAUDIOBUFFER_HPP__

#include <opencv2/opencv.hpp>
#include <mutex>
#include <queue>

using std::array;
using std::vector;
using std::mutex;
using namespace cv;

struct VABuffer
{
    std::queue<Mat> vecframes;
    std::queue<short> AudioSignal;
};

class VideoAudioBuffer
{
public:
    void AddAFrame(Mat ANewFrame);
    void AddAudio(const short *pShort, long long sampleCount);
    void ReduceFrameBuffer(std::size_t uiReduceToCount);
    void ReduceAudioBuffer(std::size_t uiReduceToCount);
    VABuffer GetVideoAudioBuffer();

protected:
    std::size_t uiFrameBufferSize = 300;
    std::size_t uiSamepleSize = 160000;
    VABuffer mInternalBuffer;
    mutex mtx_video;
    mutex mtx_audio;
};

#endif
