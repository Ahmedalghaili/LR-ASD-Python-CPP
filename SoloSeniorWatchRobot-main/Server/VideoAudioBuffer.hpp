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
    // LR-ASD needs 25 frames and about 1.02 seconds of 16 kHz audio. Keep a
    // small safety margin instead of retaining 12 seconds of stale data.
    std::size_t uiFrameBufferSize = 50;
    std::size_t uiSamepleSize = 32000;
    VABuffer mInternalBuffer;
    mutex mtx_video;
    mutex mtx_audio;
};

#endif
