#include "VideoAudioBuffer.hpp"

VABuffer VideoAudioBuffer::GetVideoAudioBuffer()
{
    std::scoped_lock lock(mtx_video, mtx_audio);
    VABuffer temp = mInternalBuffer;
    return temp;
}

void VideoAudioBuffer::AddAFrame(Mat ANewFrame)
{
    mtx_video.lock();
    // cv::Mat is reference-counted, so this keeps the decoded pixels alive
    // without an unnecessary full-frame copy.
    mInternalBuffer.vecframes.push(ANewFrame);
    ReduceFrameBuffer(uiFrameBufferSize);
    mtx_video.unlock();
}

void VideoAudioBuffer::AddAudio(const short *pShort, long long sampleCount)
{
    mtx_audio.lock();
    for (long long i = 0; i < sampleCount; i++)
    {
        mInternalBuffer.AudioSignal.push(pShort[i]);
    }
    ReduceAudioBuffer(uiSamepleSize);
    mtx_audio.unlock();
}

void VideoAudioBuffer::ReduceFrameBuffer(std::size_t uiReduceToCount)
{
    while (mInternalBuffer.vecframes.size() > uiReduceToCount)
    {
        mInternalBuffer.vecframes.pop();
    }
}

void VideoAudioBuffer::ReduceAudioBuffer(std::size_t uiReduceToCount)
{
    while (mInternalBuffer.AudioSignal.size() > uiReduceToCount)
    {
        mInternalBuffer.AudioSignal.pop();
    }
}
