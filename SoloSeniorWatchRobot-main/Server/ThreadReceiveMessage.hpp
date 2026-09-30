#ifndef __THREAD_RECEIVE_MESSAGES_hpp__
#define __THREAD_RECEIVE_MESSAGES_hpp__

#include <QThread>
#include <QTcpServer>
#include <QTcpSocket>
#include <QDebug>
#include <iostream>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <functional>
#include "SendMessageManager.hpp"
#include "SocketBufferParser.hpp"
#include "ThreadStateControl.hpp"
#include "ThreadProcessImage.hpp"
#include "ThreadSafeQueue.hpp"
#include "Logger.hpp"
#include "RobotCommand.pb.h"
#include "Setting.hpp"

using namespace std;

class ThreadReceiveMessage : public QThread
{
    Q_OBJECT

public:
    bool b_WhileLoop = true;
    condition_variable cond_var_receive_message;

    SendMessageManager *pSendMessageManager;
    // Because there are two clients connecting to this port: Tablet and Robot app, te DataFrames_queue should be inside the ThreadReceiveMessage class.
    ThreadSafeQueue<DataFrame> DataFrames_queue;
    ThreadStateControl *mpThreadStateControl;
    ThreadProcessImage *mpThreadProcessImage;
    ThreadWhisper *mpWhisper;
    Logger *mpLogger;
    // Called for every Zenbo voice-detection event that carries a direction of
    // arrival (raw SDK value). Set before the thread starts.
    std::function<void(int doa)> onVoiceDirection;
    // Called when Zenbo's listening (speakAndListen) has finished: an
    // utterance, a listen result, or the command's final state.
    std::function<void()> onListenFinished;
    // Called with each 200 ms stereo piece of Zenbo's two microphones
    // ("stereo_audio" from the Zenbo V2 app): interleaved PCM16, frames, rate.
    std::function<void(const int16_t *pcm, size_t frames, int rate)> onStereoAudio;

    void SetRobotModel(string sRobotModel)
    {
        ms_RobotModel = sRobotModel;
    }
    //    Setting *mpsetting = nullptr;

protected:
    void run();
    mutex mtx;
    string ms_RobotModel = "Unknown"; // default value is Unknown. It will be set to "Kebbi" or "Zenbo" when the robot sends the first status message.
};

#endif