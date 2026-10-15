#include "SendMessageManager.hpp"
#include <cstdlib>
#include <string>
#include <iostream>
#include <signal.h>

int message_counter = 0;

void SendMessageManager::Send()
{
    //supress the SIGPIPE signal, sometimes the socket may be closed by the robot
    //and the write command will cause a SIGPIPE signal.
    //This signal will cause the program to exit.
    //The only way to ignore this signal is to set the signal handler to SIG_IGN.
    //This is a global signal handler, so it will affect all threads.
    //It is not a good idea to set the signal handler in a thread.
    //It is better to set it in the main thread.
    //But in this case, we have to set it here.
    ::signal(SIGPIPE, SIG_IGN);
    
    std::lock_guard<std::mutex> lock(mutex_message_buffer);
    if( mQueue.size() > 0)
    {
        if( pSocket && pSocket->isValid())          //socket can disconnect any time, it is in another thread
        {
            // Send every waiting command, in order. (Keeping only the last two
            // dropped commands added together, e.g. a body turn followed by a
            // head command.) Only a large backlog loses its oldest commands.
            while(mQueue.size() > 30)
            {
                mQueue.pop();
            }
            QDataStream socketStream(pSocket);
            while (!mQueue.empty())
            {
            RobotCommandProtobuf::RobotCommand message = mQueue.front();
            mQueue.pop();
            str_results_len = message.ByteSizeLong();
            if (str_results_len > static_cast<int>(sizeof(str_results)))
                continue;
            message.SerializeToArray(str_results,message.ByteSizeLong());

            pSocket->write("BeginOfADataFrame");
            socketStream.writeRawData(str_results, str_results_len);
            pSocket->write("EndOfADataFrame");
            }
            //debug
            //cout << "send a message " << endl;

            //The robot may close the connection suddenly.
            //pSocket->isValid cannot prevent it.
            //The only way is to igoore the SIGPIPE signal.
            pSocket->flush();       //This command is required to send out data in the buffer.
        }
        else
        {
            cout << "Socket is NULL or is not valid" << endl;
        }
    }
}

void SendMessageManager::AddMessage(RobotCommandProtobuf::RobotCommand message)
{
    // SSWR_ROBOT_SILENT=1: the robot never speaks or plays motion animations
    // (used when following a voice: Zenbo stops reporting voice directions
    // while it talks or moves on its own).
    static const bool silent = std::getenv("SSWR_ROBOT_SILENT") && std::string(std::getenv("SSWR_ROBOT_SILENT")) == "1";
    if (silent)
    {
        message.clear_speak_sentence();
        message.clear_smotion();
        message.clear_motion();
        message.clear_predefined_action();
        message.clear_dancetype();
        message.clear_song();
        if (message.ByteSizeLong() == 0)
            return;
    }
    mutex_message_buffer.lock();
    if(message.has_speak_sentence())
    {
        if (message.speak_sentence() != "")
        {
            //Debug
            //cout << "(A) " << message.speak_sentence() << "\n";
            //cout << "Add a Message to mQueue" << endl;
            mQueue.push(message);
        }
    }
    else
    {
        mQueue.push(message);
    }
    //cout << "Add a Message to mQueue" << endl;
    //mQueue.push(message);
    mutex_message_buffer.unlock();
}