#include "ThreadReceiveMessage.hpp"
#include "utility_time.hpp"

void ThreadReceiveMessage::run()
{
    while (b_WhileLoop)
    {
        std::unique_lock<std::mutex> lk(mtx);
        cond_var_receive_message.wait(lk);

        while (DataFrames_queue.size() > 0) // process all messages in the queue{
        {
            // Get message from the queue
            DataFrame dataframe; // = DataFrames_queue.front();
            DataFrames_queue.pop(dataframe);
            char *data_ = dataframe.data.get();
            size_t data_length = dataframe.length;

            // Here, I need to parse the protobuf object
            RobotCommandProtobuf::RobotToServerMessage RTSmessage;
            RTSmessage.ParseFromArray(data_, static_cast<int>(data_length));

            if (RTSmessage.has_description())
            {
                if (RTSmessage.description() == "onTTSComplete")
                {
                    mpThreadStateControl->NotifyEvent("onTTSComplete", chrono::system_clock::now());
                    if (ms_RobotModel == "ZenboJrII")
                        mpWhisper->SkipCurrentSpeech(); // skip the current speech, because the robot has hear its own voice.
                    mpLogger->LogToFile("Receive onTTSComplete signal");
                }
                else if (RTSmessage.description() == "onCompleteOfMotionPlay")
                {
                    mpThreadProcessImage->NotifyEvent("onCompleteOfMotionPlay", chrono::system_clock::now(), RTSmessage.yaw(), RTSmessage.pitch());
                }
                else if (RTSmessage.description() == "onVoiceDetect" ||
                         RTSmessage.description() == "onEventUserUtterance" ||
                         RTSmessage.description() == "onListenResult")
                {
                    // Zenbo voice detection. Log every event so the DOA can be
                    // checked on the robot even when following is disabled.
                    cout << "{\"component\":\"ZenboVoice\",\"event\":\"" << RTSmessage.description()
                         << "\",\"vad_status\":\"" << RTSmessage.vad_status()
                         << "\",\"sound_level\":" << RTSmessage.sound_level();
                    if (RTSmessage.has_doa())
                        cout << ",\"doa\":" << RTSmessage.doa();
                    cout << "}" << endl;
                    // Only "onDOA" events carry a fresh direction; the other
                    // events repeat the last one (or 0).
                    if (RTSmessage.has_doa() && RTSmessage.vad_status() == "onDOA" && onVoiceDirection)
                        onVoiceDirection(RTSmessage.doa());
                    if (RTSmessage.description() != "onVoiceDetect" && onListenFinished)
                        onListenFinished();
                }
                else if (RTSmessage.description().rfind("poc_result", 0) == 0 ||
                         RTSmessage.description().rfind("robot_state cmd=UTILITY", 0) == 0)
                {
                    // Zenbo V2 app: result of a test/utility action, or of lookAtUser.
                    cout << "{\"component\":\"ZenboApp\",\"message\":\"" << RTSmessage.description() << "\"}" << endl;
                }
                else if (RTSmessage.description().rfind("robot_state cmd=DS_SPEAK_AND_LISTEN", 0) == 0)
                {
                    cout << "{\"component\":\"ZenboApp\",\"message\":\"" << RTSmessage.description() << "\"}" << endl;
                    // Command states forwarded by the Zenbo V2 app (poc_action "states_on").
                    const string &d = RTSmessage.description();
                    if ((d.find("SUCCEED") != string::npos || d.find("FAILED") != string::npos) &&
                        onListenFinished)
                        onListenFinished();
                }
                else if (RTSmessage.description() == "onActivityRestart") // The mbtx activity complete
                {
                    // debug
                    cout << "(C) Receive onActivityRestart signal" << endl;
                    mpThreadStateControl->NotifyEvent("onActivityRestart", chrono::system_clock::now());
                }
            }

            if (RTSmessage.has_numberpressed())
            {
                int numberpressed = RTSmessage.numberpressed();

                std::string str_RobotSpeakSentence;
                int RobotExpressionIndex = 0;
                string sFace = "Unknown";
                cout << "Receive number " << numberpressed << endl;
                switch (numberpressed)
                {
                case 1:
                    str_RobotSpeakSentence = "一";
                    if (ms_RobotModel == "Zenbo" || ms_RobotModel == "ZenboJrII")
                        sFace = "ACTIVE";
                    else if (ms_RobotModel == "Kebbi")
                        sFace = "TTS_JoyA";
                    else
                        cout << "Unknown robot model." << endl;
                    break;
                case 2:
                    str_RobotSpeakSentence = "二";
                    if (ms_RobotModel == "Zenbo" || ms_RobotModel == "ZenboJrII")
                        sFace = "AWARE_LEFT";
                    else if (ms_RobotModel == "Kebbi")
                        sFace = "TTS_JoyB";
                    else
                        cout << "Unknown robot model." << endl;
                    break;
                case 3:
                    str_RobotSpeakSentence = "三";
                    if (ms_RobotModel == "Zenbo" || ms_RobotModel == "ZenboJrII")
                        sFace = "AWARE_RIGHT";
                    else if (ms_RobotModel == "Kebbi")
                        sFace = "TTS_JoyC";
                    else
                        cout << "Unknown robot model." << endl;
                    break;
                case 4:
                    str_RobotSpeakSentence = "四";
                    if (ms_RobotModel == "Zenbo" || ms_RobotModel == "ZenboJrII")
                        sFace = "CONFIDENT";
                    else if (ms_RobotModel == "Kebbi")
                        sFace = "TTS_SadnessA";
                    else
                        cout << "Unknown robot model." << endl;
                    break;
                case 5:
                    str_RobotSpeakSentence = "五";
                    if (ms_RobotModel == "Zenbo" || ms_RobotModel == "ZenboJrII")
                        sFace = "DOUBTING";
                    else if (ms_RobotModel == "Kebbi")
                        sFace = "TTS_SadnessB";
                    else
                        cout << "Unknown robot model." << endl;
                    break;
                }
                RobotCommandProtobuf::RobotCommand robot_command;
                robot_command.set_speak_sentence(str_RobotSpeakSentence);
                robot_command.set_sface(sFace);
                pSendMessageManager->AddMessage(robot_command);
            }

            if (RTSmessage.has_tabletcommand())
            {
                std::string str_tabletcommand = RTSmessage.tabletcommand();
                int RobotExpressionIndex = 0;
                cout << "Receive tabletcommand: " << str_tabletcommand << endl;
                if (str_tabletcommand == "Restart")
                {
                    // Restart robot's control state.
                    mpThreadStateControl->Restart();
                }
                else if (str_tabletcommand == "Mandarin")
                {
                    // TODO: change language
                }
                else if (str_tabletcommand == "English")
                {
                    // TODO: change language
                }
                else
                {
                    cout << "Unknown tablet command." << endl;
                }
            }
        }
    }
    cout << "Exit ReceiveMessages loop." << std::endl;
}
