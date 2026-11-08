#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QDebug>
#include <QFile>
#include <QFileDialog>
#include <QMessageBox>
#include <QMetaType>
#include <QSet>
#include <QStandardPaths>
#include <QTcpServer>
#include <QTcpSocket>
#include <QUdpSocket>
#include "ThreadProcessImage.hpp"
#include "ThreadPortAudio.hpp"
#include "ThreadReceiveMessage.hpp"
#include "ThreadWhisper.hpp"
#include "ThreadLLM.hpp"
#include "ThreadStateControl.hpp"
#include <queue>
#include <QMediaDevices>
#include <QAudioDevice>
#include <QAudioSource>
#include "utility_directory.hpp"
#include "SocketBufferParser.hpp"
#include "SendMessageManager.hpp"
#include "SocketClientHandler.hpp"
#include <memory>
#include <atomic>
#include <mutex>
#include "../VideoWindow.hpp"
#include "Setting.hpp"
#include "VideoAudioBuffer.hpp"
#include "LRASDWorker.hpp"
#include "ActiveSpeakerFollower.hpp"
#include "StereoDoa.hpp"
#include <deque>

using namespace std;

QT_BEGIN_NAMESPACE
namespace Ui
{
    class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void setLanguage(QString Language);
    void setSettingFile(const QString &filePath);
    void startThreads();

protected:
    void closeEvent(QCloseEvent *event) override;
    QAudioDevice devAudio;
    QAudioSource *audioSrc = nullptr;
    bool bListening = false;
    void UISetting(Ui::MainWindow *ui);

    Ui::MainWindow *ui;
    QTimer *timer;

    QTcpServer *m_server_receive_image;
    QSet<QTcpSocket *> connection_set;
    ThreadProcessImage thread_process_image;

    QTcpServer *m_server_send_command;
    QSet<QTcpSocket *> connection_set2; // for send back command

    QTcpServer *m_server_receive_audio;
    QSet<QTcpSocket *> connection_set3; // for receive audio
    ThreadProcessAudio thread_process_audio;

    QTcpServer *m_server_receive_message;
    QSet<QTcpSocket *> connection_set4; // for Tablet
    SocketBufferParser socketHandler4;
    ThreadReceiveMessage thread_receive_message;

    QSet<SocketClientHandler *> Handler_set;

    ThreadWhisper thread_whisper;

    ThreadLLM thread_LLM;
    ThreadStateControl thread_state_control;

    std::unique_ptr<VideoWindow> pVideoWindow;

    QString QString_SentCommands;
    void send_move_body_command(float x, float y, int degree, int speed);
    void send_move_head_command(int yaw, int pitch, int speed);

    SendMessageManager sendMessageManager;
    bool bstream_recognition = true; // whether to stream the voice recognition result

    WhisperData oldWhisperData; // for timer_event update UI

    Setting msetting;
    int current_body_angle = 0;
    void rotateAndTakePhoto(int targetAngle, const QString &prefix);

    Logger mlogger;

    VideoAudioBuffer mVABuffer;
    std::unique_ptr<LRASDWorker> mLRASDWorker;

    // Active Speaker Following: turns the robot toward the person LR-ASD
    // marks as speaking. Runs on a 100 ms timer; see ActiveSpeakerFollower.hpp.
    ActiveSpeakerFollower mSpeakerFollower;
    std::uint64_t mFollowSequence = 0;
    bool mFollowActive = false;
    ActiveSpeakerDecision mFollowDecision;
    std::string mFollowLastAction;
    // Simulation mode has no real motion, so it keeps a virtual body yaw and
    // head pitch and shifts the observed faces as a real turn would.
    double mSimYaw = 0, mSimYawTarget = 0, mSimLastMs = 0;
    int mSimPitch = 0, mSimPitchStart = 0;
    // Zenbo voice-detection DOA reports (raw value, arrival time in ms),
    // written by the message thread and read on the UI thread.
    std::mutex mVoiceMutex;
    std::vector<std::pair<int, double>> mVoiceDoas;
    int mBodySpeedLevel = 0;  // Zenbo moveBody speed L1..L7; 0 = app default
    // Zenbo listening loop (original Zenbo reports DOA only while listening).
    bool mListenActive = false;
    double mListenStartMs = 0;
    double mNextListenMs = 0;
    double mStatesOnMs = -1e9;
    std::atomic<bool> mListenEnded{false};
    // Voice direction from Zenbo's two microphones (ACTIVE_SPEAKER_DOA=stereo).
    bool mStereoDoaEnabled = false;
    // ACTIVE_SPEAKER_DOA=zenbo: Zenbo's own DOA (all 4 microphones), turned
    // with moveBody; Zenbo must be kept listening for it to report.
    bool mZenboDoaRotate = false;
    // ACTIVE_SPEAKER_DOA=respeaker: voice direction from the reSpeaker XVF3800
    // array on Zenbo, sent by tests/respeaker_doa_sender.py as UDP text
    // (degrees, + = left, 0 = Zenbo's front) to 127.0.0.1:kRespeakerDoaPort.
    QUdpSocket *mRespeakerSocket = nullptr;
    static constexpr quint16 kRespeakerDoaPort = 5006;
    // The same array plugged into Zenbo's USB: the Zenbo app sends its raw
    // angle ("respeaker_doa"); the front (array angle that is Zenbo's front,
    // ~/SoloSeniorWatchRobot_build/respeaker_seeed_front.json or
    // ACTIVE_SPEAKER_RESPEAKER_FRONT) and mirror are applied here. The angle
    // jumps from reading to reading, so a direction is used only when, of the
    // speech readings in the last kRespeakerWindowMs, at least kRespeakerVotes
    // and kRespeakerVoteShare of them agree within kRespeakerSteadyDeg; their
    // median is used. Readings within kRespeakerNoiseDeg of the array angle
    // ACTIVE_SPEAKER_RESPEAKER_NOISE (Zenbo's own noise, seen at a fixed
    // angle; unset = none) are dropped.
    void onRespeakerReading(int angle, bool speech);
    void addVoiceDoa(int doa);
    int mRespeakerFront = 90;
    bool mRespeakerMirror = false;
    int mRespeakerNoise = -1;
    std::deque<std::pair<int, double>> mRespeakerRecent;   // doa, time (ms)
    int mRespeakerLastDoa = 1000;
    double mRespeakerShownMs = -1e9;   // last voice_show sent to the app's panel
    double mRespeakerBlockedLogMs = -1e9;   // rate limit for "voice_blocked" log lines
    double mVoiceSkipLogMs = -1e9;          // rate limit for "voice_skipped" log lines
    static constexpr double kRespeakerWindowMs = 2000;
    static constexpr size_t kRespeakerVotes = 5;
    static constexpr double kRespeakerVoteShare = 0.6;
    static constexpr int kRespeakerSteadyDeg = 20;
    static constexpr int kRespeakerNoiseDeg = 10;
    // Zenbo's own motors sound like speech to the array (seen: a "voice" at
    // the same array angle after every turn, so Zenbo kept turning), so the
    // array is ignored while Zenbo moves and for kRespeakerAfterMoveMs after.
    bool mRobotMoving = false;
    double mRobotMoveEndMs = -1e9;
    // 1 s: with the voice gate (VAD + CAM++) the motor noise is not taken
    // for a voice anyway.
    static constexpr double kRespeakerAfterMoveMs = 1000;
    // ACTIVE_SPEAKER_VOICE_GATE=1: the audio of Zenbo's microphone is also
    // sent to tests/voice_gate.py (UDP kVoiceGateAudioPort); =respeaker: the
    // gate gets the reSpeaker's own audio over Wi-Fi from
    // tests/respeaker_wifi_bridge.py instead. The gate answers on
    // kVoiceGateResultPort "SPEECH <name> <score>" while a real voice talks
    // (Silero VAD + CAM++ voiceprint) and "QUIET" after. A reSpeaker direction
    // is then used only within kVoiceGateHoldMs of a SPEECH, and with
    // ACTIVE_SPEAKER_KNOWN_ONLY=1 only for an enrolled person.
    QUdpSocket *mVoiceGateAudio = nullptr;   // null: Zenbo's audio is not forwarded
    bool mVoiceGateOn = false;
    QUdpSocket *mVoiceGateResult = nullptr;
    static constexpr quint16 kVoiceGateAudioPort = 5008;
    static constexpr quint16 kVoiceGateResultPort = 5009;
    static constexpr double kVoiceGateHoldMs = 600;
    bool mVoiceGateKnownOnly = false;
    std::atomic<double> mVoiceGateSpeechMs{-1e9};
    // The gate applies only while it receives audio (its "ALIVE" within
    // kVoiceGateAudioMs); without it the reSpeaker direction is used alone.
    std::atomic<double> mVoiceGateAudioMs{-1e9};
    static constexpr double kVoiceGateAudioMs = 3000;
    std::mutex mVoiceGateMutex;
    std::string mVoiceGateSpeaker = "?";
    double mVoiceGateScore = 0;          // guarded by mVoiceGateMutex
    // Last direction used for a voice (+ = left) and when, for the video overlay.
    std::atomic<int> mRespeakerShownDoa{0};
    std::atomic<double> mRespeakerDoaMs{-1e9};
    void drawVoiceBanner(cv::Mat &preview);
    // record_demo.sh: while ~/SoloSeniorWatchRobot_build/record_demo.path
    // exists, the preview (with the overlays) is saved at kDemoFps to
    // <path in the file>.avi.
    void recordDemoFrame(const cv::Mat &preview);
    cv::VideoWriter mDemoWriter;
    std::string mDemoPath;
    double mDemoNextMs = 0;
    int mDemoTicks = 0;
    static constexpr double kDemoFps = 15;
    // Zenbo's own listening moves the head (it looked down at the floor), so
    // in zenbo mode the head is put back to eye level after a voice turn and
    // while no face is seen.
    int mEyeLevelPitch = 35;
    // While Zenbo listens it ignores moveBody/moveHead, so the app listens
    // only while nobody is in view (see updateActiveSpeakerFollowing).
    bool mAppListening = false;
    double mFaceVisibleSinceMs = -1;   // start of the current run of seen faces
    double mListenSwitchMs = -1e9;     // last time listening was switched
    double mLastFaceSeenMs = 0;
    double mLastHeadResetMs = -1e9;
    StereoDoa mStereoDoa;
    std::deque<double> mStereoLevels;  // recent piece levels: background noise
    double mStereoLastAngle = 1e9;
    void onStereoAudio(const int16_t *pcm, size_t frames, int rate);
    void updateActiveSpeakerFollowing();
    void keepZenboListening(double nowMs);
    void sendPocAction(const char *action);
    void drawActiveSpeakerOverlay(cv::Mat &preview);

signals:
    void newMessage(QString); // where is the connect for this signal?
    void addSendCommandMessage(RobotCommandProtobuf::RobotCommand);

private slots:
    void onPlayVideoRequested(const QString &videoPath);
    void onPlayImageRequested(const QString &imagePath);
    void onPlayTextRequested(const QString &ShowString);

    void newConnection_receive_image();
    void newConnection_send_command();
    void newConnection_receive_audio();
    void newConnection_receive_message();
    void appendToSocketList2(QTcpSocket *socket);
    void appendToSocketList3(QTcpSocket *socket);

    void readSocket3();

    void discardSocket2();
    void discardSocket3();

    void displayError(QAbstractSocket::SocketError socketError);

    void on_pushButton_speak_clicked();
    void on_pushButton_movebody_clicked();
    void on_pushButton_movehead_clicked();
    void on_pushButton_stop_action_clicked();
    void on_pushButton_voice_to_text_clicked();
    void on_pushButton_killapp_clicked();
    void on_pushButton_onTTSComplete_clicked();

    void on_listView_FacialExpressions_doubleClicked(const QModelIndex &index);
    void on_listView_PredefinedAction_doubleClicked(const QModelIndex &index);
    void on_listView_Content_doubleClicked(const QModelIndex &index); // Kebbi's mbtx file
    void on_listView_Song_doubleClicked(const QModelIndex &index);    // Zenbo's song file
    void on_listView_Sentence1_doubleClicked(const QModelIndex &index);
    void on_listView_Sentence1_clicked(const QModelIndex &index);
    void on_listView_Sentence2_doubleClicked(const QModelIndex &index);
    void on_listView_Sentence2_clicked(const QModelIndex &index);
    void on_listView_Sentence3_doubleClicked(const QModelIndex &index);
    void on_listView_Sentence3_clicked(const QModelIndex &index);

    void on_checkBox_SaveImages_clicked();
    void on_checkBox_UseVisualCompass_clicked();

    void timer_event();

    void comboBox_MoveMode_changed();
    void comboBox_DetectionMode_changed();
    void comboBox_Processor_changed();
    void comboBox_Language_changed();

    void keyPressEvent(QKeyEvent *event);
    void on_checkBox_stream_clicked(bool checked);
    void on_pushButton_generate_response_clicked();
    void on_pushButton_speak_2_clicked();
    void on_pushButton_hideface_clicked();
    void on_pushButton_stop_song_clicked();
    void on_pushButton_stop_speak_clicked();
    void on_pushButton_take_photo_clicked();
    void on_pushButton_test_clicked();
};
#endif // MAINWINDOW_H
