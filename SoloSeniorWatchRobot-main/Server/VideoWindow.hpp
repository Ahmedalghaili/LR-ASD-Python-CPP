#ifndef __VIDEO_WINDOW_HPP__
#define __VIDEO_WINDOW_HPP__

#include <QMainWindow>
#include <QMediaPlayer>
#include <opencv2/core.hpp>

class QStackedWidget;
class QVideoWidget;
class QAudioOutput;
class QLabel;
class QKeyEvent;
class ThreadStateControl;

class VideoWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit VideoWindow(QWidget *parent = nullptr);
    void playVideo(const QString &fileName);
    void showImage(const QString &fileName);
    void showFrame(const cv::Mat &frame);
    void showString(const QString &ShowString);
    ThreadStateControl* pThreadStateControl = nullptr;

public slots:
    void onMediaStatusChanged(QMediaPlayer::MediaStatus status);

protected:
    void keyPressEvent(QKeyEvent *event) override;

private:
    QMediaPlayer *player;
    QVideoWidget *videoWidget;
    QLabel *imageLabel;
    QAudioOutput *audioOutput;
    QStackedWidget *stackedWidget;
};

#endif // __VIDEO_WINDOW_HPP__
