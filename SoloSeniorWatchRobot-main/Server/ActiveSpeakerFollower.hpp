#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

// Decides where the robot should direct its attention from LR-ASD output.
//
// The follower is deliberately free of Qt, OpenCV and protobuf so that the
// same decision logic runs in the server, in unit tests, and in the offline
// replay tool. The caller converts a decision into robot commands (or only
// logs it in simulation mode).
//
// Angle conventions follow Zenbo/LandmarkToRobotAction.cpp: a positive body
// degree turns the robot to its left, and a positive head pitch looks up.

struct SpeakerObservation
{
    int trackId;
    float logit;              // LR-ASD class-1 logit; >= 0 means speaking
    float x1, y1, x2, y2;     // face box in image pixels
};

struct ActiveSpeakerConfig
{
    // Zenbo camera field of view, as used by the existing tracking code.
    double hfovDeg = 62.5;
    double vfovDeg = 48.9;

    // Speaking decision on the smoothed LR-ASD logit.
    float speakingLogit = 0.0f;
    double scoreAlpha = 0.5;       // EMA weight of the newest logit

    // Attention timing (milliseconds).
    double acquireMs = 400;        // speech needed before attending to anyone
    double switchMs = 600;         // speech needed before leaving a current speaker
    double minDwellMs = 1500;      // minimum attention time after a switch
    double switchMargin = 1.0;     // logit lead needed to interrupt an active speaker
    double lostMs = 3000;          // release the target when unseen this long

    // Movement limits.
    double bodyDeadbandDeg = 10;   // do not rotate for smaller offsets
    double maxBodyStepDeg = 60;    // largest single rotation
    double minBodyStepDeg = 5;     // skip rotations smaller than this

    // Group framing: when other people are visible, turn only as far as keeps
    // them inside the camera view (with this margin from the image edge), so
    // the robot can still see a new speaker. The gaze expression covers the
    // remaining angle. A single person is always fully centred.
    bool keepOthersInView = true;
    double edgeMarginDeg = 6;
    double othersRecentMs = 1000;  // only people seen this recently constrain the turn
    double bodyDegPerSec = 45;     // assumed Zenbo rotation speed
    double settleMs = 600;         // extra wait after a rotation ends
    double minBodyGapMs = 1000;    // minimum time between rotation commands
    double pitchDeadbandDeg = 6;
    double maxPitchStepDeg = 8;    // largest single head-pitch change
    int pitchMin = -10;            // Zenbo Junior head pitch range (PyZenbo
    int pitchMax = 50;             // motion.move_head: -10..50 on Junior)
    double minPitchGapMs = 800;

    // Gaze expression: glance to the side when the speaker is off-centre.
    double gazeSideDeg = 8;
    double gazeHysteresisDeg = 3;  // avoid flicker around gazeSideDeg
    bool mirrorGaze = false;       // swap AWARE_LEFT/RIGHT if the face looks the wrong way

    // After a rotation the tracker may assign a new id to the same person;
    // adopt a new track within this angle of the expected position.
    double reacquireDeg = 15;

    // Sound direction (DOA) from Zenbo's voice detection. DOA only brings an
    // unseen speaker into view; LR-ASD still decides who is speaking.
    //   Off        ignore DOA
    //   Log        record DOA (and, when exactly one visible face is
    //              speaking, its bearing for calibration); never move
    //   LookAtUser turn with Zenbo's built-in utility.lookAtUser(doa)
    //   Rotate     turn the body by doaSign * doa + doaOffsetDeg degrees
    enum class DoaMode { Off, Log, LookAtUser, Rotate };
    DoaMode doaMode = DoaMode::Log;
    double doaSign = 1;
    double doaOffsetDeg = 0;
    // False until doaSign/doaOffsetDeg are measured: the follower cannot tell
    // whether a voice comes from inside the camera view, so it turns toward
    // any voice while no visible face is speaking (LookAtUser needs no
    // calibration; it uses Zenbo's own DOA).
    bool doaCalibrated = true;
    // A two-microphone DOA cannot tell front from back. When a voice comes
    // from inside the camera view but nobody at all has been seen for
    // rearNoFaceMs, it is behind the robot: turn by rearTurnDeg (0 = off).
    double rearTurnDeg = 0;
    double rearNoFaceMs = 2000;
    double doaMaxAgeMs = 1000;     // ignore older DOA reports
    double voiceQuietMs = 800;     // no visible face may have spoken for this long
    double voiceTurnGapMs = 4000;  // minimum time between voice turns
    double voiceTurnMinSettleMs = 1000; // minimum wait after a voice turn
    // A voice from inside the camera view (|calibrated DOA| below this) comes
    // from someone the camera can see: no voice turn, LR-ASD decides.
    double voiceInViewDeg = 25;
    // After a voice turn, a face within voiceFocusDeg of the centre is
    // attended as soon as LR-ASD marks it speaking (no acquire delay).
    double voiceFocusMs = 3000;
    double voiceFocusDeg = 20;
    // A visible face this close to the (calibrated) voice direction explains
    // the voice even before LR-ASD confirms it: no voice turn.
    double voiceFaceMatchDeg = 15;
    // No voice turns until the camera and LR-ASD have run this long, so that
    // faces already in view are known before a voice counts as unexplained.
    double voiceWarmupMs = 1500;

    // Safety: if two rotations in a row barely move the target in the image
    // (blocked by an obstacle, command rejected), stop rotating for a while.
    double ineffectiveFraction = 0.5;  // moved less than this share of the turn
    double ineffectivePauseMs = 4000;

    // When the target is lost, turn back toward where it was last seen
    // (estimated from the rotations sent), so the robot does not stay facing
    // an empty part of the room. Larger returns are clamped to this.
    bool returnOnLost = true;
    double maxReturnDeg = 90;

    // While nobody is speaking, turn toward the closest visible face (after
    // it has been seen this long), so people are kept centred in the camera
    // where LR-ASD works best and the robot looks attentive.
    bool lookAtIdleFace = true;
    double idleLookMs = 1000;
};

// Zenbo face indices understood by Converter.FaceIndexToRobotFace().
namespace ZenboFace
{
constexpr int AWARE_LEFT = 1;
constexpr int AWARE_RIGHT = 2;
constexpr int DEFAULT = 4;
constexpr int INTERESTED = 13;
const char *name(int face);
}

struct ActiveSpeakerDecision
{
    std::string event = "idle";    // idle, look, wait, acquire, switch, track,
                                   // settling, reacquire, release, hold, turn_to_voice
    int targetTrack = -1;
    double bearingDeg = 0;         // target offset: + = robot's left
    double elevationDeg = 0;       // target offset: + = above camera centre
    std::optional<int> bodyDegree; // relative rotation to send
    std::optional<int> headPitch;  // absolute head pitch to send
    std::optional<int> face;       // Zenbo face index to show
    std::optional<float> lookAtUserDoa; // call utility.lookAtUser(doa)
    std::optional<double> voiceDoa;     // DOA report used in this update
    std::optional<double> voiceSpeakerBearing; // bearing of the single visible
                                               // speaker at that time (calibration)
    std::string reason;

    bool hasCommand() const { return bodyDegree || headPitch || face || lookAtUserDoa; }
};

class ActiveSpeakerFollower
{
public:
    explicit ActiveSpeakerFollower(ActiveSpeakerConfig config = {});

    // Process one LR-ASD update. nowMs must be monotonic. currentPitch is the
    // robot's current absolute head pitch in degrees.
    ActiveSpeakerDecision update(const std::vector<SpeakerObservation> &observations,
                                 int imageWidth, int imageHeight,
                                 double nowMs, int currentPitch);

    // Report a sound direction from Zenbo's voice detection (raw SDK value).
    void addVoiceDirection(double doa, double nowMs);

    void reset();
    int target() const { return target_; }

    // Camera geometry helpers (horizontal): image x <-> bearing in degrees.
    double bearingOfX(double x, int imageWidth) const;
    double xOfBearing(double bearingDeg, int imageWidth) const;
    // Vertical: image y <-> elevation in degrees (+ = above centre).
    double elevationOfY(double y, int imageHeight) const;
    double yOfElevation(double elevationDeg, int imageHeight) const;
    const ActiveSpeakerConfig &config() const { return config_; }

private:
    struct TrackState
    {
        double smoothed = -10;
        double speakingSinceMs = -1;  // start of the current speaking run
        double lastSeenMs = 0;
        double firstSeenMs = 0;
        double bearingDeg = 0;
        double elevationDeg = 0;
        double size = 0;              // face box height in pixels
        bool seenNow = false;
    };

    bool speaking(const TrackState &t) const { return t.smoothed >= config_.speakingLogit; }
    double speakingFor(const TrackState &t, double nowMs) const
    {
        return t.speakingSinceMs < 0 ? 0 : nowMs - t.speakingSinceMs;
    }
    void attend(int track, double nowMs, ActiveSpeakerDecision &d);
    void aim(const TrackState &t, double nowMs, int currentPitch, ActiveSpeakerDecision &d);
    void setFace(int face, ActiveSpeakerDecision &d);
    void returnToLastTarget(double nowMs, ActiveSpeakerDecision &d);
    void lookAtIdleFace(double nowMs, int currentPitch, ActiveSpeakerDecision &d);
    void turnBodyAndHead(double bearingDeg, double elevationDeg, double nowMs,
                         int currentPitch, ActiveSpeakerDecision &d);

    ActiveSpeakerConfig config_;
    std::map<int, TrackState> tracks_;
    int target_ = -1;
    double targetSinceMs_ = 0;
    double targetExpectedBearing_ = 0;  // where the target should appear after our rotation
    double targetLostSinceMs_ = -1;
    double motionUntilMs_ = 0;
    double lastBodyMs_ = -1e9;
    double lastPitchMs_ = -1e9;
    int lastFace_ = -1;
    // Rotation effectiveness check.
    std::optional<double> rotationCheckBearing_;  // target bearing before the turn
    int rotationCheckDegree_ = 0;
    int ineffectiveRotations_ = 0;
    double rotationPausedUntilMs_ = 0;
    std::optional<double> firstUpdateMs_;
    std::optional<double> pendingDoa_;
    double pendingDoaMs_ = 0;
    double lastVisibleSpeechMs_ = -1e9;
    double lastVoiceTurnMs_ = -1e9;
    // Estimated body yaw (sum of rotations sent, + = left) and the world
    // direction where the target was last seen.
    double yawDeg_ = 0;
    std::optional<double> lastTargetWorldDeg_;
};
