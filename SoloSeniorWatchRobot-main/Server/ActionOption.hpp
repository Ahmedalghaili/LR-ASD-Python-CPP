#ifndef __ACTION_OPTION_HPP__
#define __ACTION_OPTION_HPP__

struct ActionOption
{
    enum MOVE_MODE
    {
        MOVE_MANUAL = 0,
        MOVE_BODY = 1,
        MOVE_HEAD = 2,
        // Active Speaker Following: attend to the person LR-ASD marks as
        // speaking. SIMULATION decides and logs without moving the robot.
        FOLLOW_SPEAKER_SIMULATION = 3,
        FOLLOW_SPEAKER = 4,
    };

    // Manual is the safe default: automatic tracking must be selected
    // explicitly before it is allowed to move the robot.
    MOVE_MODE move_mode = MOVE_MANUAL;
    bool bUseVisualCompass = false;

    // Pose-based tracking (and its return-to-zero behaviour) only runs in
    // the body/head tracking modes, so it never competes with manual control
    // or with Active Speaker Following.
    bool poseTrackingEnabled() const
    {
        return move_mode == MOVE_BODY || move_mode == MOVE_HEAD;
    }
    bool followSpeakerEnabled() const
    {
        return move_mode == FOLLOW_SPEAKER_SIMULATION || move_mode == FOLLOW_SPEAKER;
    }
};

#endif
