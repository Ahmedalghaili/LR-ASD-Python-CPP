#ifndef __ACTION_OPTION_HPP__
#define __ACTION_OPTION_HPP__

struct ActionOption
{
    enum MOVE_MODE
    {
        MOVE_MANUAL = 0,
        MOVE_BODY = 1,
        MOVE_HEAD = 2,
    };

    // Manual is the safe default: automatic tracking must be selected
    // explicitly before it is allowed to move the robot.
    MOVE_MODE move_mode = MOVE_MANUAL;
    bool bUseVisualCompass = false;
};

#endif
