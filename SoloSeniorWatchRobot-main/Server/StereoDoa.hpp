#pragma once

#include <cstddef>
#include <cstdint>

// Direction of a voice from Zenbo's two microphones (original Zenbo; the
// Zenbo V2 app sends 200 ms stereo pieces as "stereo_audio" messages).
//
// GCC-PHAT gives the time difference of arrival (TDOA) between the two
// microphones; asin(TDOA / max TDOA) gives the angle. Measured on the robot
// (tests/zenbo_mic_probe.py): front ~0 us, left -182 us, right +224 us, so
// the microphones are ~8 cm apart and a negative TDOA means the voice is on
// the robot's left. With two microphones front and back give the same
// value: the angle is always in -90..90 deg.
struct StereoDoaResult
{
    bool valid = false;        // a voice with a consistent direction was found
    double tdoaUs = 0;         // + = the sound reached microphone 1 first
    double angleDeg = 0;       // + = robot's left, -90..90
    double consistency = 0;    // share of loud frames that agree (0..1)
    double levelDbfs = -120;   // loudness of the piece
};

struct StereoDoaConfig
{
    double maxTdoaUs = 230;      // delay of a voice at 90 deg (measured)
    double minLevelDbfs = -55;   // quieter pieces are not a voice (Zenbo raw mics: speech ~-42..-49)
    double minConsistency = 0.6; // below this the direction is unreliable
    int frame = 512;             // samples per GCC-PHAT frame
    int upsample = 8;            // sub-sample resolution of the delay
};

class StereoDoa
{
public:
    explicit StereoDoa(StereoDoaConfig config = {}) : config_(config) {}

    // interleaved: frames x 2 PCM16 samples (microphone 0, microphone 1).
    StereoDoaResult process(const int16_t *interleaved, size_t frames, int sampleRate) const;

    const StereoDoaConfig &config() const { return config_; }

private:
    StereoDoaConfig config_;
};
