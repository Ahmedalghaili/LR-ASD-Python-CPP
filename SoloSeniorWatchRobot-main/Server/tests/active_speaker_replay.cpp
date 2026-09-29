// Offline simulation of Active Speaker Following on a recorded LR-ASD log.
//
// Input: the per-frame CSV written by the LR-ASD C++ runner
//   frame,time_s,track,score,speaking,x1,y1,x2,y2
// The recorded camera never moved, so the tool simulates the robot body: each
// rotation command turns a virtual robot at the configured speed, and every
// face is re-projected into the virtual camera (faces outside the field of
// view disappear). Decisions are made every 200 ms, as in the server.
//
// Build: g++ -std=c++17 -O2 -I.. ../ActiveSpeakerFollower.cpp active_speaker_replay.cpp -o replay
// Run:   ./replay log.csv 1920 1080 [--quiet] [--naive] [--no-framing] [--switch-ms N]
//   --naive       baseline without stability rules: no smoothing, dead band,
//                 dwell, or settle time; turns to the loudest speaker at once
//                 (group framing stays on unless --no-framing is also given)
//   --no-framing  disable keeping other people in view
#include "ActiveSpeakerFollower.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct Row { int frame; double t; int track; float score; float x1, y1, x2, y2; };

int main(int argc, char **argv)
{
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s log.csv width height [--quiet] [--naive] [--no-framing] [--switch-ms N]\n",
                     argv[0]);
        return 2;
    }
    const int W = std::atoi(argv[2]), H = std::atoi(argv[3]);
    bool quiet = false;
    ActiveSpeakerConfig config;
    for (int i = 4; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--quiet") quiet = true;
        else if (a == "--no-framing") config.keepOthersInView = false;
        else if (a == "--switch-ms" && i + 1 < argc) config.switchMs = std::atof(argv[++i]);
        else if (a == "--naive") {
            config.scoreAlpha = 1;
            config.acquireMs = config.switchMs = config.minDwellMs = 0;
            config.switchMargin = 0;
            config.bodyDeadbandDeg = config.minBodyStepDeg = 1;
            config.minBodyGapMs = config.settleMs = 0;
        }
    }

    std::map<int, std::vector<Row>> frames;
    std::ifstream in(argv[1]);
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string c[9];
        for (auto &v : c) std::getline(ss, v, ',');
        Row r{std::stoi(c[0]), std::stod(c[1]), std::stoi(c[2]), std::stof(c[3]),
              std::stof(c[5]), std::stof(c[6]), std::stof(c[7]), std::stof(c[8])};
        frames[r.frame].push_back(r);
    }
    if (frames.empty()) { std::fprintf(stderr, "no rows\n"); return 1; }

    ActiveSpeakerFollower follower(config);
    const auto &cfg = follower.config();
    const double tanH = std::tan(cfg.hfovDeg * M_PI / 360);
    auto bearingOf = [&](double cx) { return -std::atan((cx / W - 0.5) * 2 * tanH) * 180 / M_PI; };
    auto pixelOf = [&](double b) { return (0.5 - std::tan(b * M_PI / 180) / (2 * tanH)) * W; };

    double robotYaw = 0, commandedYaw = 0;  // + = turned left
    int pitch = 10, bodyCommands = 0, faceCommands = 0, pitchCommands = 0;
    double totalRotation = 0;
    std::map<std::string, int> events;
    int judged = 0, correct = 0, correctGrace = 0;
    int previousSpeaker = -1;
    double speakerChangedMs = -1e9;
    constexpr double kReactionMs = 1500;
    const int lastFrame = frames.rbegin()->first;

    for (int f = 0; f <= lastFrame; f += 5) {
        auto it = frames.find(f);
        if (it == frames.end()) continue;
        const double now = it->second.front().t * 1000;

        // Move the virtual body toward the commanded yaw.
        const double maxStep = cfg.bodyDegPerSec * 0.2;
        robotYaw += std::clamp(commandedYaw - robotYaw, -maxStep, maxStep);

        std::vector<SpeakerObservation> obs;
        int speakingTrack = -1, speakingCount = 0;
        for (const Row &r : it->second) {
            // Ground truth counts everyone in the recording, including people
            // the virtual robot has turned away from.
            if (r.score >= 0) { speakingTrack = r.track; ++speakingCount; }
            const double worldBearing = bearingOf((r.x1 + r.x2) / 2);
            const double visible = worldBearing - robotYaw;
            if (std::abs(visible) > cfg.hfovDeg / 2) continue;
            const double shift = pixelOf(visible) - (r.x1 + r.x2) / 2;
            obs.push_back({r.track, r.score, float(r.x1 + shift), r.y1, float(r.x2 + shift), r.y2});
        }

        const auto d = follower.update(obs, W, H, now, pitch);
        ++events[d.event];
        if (d.bodyDegree) {
            commandedYaw += *d.bodyDegree;
            totalRotation += std::abs(*d.bodyDegree);
            ++bodyCommands;
        }
        if (d.headPitch) { pitch = *d.headPitch; ++pitchCommands; }
        if (d.face) ++faceCommands;

        // Attention score: when exactly one person is speaking in this
        // update, is the robot attending to them?
        if (speakingCount == 1) {
            if (speakingTrack != previousSpeaker) {
                previousSpeaker = speakingTrack;
                speakerChangedMs = now;
            }
            ++judged;
            correct += follower.target() == speakingTrack;
            correctGrace += follower.target() == speakingTrack || now - speakerChangedMs < kReactionMs;
        }

        if (!quiet && (d.hasCommand() || d.event == "acquire" || d.event == "switch" ||
                       d.event == "reacquire" || d.event == "release")) {
            std::printf("t=%6.2fs %-9s target=%2d bearing=%+6.1f", now / 1000, d.event.c_str(),
                        d.targetTrack, d.bearingDeg);
            if (d.bodyDegree) std::printf("  ROTATE %+d", *d.bodyDegree);
            if (d.headPitch) std::printf("  PITCH %d", *d.headPitch);
            if (d.face) std::printf("  FACE %s", ZenboFace::name(*d.face));
            if (!d.reason.empty()) std::printf("  (%s)", d.reason.c_str());
            std::printf("\n");
        }
    }

    const double seconds = frames.rbegin()->second.front().t;
    std::printf("\nduration %.1fs | body rotations %d (%.0f deg total, %.2f per 10 s) | "
                "pitch commands %d | face changes %d\n",
                seconds, bodyCommands, totalRotation, bodyCommands / seconds * 10, pitchCommands,
                faceCommands);
    std::printf("events:");
    for (auto &[k, v] : events) std::printf(" %s=%d", k.c_str(), v);
    std::printf("\nattention on the single active speaker: %d/%d updates (%.1f%%); "
                "allowing %.1f s to react: %.1f%%\n", correct, judged,
                judged ? 100.0 * correct / judged : 0.0, kReactionMs / 1000,
                judged ? 100.0 * correctGrace / judged : 0.0);
    return 0;
}
