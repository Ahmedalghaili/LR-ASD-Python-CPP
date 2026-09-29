// Unit tests for ActiveSpeakerFollower. Build and run:
//   g++ -std=c++17 -I.. ../ActiveSpeakerFollower.cpp test_active_speaker_follower.cpp -o t && ./t
#include "ActiveSpeakerFollower.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace
{
int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

constexpr int W = 640, H = 480;
constexpr double STEP = 200;  // LR-ASD update period in the server

// Face box whose centre is at the given bearing (degrees, + = robot's left).
SpeakerObservation face(int id, float logit, double bearingDeg, double size = 120)
{
    const double tanH = std::tan(62.5 * M_PI / 360);
    const double cx = (0.5 - std::tan(bearingDeg * M_PI / 180) / (2 * tanH)) * W;
    const double cy = H * 0.45;
    return {id, logit, float(cx - size / 2), float(cy - size / 2), float(cx + size / 2),
            float(cy + size / 2)};
}

struct Run
{
    ActiveSpeakerFollower f;
    double now = 0;
    int pitch = 10;
    std::vector<ActiveSpeakerDecision> log;
    ActiveSpeakerDecision step(const std::vector<SpeakerObservation> &obs)
    {
        now += STEP;
        auto d = f.update(obs, W, H, now, pitch);
        if (d.headPitch) pitch = *d.headPitch;
        log.push_back(d);
        return d;
    }
    int bodyCommands() const
    {
        int n = 0;
        for (auto &d : log) n += d.bodyDegree.has_value();
        return n;
    }
    int count(const std::string &event) const
    {
        int n = 0;
        for (auto &d : log) n += d.event == event;
        return n;
    }
};

void test(const char *name, const std::function<void()> &body)
{
    const int before = failures;
    body();
    std::printf("%s %s\n", failures == before ? "PASS" : "FAIL", name);
}
}

int main()
{
    test("acquires a sustained speaker and turns toward them", [] {
        Run r;
        ActiveSpeakerDecision first;
        for (int i = 0; i < 6; ++i) {
            auto d = r.step({face(3, 2.0f, 25)});
            if (d.event == "acquire") first = d;
        }
        CHECK(first.event == "acquire");
        CHECK(first.targetTrack == 3);
        CHECK(first.bodyDegree && std::abs(*first.bodyDegree - 25) <= 1);
        CHECK(first.face && *first.face == ZenboFace::AWARE_LEFT);
    });

    test("ignores speech shorter than the acquire time", [] {
        Run r;
        r.step({face(3, 2.0f, 25)});
        r.step({face(3, 2.0f, 25)});  // 200 ms of speech
        for (int i = 0; i < 5; ++i) r.step({face(3, -3.0f, 25)});
        CHECK(r.count("acquire") == 0);
        // Any turn is only the idle look at a silent face, never an acquire.
        for (auto &d : r.log) CHECK(!d.bodyDegree || d.event == "look");
    });

    test("centres the closest silent face when nobody speaks", [] {
        Run r;
        std::optional<int> turn;
        for (int i = 0; i < 10 && !turn; ++i)
        {
            auto d = r.step({face(1, -3.0f, -20, 60), face(2, -3.0f, 25, 150)});
            if (d.bodyDegree) { turn = d.bodyDegree; CHECK(d.event == "look"); }
        }
        CHECK(turn && std::abs(*turn - 25) <= 1);  // the larger (closer) face
        CHECK(r.count("acquire") == 0);
    });

    test("waits before looking at a newly seen silent face", [] {
        Run r;
        for (int i = 0; i < 4; ++i) r.step({face(1, -3.0f, 30)});  // 800 ms
        CHECK(r.bodyCommands() == 0);
    });

    test("idle look can be disabled", [] {
        ActiveSpeakerConfig c;
        c.lookAtIdleFace = false;
        Run r{ActiveSpeakerFollower(c)};
        for (int i = 0; i < 20; ++i) r.step({face(1, -3.0f, 30)});
        CHECK(r.bodyCommands() == 0);
    });

    test("does not rotate for a speaker inside the dead band", [] {
        Run r;
        for (int i = 0; i < 20; ++i) r.step({face(1, 2.0f, (i % 2) ? 4 : -4)});
        CHECK(r.f.target() == 1);
        CHECK(r.bodyCommands() == 0);
        bool interested = false;
        for (auto &d : r.log) interested |= d.face && *d.face == ZenboFace::INTERESTED;
        CHECK(interested);
    });

    test("gazes right for a speaker on the robot's right", [] {
        Run r;
        ActiveSpeakerDecision d;
        for (int i = 0; i < 4; ++i) d = r.step({face(2, 2.0f, -30)});
        CHECK(r.f.target() == 2);
        bool right = false;
        for (auto &x : r.log) right |= x.face && *x.face == ZenboFace::AWARE_RIGHT;
        CHECK(right);
        bool negative = false;
        for (auto &x : r.log) negative |= x.bodyDegree && *x.bodyDegree < 0;
        CHECK(negative);
    });

    test("waits for the body to settle before the next rotation", [] {
        Run r;
        int commandsDuringSettle = 0;
        double settleUntil = -1;
        for (int i = 0; i < 12; ++i) {
            // Camera never moves, so the target stays off-centre.
            auto d = r.step({face(1, 2.0f, 40)});
            if (d.bodyDegree) {
                if (r.now < settleUntil) ++commandsDuringSettle;
                settleUntil = r.now + 40.0 / 45 * 1000 + 600;
            }
        }
        CHECK(commandsDuringSettle == 0);
        CHECK(r.count("settling") > 0);
    });

    test("does not switch for a short interruption", [] {
        Run r;
        for (int i = 0; i < 12; ++i) r.step({face(1, 2.0f, 0), face(2, -3.0f, 30)});
        CHECK(r.f.target() == 1);
        // Person 2 says one short word (400 ms) while person 1 pauses.
        r.step({face(1, -1.0f, 0), face(2, 3.0f, 30)});
        r.step({face(1, -1.0f, 0), face(2, 3.0f, 30)});
        for (int i = 0; i < 5; ++i) r.step({face(1, 2.0f, 0), face(2, -3.0f, 30)});
        CHECK(r.f.target() == 1);
        CHECK(r.count("switch") == 0);
    });

    test("switches to a new sustained speaker", [] {
        Run r;
        for (int i = 0; i < 12; ++i) r.step({face(1, 2.0f, 0), face(2, -3.0f, 30)});
        ActiveSpeakerDecision sw;
        for (int i = 0; i < 10; ++i) {
            auto d = r.step({face(1, -3.0f, 0), face(2, 2.5f, 30)});
            if (d.event == "switch") sw = d;
        }
        CHECK(sw.event == "switch");
        CHECK(sw.targetTrack == 2);
        // Person 1 at 0 deg must stay in view: at most 62.5/2 - 6 = 25.25 deg.
        CHECK(sw.bodyDegree && *sw.bodyDegree > 0 && *sw.bodyDegree <= 25);
        CHECK(sw.face && *sw.face == ZenboFace::AWARE_LEFT);
        CHECK(r.f.target() == 2);
    });

    test("turns fully to the new speaker when group framing is off", [] {
        ActiveSpeakerConfig c;
        c.keepOthersInView = false;
        Run r;
        r.f = ActiveSpeakerFollower(c);
        for (int i = 0; i < 12; ++i) r.step({face(1, 2.0f, 0), face(2, -3.0f, 30)});
        ActiveSpeakerDecision sw;
        for (int i = 0; i < 10; ++i) {
            auto d = r.step({face(1, -3.0f, 0), face(2, 2.5f, 30)});
            if (d.event == "switch") sw = d;
        }
        CHECK(sw.bodyDegree && std::abs(*sw.bodyDegree - 30) <= 1);
    });

    test("keeps the current speaker during overlapping speech", [] {
        Run r;
        for (int i = 0; i < 12; ++i) r.step({face(1, 2.0f, 0), face(2, -3.0f, 30)});
        for (int i = 0; i < 10; ++i) r.step({face(1, 2.0f, 0), face(2, 2.3f, 30)});
        CHECK(r.f.target() == 1);
    });

    test("respects the minimum dwell time after a switch", [] {
        Run r;
        for (int i = 0; i < 12; ++i) r.step({face(1, 2.0f, 0), face(2, -3.0f, 5)});
        // Rapid alternation every 1 s.
        int switches = 0;
        for (int k = 0; k < 10; ++k)
            for (int i = 0; i < 5; ++i) {
                const bool one = k % 2;
                auto d = r.step({face(1, one ? 2.0f : -3.0f, 0), face(2, one ? -3.0f : 2.0f, 5)});
                switches += d.event == "switch";
            }
        CHECK(switches <= 5);  // at most one switch per 1.5 s dwell + 0.8 s speech
    });

    test("re-identifies the target after its track id changes", [] {
        Run r;
        for (int i = 0; i < 4; ++i) r.step({face(7, 2.0f, 30)});
        CHECK(r.f.target() == 7);
        // Wait until settled; the rotation brought the person to the centre
        // and the tracker restarted them as track 8.
        for (int i = 0; i < 10; ++i) r.step({face(8, 2.0f, 1)});
        CHECK(r.f.target() == 8);
        CHECK(r.count("reacquire") == 1);
        CHECK(r.bodyCommands() == 1);
    });

    test("does not adopt a different person far from the target", [] {
        Run r;
        for (int i = 0; i < 12; ++i) r.step({face(7, 2.0f, 0)});
        for (int i = 0; i < 3; ++i) r.step({face(9, -3.0f, -35)});
        CHECK(r.count("reacquire") == 0);
        CHECK(r.f.target() == 7);
    });

    test("releases a lost target and returns to the default face", [] {
        Run r;
        for (int i = 0; i < 6; ++i) r.step({face(4, 2.0f, 0)});
        CHECK(r.f.target() == 4);
        ActiveSpeakerDecision rel;
        for (int i = 0; i < 20; ++i) {
            auto d = r.step({});
            if (d.event == "release") rel = d;
        }
        CHECK(rel.event == "release");
        CHECK(rel.face && *rel.face == ZenboFace::DEFAULT);
        CHECK(r.f.target() == -1);
    });

    test("tilts the head toward a face high in the image", [] {
        Run r;
        SpeakerObservation high = face(5, 2.0f, 0);
        high.y1 -= 150; high.y2 -= 150;
        for (int i = 0; i < 6; ++i) r.step({high});
        bool up = false;
        for (auto &d : r.log) up |= d.headPitch && *d.headPitch > 10;
        CHECK(up);
        CHECK(r.pitch <= 55);
    });

    test("limits each head pitch change to a small step", [] {
        Run r;
        SpeakerObservation top = face(5, 2.0f, 0);
        top.y1 = 0; top.y2 = 60;  // face at the very top of the image
        int biggest = 0, previous = r.pitch;
        for (int i = 0; i < 20; ++i) {
            auto d = r.step({top});
            if (d.headPitch) {
                biggest = std::max(biggest, std::abs(*d.headPitch - previous));
                previous = *d.headPitch;
            }
        }
        CHECK(biggest > 0 && biggest <= 8);
    });

    test("gaze does not flicker for a speaker near the side threshold", [] {
        Run r;
        int faces = 0;
        for (int i = 0; i < 30; ++i) faces += r.step({face(1, 2.0f, (i % 2) ? 7 : 9)}).face.has_value();
        CHECK(faces == 1);
    });

    test("sends each face expression only when it changes", [] {
        Run r;
        int faces = 0;
        for (int i = 0; i < 30; ++i) faces += r.step({face(1, 2.0f, 0)}).face.has_value();
        CHECK(faces == 1);
    });

    test("pauses rotation when turns do not move the target", [] {
        Run r;
        int rotations = 0;
        // The camera never moves (blocked robot): the target stays at 30 deg.
        double pausedAt = -1;
        int duringPause = 0;
        for (int i = 0; i < 60; ++i)
        {
            auto d = r.step({face(1, 2.0f, 30)});
            rotations += d.bodyDegree.has_value();
            if (pausedAt < 0 && d.reason.find("paused") != std::string::npos)
                pausedAt = r.now;
            else if (pausedAt >= 0 && r.now < pausedAt + r.f.config().ineffectivePauseMs)
                duringPause += d.bodyDegree.has_value();
        }
        CHECK(pausedAt > 0);      // two ineffective turns, then paused
        CHECK(duringPause == 0);  // no turn while paused
        CHECK(rotations >= 2);
    });

    test("turns back to where a lost speaker was last seen", [] {
        ActiveSpeakerConfig c;
        c.minBodyGapMs = 1e12;  // no tracking turns: the speaker stays at +30 deg
        Run r{ActiveSpeakerFollower(c)};
        for (int i = 0; i < 10; ++i) r.step({face(1, 2.0f, 30)});
        CHECK(r.bodyCommands() == 0);
        std::optional<int> back;
        for (int i = 0; i < 40; ++i)
            if (auto d = r.step({}); d.event == "release") { back = d.bodyDegree; break; }
        CHECK(back && std::abs(*back - 30) <= 1);
        // Only once: nobody is visible afterwards, and the robot stays put.
        for (int i = 0; i < 40; ++i) r.step({});
        CHECK(r.bodyCommands() == 1);
    });

    test("does not turn back when the speaker was lost straight ahead", [] {
        Run r;
        for (int i = 0; i < 10; ++i) r.step({face(1, 2.0f, 3)});
        for (int i = 0; i < 40; ++i) r.step({});
        CHECK(r.count("release") == 1);
        CHECK(r.bodyCommands() == 0);
    });

    test("keeps rotating while turns take effect", [] {
        Run r;
        double bearing = 30;
        int rotations = 0;
        bool paused = false;
        for (int i = 0; i < 60; ++i) {
            if (i % 12 == 0) bearing = (i / 12) % 2 ? -30 : 30;  // the person moves
            auto d = r.step({face(1, 2.0f, bearing)});
            paused |= d.reason.find("paused") != std::string::npos;
            if (d.bodyDegree) {
                ++rotations;
                bearing -= *d.bodyDegree;  // the turn works
            }
        }
        CHECK(rotations >= 4);
        CHECK(!paused);
    });

    // ---- Sound direction (DOA) ----
    using DoaMode = ActiveSpeakerConfig::DoaMode;
    auto withDoa = [](DoaMode mode) {
        ActiveSpeakerConfig c;
        c.doaMode = mode;
        Run r;
        r.f = ActiveSpeakerFollower(c);
        for (int i = 0; i < 8; ++i) r.step({});  // past the 1.5 s warm-up
        r.log.clear();
        return r;
    };

    test("turns toward an unseen voice with Zenbo's lookAtUser", [&] {
        Run r = withDoa(DoaMode::LookAtUser);
        for (int i = 0; i < 6; ++i) r.step({face(1, -3.0f, 0)});  // visible, silent
        r.f.addVoiceDirection(120, r.now + STEP);
        auto d = r.step({face(1, -3.0f, 0)});
        CHECK(d.event == "turn_to_voice");
        CHECK(d.lookAtUserDoa && std::abs(*d.lookAtUserDoa - 120) < 1e-3);
        CHECK(!d.bodyDegree);
    });

    test("ignores a voice that a visible speaking face explains", [&] {
        Run r = withDoa(DoaMode::LookAtUser);
        for (int i = 0; i < 6; ++i) r.step({face(1, 2.0f, 5)});  // visible and speaking
        r.f.addVoiceDirection(10, r.now + STEP);
        auto d = r.step({face(1, 2.0f, 5)});
        CHECK(d.event != "turn_to_voice");
        CHECK(!d.lookAtUserDoa);
        CHECK(d.voiceDoa && d.voiceSpeakerBearing);  // calibration pair
        CHECK(r.f.target() == 1);
    });

    test("log mode records DOA but never moves", [&] {
        Run r = withDoa(DoaMode::Log);
        for (int i = 0; i < 6; ++i) r.step({});
        r.f.addVoiceDirection(90, r.now + STEP);
        auto d = r.step({});
        CHECK(d.voiceDoa && *d.voiceDoa == 90);
        CHECK(!d.hasCommand());
    });

    test("off mode ignores DOA", [&] {
        Run r = withDoa(DoaMode::Off);
        r.f.addVoiceDirection(90, r.now + STEP);
        auto d = r.step({});
        CHECK(!d.voiceDoa && !d.hasCommand());
    });

    test("rotate mode applies the DOA calibration", [&] {
        ActiveSpeakerConfig c;
        c.doaMode = DoaMode::Rotate;
        c.doaSign = -1;
        c.doaOffsetDeg = 90;
        Run r;
        r.f = ActiveSpeakerFollower(c);
        for (int i = 0; i < 8; ++i) r.step({});  // past the warm-up
        r.f.addVoiceDirection(270, r.now + STEP);  // -270 + 90 = -180
        auto d = r.step({});
        CHECK(d.event == "turn_to_voice");
        CHECK(d.bodyDegree && std::abs(*d.bodyDegree) == 180);
        r.f.addVoiceDirection(150, r.now + 20000);   // after the settle time
        r.now += 20000 - STEP;
        d = r.step({});
        CHECK(d.bodyDegree && *d.bodyDegree == -60);  // -150 + 90
    });

    test("limits how often the robot turns toward voices", [&] {
        Run r = withDoa(DoaMode::LookAtUser);
        int turns = 0;
        for (int i = 0; i < 25; ++i) {  // a DOA report every 200 ms for 5 s
            r.f.addVoiceDirection(i % 2 ? 100 : -100, r.now + STEP);
            turns += r.step({}).event == "turn_to_voice";
        }
        CHECK(turns == 2);  // at 0 s and after the 4 s gap
    });

    test("no voice turn before the camera has had time to see who is there", [&] {
        ActiveSpeakerConfig c;
        c.doaMode = DoaMode::LookAtUser;
        Run r;
        r.f = ActiveSpeakerFollower(c);
        r.f.addVoiceDirection(120, r.now + STEP);
        CHECK(r.step({}).event != "turn_to_voice");
    });

    test("ignores stale DOA reports", [&] {
        Run r = withDoa(DoaMode::LookAtUser);
        r.f.addVoiceDirection(100, r.now - 5000);
        auto d = r.step({});
        CHECK(!d.voiceDoa && !d.hasCommand());
    });

    test("a visible face near the voice direction explains the voice", [&] {
        Run r = withDoa(DoaMode::LookAtUser);
        for (int i = 0; i < 6; ++i) r.step({face(1, -3.0f, -21)});
        r.f.addVoiceDirection(-30, r.now + STEP);  // noisy DOA just outside the view limit
        auto d = r.step({face(1, -3.0f, -21)});
        CHECK(d.event != "turn_to_voice" && !d.lookAtUserDoa);
    });

    test("does not turn toward a voice from inside the camera view", [&] {
        Run r = withDoa(DoaMode::LookAtUser);
        for (int i = 0; i < 6; ++i) r.step({face(1, -3.0f, 5)});  // visible, not yet confirmed
        r.f.addVoiceDirection(8, r.now + STEP);
        auto d = r.step({face(1, -3.0f, 5)});
        CHECK(d.event != "turn_to_voice" && !d.lookAtUserDoa);
    });

    test("after a voice turn, attends at once to a centred face LR-ASD marks speaking", [&] {
        Run r = withDoa(DoaMode::LookAtUser);
        r.f.addVoiceDirection(90, r.now + STEP);
        CHECK(r.step({}).event == "turn_to_voice");
        for (int i = 0; i < 12; ++i) r.step({});  // turning (90 deg at 45 deg/s + settle)
        auto d = r.step({face(6, 1.0f, 4)});     // first speaking update of this face
        CHECK(d.event == "acquire" && d.targetTrack == 6);
    });

    test("after a voice turn, attends to the speaker LR-ASD then finds", [&] {
        Run r = withDoa(DoaMode::LookAtUser);
        for (int i = 0; i < 12; ++i) r.step({face(1, 2.0f, 0)});
        CHECK(r.f.target() == 1);
        for (int i = 0; i < 6; ++i) r.step({face(1, -3.0f, 0)});  // person 1 stops
        r.f.addVoiceDirection(150, r.now + STEP);
        CHECK(r.step({face(1, -3.0f, 0)}).event == "turn_to_voice");
        CHECK(r.f.target() == -1);
        // After the turn, person 2 is in view and LR-ASD sees them speak.
        for (int i = 0; i < 20; ++i) r.step({face(2, 2.0f, 3)});
        CHECK(r.f.target() == 2);
    });

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures,
                failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
