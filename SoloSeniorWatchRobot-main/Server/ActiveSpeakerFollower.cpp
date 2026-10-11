#include "ActiveSpeakerFollower.hpp"

#include <algorithm>
#include <cmath>

namespace
{
constexpr double kPi = 3.14159265358979323846;
double deg(double rad) { return rad * 180.0 / kPi; }
double rad(double deg) { return deg * kPi / 180.0; }
}

const char *ZenboFace::name(int face)
{
    switch (face)
    {
    case AWARE_LEFT: return "AWARE_LEFT";
    case AWARE_RIGHT: return "AWARE_RIGHT";
    case DEFAULT: return "DEFAULT";
    case INTERESTED: return "INTERESTED";
    default: return "OTHER";
    }
}

ActiveSpeakerFollower::ActiveSpeakerFollower(ActiveSpeakerConfig config)
    : config_(config)
{
}

double ActiveSpeakerFollower::bearingOfX(double x, int imageWidth) const
{
    const double tanH = std::tan(rad(config_.hfovDeg) / 2);
    return -deg(std::atan((x / imageWidth - 0.5) * 2 * tanH));
}

double ActiveSpeakerFollower::xOfBearing(double bearingDeg, int imageWidth) const
{
    const double tanH = std::tan(rad(config_.hfovDeg) / 2);
    return (0.5 - std::tan(rad(bearingDeg)) / (2 * tanH)) * imageWidth;
}

double ActiveSpeakerFollower::elevationOfY(double y, int imageHeight) const
{
    const double tanV = std::tan(rad(config_.vfovDeg) / 2);
    return -deg(std::atan((y / imageHeight - 0.5) * 2 * tanV));
}

double ActiveSpeakerFollower::yOfElevation(double elevationDeg, int imageHeight) const
{
    const double tanV = std::tan(rad(config_.vfovDeg) / 2);
    return (0.5 - std::tan(rad(elevationDeg)) / (2 * tanV)) * imageHeight;
}

void ActiveSpeakerFollower::reset()
{
    tracks_.clear();
    target_ = -1;
    targetSinceMs_ = 0;
    targetExpectedBearing_ = 0;
    targetLostSinceMs_ = -1;
    motionUntilMs_ = 0;
    lastBodyMs_ = -1e9;
    lastPitchMs_ = -1e9;
    lastFace_ = -1;
    pendingDoa_.reset();
    pendingDoaMs_ = 0;
    lastVisibleSpeechMs_ = -1e9;
    lastVoiceTurnMs_ = -1e9;
    rotationCheckBearing_.reset();
    rotationCheckDegree_ = 0;
    ineffectiveRotations_ = 0;
    rotationPausedUntilMs_ = 0;
    firstUpdateMs_.reset();
    yawDeg_ = 0;
    lastTargetWorldDeg_.reset();
    noiseLevels_.clear();
    noiseNext_ = 0;
    voiceSinceMs_ = -1;
    lastVoiceMs_ = -1e9;
    lastSearchMs_ = -1e9;
    searchCooldownUntilMs_ = 0;
    searchSteps_ = 0;
}

void ActiveSpeakerFollower::addVoiceLevel(double dbfs, double nowMs)
{
    // Motor noise while turning is not a voice, and not room noise either.
    if (nowMs < motionUntilMs_)
        return;
    // Room noise: the 20th percentile of the last 50 levels without a voice
    // (~10 s). Speech is kept out, or a long talk would raise the noise level
    // until the voice no longer counts; a "voice" lasting over 30 s is a
    // steady new noise (a fan), and the noise level adapts to it.
    constexpr size_t kHistory = 50;
    double noise = dbfs;
    if (!noiseLevels_.empty())
    {
        std::vector<double> sorted = noiseLevels_;
        std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 5, sorted.end());
        noise = sorted[sorted.size() / 5];
    }
    const bool voice = noiseLevels_.size() >= 10 && dbfs >= config_.voiceMinDbfs &&
                       dbfs >= noise + config_.voiceAboveNoiseDb;
    const bool steadySound = voice && voiceSinceMs_ >= 0 && nowMs - voiceSinceMs_ > 30000;
    if (!voice || steadySound)
    {
        if (noiseLevels_.size() < kHistory)
            noiseLevels_.push_back(dbfs);
        else
            noiseLevels_[noiseNext_++ % kHistory] = dbfs;
    }
    if (voice)
    {
        if (voiceSinceMs_ < 0 || nowMs - lastVoiceMs_ > config_.voiceGapMs)
            voiceSinceMs_ = nowMs;
        lastVoiceMs_ = nowMs;
    }
    else if (nowMs - lastVoiceMs_ > config_.voiceGapMs)
    {
        voiceSinceMs_ = -1;
    }
}

void ActiveSpeakerFollower::addVoiceDirection(double doa, double nowMs)
{
    pendingDoa_ = doa;
    pendingDoaMs_ = nowMs;
}

ActiveSpeakerDecision ActiveSpeakerFollower::update(
    const std::vector<SpeakerObservation> &observations, int imageWidth, int imageHeight,
    double nowMs, int currentPitch)
{
    ActiveSpeakerDecision d;
    if (imageWidth <= 0 || imageHeight <= 0)
        return d;

    if (!firstUpdateMs_)
        firstUpdateMs_ = nowMs;

    // 1. Update per-track speaking state and geometry.
    for (auto &[id, t] : tracks_)
        t.seenNow = false;
    const double tanH = std::tan(rad(config_.hfovDeg) / 2);
    const double tanV = std::tan(rad(config_.vfovDeg) / 2);
    for (const SpeakerObservation &o : observations)
    {
        auto [it, inserted] = tracks_.try_emplace(o.trackId);
        TrackState &t = it->second;
        if (inserted)
        {
            t.firstSeenMs = nowMs;
            t.smoothed = o.logit;
        }
        else
        {
            t.smoothed = config_.scoreAlpha * o.logit + (1 - config_.scoreAlpha) * t.smoothed;
        }
        if (speaking(t))
        {
            if (t.speakingSinceMs < 0)
                t.speakingSinceMs = nowMs;
        }
        else
        {
            t.speakingSinceMs = -1;
        }
        t.lastSeenMs = nowMs;
        t.seenNow = true;
        t.size = o.y2 - o.y1;
        // Aim at eye level (40% down the face box), not the box centre.
        const double cx = (o.x1 + o.x2) / 2.0 / imageWidth;
        const double cy = (o.y1 + 0.4 * (o.y2 - o.y1)) / imageHeight;
        t.bearingDeg = -deg(std::atan((cx - 0.5) * 2 * tanH));
        t.elevationDeg = -deg(std::atan((cy - 0.5) * 2 * tanV));
    }
    int visibleSpeakers = 0, visibleSpeaker = -1;
    for (const auto &[id, t] : tracks_)
        if (t.seenNow && speaking(t))
        {
            ++visibleSpeakers;
            visibleSpeaker = id;
        }
    if (visibleSpeakers > 0)
        lastVisibleSpeechMs_ = nowMs;
    for (auto it = tracks_.begin(); it != tracks_.end();)
    {
        if (it->first != target_ && nowMs - it->second.lastSeenMs > config_.lostMs)
            it = tracks_.erase(it);
        else
            ++it;
    }

    // 1b. Sound direction. A voice that no visible face explains comes from
    // someone the camera cannot see: turn toward it and let LR-ASD pick the
    // speaker once their face is in view.
    using DoaMode = ActiveSpeakerConfig::DoaMode;
    if (pendingDoa_ && config_.doaMode != DoaMode::Off)
    {
        const double doa = *pendingDoa_;
        const bool fresh = nowMs - pendingDoaMs_ <= config_.doaMaxAgeMs;
        pendingDoa_.reset();
        if (fresh)
        {
            d.voiceDoa = doa;
            if (visibleSpeakers == 1)
                d.voiceSpeakerBearing = tracks_[visibleSpeaker].bearingDeg;
            const bool unexplained = nowMs - lastVisibleSpeechMs_ >= config_.voiceQuietMs;
            const bool acts = config_.doaMode == DoaMode::LookAtUser ||
                              config_.doaMode == DoaMode::Rotate;
            // Calibrated direction (+ = robot's left). A voice from inside the
            // camera view is left to LR-ASD.
            int rotateDegree = static_cast<int>(std::lround(
                std::remainder(config_.doaSign * doa + config_.doaOffsetDeg, 360.0)));
            bool outOfView = !config_.doaCalibrated ||
                             std::abs(rotateDegree) >= config_.voiceInViewDeg;
            bool fromBehind = false;
            if (!outOfView && config_.rearTurnDeg != 0)
            {
                bool anyoneSeen = false;
                for (const auto &[id, t] : tracks_)
                    anyoneSeen |= nowMs - t.lastSeenMs <= config_.rearNoFaceMs;
                if (!anyoneSeen)
                {
                    fromBehind = true;
                    outOfView = true;
                    rotateDegree = static_cast<int>(std::lround(config_.rearTurnDeg));
                }
            }
            bool faceThere = false;
            for (const auto &[id, t] : tracks_)
                faceThere |= config_.doaCalibrated && t.seenNow &&
                             std::abs(t.bearingDeg - rotateDegree) <= config_.voiceFaceMatchDeg;
            const bool warmedUp = nowMs - *firstUpdateMs_ >= config_.voiceWarmupMs;
            if (acts && warmedUp && unexplained && outOfView && !faceThere && nowMs >= motionUntilMs_ &&
                nowMs - lastVoiceTurnMs_ >= config_.voiceTurnGapMs)
            {
                const double turnMs = std::max(
                    config_.voiceTurnMinSettleMs,
                    std::abs(rotateDegree) / config_.bodyDegPerSec * 1000.0 + config_.settleMs);
                motionUntilMs_ = nowMs + turnMs;
                if (config_.doaMode == DoaMode::LookAtUser)
                {
                    d.lookAtUserDoa = static_cast<float>(doa);
                    yawDeg_ += rotateDegree;  // best estimate of lookAtUser's turn
                }
                else
                {
                    d.bodyDegree = rotateDegree;
                    yawDeg_ += rotateDegree;
                    lastBodyMs_ = nowMs;
                }
                lastVoiceTurnMs_ = nowMs;
                rotationCheckBearing_.reset();  // the target changes after a voice turn
                lastTargetWorldDeg_.reset();
                d.event = "turn_to_voice";
                d.reason = "voice at DOA " + std::to_string(int(std::lround(doa))) +
                           (fromBehind ? " from straight ahead but nobody in view: turning to the back"
                                       : " with no visible speaker");
                // The previous target is now behind the turn; the next speaker
                // is acquired normally once LR-ASD sees them.
                if (target_ >= 0)
                    tracks_.erase(target_);
                target_ = -1;
                targetLostSinceMs_ = -1;
                return d;
            }
        }
    }

    // 2. Keep, recover, or release the current target.
    if (target_ >= 0)
    {
        TrackState &current = tracks_[target_];
        if (!current.seenNow)
        {
            if (targetLostSinceMs_ < 0)
                targetLostSinceMs_ = nowMs;
            // The IoU tracker gives a new id when our own rotation moves the
            // face far across the image. Adopt a new track that appeared
            // after the target vanished, close to where the target should be.
            int best = -1;
            double bestErr = config_.reacquireDeg;
            for (const auto &[id, t] : tracks_)
            {
                if (id == target_ || !t.seenNow || t.firstSeenMs < current.lastSeenMs)
                    continue;
                const double err = std::abs(t.bearingDeg - targetExpectedBearing_);
                if (err < bestErr)
                {
                    bestErr = err;
                    best = id;
                }
            }
            if (best >= 0)
            {
                const int old = target_;
                tracks_.erase(old);
                target_ = best;
                targetLostSinceMs_ = -1;
                d.event = "reacquire";
                d.reason = "track " + std::to_string(old) + " re-identified as " + std::to_string(best);
            }
            else if (nowMs - current.lastSeenMs > config_.lostMs)
            {
                d.event = "release";
                d.reason = "target " + std::to_string(target_) + " not seen for " +
                           std::to_string(int(nowMs - current.lastSeenMs)) + " ms";
                tracks_.erase(target_);
                target_ = -1;
                targetLostSinceMs_ = -1;
                setFace(ZenboFace::DEFAULT, d);
                returnToLastTarget(nowMs, d);
                return d;
            }
            else
            {
                d.event = "hold";
                d.targetTrack = target_;
                d.reason = "target briefly not visible";
                return d;
            }
        }
        else
        {
            targetLostSinceMs_ = -1;
        }
        if (tracks_[target_].seenNow)
            lastTargetWorldDeg_ = yawDeg_ + tracks_[target_].bearingDeg;
    }

    // 3. While the body is still turning, the image is moving; do not act on it.
    if (nowMs < motionUntilMs_)
    {
        if (d.event != "reacquire")
            d.event = "settling";
        d.targetTrack = target_;
        if (target_ >= 0)
        {
            d.bearingDeg = tracks_[target_].bearingDeg;
            d.elevationDeg = tracks_[target_].elevationDeg;
        }
        return d;
    }

    // 4. Choose who to attend to.
    int candidate = -1;
    // Just after a voice turn, the speaker is expected near the centre.
    const bool voiceFocus = target_ < 0 && nowMs - lastVoiceTurnMs_ <= config_.voiceFocusMs;
    for (const auto &[id, t] : tracks_)
    {
        double needed = target_ < 0 ? config_.acquireMs : config_.switchMs;
        if (voiceFocus && std::abs(t.bearingDeg) <= config_.voiceFocusDeg)
            needed = 0;
        if (id == target_ || !t.seenNow || !speaking(t) || speakingFor(t, nowMs) < needed)
            continue;
        if (candidate < 0 || t.smoothed > tracks_[candidate].smoothed)
            candidate = id;
    }

    if (candidate >= 0)
    {
        if (target_ < 0)
        {
            d.event = "acquire";
            d.reason = "track " + std::to_string(candidate) + " speaking for " +
                       std::to_string(int(speakingFor(tracks_[candidate], nowMs))) + " ms";
            attend(candidate, nowMs, d);
        }
        else
        {
            const TrackState &current = tracks_[target_];
            const TrackState &challenger = tracks_[candidate];
            const bool dwellDone = nowMs - targetSinceMs_ >= config_.minDwellMs;
            const bool clearlyLouder = challenger.smoothed > current.smoothed + config_.switchMargin;
            if (dwellDone && (!speaking(current) || clearlyLouder))
            {
                d.event = "switch";
                d.reason = "track " + std::to_string(candidate) + " speaking for " +
                           std::to_string(int(speakingFor(challenger, nowMs))) + " ms; track " +
                           std::to_string(target_) + (speaking(current) ? " quieter" : " silent");
                attend(candidate, nowMs, d);
            }
        }
    }

    if (target_ < 0)
    {
        bool anyoneSpeaking = false;
        for (const auto &[id, t] : tracks_)
            anyoneSpeaking |= t.seenNow && speaking(t);
        d.event = anyoneSpeaking ? "wait" : "idle";
        if (anyoneSpeaking)
            d.reason = "speech shorter than " + std::to_string(int(config_.acquireMs)) + " ms";
        else if (!searchForVoice(nowMs, d))
            lookAtIdleFace(nowMs, currentPitch, d);
        return d;
    }

    // A voice keeps going while the target has been silent for a long time:
    // the speaker is someone the camera cannot see.
    if (d.event == "idle" && speakingFor(tracks_[target_], nowMs) == 0 &&
        nowMs - targetSinceMs_ >= config_.searchTargetSilentMs &&
        nowMs - lastVisibleSpeechMs_ >= config_.searchTargetSilentMs && searchForVoice(nowMs, d))
    {
        target_ = -1;
        targetLostSinceMs_ = -1;
        lastTargetWorldDeg_.reset();
        return d;
    }

    // 5. Keep the target centred.
    if (d.event == "idle")
        d.event = "track";
    aim(tracks_[target_], nowMs, currentPitch, d);
    return d;
}

void ActiveSpeakerFollower::attend(int track, double nowMs, ActiveSpeakerDecision &)
{
    rotationCheckBearing_.reset();
    searchSteps_ = 0;
    target_ = track;
    targetSinceMs_ = nowMs;
    targetLostSinceMs_ = -1;
}

void ActiveSpeakerFollower::aim(const TrackState &t, double nowMs, int currentPitch,
                                ActiveSpeakerDecision &d)
{
    d.targetTrack = target_;
    d.bearingDeg = t.bearingDeg;
    d.elevationDeg = t.elevationDeg;

    // Eyes first: glance toward an off-centre speaker immediately, then the
    // body follows. Once centred, show attentive interest.
    // Hysteresis: a side glance needs gazeSide + h to start and ends below
    // gazeSide - h, so a speaker near the threshold does not flicker the face.
    const bool glancing = lastFace_ == ZenboFace::AWARE_LEFT || lastFace_ == ZenboFace::AWARE_RIGHT;
    const double threshold = config_.gazeSideDeg +
                             (glancing ? -config_.gazeHysteresisDeg : config_.gazeHysteresisDeg);
    if (std::abs(t.bearingDeg) > threshold)
    {
        const bool left = (t.bearingDeg > 0) != config_.mirrorGaze;
        setFace(left ? ZenboFace::AWARE_LEFT : ZenboFace::AWARE_RIGHT, d);
    }
    else
    {
        setFace(ZenboFace::INTERESTED, d);
    }

    // Did the previous turn actually move this target in the image?
    if (rotationCheckBearing_)
    {
        const double moved = *rotationCheckBearing_ - t.bearingDeg;
        const bool effective =
            moved * rotationCheckDegree_ > 0 &&
            std::abs(moved) >= config_.ineffectiveFraction * std::abs(rotationCheckDegree_);
        ineffectiveRotations_ = effective ? 0 : ineffectiveRotations_ + 1;
        rotationCheckBearing_.reset();
        if (ineffectiveRotations_ >= 2)
        {
            rotationPausedUntilMs_ = nowMs + config_.ineffectivePauseMs;
            ineffectiveRotations_ = 0;
            d.reason += (d.reason.empty() ? "" : "; ") +
                        std::string("turns had no effect; body rotation paused");
        }
    }

    targetExpectedBearing_ = t.bearingDeg;
    if (std::abs(t.bearingDeg) > config_.bodyDeadbandDeg &&
        nowMs - lastBodyMs_ >= config_.minBodyGapMs && nowMs >= rotationPausedUntilMs_)
    {
        double step = std::clamp(t.bearingDeg, -config_.maxBodyStepDeg, config_.maxBodyStepDeg);
        if (config_.keepOthersInView)
        {
            // Every other recently seen person must stay within +/-limit
            // after the turn: rotation r must satisfy |b - r| <= limit.
            const double limit = config_.hfovDeg / 2 - config_.edgeMarginDeg;
            double lo = -1e9, hi = 1e9;
            for (const auto &[id, other] : tracks_)
            {
                if (id == target_ || nowMs - other.lastSeenMs > config_.othersRecentMs)
                    continue;
                lo = std::max(lo, other.bearingDeg - limit);
                hi = std::min(hi, other.bearingDeg + limit);
            }
            if (lo <= hi)
            {
                const double framed = std::clamp(step, lo, hi);
                if (framed != step)
                    d.reason += (d.reason.empty() ? "" : "; ") +
                                std::string("turn limited to keep others in view");
                step = framed;
            }
        }
        const int degree = static_cast<int>(std::lround(step));
        if (std::abs(degree) >= config_.minBodyStepDeg)
        {
            rotationCheckBearing_ = t.bearingDeg;
            rotationCheckDegree_ = degree;
            d.bodyDegree = degree;
            yawDeg_ += degree;
            lastBodyMs_ = nowMs;
            motionUntilMs_ = nowMs + std::abs(degree) / config_.bodyDegPerSec * 1000.0 + config_.settleMs;
            targetExpectedBearing_ = t.bearingDeg - degree;
        }
    }

    const double pitchStep = std::clamp(t.elevationDeg, -config_.maxPitchStepDeg, config_.maxPitchStepDeg);
    const int desired = std::clamp(static_cast<int>(std::lround(currentPitch + pitchStep)),
                                   config_.pitchMin, config_.pitchMax);
    if (std::abs(desired - currentPitch) > config_.pitchDeadbandDeg &&
        nowMs - lastPitchMs_ >= config_.minPitchGapMs)
    {
        d.headPitch = desired;
        lastPitchMs_ = nowMs;
    }
}

void ActiveSpeakerFollower::returnToLastTarget(double nowMs, ActiveSpeakerDecision &d)
{
    if (!config_.returnOnLost || !lastTargetWorldDeg_)
        return;
    const double offset = std::remainder(*lastTargetWorldDeg_ - yawDeg_, 360.0);
    lastTargetWorldDeg_.reset();
    const int degree = static_cast<int>(
        std::lround(std::clamp(offset, -config_.maxReturnDeg, config_.maxReturnDeg)));
    if (std::abs(degree) <= config_.bodyDeadbandDeg)
        return;
    d.bodyDegree = degree;
    yawDeg_ += degree;
    lastBodyMs_ = nowMs;
    rotationCheckBearing_.reset();
    ineffectiveRotations_ = 0;
    rotationPausedUntilMs_ = 0;
    motionUntilMs_ = nowMs + std::abs(degree) / config_.bodyDegPerSec * 1000.0 + config_.settleMs;
    d.reason += "; turning " + std::to_string(degree) + " deg back to where the speaker was last seen";
}

bool ActiveSpeakerFollower::searchForVoice(double nowMs, ActiveSpeakerDecision &d)
{
    if (!config_.searchOnVoice)
        return false;
    if (voiceSinceMs_ < 0 || nowMs - lastVoiceMs_ > config_.voiceGapMs)
    {
        // Quiet again: the next voice starts a new search.
        if (nowMs - lastVoiceMs_ > 3000)
            searchSteps_ = 0;
        return false;
    }
    if (nowMs - voiceSinceMs_ < config_.searchVoiceMs ||
        nowMs - lastVisibleSpeechMs_ < config_.voiceQuietMs || nowMs < motionUntilMs_ ||
        nowMs - *firstUpdateMs_ < config_.voiceWarmupMs ||
        nowMs - lastSearchMs_ < config_.searchLookMs || nowMs < searchCooldownUntilMs_)
        return false;
    if (searchSteps_ >= config_.searchMaxSteps)
    {
        searchSteps_ = 0;
        searchCooldownUntilMs_ = nowMs + config_.searchCooldownMs;
        d.reason = "searched a full turn without finding the speaker; resting";
        return false;
    }
    const int degree = static_cast<int>(std::lround(config_.searchStepDeg));
    ++searchSteps_;
    d.event = "search";
    d.bodyDegree = degree;
    d.reason = "voice heard for " + std::to_string(int(nowMs - voiceSinceMs_)) +
               " ms but no visible face speaking; search step " + std::to_string(searchSteps_);
    yawDeg_ += degree;
    lastBodyMs_ = nowMs;
    lastSearchMs_ = nowMs + std::abs(degree) / config_.bodyDegPerSec * 1000.0;
    motionUntilMs_ = lastSearchMs_ + config_.settleMs;
    rotationCheckBearing_.reset();
    // A new view: the voice must be heard again before the next step.
    voiceSinceMs_ = -1;
    return true;
}

void ActiveSpeakerFollower::lookAtIdleFace(double nowMs, int currentPitch,
                                           ActiveSpeakerDecision &d)
{
    if (!config_.lookAtIdleFace)
        return;
    const TrackState *closest = nullptr;
    int closestId = -1;
    for (const auto &[id, t] : tracks_)
        if (t.seenNow && nowMs - t.firstSeenMs >= config_.idleLookMs &&
            (!closest || t.size > closest->size))
        {
            closest = &t;
            closestId = id;
        }
    if (!closest)
        return;
    turnBodyAndHead(closest->bearingDeg, closest->elevationDeg, nowMs, currentPitch, d);
    if (d.bodyDegree || d.headPitch)
    {
        d.event = "look";
        d.targetTrack = closestId;
        d.bearingDeg = closest->bearingDeg;
        d.elevationDeg = closest->elevationDeg;
        d.reason = "nobody speaking; centring the closest face";
    }
}

void ActiveSpeakerFollower::turnBodyAndHead(double bearingDeg, double elevationDeg, double nowMs,
                                            int currentPitch, ActiveSpeakerDecision &d)
{
    if (std::abs(bearingDeg) > config_.bodyDeadbandDeg &&
        nowMs - lastBodyMs_ >= config_.minBodyGapMs && nowMs >= rotationPausedUntilMs_)
    {
        const int degree = static_cast<int>(std::lround(
            std::clamp(bearingDeg, -config_.maxBodyStepDeg, config_.maxBodyStepDeg)));
        if (std::abs(degree) >= config_.minBodyStepDeg)
        {
            d.bodyDegree = degree;
            yawDeg_ += degree;
            lastBodyMs_ = nowMs;
            motionUntilMs_ = nowMs + std::abs(degree) / config_.bodyDegPerSec * 1000.0 + config_.settleMs;
        }
    }
    const double pitchStep = std::clamp(elevationDeg, -config_.maxPitchStepDeg, config_.maxPitchStepDeg);
    const int desired = std::clamp(static_cast<int>(std::lround(currentPitch + pitchStep)),
                                   config_.pitchMin, config_.pitchMax);
    if (std::abs(desired - currentPitch) > config_.pitchDeadbandDeg &&
        nowMs - lastPitchMs_ >= config_.minPitchGapMs)
    {
        d.headPitch = desired;
        lastPitchMs_ = nowMs;
    }
}

void ActiveSpeakerFollower::setFace(int face, ActiveSpeakerDecision &d)
{
    if (face != lastFace_)
    {
        d.face = face;
        lastFace_ = face;
    }
}
