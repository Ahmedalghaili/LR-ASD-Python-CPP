#include "StereoDoa.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>

StereoDoaResult StereoDoa::process(const int16_t *interleaved, size_t frames, int sampleRate) const
{
    StereoDoaResult r;
    const int n = config_.frame;
    const int up = std::max(config_.upsample, 1);
    if (!interleaved || sampleRate <= 0 || frames < static_cast<size_t>(n))
        return r;

    double sum = 0;
    for (size_t i = 0; i < frames * 2; ++i)
        sum += double(interleaved[i]) * interleaved[i];
    r.levelDbfs = 20 * std::log10(std::max(std::sqrt(sum / (frames * 2)), 1.0) / 32768.0);
    if (r.levelDbfs < config_.minLevelDbfs)
        return r;

    // Largest delay searched: a little beyond a voice at 90 deg.
    const int maxShift = static_cast<int>(std::ceil(config_.maxTdoaUs * 1.3e-6 * sampleRate * up));
    const int big = n * up;
    std::vector<double> lags, energies;
    cv::Mat window(1, n, CV_64F), a(1, n, CV_64F), b(1, n, CV_64F);
    for (int i = 0; i < n; ++i)
        window.at<double>(i) = 0.5 - 0.5 * std::cos(2 * CV_PI * i / (n - 1));

    for (size_t start = 0; start + n <= frames; start += n / 2)
    {
        double e = 0;
        for (int i = 0; i < n; ++i)
        {
            const double x = interleaved[2 * (start + i)];
            const double y = interleaved[2 * (start + i) + 1];
            e += x * x;
            a.at<double>(i) = x * window.at<double>(i);
            b.at<double>(i) = y * window.at<double>(i);
        }
        cv::Mat A, B;
        cv::dft(a, A, cv::DFT_COMPLEX_OUTPUT);
        cv::dft(b, B, cv::DFT_COMPLEX_OUTPUT);
        // PHAT-weighted cross spectrum A * conj(B), zero-padded in the middle
        // (between positive and negative frequencies) for sub-sample lags.
        cv::Mat R = cv::Mat::zeros(1, big, CV_64FC2);
        for (int k = 0; k < n; ++k)
        {
            const cv::Vec2d x = A.at<cv::Vec2d>(k), y = B.at<cv::Vec2d>(k);
            cv::Vec2d c(x[0] * y[0] + x[1] * y[1], x[1] * y[0] - x[0] * y[1]);
            const double m = std::hypot(c[0], c[1]) + 1e-9;
            c /= m;
            const int dst = k < n / 2 ? k : big - (n - k);
            R.at<cv::Vec2d>(dst) = c;
        }
        cv::Mat cc;
        cv::idft(R, cc, cv::DFT_REAL_OUTPUT);
        int best = 0;
        double bestValue = -1e300;
        for (int s = -maxShift; s <= maxShift; ++s)
        {
            const double v = cc.at<double>((s + big) % big);
            if (v > bestValue)
            {
                bestValue = v;
                best = s;
            }
        }
        lags.push_back(double(best) / up);  // samples
        energies.push_back(e);
    }
    if (lags.empty())
        return r;

    // Use the louder frames (the voice), as in tests/zenbo_mic_probe.py.
    std::vector<double> sorted = energies;
    std::nth_element(sorted.begin(), sorted.begin() + sorted.size() * 6 / 10, sorted.end());
    const double threshold = sorted[sorted.size() * 6 / 10];
    std::vector<double> loud;
    for (size_t i = 0; i < lags.size(); ++i)
        if (energies[i] >= threshold)
            loud.push_back(lags[i]);
    if (loud.empty())
        return r;
    std::vector<double> m = loud;
    std::nth_element(m.begin(), m.begin() + m.size() / 2, m.end());
    const double median = m[m.size() / 2];
    int agree = 0;
    for (double l : loud)
        agree += std::abs(l - median) <= 1.0;
    r.consistency = double(agree) / loud.size();
    // The correlation peak of A * conj(B) lies at a negative lag when
    // microphone 1 hears the sound later, i.e. a negative TDOA means the
    // sound reached microphone 0 first, the robot's left.
    r.tdoaUs = median / sampleRate * 1e6;
    const double s = std::clamp(-r.tdoaUs / config_.maxTdoaUs, -1.0, 1.0);
    r.angleDeg = std::asin(s) * 180.0 / CV_PI;
    r.valid = r.consistency >= config_.minConsistency;
    return r;
}
