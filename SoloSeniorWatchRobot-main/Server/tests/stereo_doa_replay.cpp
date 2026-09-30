// Run StereoDoa on a raw recording: interleaved PCM16 stereo (microphone 0,
// microphone 1), cut into 200 ms pieces like the Zenbo V2 app sends them.
//   stereo_doa_replay FILE.raw RATE
//   stereo_doa_replay --selftest
#include "StereoDoa.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>

static int selftest()
{
    // White noise with microphone 1 delayed by 3 samples at 16 kHz (187.5 us):
    // the sound reaches microphone 0 first, so the voice is on the left.
    const int rate = 16000, frames = rate / 5;
    std::mt19937 rng(1);
    std::normal_distribution<double> noise(0, 3000);
    std::vector<double> s(frames + 8);
    for (double &v : s)
        v = noise(rng);
    std::vector<int16_t> pcm(frames * 2);
    for (int i = 0; i < frames; ++i)
    {
        pcm[2 * i] = static_cast<int16_t>(s[i + 3]);
        pcm[2 * i + 1] = static_cast<int16_t>(s[i]);
    }
    const StereoDoaResult r = StereoDoa().process(pcm.data(), frames, rate);
    std::printf("selftest: valid=%d tdoa=%.1f us angle=%.1f deg consistency=%.2f\n", r.valid, r.tdoaUs,
                r.angleDeg, r.consistency);
    const bool ok = r.valid && std::abs(r.tdoaUs + 187.5) < 20 && r.angleDeg > 40;
    std::printf(ok ? "SELFTEST PASSED\n" : "SELFTEST FAILED\n");
    return ok ? 0 : 1;
}

int main(int argc, char **argv)
{
    if (argc >= 2 && std::strcmp(argv[1], "--selftest") == 0)
        return selftest();
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: %s FILE.raw RATE | --selftest\n", argv[0]);
        return 2;
    }
    const int rate = std::atoi(argv[2]);
    std::ifstream in(argv[1], std::ios::binary);
    std::vector<int16_t> pcm((std::istreambuf_iterator<char>(in)), {});
    // istreambuf_iterator<char> reads bytes: rebuild the samples.
    in.clear();
    in.seekg(0, std::ios::end);
    const size_t bytes = static_cast<size_t>(in.tellg());
    in.seekg(0);
    pcm.assign(bytes / 2, 0);
    in.read(reinterpret_cast<char *>(pcm.data()), bytes / 2 * 2);
    const size_t piece = rate / 5;
    StereoDoa doa;
    for (size_t start = 0; (start + piece) * 2 <= pcm.size(); start += piece)
    {
        const StereoDoaResult r = doa.process(pcm.data() + start * 2, piece, rate);
        std::printf("t=%.1fs level=%.1f dBFS valid=%d tdoa=%.1f us angle=%.1f deg consistency=%.2f\n",
                    double(start) / rate, r.levelDbfs, r.valid, r.tdoaUs, r.angleDeg, r.consistency);
    }
    return 0;
}
