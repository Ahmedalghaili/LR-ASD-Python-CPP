// XIAO ESP32S3 on the reSpeaker XVF3800 board: send the array's processed
// voice (I2S) and its voice direction (I2C) over Wi-Fi to the PC, where
// tests/respeaker_wifi_bridge.py passes them to the voice gate (VAD + CAM++)
// and the server.
//
// The XVF3800 must run the I2S *slave* firmware at 16 kHz
// (respeaker_xvf3800_i2s_slave_v1.0.9_16k.bin, flashed with dfu-util through
// the XMOS USB-C port): this XIAO is the I2S master and gives the clocks, so
// the audio is 16 kHz, the rate VAD and CAM++ need. Both I2S channels carry
// the ASR output beam (firmware v1.0.7+); the left one is sent.
// Pins (Seeed wiki, respeaker_xvf3800_xiao_record_playback): BCK 8, WS 7,
// DATA OUT 44, DATA IN 43. I2C (XVF3800 at 0x2C) on the default pins.
//
// Wi-Fi name and password: wifi_secrets.h (not in git), see
// wifi_secrets.h.example.
//
// The PC finds the board by broadcasting "HELLO" to UDP port 5010; the board
// then streams to the PC that said hello (so the PC's IP may change):
//   UDP 5011: audio, 20 ms packets: "AUD" + uint16 sequence + 320 int16 samples
//   UDP 5007: every 100 ms "DOA <doa_value> <speech 0/1> <free-running beam deg, -1 = none>"
#include <ESP_I2S.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <math.h>
#include "wifi_secrets.h"

static const uint8_t XMOS_ADDR = 0x2C;   // XVF3800 I2C 7-bit address
static const uint16_t HELLO_PORT = 5010, AUDIO_PORT = 5011, DOA_PORT = 5007;
static const uint32_t RATE = 16000;
static const int SAMPLES = 320;          // 20 ms

I2SClass i2s;
WiFiUDP hello, out;
IPAddress pc;
bool havePc = false;
uint32_t lastHelloMs = 0, lastDoaMs = 0;
uint16_t sequence = 0;
int32_t frames[SAMPLES * 2];             // stereo, 32-bit slots
uint8_t packet[5 + SAMPLES * 2];

// Read `len` bytes of resource `resid`, command `cmd` (status byte first:
// 0 = ok, 64 = retry).
static bool xmosRead(uint8_t resid, uint8_t cmd, uint8_t *data, uint8_t len)
{
    for (int attempt = 0; attempt < 20; ++attempt)
    {
        Wire.beginTransmission(XMOS_ADDR);
        Wire.write(resid);
        Wire.write(cmd | 0x80);
        Wire.write(len + 1);
        if (Wire.endTransmission() != 0)
            return false;
        if (Wire.requestFrom(XMOS_ADDR, (uint8_t)(len + 1)) != len + 1)
            return false;
        const uint8_t status = Wire.read();
        for (uint8_t i = 0; i < len; ++i)
            data[i] = Wire.read();
        if (status == 0)
            return true;
        delay(2);
    }
    return false;
}

static void sendDirection()
{
    uint8_t d[4], beams[16];
    if (!xmosRead(20, 18, d, 4))         // DOA_VALUE: angle (uint16), speech (uint16)
    {
        Serial.println("no answer from the XVF3800 on I2C (I2S firmware flashed?)");
        return;
    }
    int freeBeam = -1;
    if (xmosRead(33, 75, beams, 16))     // AEC_AZIMUTH_VALUES: 4 floats, radians; [2] = free-running
    {
        float r;
        memcpy(&r, beams + 8, 4);
        if (!isnan(r))
            freeBeam = ((int)lroundf(r * 180.0f / M_PI) % 360 + 360) % 360;
    }
    char msg[48];
    const int n = snprintf(msg, sizeof msg, "DOA %u %u %d", d[0] | (d[1] << 8), d[2], freeBeam);
    out.beginPacket(pc, DOA_PORT);
    out.write((const uint8_t *)msg, n);
    out.endPacket();
}

void setup()
{
    Serial.begin(115200);
    Wire.begin();
    i2s.setPins(8, 7, 44, 43);
    if (!i2s.begin(I2S_MODE_STD, RATE, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO))
        Serial.println("I2S start failed");
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);                // no power-save gaps in the audio
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }
    Serial.printf("\nWi-Fi connected: %s\n", WiFi.localIP().toString().c_str());
    hello.begin(HELLO_PORT);
    memcpy(packet, "AUD", 3);
}

void loop()
{
    if (WiFi.status() != WL_CONNECTED)
    {
        WiFi.reconnect();
        delay(1000);
        return;
    }
    if (hello.parsePacket() > 0)
    {
        if (!havePc || hello.remoteIP() != pc)
            Serial.printf("streaming to %s\n", hello.remoteIP().toString().c_str());
        pc = hello.remoteIP();
        havePc = true;
        lastHelloMs = millis();
        hello.flush();
    }
    if (havePc && millis() - lastHelloMs > 10000)
    {
        havePc = false;                  // the PC went away; wait for a new hello
        Serial.println("no hello from the PC; stopped streaming");
    }

    // 20 ms of audio (blocks until the I2S DMA has it, so this paces the loop).
    const size_t got = i2s.readBytes((char *)frames, sizeof frames);
    if (!havePc || got != sizeof frames)
        return;
    packet[3] = sequence & 0xff;
    packet[4] = sequence >> 8;
    ++sequence;
    int16_t *pcm = (int16_t *)(packet + 5);
    for (int i = 0; i < SAMPLES; ++i)
        pcm[i] = frames[2 * i] >> 16;    // left channel, top 16 bits
    out.beginPacket(pc, AUDIO_PORT);
    out.write(packet, sizeof packet);
    out.endPacket();

    if (millis() - lastDoaMs >= 100)
    {
        lastDoaMs = millis();
        sendDirection();
    }
}
