#include <Arduino.h>
#include <WiFi.h>
#include <time.h>
#define BLYNK_TEMPLATE_ID "TMPL6Aq_bFCm0"
#define BLYNK_TEMPLATE_NAME "smart cold room monitoring by stephen chuang"
#define BLYNK_PRINT Serial
#include <BlynkSimpleEsp32.h>
#include <DHT.h>
#include <deque>

#ifdef EI_PORTING_ARDUINO
#undef EI_PORTING_ARDUINO
#endif
#include "edge-impulse-sdk/dsp/numpy.hpp"
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "model-parameters/model_metadata.h"

const char *WIFI_SSID = "M6";
const char *WIFI_PASS = "STep1234";

// Auth tokens for each room
const char *AUTH_TOKEN_ROOM1 = "qHDBumZ5WhCTTSdFd7AYHJh6jM9e1JZF";
const char *AUTH_TOKEN_ROOM2 = "VQslhXXlQDUmaQrKr_xikqRU-FqyPPfj";
const char *AUTH_TOKEN_ROOM3 = "kLDL8E1tXpYkuY70s2UYL1cxnvEXxYKw";

constexpr uint8_t PIN_DHT1 = 4;
constexpr uint8_t PIN_DHT2 = 5;

// Virtual pins (adjust in Blynk dashboard later)
constexpr uint8_t VP_ROOM1_TEMP = V0;
constexpr uint8_t VP_ROOM1_HUM = V1;
constexpr uint8_t VP_ROOM1_STATUS = V2;

constexpr uint8_t VP_ROOM2_TEMP = V3;
constexpr uint8_t VP_ROOM2_HUM = V4;
constexpr uint8_t VP_ROOM2_STATUS = V5;

constexpr uint8_t VP_ROOM3_TEMP = V6;
constexpr uint8_t VP_ROOM3_HUM = V7;
constexpr uint8_t VP_ROOM3_STATUS = V8;

// Virtual pins for diff_t (temperature difference)
constexpr uint8_t VP_ROOM1_DIFF_T = V9;
constexpr uint8_t VP_ROOM2_DIFF_T = V10;
constexpr uint8_t VP_ROOM3_DIFF_T = V11;

// Virtual pin for LED status (0=Normal, 1=Error)
constexpr uint8_t VP_LED_STATUS = V12;

// Virtual pin for last update timestamp
constexpr uint8_t VP_LAST_UPDATE = V13;

// Pilih ruangan aktif (1, 2, atau 3). Satu device mengirim ke satu ruangan.
int activeRoom = 1;

const int psdht = 26;

DHT dht1(PIN_DHT1, DHT22);
DHT dht2(PIN_DHT2, DHT11);
constexpr size_t FEATURE_SIZE = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
constexpr size_t SAMPLE_COUNT = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
constexpr size_t FEATURES_PER_SAMPLE = EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME;
constexpr uint32_t SAMPLE_INTERVAL_MS = EI_CLASSIFIER_INTERVAL_MS;

float feature_buffer[FEATURE_SIZE] = {0};
size_t sample_index = 0;

// Rolling stats per sensor
struct RollingStats {
	float buf[5] = {0};
	size_t idx = 0;
	size_t count = 0;
};

RollingStats rollT1;
RollingStats rollT2;

float prev_t1 = NAN;
float prev_t2 = NAN;
uint16_t stuck1 = 0;
uint16_t stuck2 = 0;

std::deque<int> recent_raw;
constexpr size_t VOTE_WINDOW = 5;

BlynkTimer timer;

static float safeFloat(float v, float fallback) {
	return isfinite(v) ? v : fallback;
}

static void pushRolling(RollingStats &rs, float v) {
	rs.buf[rs.idx] = v;
	rs.idx = (rs.idx + 1) % 5;
	if (rs.count < 5) rs.count++;
}

static void rollingMeanStd(const RollingStats &rs, float &mean, float &stddev) {
	mean = 0;
	stddev = 0;
	if (rs.count == 0) return;
	for (size_t i = 0; i < rs.count; ++i) mean += rs.buf[i];
	mean /= rs.count;
	for (size_t i = 0; i < rs.count; ++i) {
		float d = rs.buf[i] - mean;
		stddev += d * d;
	}
	stddev = sqrtf(stddev / rs.count);
}

static uint16_t updateStuck(float current, float previous, uint16_t prevCount) {
	if (!isfinite(current) || !isfinite(previous)) return prevCount;
	const float tol = 0.05f;
	if (fabsf(current - previous) < tol) {
		return prevCount + 1;
	}
	return 0;
}

static void add_feature_sample(float t1, float h1, float t2, float h2, bool fail1, bool fail2) {
	float dt1 = isfinite(prev_t1) ? (t1 - prev_t1) : 0;
	float dt2 = isfinite(prev_t2) ? (t2 - prev_t2) : 0;

	if (fail1) stuck1 = 50; else stuck1 = updateStuck(t1, prev_t1, stuck1);
	if (fail2) stuck2 = 50; else stuck2 = updateStuck(t2, prev_t2, stuck2);

	prev_t1 = t1;
	prev_t2 = t2;

	pushRolling(rollT1, t1);
	pushRolling(rollT2, t2);

	float meanT1 = 0, meanT2 = 0, stdT1 = 0, stdT2 = 0;
	rollingMeanStd(rollT1, meanT1, stdT1);
	rollingMeanStd(rollT2, meanT2, stdT2);

	float diffT = t1 - t2;
	float abs_dT1 = fabsf(dt1);
	float abs_dT2 = fabsf(dt2);

	size_t base = sample_index * FEATURES_PER_SAMPLE;
	feature_buffer[base + 0] = t1;
	feature_buffer[base + 1] = h1;
	feature_buffer[base + 2] = t2;
	feature_buffer[base + 3] = h2;
	feature_buffer[base + 4] = dt1;
	feature_buffer[base + 5] = dt2;
	feature_buffer[base + 6] = meanT1;
	feature_buffer[base + 7] = meanT2;
	feature_buffer[base + 8] = stdT1;
	feature_buffer[base + 9] = stdT2;
	feature_buffer[base +10] = static_cast<float>(stuck1);
	feature_buffer[base +11] = static_cast<float>(stuck2);
	feature_buffer[base +12] = diffT;
	feature_buffer[base +13] = abs_dT1;
	feature_buffer[base +14] = abs_dT2;
}

static int applyVoting(int rawClass) {
	recent_raw.push_back(rawClass);
	if (recent_raw.size() > VOTE_WINDOW) recent_raw.pop_front();

	int counts[3] = {0, 0, 0};
	for (int c : recent_raw) counts[c]++;
	int majority = 0;
	for (int i = 1; i < 3; ++i) if (counts[i] > counts[majority]) majority = i;

	bool twoConsecutiveClass2 = false;
	for (size_t i = 1; i < recent_raw.size(); ++i) {
		if (recent_raw[i] == 2 && recent_raw[i - 1] == 2) {
			twoConsecutiveClass2 = true;
			break;
		}
	}

	if (majority == 2 && !twoConsecutiveClass2) {
		majority = (counts[1] >= counts[0]) ? 1 : 0;
	}
	return majority;
}

static const char *className(int cls) {
	switch (cls) {
		case 0: return "Normal";
		case 1: return "Cooling Failure";
		case 2: return "Sensor Fault";
		default: return "Unknown";
	}
}

struct RoomPins {
	uint8_t vpTemp;
	uint8_t vpHum;
	uint8_t vpStatus;
	uint8_t vpDiffT;
};

static RoomPins selectRoomPins(int room) {
	switch (room) {
		case 2: return {VP_ROOM2_TEMP, VP_ROOM2_HUM, VP_ROOM2_STATUS, VP_ROOM2_DIFF_T};
		case 3: return {VP_ROOM3_TEMP, VP_ROOM3_HUM, VP_ROOM3_STATUS, VP_ROOM3_DIFF_T};
		default: return {VP_ROOM1_TEMP, VP_ROOM1_HUM, VP_ROOM1_STATUS, VP_ROOM1_DIFF_T};
	}
}

static const char* selectAuthToken(int room) {
	switch (room) {
		case 2: return AUTH_TOKEN_ROOM2;
		case 3: return AUTH_TOKEN_ROOM3;
		default: return AUTH_TOKEN_ROOM1;
	}
}

static String getFormattedTimestamp() {
	time_t now = time(nullptr);
	struct tm* timeinfo = localtime(&now);
	
	char buffer[30];
	strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeinfo);
	return String(buffer);
}

static void runInference() {
	ei::signal_t signal;
	ei_impulse_result_t result = {0};
	EI_IMPULSE_ERROR ei_status = EI_IMPULSE_OK;
	ei_status = static_cast<EI_IMPULSE_ERROR>(ei::numpy::signal_from_buffer(feature_buffer, FEATURE_SIZE, &signal));
	if (ei_status != EI_IMPULSE_OK) {
		Serial.print("Signal error: ");
		Serial.println(ei_status);
		return;
	}

	ei_status = run_classifier(&signal, &result, false);
	if (ei_status != EI_IMPULSE_OK) {
		Serial.print("Classifier error: ");
		Serial.println(ei_status);
		return;
	}

	int rawClass = 0;
	float best = result.classification[0].value;
	for (size_t ix = 1; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
		if (result.classification[ix].value > best) {
			best = result.classification[ix].value;
			rawClass = static_cast<int>(ix);
		}
	}

	int finalClass = applyVoting(rawClass);
	Serial.print("Raw class: ");
	Serial.print(rawClass);
	Serial.print(" Final class: ");
	Serial.println(finalClass);

	// Publish data to the selected room only
	float t1 = feature_buffer[(SAMPLE_COUNT - 1) * FEATURES_PER_SAMPLE];
	float h1 = feature_buffer[(SAMPLE_COUNT - 1) * FEATURES_PER_SAMPLE + 1];
	float t2 = feature_buffer[(SAMPLE_COUNT - 1) * FEATURES_PER_SAMPLE + 2];
	float h2 = feature_buffer[(SAMPLE_COUNT - 1) * FEATURES_PER_SAMPLE + 3];
	float diffT = t1 - t2; // Temperature difference

	RoomPins pins = selectRoomPins(activeRoom);
	Blynk.virtualWrite(pins.vpTemp, t1);
	Blynk.virtualWrite(pins.vpHum, h1);
	Blynk.virtualWrite(pins.vpStatus, className(finalClass));
	Blynk.virtualWrite(pins.vpDiffT, diffT);
	
	// LED status: 1 if error (finalClass != 0), 0 if normal
	int ledStatus = (finalClass != 0) ? 1 : 0;
	Blynk.virtualWrite(VP_LED_STATUS, ledStatus);
	
	// Send last update timestamp
	Blynk.virtualWrite(VP_LAST_UPDATE, getFormattedTimestamp());
}

static void sampleTask() {
	float t1 = dht1.readTemperature();
	float h1 = dht1.readHumidity();
	float t2 = dht2.readTemperature();
	float h2 = dht2.readHumidity();

	bool fail1 = isnan(t1) || isnan(h1);
	bool fail2 = isnan(t2) || isnan(h2);

	t1 = safeFloat(t1, 0);
	h1 = safeFloat(h1, 0);
	t2 = safeFloat(t2, 0);
	h2 = safeFloat(h2, 0);

	Serial.print("S");
	Serial.print(sample_index + 1);
	Serial.print(" T1:"); Serial.print(t1);
	Serial.print(" H1:"); Serial.print(h1);
	Serial.print(" T2:"); Serial.print(t2);
	Serial.print(" H2:"); Serial.print(h2);
	Serial.print(fail1 ? " [F1]" : "");
	Serial.println(fail2 ? " [F2]" : "");

	add_feature_sample(t1, h1, t2, h2, fail1, fail2);
	sample_index++;

	if (sample_index >= SAMPLE_COUNT) {
		runInference();
		sample_index = 0;
		rollT1 = RollingStats();
		rollT2 = RollingStats();
		prev_t1 = NAN;
		prev_t2 = NAN;
		stuck1 = 0;
		stuck2 = 0;
	}
}

void setup() {
	Serial.begin(115200);
	delay(200);
	dht1.begin();
	dht2.begin();
	pinMode(psdht, OUTPUT);
	digitalWrite(psdht, HIGH); // power DHT
	WiFi.begin(WIFI_SSID, WIFI_PASS);
	Serial.print("Connecting WiFi");
	while (WiFi.status() != WL_CONNECTED) {
		delay(500);
		Serial.print(".");
	}
	Serial.println(" connected");

	// Configure time with NTP
	configTime(7 * 3600, 0, "pool.ntp.org", "time.nist.gov"); // GMT+7 for Indonesia
	Serial.print("Waiting for NTP time sync: ");
	time_t now = time(nullptr);
	int maxWait = 20;
	while (now < 24 * 3600 && maxWait-- > 0) {
		Serial.print(".");
		delay(500);
		now = time(nullptr);
	}
	Serial.println();
	Serial.println(ctime(&now));

	const char* authToken = selectAuthToken(activeRoom);
	Serial.print("Active Room: ");
	Serial.println(activeRoom);
	Serial.print("Auth Token: ");
	Serial.println(authToken);
	
	Blynk.begin(authToken, WIFI_SSID, WIFI_PASS);

	timer.setInterval(SAMPLE_INTERVAL_MS, sampleTask);
}

void loop() {
	Blynk.run();
	timer.run();
}
