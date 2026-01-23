#include <Arduino.h>
#include <WebServer.h>
#include <WiFi.h>
#include <DHT.h>

// ==================== Configuration ====================
#define DHT1_PIN 4
#define DHT1_TYPE DHT11
#define DHT2_PIN 2
#define DHT2_TYPE DHT22
#define SAMPLING_INTERVAL 5000
#define MAX_LOG_ROWS 200
#define CSV_HEADER "index,timestamp_ms,temperature1_c,humidity1_percent,temperature2_c,humidity2_percent,status"

// ==================== Global Objects ====================
DHT dht1(DHT1_PIN, DHT1_TYPE);
DHT dht2(DHT2_PIN, DHT2_TYPE);
WebServer server(80);

// ==================== Data Structure ====================
struct SensorData {
  uint32_t index;
  uint32_t timestamp_ms;
  float temperature1_c;
  float humidity1_percent;
  float temperature2_c;
  float humidity2_percent;
  String status;
};

// ==================== Ring Buffer Implementation ====================
class RingBuffer {
private:
  SensorData buffer[MAX_LOG_ROWS];
  uint16_t head = 0;
  uint16_t count = 0;
  uint32_t totalIndex = 0;

public:
  void add(float t1, float h1, float t2, float h2, String stat) {
    buffer[head].index = totalIndex++;
    buffer[head].timestamp_ms = millis();
    buffer[head].temperature1_c = t1;
    buffer[head].humidity1_percent = h1;
    buffer[head].temperature2_c = t2;
    buffer[head].humidity2_percent = h2;
    buffer[head].status = stat;

    head = (head + 1) % MAX_LOG_ROWS;
    if (count < MAX_LOG_ROWS) {
      count++;
    }
  }

  uint16_t getCount() const {
    return count;
  }

  SensorData get(uint16_t idx) const {
    if (idx >= count) {
      return SensorData{0, 0, 0, 0, 0, 0, "INVALID"};
    }
    uint16_t actual = (head - count + idx) % MAX_LOG_ROWS;
    return buffer[actual];
  }

  void clear() {
    head = 0;
    count = 0;
    totalIndex = 0;
  }

  String toCSV() {
    String csv = String(CSV_HEADER) + "\n";
    for (uint16_t i = 0; i < count; i++) {
      SensorData data = get(i);
      csv += String(data.index) + ",";
      csv += String(data.timestamp_ms) + ",";
      csv += String(data.temperature1_c, 2) + ",";
      csv += String(data.humidity1_percent, 2) + ",";
      csv += String(data.temperature2_c, 2) + ",";
      csv += String(data.humidity2_percent, 2) + ",";
      csv += data.status + "\n";
    }
    return csv;
  }
};

RingBuffer logBuffer;
unsigned long lastSampleTime = 0;

// ==================== HTML Page ====================
String getHTMLPage() {
  String html = R"(
<!DOCTYPE html>
<html>
<head>
  <meta charset='UTF-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1.0'>
  <title>DHT11 & DHT22 Dual Logger</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
      padding: 20px;
      display: flex;
      justify-content: center;
      align-items: center;
    }
    .container {
      background: white;
      border-radius: 12px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.3);
      max-width: 1000px;
      width: 100%;
      padding: 30px;
    }
    h1 {
      color: #333;
      margin-bottom: 10px;
      text-align: center;
      font-size: 28px;
    }
    .status-bar {
      display: flex;
      justify-content: space-between;
      margin: 20px 0;
      padding: 15px;
      background: #f5f5f5;
      border-radius: 8px;
      flex-wrap: wrap;
      gap: 15px;
    }
    .stat-item {
      flex: 1;
      min-width: 150px;
      text-align: center;
    }
    .stat-label { color: #666; font-size: 12px; font-weight: bold; }
    .stat-value { color: #667eea; font-size: 20px; font-weight: bold; }
    .controls {
      display: flex;
      gap: 10px;
      justify-content: center;
      margin: 20px 0;
      flex-wrap: wrap;
    }
    button {
      padding: 10px 20px;
      border: none;
      border-radius: 6px;
      cursor: pointer;
      font-size: 14px;
      font-weight: bold;
      transition: all 0.3s;
    }
    .btn-primary {
      background: #667eea;
      color: white;
    }
    .btn-primary:hover { background: #5568d3; transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102,126,234,0.4); }
    .btn-danger {
      background: #ff6b6b;
      color: white;
    }
    .btn-danger:hover { background: #ee5a52; transform: translateY(-2px); box-shadow: 0 5px 15px rgba(255,107,107,0.4); }
    .btn-success {
      background: #51cf66;
      color: white;
    }
    .btn-success:hover { background: #40c057; transform: translateY(-2px); box-shadow: 0 5px 15px rgba(81,207,102,0.4); }
    .table-wrapper {
      overflow-x: auto;
      margin-top: 20px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    thead {
      background: #667eea;
      color: white;
      position: sticky;
      top: 0;
    }
    th {
      padding: 12px;
      text-align: left;
      font-weight: bold;
    }
    td {
      padding: 10px 12px;
      border-bottom: 1px solid #eee;
    }
    tbody tr:hover {
      background: #f9f9f9;
    }
    tbody tr:nth-child(even) {
      background: #f5f5f5;
    }
    .status-ok { color: #51cf66; font-weight: bold; }
    .status-error { color: #ff6b6b; font-weight: bold; }
    .refresh-indicator {
      display: inline-block;
      width: 8px;
      height: 8px;
      background: #51cf66;
      border-radius: 50%;
      margin-left: 10px;
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }
    .footer {
      text-align: center;
      margin-top: 20px;
      font-size: 12px;
      color: #999;
    }
    @media (max-width: 600px) {
      .container { padding: 15px; }
      h1 { font-size: 20px; }
      table { font-size: 11px; }
      th, td { padding: 8px; }
      .controls { flex-direction: column; }
      button { width: 100%; }
    }
  </style>
</head>
<body>
  <div class='container'>
    <h1>DHT11 & DHT22 Logger <span class='refresh-indicator'></span></h1>
    
    <div class='status-bar'>
      <div class='stat-item'>
        <div class='stat-label'>RECORDS</div>
        <div class='stat-value' id='recordCount'>0</div>
      </div>
      <div class='stat-item'>
        <div class='stat-label'>T1 / H1 (DHT11)</div>
        <div class='stat-value' id='lastTH1'>-- / --</div>
      </div>
      <div class='stat-item'>
        <div class='stat-label'>T2 / H2 (DHT22)</div>
        <div class='stat-value' id='lastTH2'>-- / --</div>
      </div>
      <div class='stat-item'>
        <div class='stat-label'>AUTO REFRESH</div>
        <div class='stat-value'>3s</div>
      </div>
    </div>

    <div class='controls'>
      <button class='btn-success' onclick='downloadCSV()'>📥 Download CSV</button>
      <button class='btn-danger' onclick='clearData()'>🗑️ Clear All</button>
    </div>

    <div class='table-wrapper'>
      <table>
        <thead>
          <tr>
            <th>Index</th>
            <th>Timestamp (ms)</th>
            <th>T1 (°C)</th>
            <th>H1 (%)</th>
            <th>T2 (°C)</th>
            <th>H2 (%)</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody id='dataTable'>
          <tr><td colspan='5' style='text-align: center; color: #999;'>Waiting for data...</td></tr>
        </tbody>
      </table>
    </div>

    <div class='footer'>
      Auto-refreshing every 3 seconds | Sampling every 5 seconds
    </div>
  </div>

  <script>
    function loadData() {
      fetch('/csv')
        .then(r => r.text())
        .then(csv => {
          const lines = csv.trim().split('\n').slice(1);
          const tbody = document.getElementById('dataTable');
          
          if (lines.length === 0 || lines[0].trim() === '') {
            tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: #999;">No data yet</td></tr>';
            document.getElementById('recordCount').innerText = '0';
            return;
          }

          let rows = lines.map(line => {
            const [idx, ts, t1, h1, t2, h2, status] = line.split(',');
            const statusClass = status.includes('ERROR') ? 'status-error' : 'status-ok';
            return `
              <tr>
                <td>${idx}</td>
                <td>${ts}</td>
                <td>${t1}</td>
                <td>${h1}</td>
                <td>${t2}</td>
                <td>${h2}</td>
                <td><span class='${statusClass}'>${status}</span></td>
              </tr>
            `;
          });

          tbody.innerHTML = rows.reverse().join('');
          document.getElementById('recordCount').innerText = lines.length;
          
          if (lines.length > 0) {
            const lastLine = lines[lines.length - 1].split(',');
            document.getElementById('lastTH1').innerText = `${lastLine[2]}°C / ${lastLine[3]}%`;
            document.getElementById('lastTH2').innerText = `${lastLine[4]}°C / ${lastLine[5]}%`;
          }
        })
        .catch(e => console.error('Error loading data:', e));
    }

    function downloadCSV() {
      fetch('/csv')
        .then(r => r.text())
        .then(csv => {
          const blob = new Blob([csv], { type: 'text/csv' });
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = 'sensor_data_' + new Date().toISOString().slice(0,10) + '.csv';
          a.click();
          window.URL.revokeObjectURL(url);
        });
    }

    function clearData() {
      if (confirm('Are you sure you want to clear all data?')) {
        fetch('/clear').then(() => {
          alert('Data cleared!');
          loadData();
        });
      }
    }

    loadData();
    setInterval(loadData, 3000);
  </script>
</body>
</html>
  )";
  return html;
}

// ==================== HTTP Handlers ====================
void handleRoot() {
  server.send(200, "text/html; charset=utf-8", getHTMLPage());
}

void handleCSV() {
  server.send(200, "text/csv; charset=utf-8", logBuffer.toCSV());
}

void handleClear() {
  logBuffer.clear();
  server.send(200, "text/plain", "Log cleared");
  Serial.println("\n[INFO] Log cleared via /clear endpoint\n");
}

void handleNotFound() {
  server.send(404, "text/plain", "404: Not Found");
}

// ==================== Sensor Reading ====================
void readSensor() {
  if (millis() - lastSampleTime < SAMPLING_INTERVAL) {
    return;
  }
  lastSampleTime = millis();

  float t1 = dht1.readTemperature();
  if (!isnan(t1)) {
    t1 += 0.3; // setelah di baca data nya DHT 11 selalu lebih kecil 0.3 derajat Celsius
  }
  float h1 = dht1.readHumidity();
  float t2 = dht2.readTemperature();
  float h2 = dht2.readHumidity();

  bool err1 = isnan(t1) || isnan(h1);
  bool err2 = isnan(t2) || isnan(h2);
  String status = "OK";

  if (err1 && err2) {
    t1 = h1 = t2 = h2 = -999.0;
    status = "BOTH_ERROR";
  } else if (err1) {
    t1 = h1 = -999.0;
    status = "S1_ERROR";
  } else if (err2) {
    t2 = h2 = -999.0;
    status = "S2_ERROR";
  }

  logBuffer.add(t1, h1, t2, h2, status);

  uint16_t idx = logBuffer.getCount() - 1;
  SensorData last = logBuffer.get(idx);
  Serial.print(last.index);
  Serial.print(",");
  Serial.print(last.timestamp_ms);
  Serial.print(",");
  Serial.print(last.temperature1_c);
  Serial.print(",");
  Serial.print(last.humidity1_percent);
  Serial.print(",");
  Serial.print(last.temperature2_c);
  Serial.print(",");
  Serial.print(last.humidity2_percent);
  Serial.print(",");
  Serial.println(last.status);
}

// ==================== Setup ====================
void setup() {
  Serial.begin(115200);
  delay(100);
  
  Serial.println("\n\n");
  Serial.println("====== Dual DHT Logger (DHT11 + DHT22) ======");
  Serial.println("Initializing sensors...");

  dht1.begin();
  dht2.begin();
  delay(1000);

  Serial.println("Sensors initialized");
  Serial.println();
  Serial.println(CSV_HEADER);

  // Connect to WiFi (Access Point mode)
  WiFi.mode(WIFI_AP);
  WiFi.softAP("DHT11-Logger", "12345678");
  
  Serial.println("WiFi AP Started");
  Serial.println("  SSID: DHT11-Logger");
  Serial.println("  Password: 12345678");
  Serial.print("  IP: ");
  Serial.println(WiFi.softAPIP());
  Serial.println();

  // Setup web server
  server.on("/", handleRoot);
  server.on("/csv", handleCSV);
  server.on("/clear", handleClear);
  server.onNotFound(handleNotFound);
  server.begin();

  Serial.println("Web server started");
  Serial.println("http://192.168.4.1/");
  Serial.println("======================================");
  Serial.println();
}

// ==================== Loop ====================
void loop() {
  server.handleClient();
  readSensor();
  delay(10);
}