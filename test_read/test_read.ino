const int clockPin = 9;        // GPIO pin for clock signal output
const int samplePin = 10;      // Optional LED to indicate sampling
float sampleRateHz = 120.0;     // Sampling rate in Hz (e.g., 60 Hz = 1/60 seconds)
unsigned long runTimeSeconds = 10; // Run time in seconds (e.g., 60 seconds)
unsigned long startTime;
bool isRunning = true;
int lastValue = 0;
int lastStatus = 0;
unsigned long lastSampleTime = 0;
int successCount = 0;          // Counter for successful samples
int failCount = 0;             // Counter for failed samples

void setup() {
  pinMode(8, OUTPUT);         // Enable 3.3V on UEXT
  digitalWrite(8, LOW);
  pinMode(clockPin, OUTPUT);  // Clock signal pin
  digitalWrite(clockPin, LOW);
  pinMode(samplePin, OUTPUT); // Sampling indicator LED
  digitalWrite(samplePin, LOW);

  Serial.begin(9600);         // USB serial for debugging
  while (!Serial) {}
  Serial1.begin(115200);      // MOD-PULSE UART at 115200 baud
  Serial.println("Starting MOD-PULSE continuous sampler with clock output...");
  Serial.print("Sample Rate: "); Serial.print(sampleRateHz); Serial.println(" Hz");
  Serial.print("Run Time: "); Serial.print(runTimeSeconds); Serial.println(" seconds");

  startTime = millis();       // Record start time
}

void loop() {
  if (isRunning) {
    unsigned long currentTime = millis();
    unsigned long sampleInterval = 1000 / sampleRateHz; // Interval in milliseconds

    // Check if it's time to sample
    if (currentTime - lastSampleTime >= sampleInterval) {
      lastSampleTime = currentTime;

      // Generate clock pulse (HIGH during sampling)
      digitalWrite(clockPin, HIGH);

      // Check for available data
      if (Serial1.available() >= 4) {
        byte packet[4];
        for (int i = 0; i < 4; i++) {
          packet[i] = Serial1.read();
        }
        if (packet[0] == 0xFF) {
          lastValue = packet[1];    // Store value (e.g., HR)
          lastStatus = packet[2];   // Store status (e.g., SpO2)
          digitalWrite(samplePin, HIGH); // Indicate sampling
          Serial.print("Sampled - Value: ");
          Serial.print(lastValue, DEC);
          Serial.print(" Status: ");
          Serial.println(lastStatus, DEC);
          successCount++;            // Increment success counter
        } else {
          Serial.println("Failed - Invalid packet (no 0xFF header)");
          failCount++;               // Increment fail counter
        }
      } else {
        Serial.println("Failed - Insufficient data");
        failCount++;                 // Increment fail counter for no data
      }

      // End clock pulse
      digitalWrite(clockPin, LOW);
      digitalWrite(samplePin, LOW); // Turn off LED after sampling
    }

    // Check runtime
    //    if (currentTime - startTime >= runTimeSeconds * 1000) {
    //      isRunning = false;
    //      Serial.print("Stopped after " + String(runTimeSeconds) + " seconds. ");
    //      Serial.print("Success Count: "); Serial.print(successCount);
    //      Serial.print(" Fail Count: "); Serial.println(failCount);
    //      digitalWrite(clockPin, LOW); // Ensure clock is off
    //      digitalWrite(samplePin, LOW); // Ensure LED is off
    //    }
  }
}
