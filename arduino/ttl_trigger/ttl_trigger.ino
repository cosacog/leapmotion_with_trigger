/*
 * TTL Trigger Detection for Leap Motion Recording
 *
 * Hardware: Arduino Nano (or compatible)
 * Purpose: High-precision TTL pulse detection using hardware interrupt
 *
 * Wiring:
 *   - TTL Input: Pin 2 (INT0) - Connect to TTL signal
 *   - GND: Connect to TTL ground
 *
 * Serial Protocol (115200 baud):
 *   Commands from PC:
 *     'S' - Sync start: Returns "S,<micros>\n"
 *     'E' - Sync end:   Returns "E,<micros>\n" then all trigger times
 *     'R' - Reset:      Clears trigger buffer
 *     'C' - Count:      Returns current trigger count
 *     'P' - Ping:       Returns "PONG\n" (connection test)
 *
 *   Events to PC:
 *     "T,<micros>\n" - Trigger detected at <micros> (sent in real-time)
 *
 * Note: micros() overflows after ~70 minutes. For longer recordings,
 *       use sync points at start/end for time alignment.
 */

#define TTL_PIN 2          // Hardware interrupt pin (INT0)
#define MAX_TRIGGERS 200   // Maximum triggers to store (limited by Arduino Nano RAM)
#define BAUD_RATE 115200

// Trigger storage
volatile unsigned long trigger_times[MAX_TRIGGERS];
volatile int trigger_count = 0;
volatile bool new_trigger = false;
volatile unsigned long last_trigger_time = 0;

// Sync points
unsigned long sync_start_time = 0;
unsigned long sync_end_time = 0;

// Debounce (optional, set to 0 to disable)
#define DEBOUNCE_US 1000  // 1ms debounce

void setup() {
    Serial.begin(BAUD_RATE);
    while (!Serial) {
        ; // Wait for serial port (for Leonardo/Micro)
    }

    // Configure TTL input pin
    pinMode(TTL_PIN, INPUT);

    // Attach interrupt on rising edge
    attachInterrupt(digitalPinToInterrupt(TTL_PIN), ttl_isr, RISING);

    Serial.println("TTL_TRIGGER_READY");
    Serial.print("MAX_TRIGGERS=");
    Serial.println(MAX_TRIGGERS);
}

// Interrupt Service Routine - keep minimal for speed
void ttl_isr() {
    unsigned long now = micros();

    // Debounce check
    if (DEBOUNCE_US > 0 && (now - last_trigger_time) < DEBOUNCE_US) {
        return;
    }

    if (trigger_count < MAX_TRIGGERS) {
        trigger_times[trigger_count] = now;
        trigger_count++;
        new_trigger = true;
    }
    last_trigger_time = now;
}

void loop() {
    // Send new trigger immediately (real-time notification)
    if (new_trigger) {
        noInterrupts();
        int count = trigger_count;
        unsigned long t = trigger_times[count - 1];
        new_trigger = false;
        interrupts();

        Serial.print("T,");
        Serial.println(t);
    }

    // Process serial commands
    if (Serial.available() > 0) {
        char cmd = Serial.read();
        processCommand(cmd);
    }
}

void processCommand(char cmd) {
    unsigned long now;

    switch (cmd) {
        case 'S':  // Sync start
            now = micros();
            sync_start_time = now;
            Serial.print("S,");
            Serial.println(now);
            break;

        case 'E':  // Sync end - returns sync time then all triggers
            now = micros();
            sync_end_time = now;
            Serial.print("E,");
            Serial.println(now);

            // Send all stored triggers
            Serial.print("COUNT,");
            Serial.println(trigger_count);

            for (int i = 0; i < trigger_count; i++) {
                Serial.print("T,");
                Serial.println(trigger_times[i]);
            }
            Serial.println("END");
            break;

        case 'R':  // Reset
            noInterrupts();
            trigger_count = 0;
            new_trigger = false;
            last_trigger_time = 0;
            interrupts();
            sync_start_time = 0;
            sync_end_time = 0;
            Serial.println("RESET_OK");
            break;

        case 'C':  // Count
            Serial.print("COUNT,");
            Serial.println(trigger_count);
            break;

        case 'P':  // Ping
            Serial.println("PONG");
            break;

        case '\n':  // Ignore newlines
        case '\r':
            break;

        default:
            Serial.print("UNKNOWN_CMD:");
            Serial.println(cmd);
            break;
    }
}
