const int TRIGGER_PIN = 2;
volatile bool triggered = false;
volatile unsigned long triggerTime = 0;

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    pinMode(TRIGGER_PIN, INPUT);
    attachInterrupt(digitalPinToInterrupt(TRIGGER_PIN), onTrigger, RISING);
    
    Serial.println("Ready - Waiting for trigger...");
}

void loop() {
    // 現在のピン状態を表示
    int state = digitalRead(TRIGGER_PIN);
    Serial.print("Pin state: ");
    Serial.println(state);
    
    if (triggered) {
        triggered = false;
        Serial.print(">>> Trigger detected at: ");
        Serial.println(triggerTime);
    }
    
    delay(500);
}

void onTrigger() {
    triggerTime = micros();
    triggered = true;
}

