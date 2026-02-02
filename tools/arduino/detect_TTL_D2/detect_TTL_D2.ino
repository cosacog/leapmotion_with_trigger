const int TRIGGER_PIN = 2;  // 割り込み対応ピン
volatile bool triggered = false;
volatile unsigned long triggerTime = 0;

void setup() {
    Serial.begin(115200);
    pinMode(TRIGGER_PIN, INPUT);
    
    // 立ち上がりエッジで割り込み
    attachInterrupt(digitalPinToInterrupt(TRIGGER_PIN), onTrigger, RISING);
    
    Serial.println("Ready");
}

void loop() {
    if (triggered) {
        triggered = false;
        Serial.print("Trigger detected at: ");
        Serial.println(triggerTime);
        
        // ここにトリガー検出時の処理を追加
    }
}

void onTrigger() {
    triggerTime = micros();
    triggered = true;
}
