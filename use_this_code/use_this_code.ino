#include <Wire.h>
#include <LiquidCrystal.h>

const int ledPin = LED_BUILTIN;  // 아두이노 내장 LED 핀 번호
const int mq3Pin = A5;           // MQ-3 센서핀을 아두이노 보드의 A5 핀으로 설정
const int piezo = 7;             // Piezo 부저 핀 번호

void Alcol_Sensor();
void CO2_Sensor();
void Sleep_Sensor();

void setup() {
  pinMode(ledPin, OUTPUT);  // LED 핀을 출력으로 설정
  Serial.begin(9600);       // 시리얼 통신 시작
  Serial.println("Arduino ready");  // 아두이노가 준비되었음을 알림
}

void loop() {
  Alcol_Sensor();
  CO2_Sensor();
  Sleep_Sensor();
  delay(1000);  // 1초 대기
}

void Sleep_Sensor() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');  // 시리얼 입력을 읽음
    input.trim();  // 입력 문자열의 앞뒤 공백 제거

    Serial.print("Received: ");  // 디버그 메시지 추가
    Serial.println(input);

    if (input == "SLEEP_TRUE") {
      Serial.println("Processing SLEEP_TRUE");  // 디버그 메시지
      digitalWrite(ledPin, HIGH);  // LED 켜기
      delay(2000);                 // 2초 동안 대기
      digitalWrite(ledPin, LOW);   // LED 끄기
      Sound_Shi5(0.2);
      delay(200);
      Sound_Do6(0.2);
      delay(200);
      
      // 처리 후 입력 문자열 초기화
      input = "";
    }
  }
}

void Alcol_Sensor() {
  int alcol_val = analogRead(mq3Pin); 
  Serial.print("alcol_value: ");
  Serial.println(alcol_val);   // MQ-3 센서 출력값을 시리얼 모니터로 출력
  
  if(alcol_val >= 600) {  // 센서 값이 700 이상이면
    Serial.println("Processing Alcol_Sensor (High)");
    Sound_Do5(0.2);
    delay(100);
    Sound_Shi5(0.2);
    delay(100);
    Sound_La5(0.2);
    delay(100);
  }
  else{  // 센서 값이 200 미만이면
    Serial.println("Processing Alcol_Sensor (Low)");
    delay(1000);
  }
}

void CO2_Sensor() {
  int co2_val = analogRead(A1);
  int co2ppm = map(co2_val, 0, 1023, 400, 5000); // map(value, fromLow, fromHigh, toLow, toHigh)
  
  Serial.print("co2_value: ");
  Serial.println(co2ppm);

  if(co2ppm > 920) {
    Serial.println("Processing CO2_Sensor (High)");
    Sound_Mi5(0.2);
    delay(100);
    Sound_Pa5(0.2);
    delay(100);
    Sound_Mi5(0.2);
    delay(100);
    Sound_Pa5(0.2);
    delay(100);
  } else{
    Serial.println("Processing CO2_Sensor (Low)");
    delay(1000);
  }
}

// 부저의 소리를 담당하는 함수
void Sound_Do5(double sec) {
  tone(piezo, 523); // 5옥타브 도
  delay(1000 * sec);
  noTone(piezo);
}

void Sound_Re5(double sec) {
  tone(piezo, 587); // 레
  delay(1000 * sec);
  noTone(piezo);
}

void Sound_Mi5(double sec) {
  tone(piezo, 659); // 미
  delay(1000 * sec);
  noTone(piezo);
}

void Sound_Pa5(double sec) {
  tone(piezo, 698); // 파
  delay(1000 * sec);
  noTone(piezo);
}

void Sound_Sol5(double sec) {
  tone(piezo, 784); // 솔
  delay(1000 * sec);
  noTone(piezo);
}

void Sound_La5(double sec) {
  tone(piezo, 880); // 라
  delay(1000 * sec);
  noTone(piezo);
}

void Sound_Shi5(double sec) {
  tone(piezo, 988); // 시
  delay(1000 * sec);
  noTone(piezo);
}

void Sound_Do6(double sec) {
  tone(piezo, 1046); // 6옥타브 도
  delay(1000 * sec);
  noTone(piezo);
}
