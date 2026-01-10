#include <SoftwareSerial.h>
SoftwareSerial BTSerial(8, 9);   //tx, rx
const int emgPin = A0;
int emgValue = 0;


void setup() {
  // put your setup code here, to run once:

  Serial.begin(9600);
  BTSerial.begin(9600);


  Serial.println("Initialisation");
}

void loop() {
  // put your main code here, to run repeatedly:

  emgValue = analogRead(emgPin);

  BTSerial.print("Emg signals: ");
  BTSerial.println(emgValue);

  Serial.println(emgValue);
  delay(10); //100Hz
}
