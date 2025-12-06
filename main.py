#!/usr/bin/env python3
import sys
import os
import threading
import time

sys.path.append(os.path.dirname(__file__))

from audio.recorder import AudioRecorder
from emotions.detector import EmotionDetector
from utils.logger import setup_logging
from utils.config import Config

class VoiceEmotionSystem:
    def __init__(self, use_real_model=True):
        print("Инициализация системы распознавания речи и эмоций...")
        
        self.logger = setup_logging()
        
        Config.print_config()
        
        self.recorder = AudioRecorder()

        if use_real_model and os.path.exists("emotions/real_emotion_model.pkl"):
            print("Использую обученную модель на реальных данных")
            self.emotion_detector = EmotionDetector(model_path="emotions/real_emotion_model.pkl")  
        else:
            print("Использую ДЕМО-модель")
            self.emotion_detector = EmotionDetector()  
        
        print("Система готова к работе!")
    
    def process_audio(self, audio_data):

        results = {}
        
        def recognize_speech():
            try:
                results['text'] = self.recorder.recognize_speech(audio_data)
            except Exception as e:
                print(f"Ошибка распознавания речи: {e}")
                results['text'] = None

        def detect_emotion():
            try:
                results['emotion'], results['confidence'] = self.emotion_detector.detect_emotion(audio_data)
            except Exception as e:
                print(f"Ошибка определения эмоций: {e}")
                results['emotion'] = "неопределено"
                results['confidence'] = 0.0
        
        speech_thread = threading.Thread(target=recognize_speech)
        emotion_thread = threading.Thread(target=detect_emotion)
        
        speech_thread.start()
        emotion_thread.start()
     
        speech_thread.join()
        emotion_thread.join()
        
        return results
    
    def run(self):
        try:
            # Показываем доступные микрофоны
            self.recorder.list_microphones()
            
            while True:
                print("\n" + "="*60)
                print("СИСТЕМА РАСПОЗНАВАНИЯ РЕЧИ И ЭМОЦИЙ")
                print("="*60)
                
                audio_data = self.recorder.record_audio()
                if audio_data is None:
                    print("Не удалось записать аудио. Попробуйте еще раз.")
                    continue
                
                print("Обрабатываю аудио (речь + эмоции)...")
                start_time = time.time()
                
                results = self.process_audio(audio_data)
                
                processing_time = time.time() - start_time

                self.show_results(results, processing_time)

                if not self.ask_continue():
                    break
                    
        except KeyboardInterrupt:
            print("\nЗавершение работы системы")
        except Exception as e:
            print(f"Критическая ошибка: {e}")
    
    def show_results(self, results, processing_time):
        print("\n" + "═" * 60)
        print("РЕЗУЛЬТАТЫ ОБРАБОТКИ:")
        print("═" * 60)
        
        if results['text']:
            print(f"РАСПОЗНАННАЯ РЕЧЬ: {results['text']}")
        else:
            print("Речь не распознана")
        
        emotion_display = results['emotion'].upper()
        if results['confidence'] > 0.7:
            emotion_display = f" {emotion_display}"
        elif results['confidence'] > 0.5:
            emotion_display = f" {emotion_display}"
        else:
            emotion_display = f" {emotion_display}"
        
        print(f"ЭМОЦИЯ: {emotion_display}")
        print(f"УВЕРЕННОСТЬ: {results['confidence']:.2f}")
        print(f"ВРЕМЯ ОБРАБОТКИ: {processing_time:.2f} сек.")
        
        if results['text'] and results['emotion'] != 'неопределено':
            print(f"   Чат-бот получит: '{results['text']}'")
            print(f"   С эмоциональным контекстом: '{results['emotion']}'")
            print(f"   Уверенность: {results['confidence']:.1%}")
        else:
            print("Недостаточно данных для передачи в чат-бот")
    
    def ask_continue(self):
        try:
            response = input("\nПродолжить работу? (y/n): ").strip().lower()
            return response in ['y', 'yes', 'д', 'да']
        except (KeyboardInterrupt, EOFError):
            return False

def main():
    # Проверяем, есть ли реальная модель
    use_real_model = os.path.exists("emotions/real_emotion_model.pkl")
    
    if use_real_model:
        print("Обнаружена модель, обученная на реальных данных!")
    else:
        print("ℹРеальная модель не найдена. Использую демо-версию.")
        print("   Для лучших резуль татов создайте датасет и обучите модель:")
        print("   1. python create_dataset.py")
        print("   2. python train_real_model.py")
    
    system = VoiceEmotionSystem(use_real_model=use_real_model)
    system.run()

if __name__ == "__main__":
    main()