"""
Модуль для записи и обработки аудио - УПРОЩЕННАЯ ВЕРСИЯ
"""
import speech_recognition as sr
import logging

class AudioRecorder:
    """Класс для работы с аудио записью"""
    
    def __init__(self):
        self.logger = logging.getLogger('AudioRecorder')
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Настройка распознавателя
        self.recognizer.energy_threshold = 1000
        self.recognizer.pause_threshold = 0.8
        self.recognizer.dynamic_energy_threshold = False
        
        self._setup_microphone()
    
    def _setup_microphone(self):
        """Настройка микрофона и калибровка"""
        try:
            print("🎧 Калибрую микрофон...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print("✅ Микрофон настроен")
        except Exception as e:
            print(f"❌ Ошибка настройки микрофона: {e}")
            raise
    
    def list_microphones(self):
        """Показать доступные микрофоны"""
        print("🎧 Доступные аудиоустройства:")
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"  {index}: {name}")
    
    def record_audio(self):
        """
        Запись аудио с микрофона
        """
        try:
            print("🎤 Начинаю запись...")
            
            with self.microphone as source:
                # Слушаем микрофон
                audio = self.recognizer.listen(
                    source, 
                    timeout=10,
                    phrase_time_limit=10
                )
            
            print("✅ Аудио записано, распознаю...")
            return audio
            
        except sr.WaitTimeoutError:
            print("⏰ Время ожидания истекло")
            return None
        except Exception as e:
            print(f"❌ Ошибка записи: {e}")
            return None
    
    def recognize_speech(self, audio_data):
        """
        Распознавание речи из аудио данных
        """
        try:
            text = self.recognizer.recognize_google(audio_data, language="ru-RU")
            print(f"✅ Речь распознана: '{text}'")
            return text
            
        except sr.UnknownValueError:
            print("❌ Речь не распознана")
            return None
        except sr.RequestError as e:
            print(f"❌ Ошибка сервиса распознавания: {e}")
            return None
    
    def simple_record_and_recognize(self):
        """
        Простая запись и распознавание в одном методе
        """
        print("\n" + "="*50)
        print("🎤 СИСТЕМА РАСПОЗНАВАНИЯ РЕЧИ")
        print("="*50)
        print("Инструкция:")
        print("1. Нажмите Enter чтобы начать запись")
        print("2. Говорите четко в микрофон")
        print("3. Система автоматически остановит запись")
        print("4. Для выхода нажмите Ctrl+C")
        print("="*50)
        
        try:
            input("\nНажмите Enter чтобы начать говорить...")
            
            # Записываем аудио
            audio = self.record_audio()
            if audio is None:
                return None
            
            # Распознаем речь
            text = self.recognize_speech(audio)
            return text
            
        except KeyboardInterrupt:
            print("👋 Завершение работы по запросу пользователя")
            return "exit"
        except Exception as e:
            print(f"❌ Неожиданная ошибка: {e}")
            return None