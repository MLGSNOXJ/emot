"""
Скрипт для отладки системы эмоций
"""
import sys
import os

# Добавляем текущую директорию в путь
sys.path.append(os.path.dirname(__file__))

from emotions.feature_extractor import AudioFeatureExtractor
from emotions.detector import EmotionDetector
import speech_recognition as sr

def test_feature_extraction():
    """Тестируем извлечение признаков"""
    print("🧪 ТЕСТ ИЗВЛЕЧЕНИЯ ПРИЗНАКОВ")
    print("=" * 50)
    
    extractor = AudioFeatureExtractor()
    
    # Создаем тестовое аудио
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Запиши тестовую фразу для анализа...")
        audio = r.listen(source, timeout=10, phrase_time_limit=5)
        
        # Сохраняем аудио
        test_file = "test_audio.wav"
        with open(test_file, "wb") as f:
            f.write(audio.get_wav_data())
        
        print(f"✅ Аудио сохранено: {test_file}")
    
    # Тестируем извлечение признаков
    features = extractor.debug_features(test_file)
    
    if features is not None:
        print(f"🎯 УСПЕХ: Извлечено {len(features)} признаков")
    else:
        print("❌ ПРОВАЛ: Не удалось извлечь признаки")
    
    # Удаляем тестовый файл
    if os.path.exists(test_file):
        os.remove(test_file)
    
    return features is not None

def test_emotion_detection():
    """Тестируем определение эмоций"""
    print("\n🧪 ТЕСТ ОПРЕДЕЛЕНИЯ ЭМОЦИЙ")
    print("=" * 50)
    
    detector = EmotionDetector()
    
    # Создаем тестовое аудио
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Запиши фразу для определения эмоции...")
        print("💡 Попробуй сказать с разной интонацией!")
        audio = r.listen(source, timeout=10, phrase_time_limit=5)
    
    # Тестируем определение эмоции
    emotion, confidence = detector.detect_emotion(audio)
    
    print(f"🎭 РЕЗУЛЬТАТ: {emotion} (уверенность: {confidence:.2f})")
    
    return emotion != "неопределено"

if __name__ == "__main__":
    print("🔧 ЗАПУСК ОТЛАДКИ СИСТЕМЫ ЭМОЦИЙ")
    print("=" * 60)
    
    # Тестируем извлечение признаков
    features_ok = test_feature_extraction()
    
    # Тестируем определение эмоций
    emotion_ok = test_emotion_detection()
    
    print("\n" + "=" * 60)
    print("📊 ИТОГИ ОТЛАДКИ:")
    print(f"   Извлечение признаков: {'✅ УСПЕХ' if features_ok else '❌ ПРОВАЛ'}")
    print(f"   Определение эмоций: {'✅ УСПЕХ' if emotion_ok else '❌ ПРОВАЛ'}")
    
    if features_ok and emotion_ok:
        print("\n🎉 СИСТЕМА ЭМОЦИЙ РАБОТАЕТ КОРРЕКТНО!")
    else:
        print("\n❌ ЕСТЬ ПРОБЛЕМЫ. Проверьте ошибки выше.")