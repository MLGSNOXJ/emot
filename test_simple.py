"""
Простой тест для проверки работы системы с эмоциями
"""
import sys
import os

# Добавляем текущую директорию в путь
sys.path.append(os.path.dirname(__file__))

def test_emotion_imports():
    """Тест импортов для эмоций"""
    print("🧪 Тестируем импорты для эмоций...")
    
    try:
        from emotions.detector import EmotionDetector
        print("✅ EmotionDetector - импорт успешен")
    except ImportError as e:
        print(f"❌ EmotionDetector - ошибка: {e}")
        return False
    
    try:
        from emotions.feature_extractor import AudioFeatureExtractor
        print("✅ AudioFeatureExtractor - импорт успешен")
    except ImportError as e:
        print(f"❌ AudioFeatureExtractor - ошибка: {e}")
        return False
    
    return True

def test_emotion_detector():
    """Тест создания детектора эмоций"""
    print("\n🧪 Тестируем создание EmotionDetector...")
    
    try:
        from emotions.detector import EmotionDetector
        detector = EmotionDetector()
        print("✅ EmotionDetector создан успешно")
        print(f"   Доступные эмоции: {list(detector.emotions.values())}")
        return True
    except Exception as e:
        print(f"❌ Ошибка создания EmotionDetector: {e}")
        return False

def test_feature_extractor():
    """Тест экстрактора признаков"""
    print("\n🧪 Тестируем AudioFeatureExtractor...")
    
    try:
        from emotions.feature_extractor import AudioFeatureExtractor
        extractor = AudioFeatureExtractor()
        feature_names = extractor.get_feature_names()
        print(f"✅ AudioFeatureExtractor создан успешно")
        print(f"   Количество признаков: {len(feature_names)}")
        return True
    except Exception as e:
        print(f"❌ Ошибка создания AudioFeatureExtractor: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Запуск тестов системы с эмоциями...")
    
    # Проверяем существование файлов
    files_to_check = [
        'emotions/detector.py',
        'emotions/feature_extractor.py',
        'audio/recorder.py',
        'main.py'
    ]
    
    print("\n📁 Проверка файлов...")
    for file in files_to_check:
        if os.path.exists(file):
            print(f"✅ {file} - существует")
        else:
            print(f"❌ {file} - не найден")
    
    # Тестируем импорты
    emotion_imports_ok = test_emotion_imports()
    detector_ok = test_emotion_detector()
    extractor_ok = test_feature_extractor()
    
    print(f"\n📊 Результаты тестов эмоций:")
    print(f"   Импорты: {'✅ Успешно' if emotion_imports_ok else '❌ Ошибки'}")
    print(f"   Детектор: {'✅ Успешно' if detector_ok else '❌ Ошибки'}")
    print(f"   Экстрактор: {'✅ Успешно' if extractor_ok else '❌ Ошибки'}")
    
    if emotion_imports_ok and detector_ok and extractor_ok:
        print("\n🎉 Все тесты пройдены! Система эмоций готова!")
        print("🚀 Запускайте main.py для работы с распознаванием речи и эмоций")
    else:
        print("\n❌ Есть проблемы с системой эмоций.")