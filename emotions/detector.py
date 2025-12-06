"""
Детектор эмоций по голосу - ОБНОВЛЕННАЯ ВЕРСИЯ С ПОДДЕРЖКОЙ model_path
"""
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from .feature_extractor import AudioFeatureExtractor

class EmotionDetector:
    def __init__(self, model_path=None):  # ДОБАВИЛ ПАРАМЕТР model_path
        self.feature_extractor = AudioFeatureExtractor()
        self.model = None
        self.scaler = StandardScaler()
        
        # Базовые эмоции
        self.emotions = {
            0: 'нейтрально',
            1: 'радость', 
            2: 'грусть',
            3: 'злость',
            4: 'удивление'
        }
        
        # Загружаем модель если указан путь
        if model_path:
            self.load_model(model_path)
        else:
            # Или загружаем модель по умолчанию
            self.load_model()
    
    def create_simple_model(self):
        """Создает простую модель для классификации эмоций"""
        print("🤖 Создаю модель для определения эмоций...")
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        
        self.scaler = StandardScaler()
        print("✅ Базовая модель создана")
        return True
    
    def train_demo_model(self):
        """Создает демо-модель с правильной размерностью"""
        if self.model is None:
            self.create_simple_model()
        
        print("🎯 Создаю демо-данные для тестирования...")
        
        # Получаем актуальное количество признаков
        n_features = len(self.feature_extractor.get_feature_names())
        n_samples = 300
        
        print(f"   Количество признаков: {n_features}")
        print(f"   Количество образцов: {n_samples}")
        
        # Генерируем случайные признаки с правильной размерностью
        X_demo = np.random.randn(n_samples, n_features)
        
        # Создаем более "реалистичные" метки эмоций
        for i in range(n_samples):
            if i % 5 == 0:  # радость - высокие энергии
                X_demo[i, -5:] += 0.5
            elif i % 5 == 1:  # грусть - низкие энергии
                X_demo[i, -5:] -= 0.3
            elif i % 5 == 2:  # злость - высокие ZCR
                X_demo[i, -6] += 0.4
            elif i % 5 == 3:  # удивление - высокий темп
                X_demo[i, -3] += 50
        
        # Создаем метки эмоций
        y_demo = np.array([i % 5 for i in range(n_samples)])
        
        # Масштабируем и обучаем
        X_scaled = self.scaler.fit_transform(X_demo)
        self.model.fit(X_scaled, y_demo)
        
        train_accuracy = self.model.score(X_scaled, y_demo)
        print(f"✅ Демо-модель обучена (точность: {train_accuracy:.2f})")
        
        return True
    
    def detect_emotion(self, audio_data):
        """Определяет эмоцию по аудио данным"""
        try:
            if self.model is None:
                print("❌ Модель не загружена. Использую базовое определение...")
                return self._basic_emotion_detection(audio_data)
            
            # Извлекаем признаки
            features = self.feature_extractor.extract_from_audio_data(audio_data)
            
            if features is None:
                print("❌ Не удалось извлечь признаки")
                return "неопределено", 0.0
            
            # Проверяем размерность
            expected_features = len(self.feature_extractor.get_feature_names())
            if len(features) != expected_features:
                print(f"❌ Неправильная размерность: {len(features)} вместо {expected_features}")
                return "неопределено", 0.0
            
            # Масштабируем и предсказываем
            features_reshaped = features.reshape(1, -1)
            features_scaled = self.scaler.transform(features_reshaped)
            
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            emotion = self.emotions.get(prediction, 'неопределено')
            confidence = max(probabilities)
            
            print(f"🔍 Предсказание: {emotion} (уверенность: {confidence:.2f})")
            
            if confidence < 0.3:
                emotion = "неопределено"
                confidence = 0.0
            
            return emotion, confidence
            
        except Exception as e:
            print(f"❌ Ошибка определения эмоции: {e}")
            # Пробуем базовый метод как запасной вариант
            return self._basic_emotion_detection(audio_data)
    
    def _basic_emotion_detection(self, audio_data):
        """Базовая эвристическая оценка эмоций"""
        try:
            features = self.feature_extractor.extract_from_audio_data(audio_data)
            
            if features is None:
                return "нейтрально", 0.5
            
            feature_names = self.feature_extractor.get_feature_names()
            
            # Находим индексы нужных признаков
            rms_idx = feature_names.index("rms_energy")
            tempo_idx = feature_names.index("tempo")
            zcr_idx = feature_names.index("zero_crossing_rate")
            
            rms_energy = features[rms_idx]
            tempo = features[tempo_idx]
            zcr = features[zcr_idx]
            
            print(f"🔍 Базовый анализ: RMS={rms_energy:.3f}, Tempo={tempo:.1f}, ZCR={zcr:.3f}")
            
            # Улучшенные правила
            if rms_energy > 0.1 and tempo > 140 and zcr > 0.1:
                return "радость", 0.7
            elif rms_energy < 0.04 and tempo < 100 and zcr < 0.05:
                return "грусть", 0.7
            elif rms_energy > 0.15 and zcr > 0.15:
                return "злость", 0.6
            elif tempo > 160 and rms_energy > 0.08:
                return "удивление", 0.6
            elif 0.05 <= rms_energy <= 0.09 and 100 <= tempo <= 130:
                return "нейтрально", 0.8
            else:
                return "нейтрально", 0.5
                
        except Exception as e:
            print(f"❌ Ошибка базового определения: {e}")
            return "нейтрально", 0.5
    
    def save_model(self, model_path="emotions/emotion_model.pkl"):
        """Сохраняет модель"""
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'emotions': self.emotions,
                'feature_count': len(self.feature_extractor.get_feature_names())
            }
            
            joblib.dump(model_data, model_path)
            print(f"✅ Модель сохранена: {model_path}")
            print(f"   Количество признаков: {model_data['feature_count']}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка сохранения модели: {e}")
            return False
    
    def load_model(self, model_path="emotions/real_emotion_model.pkl"):  # ИЗМЕНИЛ ПУТЬ ПО УМОЛЧАНИЮ
        """Загружает модель с проверкой совместимости"""
        try:
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                
                # Проверяем совместимость признаков
                current_features = len(self.feature_extractor.get_feature_names())
                saved_features = model_data.get('feature_count', current_features)
                
                if current_features != saved_features:
                    print(f"⚠️  Несовместимость признаков: текущие {current_features}, сохраненные {saved_features}")
                    print("   Переобучаю модель с новой размерностью...")
                    self.create_simple_model()
                    self.train_demo_model()
                    self.save_model(model_path)
                else:
                    self.model = model_data['model']
                    self.scaler = model_data['scaler']
                    self.emotions = model_data.get('emotions', self.emotions)  # Используем сохраненные эмоции если есть
                    print(f"✅ Модель загружена: {model_path}")
                    print(f"   Количество признаков: {current_features}")
                    print(f"   Эмоции: {list(self.emotions.values())}")
                
                return True
            else:
                print(f"📝 Модель не найдена: {model_path}")
                print("   Создам базовую модель для демонстрации...")
                self.create_simple_model()
                self.train_demo_model()
                self.save_model(model_path)
                return False
                
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            print("   Создам базовую модель для демонстрации...")
            self.create_simple_model()
            self.train_demo_model()
            return False
    
    def get_model_info(self):
        """Возвращает информацию о модели"""
        if self.model is None:
            return "Модель не загружена"
        
        feature_count = len(self.feature_extractor.get_feature_names())
        emotion_count = len(self.emotions)
        
        return f"Модель: {feature_count} признаков, {emotion_count} эмоций"

# Тестируем детектор
if __name__ == "__main__":
    detector = EmotionDetector()
    print("🔧 Тестируем детектор эмоций...")
    print(f"Доступные эмоции: {list(detector.emotions.values())}")
    print(f"Количество признаков: {len(detector.feature_extractor.get_feature_names())}")
    print(detector.get_model_info())