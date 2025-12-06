"""
Извлечение признаков из аудио для определения эмоций - ПЕРЕПИСАННАЯ ВЕРСИЯ
"""
import librosa
import numpy as np
import os

class AudioFeatureExtractor:
    def __init__(self):
        self.sample_rate = 22050
        self.n_mfcc = 13
        self.hop_length = 512
        
    def extract_features(self, audio_path):
        """
        Извлекает ключевые признаки из аудиофайла для определения эмоций
        """
        try:
            print(f"🔍 Извлекаю признаки из аудио: {audio_path}")
            
            # Загружаем аудио
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=3.0)
            
            features = []
            
            # 1. MFCC (Мел-кепстральные коэффициенты)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc, 
                                      hop_length=self.hop_length)
            # Берем средние значения по времени для каждого коэффициента
            mfcc_mean = np.mean(mfcc, axis=1)
            features.extend(mfcc_mean.tolist())  # Явно преобразуем в список
            
            # 2. Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, 
                                               hop_length=self.hop_length)
            chroma_mean = np.mean(chroma, axis=1)
            features.extend(chroma_mean.tolist())
            
            # 3. Spectral Contrast
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, 
                                                       hop_length=self.hop_length)
            contrast_mean = np.mean(contrast, axis=1)
            features.extend(contrast_mean.tolist())
            
            # 4. Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(audio, 
                                                   hop_length=self.hop_length)
            zcr_mean = float(np.mean(zcr))  # Явно преобразуем в float
            features.append(zcr_mean)
            
            # 5. RMS Energy
            rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)
            rms_mean = float(np.mean(rms))
            features.append(rms_mean)
            
            # 6. Spectral Centroid
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, 
                                                       hop_length=self.hop_length)
            centroid_mean = float(np.mean(centroid))
            features.append(centroid_mean)
            
            # 7. Spectral Rolloff
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr,
                                                     hop_length=self.hop_length)
            rolloff_mean = float(np.mean(rolloff))
            features.append(rolloff_mean)
            
            # 8. Spectral Bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr,
                                                         hop_length=self.hop_length)
            bandwidth_mean = float(np.mean(bandwidth))
            features.append(bandwidth_mean)
            
            # 9. Tempo
            try:
                tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
                features.append(float(tempo))
            except:
                features.append(120.0)  # значение по умолчанию
            
            # 10. Harmonics and Percussive
            y_harmonic, y_percussive = librosa.effects.hpss(audio)
            harmonic_mean = float(np.mean(y_harmonic))
            percussive_mean = float(np.mean(y_percussive))
            features.append(harmonic_mean)
            features.append(percussive_mean)
            
            # Проверяем, что все признаки - числа
            self._validate_features(features)
            
            print(f"✅ Извлечено {len(features)} признаков")
            return np.array(features, dtype=np.float64)
            
        except Exception as e:
            print(f"❌ Ошибка извлечения признаков: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _validate_features(self, features):
        """Проверяет, что все признаки являются числами"""
        for i, feature in enumerate(features):
            if not isinstance(feature, (int, float, np.number)):
                print(f"⚠️  Признак {i} не число: {type(feature)} = {feature}")
                # Пытаемся исправить
                if isinstance(feature, (list, np.ndarray)):
                    features[i] = float(np.mean(feature))
                else:
                    features[i] = 0.0
    
    def extract_from_audio_data(self, audio_data, temp_file="temp_audio.wav"):
        """
        Извлекает признаки из аудио данных (из speech_recognition)
        """
        try:
            # Сохраняем аудио во временный файл
            with open(temp_file, "wb") as f:
                f.write(audio_data.get_wav_data())
            
            # Извлекаем признаки
            features = self.extract_features(temp_file)
            
            # Удаляем временный файл
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            return features
            
        except Exception as e:
            print(f"❌ Ошибка обработки аудио данных: {e}")
            return None
    
    def get_feature_names(self):
        """Возвращает названия признаков"""
        feature_names = []
        
        # MFCC (13 коэффициентов)
        for i in range(self.n_mfcc):
            feature_names.append(f"mfcc_{i+1}")
        
        # Chroma (12 нот)
        for i in range(12):
            feature_names.append(f"chroma_{i+1}")
            
        # Spectral Contrast (7 полос)
        for i in range(7):
            feature_names.append(f"spectral_contrast_{i+1}")
            
        # Остальные признаки
        feature_names.extend([
            "zero_crossing_rate", 
            "rms_energy", 
            "spectral_centroid",
            "spectral_rolloff",
            "spectral_bandwidth",
            "tempo",
            "harmonic",
            "percussive"
        ])
        
        return feature_names

    def debug_features(self, audio_path):
        """Отладочная функция для проверки признаков"""
        print(f"🔍 ДЕТАЛЬНАЯ ОТЛАДКА ПРИЗНАКОВ:")
        print(f"   Аудио файл: {audio_path}")
        
        features = self.extract_features(audio_path)
        if features is not None:
            print(f"   ✅ УСПЕХ: Извлечено {len(features)} признаков")
            print(f"   Тип: {type(features)}")
            print(f"   Форма: {features.shape}")
            print(f"   Размер: {features.size}")
            print(f"   Тип данных: {features.dtype}")
            print(f"   Минимальное значение: {np.min(features):.4f}")
            print(f"   Максимальное значение: {np.max(features):.4f}")
            print(f"   Среднее значение: {np.mean(features):.4f}")
            
            # Проверяем первые несколько признаков
            feature_names = self.get_feature_names()
            print(f"\n   ПЕРВЫЕ 10 ПРИЗНАКОВ:")
            for i in range(min(10, len(features))):
                print(f"     {feature_names[i]}: {features[i]:.4f}")
                
        else:
            print("   ❌ ПРОВАЛ: Не удалось извлечь признаки")
        
        return features

# Тестируем экстрактор
if __name__ == "__main__":
    extractor = AudioFeatureExtractor()
    print("🔧 Тестируем экстрактор признаков...")
    feature_names = extractor.get_feature_names()
    print(f"Всего признаков: {len(feature_names)}")
    print("Пример имен признаков:", feature_names[:5], "...")