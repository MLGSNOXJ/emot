import os
import shutil
import sys
sys.path.append(os.path.dirname(__file__))

from emotions.feature_extractor import AudioFeatureExtractor
import pandas as pd
import numpy as np

class DatasetCreator:
    def __init__(self):
        self.feature_extractor = AudioFeatureExtractor()
        self.dataset_path = "data"
        self.raw_path = os.path.join(self.dataset_path, "raw")
        self.processed_path = os.path.join(self.dataset_path, "processed")

        self.create_folder_structure()
    
    def create_folder_structure(self):
        emotions = ['neutral', 'happy', 'sad', 'angry', 'surprise', 'fear', 'disgust']
        
        for emotion in emotions:
            emotion_path = os.path.join(self.raw_path, emotion)
            os.makedirs(emotion_path, exist_ok=True)
            print(f"Создана папка: {emotion_path}")
        
        os.makedirs(self.processed_path, exist_ok=True)
        print(f"Структура датасета создана в: {self.dataset_path}")
    
    def scan_existing_files(self):
        print("\nСКАНИРУЮ СУЩЕСТВУЮЩИЕ ФАЙЛЫ...")
        
        emotions = ['neutral', 'happy', 'sad', 'angry', 'surprise', 'fear', 'disgust']
        stats = {}
        
        for emotion in emotions:
            emotion_path = os.path.join(self.raw_path, emotion)
            if os.path.exists(emotion_path):
                files = [f for f in os.listdir(emotion_path) if f.endswith(('.wav', '.mp3', '.m4a'))]
                stats[emotion] = len(files)
                print(f"   {emotion}: {len(files)} файлов")
            else:
                stats[emotion] = 0
                print(f"   {emotion}: папка не существует")
        
        total_files = sum(stats.values())
        print(f"\nВСЕГО ФАЙЛОВ: {total_files}")
        
        return stats
    
    def organize_files_by_name(self, source_folder):
        """
        Организует файлы из исходной папки по эмоциям на основе названий
        """
        if not os.path.exists(source_folder):
            print(f"Исходная папка не существует: {source_folder}")
            return
        
        print(f"\nОРГАНИЗАЦИЯ ФАЙЛОВ ИЗ: {source_folder}")
        
        emotion_keywords = {
            'neutral': ['neutral', 'нормальный', 'обычный', 'нейтрал', 'normal'],
            'happy': ['happy', 'радость', 'веселый', 'счастливый', 'joy', 'fun'],
            'sad': ['sad', 'грусть', 'печаль', 'грустный', 'sorrow'],
            'angry': ['angry', 'злость', 'злой', 'гнев', 'anger', 'mad'],
            'surprise': ['surprise', 'удивление', 'удивленный', 'surprised'],
            'fear': ['fear', 'страх', 'боязнь', 'испуг', 'scared', 'fright'],
            'disgust': ['disgust', 'отвращение', 'омерзение', 'disgusted']
        }
        
        moved_files = 0
        
        for filename in os.listdir(source_folder):
            if filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                file_path = os.path.join(source_folder, filename)
                filename_lower = filename.lower()
                
                # Ищем соответствующую эмоцию
                target_emotion = None
                for emotion, keywords in emotion_keywords.items():
                    for keyword in keywords:
                        if keyword in filename_lower:
                            target_emotion = emotion
                            break
                    if target_emotion:
                        break
                
                # Если эмоция не определена, используем "neutral"
                if not target_emotion:
                    target_emotion = 'neutral'
                    print(f"Неопределенная эмоция для файла: {filename} -> neutral")
                
                target_folder = os.path.join(self.raw_path, target_emotion)
                target_path = os.path.join(target_folder, filename)
                
                shutil.copy2(file_path, target_path)
                print(f"{filename} -> {target_emotion}")
                moved_files += 1
        
        print(f"\nПЕРЕМЕЩЕНО ФАЙЛОВ: {moved_files}")
        return moved_files
    
    def extract_features_from_dataset(self):
        print("\nИЗВЛЕЧЕНИЕ ПРИЗНАКОВ ИЗ ДАТАСЕТА...")
        
        emotions = ['neutral', 'happy', 'sad', 'angry', 'surprise', 'fear', 'disgust']
        features_list = []
        labels_list = []
        file_names = []
        
        total_files = 0
        successful_extractions = 0
        
        for emotion in emotions:
            emotion_path = os.path.join(self.raw_path, emotion)
            
            if not os.path.exists(emotion_path):
                print(f"Папка не существует: {emotion_path}")
                continue
            
            files = [f for f in os.listdir(emotion_path) if f.endswith(('.wav', '.mp3', '.m4a', '.flac'))]
            total_files += len(files)
            
            print(f"\nОбрабатываю {emotion}: {len(files)} файлов")
            
            for i, filename in enumerate(files):
                file_path = os.path.join(emotion_path, filename)
                
                print(f"   {i+1}/{len(files)}: {filename}", end="")
                
                features = self.feature_extractor.extract_features(file_path)
                
                if features is not None:
                    features_list.append(features)
                    labels_list.append(emotion)
                    file_names.append(filename)
                    successful_extractions += 1
                    print(" Да")
                else:
                    print(" Нет")
        
        if features_list:
            feature_names = self.feature_extractor.get_feature_names()
            df_features = pd.DataFrame(features_list, columns=feature_names)
            df_features['emotion'] = labels_list
            df_features['filename'] = file_names

            csv_path = os.path.join(self.processed_path, "emotion_dataset.csv")
            df_features.to_csv(csv_path, index=False)
            
            print(f"\nДАТАСЕТ СОХРАНЕН!")
            print(f"   Файл: {csv_path}")
            print(f"   Записей: {successful_extractions}/{total_files}")
            print(f"   Признаков: {len(feature_names)}")
            print(f"   Эмоций: {len(set(labels_list))}")
            
            print(f"\nСТАТИСТИКА ПО ЭМОЦИЯМ:")
            emotion_counts = df_features['emotion'].value_counts()
            for emotion, count in emotion_counts.items():
                print(f"   {emotion}: {count} записей")
            
            return df_features
        else:
            print("Не удалось извлечь признаки ни из одного файла")
            return None
    
    def show_instructions(self):
        """Показывает инструкции по использованию"""
        print("\n" + "="*60)
        print("ИНСТРУКЦИЯ ПО СОЗДАНИЮ ДАТАСЕТА")
        print("="*60)
        print("1. Скачайте аудиофайлы с разными эмоциями из интернета")
        print("2. Сохраните их в одну папку (например: downloaded_audio/)")
        print("3. Назовите файлы так, чтобы в названии была эмоция:")
        print("   Примеры названий:")
        print("   - happy_voice_1.wav")
        print("   - sad_speech_2.mp3") 
        print("   - angry_man_1.m4a")
        print("   - neutral_tone_3.wav")
        print("4. Запустите организацию файлов:")
        print("   creator.organize_files_by_name('downloaded_audio')")
        print("5. Извлеките признаки:")
        print("   creator.extract_features_from_dataset()")
        print("="*60)

def create_dataset_from_folder(source_folder):
    creator = DatasetCreator()
    creator.organize_files_by_name(source_folder)
    return creator.extract_features_from_dataset()

def show_dataset_stats():
    creator = DatasetCreator()
    return creator.scan_existing_files()

if __name__ == "__main__":
    creator = DatasetCreator()
    creator.show_instructions()
    
    creator.scan_existing_files()
    
    print("\nДЛЯ СОЗДАНИЯ ДАТАСЕТА ВЫПОЛНИТЕ:")
    print("1. creator.organize_files_by_name('ваша_папка_с_аудио')")
    print("2. creator.extract_features_from_dataset()")