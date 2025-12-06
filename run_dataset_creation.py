"""
Простой скрипт для создания датасета
"""
import os
import sys

# Добавляем текущую директорию в путь
sys.path.append(os.path.dirname(__file__))

from create_dataset import DatasetCreator

def main():
    print("🎯 СОЗДАНИЕ ДАТАСЕТА ЭМОЦИЙ")
    print("=" * 50)
    
    # Создаем организатор датасета
    creator = DatasetCreator()
    
    # Показываем текущую статистику
    print("\n📊 ТЕКУЩАЯ СТАТИСТИКА ДАТАСЕТА:")
    stats = creator.scan_existing_files()
    
    # Проверяем, есть ли папка с аудио
    audio_folder = "downloaded_audio"
    if not os.path.exists(audio_folder):
        print(f"\n❌ Папка '{audio_folder}' не найдена!")
        print("📋 Создайте папку 'downloaded_audio' и добавьте туда аудиофайлы.")
        print("   Названия файлов должны содержать эмоции:")
        print("   - happy_voice.wav")
        print("   - sad_speech.mp3")
        print("   - angry_man.m4a")
        print("   - neutral_tone.wav")
        return
    
    # Показываем файлы в папке
    audio_files = [f for f in os.listdir(audio_folder) if f.endswith(('.wav', '.mp3', '.m4a', '.flac'))]
    print(f"\n📁 Найдено файлов в '{audio_folder}': {len(audio_files)}")
    
    if len(audio_files) == 0:
        print("❌ В папке нет аудиофайлов!")
        print("   Добавьте файлы с расширениями: .wav, .mp3, .m4a, .flac")
        return
    
    print("📋 Файлы:")
    for file in audio_files[:10]:  # Показываем первые 10 файлов
        print(f"   - {file}")
    if len(audio_files) > 10:
        print(f"   ... и еще {len(audio_files) - 10} файлов")
    
    # Спрашиваем подтверждение
    response = input(f"\n🚀 Начать организацию {len(audio_files)} файлов? (y/n): ")
    if response.lower() not in ['y', 'yes', 'д', 'да']:
        print("❌ Отменено пользователем")
        return
    
    # Организуем файлы
    print("\n🔄 ОРГАНИЗАЦИЯ ФАЙЛОВ...")
    moved_count = creator.organize_files_by_name(audio_folder)
    
    if moved_count > 0:
        print(f"\n✅ УСПЕХ! Организовано {moved_count} файлов")
        
        # Извлекаем признаки
        response = input("\n🚀 Извлечь признаки из датасета? (y/n): ")
        if response.lower() in ['y', 'yes', 'д', 'да']:
            print("\n🔍 ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ...")
            df = creator.extract_features_from_dataset()
            
            if df is not None:
                print(f"\n🎉 ДАТАСЕТ СОЗДАН УСПЕШНО!")
                print("📋 Дальнейшие действия:")
                print("   1. Запусти обучение модели: python train_real_model.py")
                print("   2. Запусти систему: python main.py")
            else:
                print("❌ Не удалось создать датасет")
        else:
            print("ℹ️  Признаки не извлечены. Запустите позже:")
            print("   creator.extract_features_from_dataset()")
    else:
        print("❌ Не удалось организовать файлы")

if __name__ == "__main__":
    main()