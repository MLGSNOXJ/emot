import os
import sys
sys.path.append(os.path.dirname(__file__))

from train_real_model import RealEmotionTrainer

def main():
    print("ОБУЧЕНИЕ МОДЕЛИ НА РЕАЛЬНЫХ ДАННЫХ")
    print("=" * 50)
    
    trainer = RealEmotionTrainer()
    success = trainer.train_complete_pipeline()
    
    if success:
        print("\nМОДЕЛЬ ОБУЧЕНА!")
        print("Дальнейшие действия:")
        print("   1. Запусти систему: python main.py")
        print("   2. Протестируй на разных голосах и эмоциях")
    else:
        print("\nОБУЧЕНИЕ НЕ УДАЛОСЬ")
        print("Возможные причины:")
        print("   - Недостаточно данных в датасете")
        print("   - Проблемы с качеством аудиофайлов")
        print("   - Неправильные названия файлов")

if __name__ == "__main__":
    main()