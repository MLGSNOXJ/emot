import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

sys.path.append(os.path.dirname(__file__))

class RealEmotionTrainer:
    def __init__(self):
        self.dataset_path = "data/processed/emotion_dataset.csv"
        self.model_path = "emotions/real_emotion_model.pkl"
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        
    def load_dataset(self):
        """Загружает датасет из CSV файла"""
        if not os.path.exists(self.dataset_path):
            print(f"Файл датасета не найден: {self.dataset_path}")
            print("   Сначала создайте датасет с помощью create_dataset.py")
            return None
        
        print("Загружаю датасет...")
        df = pd.read_csv(self.dataset_path)
        
        print(f"Датасет загружен: {len(df)} записей")
        print(f"   Эмоции: {df['emotion'].unique()}")
        print(f"   Признаков: {len(df.columns) - 2}")  
        
        print("\nРАСПРЕДЕЛЕНИЕ ПО ЭМОЦИЯМ:")
        emotion_counts = df['emotion'].value_counts()
        for emotion, count in emotion_counts.items():
            print(f"   {emotion}: {count} записей ({count/len(df)*100:.1f}%)")
        
        return df
    
    def prepare_data(self, df):
        feature_columns = [col for col in df.columns if col not in ['emotion', 'filename']]
        X = df[feature_columns].values
        y = df['emotion'].values
        

        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Данные подготовлены:")
        print(f"   Признаки: {X.shape}")
        print(f"   Метки: {len(np.unique(y_encoded))} классов")
        print(f"   Соответствие меток: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        return X, y_encoded, feature_columns
    
    def train_model(self, X, y, test_size=0.3): 
        if len(X) < 50:
            test_size = 0.2  
        elif len(X) < 100:
            test_size = 0.25
        

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"РАЗДЕЛЕНИЕ ДАННЫХ:")
        print(f"   Обучающая выборка: {X_train.shape[0]} записей")
        print(f"   Тестовая выборка: {X_test.shape[0]} записей")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if len(X) < 100:
            print("🔧 Использую Random Forest (лучше для маленьких датасетов)")
            self.model = RandomForestClassifier(
                n_estimators=50, 
                max_depth=10,     
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train_scaled, y_train)
            
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Random Forest точность: {accuracy:.3f}")
            
        else:
            # Для больших датасетов 
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                'SVM': SVC(kernel='rbf', probability=True, random_state=42)
            }
            
            best_model = None
            best_accuracy = 0
            best_model_name = ""
            
            for name, model in models.items():
                print(f"\nОбучаю {name}...")
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                print(f"{name} точность: {accuracy:.3f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_model_name = name
            
            self.model = best_model
            print(f"\nЛУЧШАЯ МОДЕЛЬ: {best_model_name} (точность: {best_accuracy:.3f})")
            accuracy = best_accuracy
    
        print(f"\nДЕТАЛЬНЫЙ ОТЧЕТ:")
        y_pred = self.model.predict(X_test_scaled)
        print(classification_report(y_test, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        return accuracy
    
    def save_model(self):
        if self.model is None:
            print("Модель не обучена")
            return False
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_count': len([col for col in pd.read_csv(self.dataset_path).columns 
                                if col not in ['emotion', 'filename']])
        }
        
        joblib.dump(model_data, self.model_path)
        print(f"Модель сохранена: {self.model_path}")
        return True
    
    def train_complete_pipeline(self):
        print("ЗАПУСК ОБУЧЕНИЯ НА РЕАЛЬНЫХ ДАННЫХ")
        print("=" * 50)
        
        df = self.load_dataset()
        if df is None:
            return False
        
        if len(df) < 10:
            print(f"Очень мало данных: {len(df)} записей")
            print("   Модель может работать неточно, но попробуем обучить...")
        elif len(df) < 30:
            print(f"Умеренное количество данных: {len(df)} записей")
            print("   Модель будет обучаться, точность может быть средней")
        else:
            print(f"Хорошее количество данных: {len(df)} записей")

        X, y, feature_columns = self.prepare_data(df)
        
        accuracy = self.train_model(X, y)
        
        # Сохраняем модель ВСЕГДА (даже если точность низкая)
        self.save_model()
        print(f"\nОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print(f"   Точность модели: {accuracy:.3f}")
        print(f"   Модель сохранена и готова к использованию")
        
        if accuracy < 0.4:
            print(f"\nСОВЕТ: Точность низкая. Попробуйте:")
            print("   - Добавить больше аудио файлов (хотя бы 50-100)")
            print("   - Убедиться, что файлы хорошего качества")
            print("   - Проверить правильность названий файлов")
        elif accuracy < 0.6:
            print(f"\nСОВЕТ: Точность средняя. Можете:")
            print("   - Добавить еще данных для улучшения")
            print("   - Уже можно тестировать систему")
        else:
            print(f"\nОтличная точность! Система готова к работе!")
    
        return True

def main():
    trainer = RealEmotionTrainer()
    success = trainer.train_complete_pipeline()
    
    if success:
        print("\nДАЛЕЕ:")
        print("1. Запустите систему: python main.py")
        print("2. Тестируйте на реальных голосах!")
        print("3. Если точность низкая - добавьте больше данных и переобучите")
    else:
        print("\nОБУЧЕНИЕ НЕ УДАЛОСЬ")
        print("   Проверьте наличие данных и попробуйте снова")

if __name__ == "__main__":
    main()