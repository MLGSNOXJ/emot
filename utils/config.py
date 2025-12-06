"""
Загрузка конфигурации приложения - УПРОЩЕННАЯ ВЕРСИЯ
"""
import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

class Config:
    """Класс для хранения конфигурации"""
    
    # Настройки аудио
    AUDIO_SAMPLE_RATE = int(os.getenv('AUDIO_SAMPLE_RATE', '16000'))
    AUDIO_CHUNK_SIZE = int(os.getenv('AUDIO_CHUNK_SIZE', '1024'))
    
    # Настройки распознавания речи
    SPEECH_LANGUAGE = os.getenv('SPEECH_LANGUAGE', 'ru-RU')
    SPEECH_TIMEOUT = int(os.getenv('SPEECH_TIMEOUT', '10'))
    
    # Логирование
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def print_config(cls):
        """Печать текущей конфигурации"""
        print("🔧 Текущая конфигурация:")
        print(f"   Аудио: sample_rate={cls.AUDIO_SAMPLE_RATE}, chunk_size={cls.AUDIO_CHUNK_SIZE}")
        print(f"   Речь: language={cls.SPEECH_LANGUAGE}, timeout={cls.SPEECH_TIMEOUT}")
        print(f"   Логи: level={cls.LOG_LEVEL}")