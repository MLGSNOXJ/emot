"""
Простая система логирования - УПРОЩЕННАЯ ВЕРСИЯ
"""
import logging
import os
from datetime import datetime

def setup_logging():
    """Настройка логирования"""
    # Создаем папку для логов если её нет
    os.makedirs('logs', exist_ok=True)
    
    # Формат логов
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Базовая настройка
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(f'logs/assistant_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )
    
    print("✅ Логирование настроено")
    
    return logging.getLogger('VoiceAssistant')