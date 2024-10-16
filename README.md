# Прокси для использования GigaChat вместо ChatGPT

Данный проект представляет собой HTTP-прокси, который позволяет подменять использование ChatGPT на GigaChat в тех случаях, когда возможна настройка URL для взаимодействия с ChatGPT. Утилита поддерживает все основные функции взаимодействия с чат-моделями, включая поддержку работы с функциями и асинхронную обработку запросов.

## Основные возможности

	•	Полная замена: утилита подменяет использование ChatGPT на GigaChat, позволяя использовать все его функции.
	•	Поддержка функций: корректно обрабатываются вызовы функций через API, включая передачу и выполнение функций с аргументами.
	•	Асинхронный HTTP-прокси: поддерживает многопоточную обработку запросов для эффективной работы с большим количеством клиентов.
	•	Простота настройки: настройка хоста и порта через аргументы командной строки или переменные окружения.
	•	Поддержка логирования: режим подробного вывода запросов и ответов для отладки.

## Установка

1. Установите библиотеку
```pip install git+https://github.com/Rai220/Gpt2Giga.git```

2. Настройте переменные окружения, создайте файл .env в корне проекта и укажите необходимые параметры для доступа к GigaChat.

```
GIGACHAT_USER=
GIGACHAT_PASSWORD=
GIGACHAT_BASE_URL=
```

## Использование

Запуск прокси-сервера:

```python proxy.py --host <host> --port <port> --verbose <True/False>```

### Пример запуска с настройками по умолчанию:

```python proxy.py```

После запуска сервер будет слушать указанный хост и порт и перенаправлять все запросы, адресованные ChatGPT, на GigaChat.

### Пример использования

Приложение, которое взаимодействует с ChatGPT через настраиваемый URL, можно перенаправить на прокси, чтобы оно начало работать с GigaChat. Для этого достаточно указать URL сервера, запущенного через данную утилиту.

### Переменные окружения

Вы можете настроить следующие переменные окружения через файл .env:

	•	PROXY_HOST: хост, который будет прослушивать прокси (по умолчанию: localhost).
	•	PROXY_PORT: порт для работы прокси (по умолчанию: 8090).
	•	GPT2GIGA_VERBOSE: режим вывода подробной информации о запросах и ответах (по умолчанию: True).

# Лицензия

Этот проект распространяется под лицензией MIT. См. LICENSE для получения подробной информации.

Этот README файл описывает работу вашей утилиты и содержит все необходимые инструкции для пользователей.
